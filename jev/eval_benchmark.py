import os as _os
JEV_DATA = _os.environ.get('JEV_DATA', _os.path.expanduser('~/jev-data'))
STRATA = _os.environ.get('STRATA_ROOT', _os.path.expanduser('~/strata'))
JEV_PY = _os.environ.get('JEV_PY', 'python3')
"""JevBench public 231 re-measured with the frozen head on the abliterated backbone.

Same benchmark revision, same frozen prompt builder, same scorer, same head and normalization as
the 2026-10-17 run (198/231 = 85.71%); the only change is which backbone answers:
qwen3.8-flash-next-iq3_s-abliterated.

Honest scope note: here `latent_ids` come from the LIVE tokenizer pack, so this measurement is
end-to-end (live tokenizer + live backbone + frozen head).  The archive-replay harness
(`eval_replay.py`) sends the archived ids and therefore isolates the backbone alone.

No decode fallback and no threshold: head-only accuracy.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TASKS = Path(_os.environ.get('JEV_JEVBENCH_TASKS', str(HERE.parent / 'benchmarks/jevbench-public231/tasks.jsonl')))
SCORING = HERE / 'upstream_scoring.py'

from .head_loader import MeasurementHead                        # noqa: E402
from .common import volunteer_for_oom, digest                        # noqa: E402
from .jevbench import build_item, softmax                             # noqa: E402
from .upstream_scoring import score_task                             # noqa: E402

TAPS = '15,23,29,35,41,-1'


def score_one(probs, labels, expected, qtype):
    """Official JevBench scoring, via the frozen upstream scorer (identical to 2026-10-17)."""
    from types import SimpleNamespace
    task = SimpleNamespace(labels=labels, expected=expected, question={'type': qtype})
    outcome = score_task({k: float(v) for k, v in probs.items()}, task)
    return {k: outcome.get(k) for k in ('valid', 'strict_valid', 'renormalized', 'correct',
                                      'predicted', 'ordinal_ev')}


def subset_metrics(records, subset, key):
    """Score each distribution, keeping missing/error items in the denominator.

    The legacy helper expects a generic `predicted` field and counts only the
    supplied rows as planned. Neither assumption is safe for a partial live run.
    """
    rows = [r for r in records if r['id'] in subset]
    valid, bins, brier = [], {}, []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        scored = score_one(row[key], row['labels'], row['gold_label'], row['type'])
        if not scored['valid']:
            continue
        valid.append((row, scored))
        if row['gold_label'] is None:
            continue
        confidence = max(row[key].values())
        bins.setdefault(min(9, int(confidence * 10)), []).append(
            (confidence, float(bool(scored['correct']))))
        brier.append(sum((p - float(label == row['gold_label'])) ** 2
                         for label, p in row[key].items()))
    count = sum(len(bucket) for bucket in bins.values())
    calibration_error = sum(len(bucket) / count * abs(
        np.mean([p for p, _ in bucket]) - np.mean([hit for _, hit in bucket]))
        for bucket in bins.values()) if count else None
    correct = sum(bool(scored['correct']) for _, scored in valid)
    head_ms = [r['head_ms'] for r, _ in valid]
    path_ms = [r['head_path_ms'] for r, _ in valid]
    return {'planned': len(subset), 'recorded': len(rows), 'valid': len(valid),
            'errors': len(subset) - len(valid), 'missing': len(subset) - len(rows),
            'correct': correct,
            'accuracy_all_planned': correct / len(subset) if subset else None,
            'accuracy_valid': correct / len(valid) if valid else None,
            'top_label_ece': float(calibration_error) if calibration_error is not None else None,
            'brier': float(np.mean(brier)) if brier else None,
            'head_ms_p50': float(np.percentile(head_ms, 50)) if head_ms else None,
            'feature_plus_head_p50_ms': float(np.percentile(path_ms, 50)) if path_ms else None,
            'feature_plus_head_p95_ms': float(np.percentile(path_ms, 95)) if path_ms else None}


def past(deadline: str) -> bool:
    return bool(deadline) and datetime.now().strftime('%Y-%m-%d %H:%M:%S') >= deadline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['criterion_as_option', 'label_as_option'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--identity', default=str(HERE / 'runtime-identity.json'))
    parser.add_argument('--tokenizer-dir', help='served tokenizer pack; defaults to the recorded identity path')
    parser.add_argument('--exclusive', action='store_true',
                        help='use the exclusive readout path instead of shared-slots')
    parser.add_argument('--confirm-window', action='store_true',
                        help='required: acknowledges the live engine serves this run')
    parser.add_argument('--deadline', help='stop after this local time, keep partial evidence')
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if not args.confirm_window:
        raise SystemExit('refusing to start without --confirm-window')
    volunteer_for_oom()
    torch.set_num_threads(2)

    tasks = [json.loads(line) for line in TASKS.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(tasks) != 231:
        raise ValueError('expected the 231 public JevBench decisions, got %d' % len(tasks))
    if args.limit:
        tasks = tasks[:args.limit]
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate benchmark task ids')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = json.loads(Path(args.identity).read_text(encoding='utf-8'))
    tokenizer_dir = args.tokenizer_dir or identity.get('running_config', {}).get('tokenizer_dir')
    if not tokenizer_dir:
        parser.error('pass --tokenizer-dir or record running_config.tokenizer_dir in the identity')
    head = MeasurementHead(base=args.base, identity=args.identity, parallel=not args.exclusive,
                              tokenizer_dir=tokenizer_dir)
    temperature = head.temperature or 1.0
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    items, requests, lengths = {}, {}, {}
    for task in tasks:
        item = build_item(task, args.mode)
        request = head.request_for(item)
        items[task['id']] = item
        requests[task['id']] = request
        lengths[task['id']] = len(request['latent_ids'])
    if args.preflight:
        print(json.dumps({'n': len(tasks), 'mode': args.mode,
                          'tokens_min': min(lengths.values()),
                          'tokens_p50': float(np.median(list(lengths.values()))),
                          'tokens_max': max(lengths.values()),
                          'total_tokens': sum(lengths.values()),
                          'model': identity.get('running_config', {}).get('model_name'),
                          'tokenizer_dir': str(tokenizer_dir)}), flush=True)
        return 0
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (previous.get('live_identity') != identity
                or previous.get('prompt_protocol', {}).get('option_mode') != args.mode
                or previous.get('prompt_tokens_by_id') != lengths):
            raise ValueError('existing benchmark manifest differs from this identity/mode/task selection')
    else:
        manifest_path.write_text(json.dumps({
            'schema': 'jevbench-open-current-head-abliterated-v1',
            'benchmark': 'JevBench public 231 (original 72 + easy 48 + hard 111)',
            'benchmark_revision': '7625ecb1b075cfe4d09d34141d353325825eea1d',
            'benchmark_source_sha256': {'tasks.jsonl': digest(TASKS),
                                        'scoring.py': digest(SCORING)},
            'live_identity': identity,
            'identity_sha256': digest(args.identity),
            'tokenizer_dir': str(tokenizer_dir),
            'tokenizer_note': 'latent_ids are produced by the live tokenizer pack, so this run is '
                              'end-to-end (live tokenizer + live backbone + frozen head)',
            'head_files': identity.get('head_files', {}),
            'policy': identity.get('policy', {}),
            'prompt_protocol': {'builder': 'jev/readout.build',
                                'builder_sha256': digest(HERE / 'readout.py'),
                                'taps': TAPS, 'pools': 'last,mean',
                                'state_text': 'jev/common.canonical',
                                'option_mode': args.mode},
            'readout_scheduling': 'exclusive' if args.exclusive else 'shared-slots',
            'cascade_fallback': False,
            'training_or_calibration_on_benchmark': False,
            'baseline_20261017': {'accuracy': 0.8571428571428571, 'correct': 198, 'planned': 231,
                                  'pair': 'server 3ebd690b / engine ce7559e0 / config b6b687b8'},
            'prompt_tokens_by_id': lengths,
        }, ensure_ascii=False, indent=2) + '\n')

    predictions = out / 'predictions.jsonl'
    records = []
    if predictions.exists():
        records = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    done = {r['id'] for r in records}
    subset = {t['id'] for t in tasks}
    if len(done) != len(records) or not done <= subset:
        raise ValueError('existing benchmark rows contain duplicate or unselected ids')
    consecutive = 0
    started = time.monotonic()

    def write_summary():
        raw_metrics = subset_metrics(records, subset, 'probabilities_raw')
        cal_metrics = subset_metrics(records, subset, 'probabilities_calibrated')
        summary = {'mode': args.mode, 'finished': done == subset,
                   'passed': done == subset and raw_metrics['errors'] == 0,
                   'identity_sha256': digest(args.identity),
                   'deadline': args.deadline, 'elapsed_s': time.monotonic() - started,
                   'n_recorded': len(records), 'all': raw_metrics,
                   'by_tier': {tier: subset_metrics(records,
                       {t['id'] for t in tasks if t['tier'] == tier}, 'probabilities_raw')
                       for tier in sorted({t['tier'] for t in tasks})},
                   'by_type': {kind: subset_metrics(records,
                       {t['id'] for t in tasks if t['question']['type'] == kind}, 'probabilities_raw')
                       for kind in sorted({t['question']['type'] for t in tasks})},
                   'calibration_effect': {
                       'accuracy_raw': raw_metrics['accuracy_all_planned'],
                       'accuracy_calibrated': cal_metrics['accuracy_all_planned'],
                       'ece_raw': raw_metrics['top_label_ece'],
                       'ece_calibrated': cal_metrics['top_label_ece']}}
        (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    with predictions.open('a') as sink:
        for task in tasks:
            if task['id'] in done:
                continue
            if past(args.deadline):
                print('STOP deadline %s reached with %d/%d recorded'
                      % (args.deadline, len(records), len(tasks)), flush=True)
                break
            labels = [str(x) for x in task['labels']]
            gold = str(task['expected']) if task['expected'] is not None else None
            item = items[task['id']]
            row = {'id': task['id'], 'tier': task['tier'], 'family': task['family'],
                   'type': task['question']['type'], 'labels': labels, 'gold_label': gold,
                   'n_tokens': lengths[task['id']], 'status': 'error'}
            try:
                result = head.infer_request(requests[task['id']], len(item['options']))
                logits = np.asarray(result['logits'], dtype=np.float64)[:len(labels)]
                if logits.shape != (len(labels),) or not np.isfinite(logits).all():
                    raise ValueError('invalid head logits')
                raw = softmax(logits)
                calibrated = softmax(logits / temperature)
                raw_probs = {labels[i]: float(raw[i]) for i in range(len(labels))}
                cal_probs = {labels[i]: float(calibrated[i]) for i in range(len(labels))}
                scored_raw = score_one(raw_probs, labels, task['expected'], task['question']['type'])
                scored_cal = score_one(cal_probs, labels, task['expected'], task['question']['type'])
                row.update(status='ok', probabilities_raw=raw_probs, probabilities_calibrated=cal_probs,
                           valid=scored_raw['valid'], strict_valid=scored_raw['strict_valid'],
                           renormalized=scored_raw['renormalized'], correct=scored_raw['correct'],
                           predicted=scored_raw['predicted'], predicted_raw=scored_raw['predicted'],
                           predicted_calibrated=scored_cal['predicted'],
                           ordinal_ev_raw=scored_raw['ordinal_ev'],
                           ordinal_ev_calibrated=scored_cal['ordinal_ev'],
                           correct_calibrated=scored_cal['correct'],
                           logits=logits.tolist(),
                           feature_api_ms=result['feature_api_ms'], head_ms=result['head_ms'],
                           head_path_ms=result['head_path_ms'])
            except Exception as exc:                      # keep the row, count it wrong, do not stop
                row['error'] = type(exc).__name__ + ': ' + str(exc)
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            records.append(row)
            done.add(task['id'])
            consecutive = consecutive + 1 if row['status'] != 'ok' else 0
            if len(records) % 25 == 0 or consecutive:
                print('PROGRESS %d/%d %s %s %s api_ms=%.0f' % (len(records), len(tasks), task['id'],
                                                               row['status'], row.get('predicted_raw'),
                                                               row.get('feature_api_ms', 0)), flush=True)
            write_summary()
            if consecutive >= 3:
                print('STOP three consecutive failures', flush=True)
                return 2
    write_summary()
    print('DONE', str(out), flush=True)
    return 0 if done == subset and all(r['status'] == 'ok' for r in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
