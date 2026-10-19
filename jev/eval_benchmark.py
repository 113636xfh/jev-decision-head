"""JevBench public 231 re-measured with the frozen head on the CURRENT (abliterated) backbone.

Same benchmark revision, same frozen prompt builder, same scorer, same head and normalization as
the accepted 2026-10-17 run (198/231 = 85.71%); the only change is which pair answers:
engine c1d3070c / server d85622eb / config 5440f05c (qwen3.8-flash-next-iq3_s-abliterated)
instead of the certified ce7559e0 / 3ebd690b / b6b687b8.

Honest scope note: here `latent_ids` come from the LIVE tokenizer pack, so this measurement is
end-to-end (live tokenizer + live backbone + frozen head).  The archive-replay harness
(`eval_head_abliterated.py`) sends the archived ids and therefore isolates the backbone alone.

No decode fallback and no threshold: head-only accuracy.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BENCH = Path('/home/xfh/data/strata-upgrade-20261016/jevbench-current-head-20261017')
V3 = Path('/home/xfh/data/jev-v3-fast')
sys.path[:0] = [str(HERE), str(BENCH)]

import identity as IDENT                                             # noqa: E402
from head_abliterated import UncertifiedLiveHead                     # noqa: E402
from common import canonical, volunteer_for_oom, digest              # noqa: E402
from run_jevbench import (build_item, criteria_for, softmax,          # noqa: E402
                          subset_metrics, OFFICIAL)

TAPS = '15,23,29,35,41,-1'


def score_one(probs, labels, expected, qtype):
    """Official JevBench scoring, via the frozen upstream scorer (identical to 2026-10-17)."""
    from types import SimpleNamespace
    task = SimpleNamespace(labels=labels, expected=expected, question={'type': qtype})
    outcome = OFFICIAL.score_task({k: float(v) for k, v in probs.items()}, task)
    return {k: outcome.get(k) for k in ('valid', 'strict_valid', 'renormalized', 'correct', 'predicted')}


def past(deadline: str) -> bool:
    return bool(deadline) and datetime.now().strftime('%Y-%m-%d %H:%M:%S') >= deadline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['criterion_as_option', 'label_as_option'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--identity', default=str(HERE / 'runtime-identity.json'))
    parser.add_argument('--exclusive', action='store_true',
                        help='use the exclusive readout path instead of shared-slots')
    parser.add_argument('--confirm-window', action='store_true',
                        help='required: acknowledges the live engine serves this run')
    parser.add_argument('--deadline', help='stop after this local time, keep partial evidence')
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    args = parser.parse_args()
    if not args.confirm_window:
        raise SystemExit('refusing to start without --confirm-window')
    volunteer_for_oom()
    torch.set_num_threads(2)

    tasks = [json.loads(line) for line in (BENCH / 'tasks.jsonl').read_text().splitlines() if line.strip()]
    assert len(tasks) == 231, len(tasks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = json.loads(Path(args.identity).read_text())
    head = UncertifiedLiveHead(base=args.base, identity=args.identity, parallel=not args.exclusive)
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
                          'model': identity['running_config']['model_name']}), flush=True)
        return 0
    if args.limit:
        tasks = tasks[:args.limit]
    manifest_path = out / 'manifest.json'
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            'schema': 'jevbench-open-current-head-abliterated-v1',
            'benchmark': 'JevBench public 231 (original 72 + easy 48 + hard 111)',
            'benchmark_revision': '7625ecb1b075cfe4d09d34141d353325825eea1d',
            'benchmark_source_sha256': {'tasks.jsonl': digest(BENCH / 'tasks.jsonl'),
                                        'scoring.py': digest(BENCH / 'upstream_scoring.py')},
            'live_identity': identity,
            'old_certificate_not_applicable': identity['old_certificate'],
            'tokenizer_note': 'latent_ids are produced by the live tokenizer pack, so this run is '
                              'end-to-end (live tokenizer + live backbone + frozen head)',
            'head_files': identity['head_files'],
            'policy': identity['policy'],
            'prompt_protocol': {'builder': 'jev-v3-fast/readout.build',
                                'builder_sha256': digest(V3 / 'readout.py'),
                                'taps': TAPS, 'pools': 'last,mean',
                                'state_text': 'jev-v3-fast/common.canonical',
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
    consecutive = 0
    started = time.monotonic()
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
                raw = softmax(result['logits'])
                calibrated = softmax(result['logits'] / head.temperature)
                raw_probs = {labels[i]: float(raw[i]) for i in range(len(labels))}
                cal_probs = {labels[i]: float(calibrated[i]) for i in range(len(labels))}
                scored_raw = score_one(raw_probs, labels, task['expected'], task['question']['type'])
                scored_cal = score_one(cal_probs, labels, task['expected'], task['question']['type'])
                row.update(status='ok', probabilities_raw=raw_probs, probabilities_calibrated=cal_probs,
                           valid=scored_raw['valid'], strict_valid=scored_raw['strict_valid'],
                           renormalized=scored_raw['renormalized'], correct=scored_raw['correct'],
                           predicted_raw=scored_raw['predicted'], predicted_calibrated=scored_cal['predicted'],
                           logits=[float(x) for x in result['logits']],
                           feature_api_ms=result['feature_api_ms'], head_ms=result['head_ms'],
                           head_path_ms=result['head_path_ms'])
            except Exception as exc:                      # keep the row, count it wrong, do not stop
                row['error'] = type(exc).__name__ + ': ' + str(exc)
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            records.append(row)
            consecutive = consecutive + 1 if row['status'] != 'ok' else 0
            if len(records) % 25 == 0 or consecutive:
                print('PROGRESS %d/%d %s %s %s api_ms=%.0f' % (len(records), len(tasks), task['id'],
                                                               row['status'], row.get('predicted_raw'),
                                                               row.get('feature_api_ms', 0)), flush=True)
            ok = [r for r in records if r['status'] == 'ok']
            subset = {t['id'] for t in tasks}
            summary = {'mode': args.mode, 'finished': len(records) == len(tasks),
                       'deadline': args.deadline, 'elapsed_s': time.monotonic() - started,
                       'n_recorded': len(records),
                       'all': subset_metrics(ok, subset, 'probabilities_raw'),
                       'by_tier': {tier: subset_metrics(ok, {t['id'] for t in tasks if t['tier'] == tier},
                                                        'probabilities_raw')
                                   for tier in sorted({t['tier'] for t in tasks})},
                       'by_type': {kind: subset_metrics(ok, {t['id'] for t in tasks if t['question']['type'] == kind},
                                                        'probabilities_raw')
                                   for kind in sorted({t['question']['type'] for t in tasks})},
                       'calibration_effect': {
                           'accuracy_raw': subset_metrics(ok, subset, 'probabilities_raw')['accuracy_all_planned'],
                           'accuracy_calibrated': subset_metrics(ok, subset, 'probabilities_calibrated')['accuracy_all_planned'],
                           'ece_raw': subset_metrics(ok, subset, 'probabilities_raw')['top_label_ece'],
                           'ece_calibrated': subset_metrics(ok, subset, 'probabilities_calibrated')['top_label_ece']}}
            (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            if consecutive >= 3:
                print('STOP three consecutive failures', flush=True)
                return 2
    print('DONE', str(out), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
