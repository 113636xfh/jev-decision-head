"""Replay archived items against the live (abliterated) backbone and score the frozen head.

Why this harness exists next to the JevBench one: the immutable taps archive stores the exact
`latent_ids` and spans the head was trained on, so sending THOSE ids to the live engine isolates
one variable - the backbone weights.  The JevBench harness tokenizes with the live pack, so it
measures the deployed end-to-end path instead.

Per item, in one pass:
  * verify the archived record/npz digests (a changed archive is a hard stop, never silently skipped);
  * request the readout with the ARCHIVED latent_ids/spans, taps 15/23/29/35/41/-1, pools last+mean;
  * normalize both the live readout and the archived FP16 vectors with the frozen train-fit norm;
  * run the same head on both and record prediction, flip, byte equality and logit delta.

Byte equality is reported but is NOT a gate: with the official expert adaptation the same engine
differs between a long-running state and a fresh restart (measured max_abs_diff 0.5614), so the
decision-level numbers are the evidence.
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
sys.path[:0] = [str(HERE)]

import identity as IDENT                                             # noqa: E402
from head_loader import UncertifiedLiveHead, RUN, V3            # noqa: E402
from common import volunteer_for_oom, digest                         # noqa: E402
from gpu_data import LAYERS                                          # noqa: E402

ARCHIVES = {'dev': V3 / 'taps-dev2k', 'test': V3 / 'taps-test2k',
            'ood': V3 / 'taps-ood2k', 'cal': V3 / 'taps-cal2k'}
TAPS = '15,23,29,35,41,-1'
POOLS = ('span_last', 'span_mean')
SHAPE = (2, len(LAYERS), 10240)
THRESHOLD = 0.85


def softmax(x):
    """Plain softmax (scipy-free, same definition as the accepted harness)."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def past(deadline):
    return bool(deadline) and datetime.now().strftime('%Y-%m-%d %H:%M:%S') >= deadline


def ece(confidence, correct, bins=10):
    buckets = {}
    for conf, hit in zip(confidence, correct):
        buckets.setdefault(min(bins - 1, int(conf * bins)), []).append((conf, hit))
    total = len(confidence)
    return sum(len(v) / total * abs(np.mean([c for _, c in v]) - np.mean([x for x, _ in v]))
               for v in buckets.values()) if total else None


def summarize(rows, planned):
    ok = [r for r in rows if r['status'] == 'ok']
    conf = [r['confidence_new'] for r in ok]
    correct_new = [1 if r['correct_new'] else 0 for r in ok]
    accepted = [i for i, c in enumerate(conf) if c >= THRESHOLD]
    deltas = [r['max_logit_delta'] for r in ok]
    api = [r['api_ms'] for r in ok]
    return {
        'planned': planned, 'recorded': len(rows), 'valid': len(ok), 'errors': len(rows) - len(ok),
        'correct_archived_features': sum(1 for r in ok if r['correct_old']),
        'correct_live_features': sum(1 for r in ok if r['correct_new']),
        'accuracy_archived': sum(1 for r in ok if r['correct_old']) / len(ok) if ok else None,
        'accuracy_live': sum(1 for r in ok if r['correct_new']) / len(ok) if ok else None,
        'flips': sum(1 for r in ok if r['flipped']),
        'flip_rate': sum(1 for r in ok if r['flipped']) / len(ok) if ok else None,
        'vectors': sum(r['vectors'] for r in ok),
        'vectors_fp16_byte_equal': sum(r['byte_equal'] for r in ok),
        'items_all_vectors_equal': sum(1 for r in ok if r['byte_equal'] == r['vectors']),
        'max_logit_delta_p50_p95_max': [float(np.percentile(deltas, 50)), float(np.percentile(deltas, 95)),
                                        float(np.max(deltas))] if deltas else None,
        'api_ms_p50_p95': [float(np.percentile(api, 50)), float(np.percentile(api, 95))] if api else None,
        'coverage_at_0.85': len(accepted) / len(ok) if ok else None,
        'accepted_subset_accuracy': float(np.mean([correct_new[i] for i in accepted])) if accepted else None,
        'ece_raw': ece(conf, correct_new) if ok else None,
        'ece_archived': ece([r['confidence_old'] for r in ok],
                            [1 if r['correct_old'] else 0 for r in ok]) if ok else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=sorted(ARCHIVES))
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--estimate-only', action='store_true')
    parser.add_argument('--confirm-window', action='store_true',
                        help='required: acknowledges the live engine serves this sweep')
    parser.add_argument('--exclusive', action='store_true')
    parser.add_argument('--deadline')
    parser.add_argument('--identity', default=str(HERE / 'runtime-identity.json'))
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    args = parser.parse_args()
    volunteer_for_oom()
    torch.set_num_threads(2)

    archive = ARCHIVES[args.split]
    # The archive root also holds manifest.json, which is not a record: counting it made the dev
    # stage crash on `record['item']` after all 1996 real items had been measured.
    records = sorted(p for p in archive.glob('*.json') if p.name != 'manifest.json')
    picks = records[::args.stride][:args.limit] if args.limit else records[::args.stride]
    if args.estimate_only:
        print(json.dumps({'split': args.split, 'archive': str(archive), 'selected': len(picks),
                          'assumed_seconds_per_item': 2.6,
                          'estimate_minutes': round(len(picks) * 2.6 / 60, 1)}))
        return 0
    if not args.confirm_window:
        raise SystemExit('refusing to start without --confirm-window')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix('.jsonl') if out.suffix == '.json' else Path(str(out) + '.jsonl')
    summary_path = Path(str(rows_path).replace('.jsonl', '-summary.json'))

    head = UncertifiedLiveHead(base=args.base, identity=args.identity, parallel=not args.exclusive)
    with np.load(RUN / 'norm.npz', allow_pickle=False) as z:
        mu, sd = z['mu'].copy(), z['sd'].copy()

    rows, started = [], time.monotonic()
    if rows_path.exists():
        rows = [json.loads(line) for line in rows_path.open() if line.strip()]
    done = {r['uid'] for r in rows}
    consecutive = 0
    with rows_path.open('a') as sink:
        for path in picks:
            record = json.loads(path.read_text())
            item = record['item']
            uid = item['uid']
            if uid in done:
                continue
            if past(args.deadline):
                print('STOP deadline %s with %d/%d recorded' % (args.deadline, len(rows), len(picks)),
                      flush=True)
                break
            npz_path = path.with_suffix('.npz')
            row = {'uid': uid, 'split': args.split, 'status': 'error',
                   'n_options': len(item['options']), 'n_spans': len(record['spans'])}
            try:
                if digest(npz_path) != record['npz_sha256']:
                    raise ValueError('archived npz digest changed: ' + str(npz_path))
                with np.load(npz_path, allow_pickle=False) as z:
                    ids = z['latent_ids'].tolist()
                    spans = z['latent_spans'].tolist()
                    if spans != record['spans']:
                        raise ValueError('archived spans disagree with the record')
                    archived = np.stack([np.stack([np.stack(
                        [z['span%d.L%d.%s' % (s, layer, pool)] for layer in LAYERS])
                        for pool in POOLS]) for s in range(len(spans))])
                if spans != record['spans'] or len(spans) != len(item['options']) + 2:
                    raise ValueError('span/candidate mismatch')
                request = {'latent_ids': ids, 'latent_spans': spans, 'latent_taps': TAPS,
                           'latent_pool': 'last,mean', 'cache_region': 'decision'}
                values, api_ms = head.raw_request(request, spans, len(item['options']))
                live = np.empty((len(spans),) + SHAPE, dtype=np.float32)
                for s in range(len(spans)):
                    for pi, pool in enumerate(POOLS):
                        for li, layer in enumerate(LAYERS):
                            vector = values['span%d.L%d.%s' % (s, layer, pool)]
                            if vector.shape != (10240,):
                                raise ValueError('vector geometry changed: %s %s'
                                                 % ('span%d.L%d.%s' % (s, layer, pool), vector.shape))
                            live[s, pi, li] = vector
                if not np.isfinite(live).all():
                    raise ValueError('nonfinite live readout')
                live_norm = ((live - mu) / sd).astype(np.float16)
                archived_norm = ((archived.astype(np.float32) - mu) / sd).astype(np.float16)
                k = len(item['options'])
                # normalize_gpu reproduces numpy's ((FP32 - mu)/sd).astype(FP16) byte for byte
                # (gpu_data.normalize_gpu; build_cache.py asserts it), so both sides below go
                # through the accepted GPU path.
                new_logits = head.normalized_from_raw(live.astype(np.float16), k)['logits']
                old_logits = head.normalized_from_raw(archived.astype(np.float16), k)['logits']
                gold = item['options'].index(item['gold'])
                pred_new, pred_old = int(new_logits.argmax()), int(old_logits.argmax())
                row.update(status='ok', gold=gold, pred_new=pred_new, pred_old=pred_old,
                           correct_new=int(pred_new == gold), correct_old=int(pred_old == gold),
                           flipped=int(pred_new != pred_old),
                           confidence_new=float(softmax(new_logits[:k]).max()),
                           confidence_old=float(softmax(old_logits[:k]).max()),
                           max_logit_delta=float(np.abs(new_logits[:k] - old_logits[:k]).max()),
                           vectors=live_norm.size,
                           byte_equal=int((live_norm.view(np.uint16) == archived_norm.view(np.uint16)).sum()),
                           api_ms=api_ms, n_tokens=len(ids))
            except Exception as exc:
                row['error'] = type(exc).__name__ + ': ' + str(exc)
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            rows.append(row)
            consecutive = consecutive + 1 if row['status'] != 'ok' else 0
            if len(rows) % 25 == 0 or consecutive:
                print('PROGRESS %s %d/%d %s %s api_ms=%.0f' % (args.split, len(rows), len(picks),
                                                               row['status'], row.get('error', ''),
                                                               row.get('api_ms', 0)), flush=True)
            summary_path.write_text(json.dumps({'split': args.split, 'finished': len(rows) >= len(picks),
                                                'deadline': args.deadline,
                                                'elapsed_s': time.monotonic() - started,
                                                'identity_sha256': digest(args.identity),
                                                'metrics': summarize(rows, len(picks))},
                                               ensure_ascii=False, indent=2) + '\n')
            if consecutive >= 3:
                print('STOP three consecutive failures', flush=True)
                return 2
    print(json.dumps({'done': summarize(rows, len(picks)),
                      'minutes': round((time.monotonic() - started) / 60, 1)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
