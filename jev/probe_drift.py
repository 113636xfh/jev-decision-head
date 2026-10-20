"""Bound the request-history drift: measure the same 24 archived items at start / mid / end.

Byte equality against the archive cannot be the gate here: with the official expert adaptation the
same engine already differs between a long-running state and a fresh restart (measured
max_abs_diff 0.5614 on a 5000-token span).  So the report compares these three phases against each
other: if the same item measured at 04:00, after the JevBench stage, and at the end of the sweep
still yields the same decision, then the accuracy difference cannot be attributed to scheduling
state - only to the backbone.

`--chat` issues one real completion between the items so the measurement window contains foreign
request history, which is the condition that moves expert placement.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from . import identity as IDENT                                             # noqa: E402
from .head_loader import MeasurementHead, V3                    # noqa: E402
from .common import volunteer_for_oom                                 # noqa: E402
from .gpu_data import LAYERS                                          # noqa: E402

COUNT = 24
STRIDE = 83
TAPS = '15,23,29,35,41,-1'
POOLS = ('span_last', 'span_mean')


def softmax(x):
    """Plain softmax (scipy-free, same definition as the accepted harness)."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['start', 'mid', 'end'])
    parser.add_argument('--chat', action='store_true', help='send one real completion first')
    parser.add_argument('--identity', default=str(HERE / 'runtime-identity.json'))
    parser.add_argument('--tokenizer-dir', help='override the tokenizer pack recorded in the identity')
    parser.add_argument('--exclusive', action='store_true')
    parser.add_argument('--out-prefix', default='', help='write repeat-* under this directory')
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    args = parser.parse_args()
    volunteer_for_oom()
    records = sorted(p for p in (V3 / 'taps-dev2k').glob('*.json')
                     if p.name != 'manifest.json')[::STRIDE][:COUNT]
    if not records:
        raise SystemExit('no archived items selected from %s' % (V3 / 'taps-dev2k'))
    identity = json.loads(Path(args.identity).read_text(encoding='utf-8'))
    # Replay keeps the archived token IDs. Check the current pack when a path is
    # supplied/recorded, without retokenizing or requiring a pack for archive-only use.
    tokenizer_dir = args.tokenizer_dir or identity.get('running_config', {}).get('tokenizer_dir')
    head = MeasurementHead(base=args.base, identity=args.identity, parallel=not args.exclusive,
                               tokenizer_dir=tokenizer_dir)

    chat = None
    if args.chat:
        head.check_runtime()
        model = head.client.get('/v1/models')['data'][0]['id']
        start = time.monotonic()
        # Client.post returns the JSON body and raises on HTTP errors; it no
        # longer returns a (status, body) pair. Never mask a failed warm-up chat.
        body = head.client.post('/v1/chat/completions', dict(
            model=model, messages=[dict(role='user', content='Reply with exactly: ok')],
            max_tokens=16, temperature=0), timeout=300)
        chat = {'status': 200, 'seconds': round(time.monotonic() - start, 3),
                'content': body['choices'][0]['message']['content']}

    prefix = Path(args.out_prefix) if args.out_prefix else HERE
    prefix.mkdir(parents=True, exist_ok=True)
    out = prefix / ('repeat-%s.jsonl' % args.phase)
    rows = []
    with out.open('w', encoding='utf-8') as sink:
        for path in records:
            record = json.loads(path.read_text(encoding='utf-8'))
            item = record['item']
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
                ids = z['latent_ids'].tolist()
                spans = z['latent_spans'].tolist()
                archived = np.stack([np.stack([np.stack(
                    [z['span%d.L%d.%s' % (s, layer, pool)] for layer in LAYERS])
                    for pool in POOLS]) for s in range(len(spans))])
            k = len(item['options'])
            if k < 1 or len(spans) != k + 2:
                raise ValueError('span/candidate mismatch')
            request = {'latent_ids': ids, 'latent_spans': spans, 'latent_taps': TAPS,
                       'latent_pool': 'last,mean', 'cache_region': 'decision'}
            values, api_ms = head.raw_request(request, spans, k)
            live = np.stack([np.stack([np.stack(
                [values['span%d.L%d.%s' % (s, layer, pool)] for layer in LAYERS])
                for pool in POOLS]) for s in range(len(spans))]).astype(np.float16)
            if live.shape != archived.shape or not np.isfinite(archived).all():
                raise ValueError('invalid archived feature geometry or values')
            # MeasurementHead returns JSON-safe lists; legacy loaders may still
            # return arrays. Convert only here, at the arithmetic boundary.
            logits = np.asarray(head.normalized_from_raw(live, k)['logits'], dtype=np.float64)[:k]
            old = np.asarray(head.normalized_from_raw(archived.astype(np.float16), k)['logits'],
                             dtype=np.float64)[:k]
            if (logits.shape != (k,) or old.shape != (k,)
                    or not np.isfinite(logits).all() or not np.isfinite(old).all()):
                raise ValueError('invalid head logits')
            row = {'uid': item['uid'], 'gold': item['options'].index(item['gold']),
                   'pred': int(logits.argmax()), 'pred_archived': int(old.argmax()),
                   'probs': [float(x) for x in softmax(logits)],
                   'max_abs_diff_vs_archived': float(np.abs(live.astype(np.float32)
                                                            - archived.astype(np.float32)).max()),
                   'max_logit_delta': float(np.abs(logits - old).max()), 'api_ms': api_ms}
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
            rows.append(row)
    (prefix / ('repeat-%s.json' % args.phase)).write_text(json.dumps(
        {'phase': args.phase, 'n': len(rows), 'chat': chat,
         'identity_sha256': IDENT.digest(args.identity),
         'decisions_stable_vs_archived': sum(1 for r in rows if r['pred'] == r['pred_archived']),
         'max_logit_delta_max': max(r['max_logit_delta'] for r in rows)},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': args.phase, 'n': len(rows), 'chat': chat}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
