"""Per-item flip lists and per-family tables for the 2026-10-18 abliterated-backbone sweep.

Reads only the measured rows plus the archived item metadata (uid -> domain/kind), so it never
re-queries the engine.  Writes `flips-<split>.jsonl` (every item whose answer changed) and prints
the family breakdown the report needs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import identity as IDENT                                    # noqa: E402
from eval_replay import ARCHIVES                  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=sorted(ARCHIVES))
    parser.add_argument('--rows', required=True)
    parser.add_argument('--out-flips', required=True)
    parser.add_argument('--min-domain-n', type=int, default=60)
    parser.add_argument('--max-domain-rows', type=int, default=12)
    args = parser.parse_args()

    meta = {}
    for path in ARCHIVES[args.split].glob('*.json'):
        if path.name == 'manifest.json':
            continue
        item = json.loads(path.read_text())['item']
        meta[item['uid']] = {'domain': (item.get('metadata') or {}).get('domain'),
                             'kind': item.get('kind')}
    rows = [json.loads(line) for line in Path(args.rows).open() if line.strip()]
    ok = [r for r in rows if r['status'] == 'ok']
    missing = [r['uid'] for r in ok if r['uid'] not in meta]
    if missing:
        raise SystemExit('archive metadata missing for %d uids, e.g. %s' % (len(missing), missing[:3]))

    flips = [dict(r, domain=meta[r['uid']]['domain'], kind=meta[r['uid']]['kind']) for r in ok
             if r['flipped']]
    flips.sort(key=lambda r: -r['max_logit_delta'])
    with Path(args.out_flips).open('w') as sink:
        for row in flips:
            sink.write(json.dumps({k: row[k] for k in ('uid', 'domain', 'kind', 'gold', 'pred_old',
                                                       'pred_new', 'correct_old', 'correct_new',
                                                       'confidence_old', 'confidence_new',
                                                       'max_logit_delta')}, ensure_ascii=False) + '\n')

    def table(key):
        buckets = defaultdict(lambda: [0, 0, 0, 0])   # n, old_correct, new_correct, flips
        for r in ok:
            b = buckets[meta[r['uid']][key]]
            b[0] += 1
            b[1] += r['correct_old']
            b[2] += r['correct_new']
            b[3] += r['flipped']
        out = []
        for name, (n, old, new, fl) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
            if n < args.min_domain_n and key == 'domain':
                continue
            out.append({'name': name, 'n': n, 'acc_archived': round(old / n, 4),
                        'acc_live': round(new / n, 4), 'delta': round((new - old) / n, 4),
                        'flips': fl, 'flip_rate': round(fl / n, 4)})
        return out

    print(json.dumps({'split': args.split, 'n': len(ok), 'flips': len(flips),
                      'flips_file': args.out_flips,
                      'by_domain': table('domain')[:args.max_domain_rows],
                      'by_kind': table('kind'),
                      'top_flips': [{'uid': r['uid'], 'domain': meta[r['uid']]['domain'],
                                     'gold': r['gold'], 'old': r['pred_old'], 'new': r['pred_new'],
                                     'delta': round(r['max_logit_delta'], 2)} for r in flips[:8]]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
