import os as _os
JEV_DATA = _os.environ.get('JEV_DATA', _os.path.expanduser('~/jev-data'))
STRATA = _os.environ.get('STRATA_ROOT', _os.path.expanduser('~/strata'))
JEV_PY = _os.environ.get('JEV_PY', 'python3')
"""Prove the archive-replay path reproduces the ACCEPTED baseline predictions.

The comparison is only meaningful if "old features" here means the same tensor the 2026-10-15
evaluation used.  That run left per-item predictions in `runs/middle5-s1/dev-predictions.jsonl`
(1996 rows, audited).  This script checks, for every uid present in a rows file, that the gold
index and the archived-feature prediction match that accepted record exactly.

It is a gate for the harness, not for the backbone: a mismatch would mean my assembly/normalization
differs from the accepted one, and then any accuracy difference would be my bug, not the model's.
"""
import argparse
import json
from pathlib import Path

RUN = Path(JEV_DATA + '/jev-v4-gpu/runs/middle5-s1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', required=True, help='rows-*.jsonl produced by eval_replay.py')
    parser.add_argument('--accepted', default=str(RUN / 'dev-predictions.jsonl'))
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    accepted = {}
    for line in Path(args.accepted).open():
        row = json.loads(line)
        accepted[row['uid']] = row
    rows = [json.loads(line) for line in Path(args.rows).open() if line.strip()]
    ok = [r for r in rows if r['status'] == 'ok']
    compared = mismatched = gold_mismatch = 0
    details = []
    for row in ok:
        reference = accepted.get(row['uid'])
        if reference is None:
            continue
        compared += 1
        if reference['gold'] != row['gold']:
            gold_mismatch += 1
            details.append({'uid': row['uid'], 'why': 'gold', 'accepted': reference['gold'],
                            'mine': row['gold']})
        if reference['pred'] != row['pred_old']:
            mismatched += 1
            details.append({'uid': row['uid'], 'why': 'pred_archived', 'accepted': reference['pred'],
                            'mine': row['pred_old']})
    result = {'accepted_file': args.accepted, 'accepted_rows': len(accepted),
              'rows_file': args.rows, 'rows_ok': len(ok), 'compared': compared,
              'gold_mismatches': gold_mismatch, 'prediction_mismatches': mismatched,
              'passed': compared > 0 and gold_mismatch == 0 and mismatched == 0,
              'details': details[:20]}
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
