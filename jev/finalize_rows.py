"""Rewrite a stage summary from its rows file, without touching the engine or the rows.

Why this exists: `eval_replay.py` counted the archive root's `manifest.json` as a record,
so both archive-replay stages raised KeyError AFTER measuring every real item, leaving
`finished: false` and `planned` one too high in the per-stage summary.  The rows themselves are
complete (verified uid-by-uid against the archive), so the fix is to recompute the summary from the
rows with the correct planned count - no new requests, no re-measurement, rows untouched.
"""
import argparse
import json
from pathlib import Path

from . import identity as IDENT                                    # noqa: E402
from .eval_replay import ARCHIVES, summarize       # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=sorted(ARCHIVES))
    parser.add_argument('--rows', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--identity', default='runtime-identity.run-20261018.json')
    args = parser.parse_args()

    planned = sum(1 for p in ARCHIVES[args.split].glob('*.json') if p.name != 'manifest.json')
    rows = [json.loads(line) for line in Path(args.rows).open() if line.strip()]
    uids = [r['uid'] for r in rows]
    if len(uids) != len(set(uids)):
        raise SystemExit('duplicate uids in the rows file; refusing to summarize')
    metrics = summarize(rows, planned)
    text = json.dumps({'split': args.split, 'finished': len(rows) >= planned,
                       'finalized_from_rows': True,
                       'identity_sha256': IDENT.digest(args.identity),
                       'metrics': metrics}, ensure_ascii=False, indent=2) + '\n'
    Path(args.summary).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
