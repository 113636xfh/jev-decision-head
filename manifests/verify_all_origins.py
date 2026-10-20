"""Run read-only upstream checks sequentially; supports long scans without a model.

python manifests/verify_all_origins.py --raw-root /path/to/upstream/raw
Optional: --remote SSH_ALIAS (raw-root is on that host). Run from a shell that
preserves POSIX arguments, e.g. MSYS_NO_PATHCONV=1 in Git Bash on Windows.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--remote')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    for split, upstream in [('train', 'train'), ('dev', 'validation'), ('test', 'test'), ('ood', 'ood'), ('cal', 'calibration')]:
        command = [sys.executable, str(root / 'verify_upstream.py'), '--raw', args.raw_root.rstrip('/') + '/' + upstream + '.jsonl.gz',
                   '--members', str(root / 'splits' / split)]
        if args.remote:
            command += ['--remote', args.remote]
        print('checking ' + split, flush=True)
        result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        report = json.loads(result.stdout)
        target = root / (split + '-origin.json')
        temporary = target.with_suffix('.json.tmp')
        temporary.write_bytes(result.stdout)
        temporary.replace(target)
        print('verified ' + split + ': ' + str(report['matched_members']), flush=True)


if __name__ == '__main__':
    main()
