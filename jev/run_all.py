"""Orchestrate the 04:00 sweep: JevBench 231 x2 -> dev 1996 -> test 1999, one shared identity.

Every stage runs as its own process (a crash in one stage cannot corrupt the other evidence) but
they all bind to the SAME runtime-identity.json written by G0, so the report can state that one
pair answered all 4457 requests.  The deadline is propagated: when it is reached the current
stage stops and the remaining stages are skipped, keeping the partial per-item evidence.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = '/home/xfh/jevtest/.venv/bin/python3'
STAGES = [
    ('jevbench-criterion', ['run_jevbench_abliterated.py', '--mode', 'criterion_as_option',
                            '--out', 'jevbench/criterion_as_option']),
    ('jevbench-label', ['run_jevbench_abliterated.py', '--mode', 'label_as_option',
                        '--out', 'jevbench/label_as_option']),
    ('dev', ['eval_head_abliterated.py', '--split', 'dev']),
    ('test', ['eval_head_abliterated.py', '--split', 'test']),
]


def past(deadline):
    return bool(deadline) and datetime.now().strftime('%Y-%m-%d %H:%M:%S') >= deadline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-window', action='store_true')
    parser.add_argument('--deadline')
    parser.add_argument('--exclusive', action='store_true',
                        help='use the exclusive readout path (default: shared-slots)')
    parser.add_argument('--identity', default=str(HERE / 'runtime-identity.json'))
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    parser.add_argument('--skip-instrument', action='store_true')
    parser.add_argument('--limit', type=int, default=0,
                        help='items per stage (smoke test); 0 = the full stage')
    parser.add_argument('--out-prefix', default='',
                        help='write stage outputs under this directory instead of the sweep root')
    args = parser.parse_args()
    if not args.confirm_window:
        raise SystemExit('refusing to start without --confirm-window')
    identity_path = Path(args.identity)
    if not identity_path.is_file():
        raise SystemExit('run identity.py first: %s is missing' % identity_path)

    status_path = HERE / 'run-status.json'
    started = time.monotonic()
    log = []

    def record(entry):
        log.append(entry)
        status_path.write_text(json.dumps({'started_local': datetime.fromtimestamp(
            time.time() - (time.monotonic() - started)).strftime('%Y-%m-%d %H:%M:%S'),
            'deadline': args.deadline, 'identity': str(identity_path),
            'readout_scheduling': 'exclusive' if args.exclusive else 'shared-slots',
            'elapsed_s': round(time.monotonic() - started, 1), 'stages': log},
            ensure_ascii=False, indent=2) + '\n')

    def stage_out(name):
        return str((Path(args.out_prefix) / name) if args.out_prefix else name)

    def phase(name, extra=()):
        if args.skip_instrument:
            return
        command = [PY, str(HERE / 'instrument_repeat.py'), '--phase', name, '--identity', args.identity,
                   '--base', args.base] + list(extra)
        if args.out_prefix:
            command += ['--out-prefix', args.out_prefix]
        if args.exclusive:
            command.append('--exclusive')
        process = subprocess.run(command, cwd=str(HERE))
        record({'stage': 'instrument-' + name, 'returncode': process.returncode})

    phase('start', ['--chat'])
    for name, argv in STAGES:
        if past(args.deadline):
            record({'stage': name, 'skipped': 'deadline reached'})
            continue
        command = [PY, str(HERE / argv[0])] + argv[1:]
        if name.startswith('jevbench'):
            index = command.index('--out')
            command[index + 1:index + 2] = [stage_out(command[index + 1])]
        else:
            command += ['--out', stage_out('rows-%s' % name)]
        command += ['--identity', args.identity, '--base', args.base, '--confirm-window']
        if args.limit:
            command += ['--limit', str(args.limit)]
        if args.deadline:
            command += ['--deadline', args.deadline]
        if args.exclusive:
            command.append('--exclusive')
        print('STAGE %s: %s' % (name, ' '.join(command)), flush=True)
        process = subprocess.run(command, cwd=str(HERE))
        record({'stage': name, 'returncode': process.returncode})
        if name == 'jevbench-label':
            phase('mid')
    phase('end')
    record({'stage': 'all', 'finished': not past(args.deadline),
            'minutes': round((time.monotonic() - started) / 60, 1)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
