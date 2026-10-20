"""Run both JevBench renderings and archived replay with one recorded identity.

Children are package modules, not script paths: their relative imports must work
in a clean checkout. The default start/mid/end instrument is a real repeated
archive replay (same 24 dev items at stride 83), without an unrelated chat/decode
request. Probe failures and child failures stop the sweep and propagate nonzero.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = os.environ.get('JEV_PY', sys.executable)
STAGES = [
    ('jevbench-criterion', ['jev.eval_benchmark', '--mode', 'criterion_as_option',
                          '--out', 'jevbench/criterion_as_option']),
    ('jevbench-label', ['jev.eval_benchmark', '--mode', 'label_as_option',
                      '--out', 'jevbench/label_as_option']),
    ('dev', ['jev.eval_replay', '--split', 'dev']),
    ('test', ['jev.eval_replay', '--split', 'test']),
]


def past(deadline):
    return bool(deadline) and datetime.now().strftime('%Y-%m-%d %H:%M:%S') >= deadline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm-window', action='store_true')
    parser.add_argument('--deadline')
    parser.add_argument('--exclusive', action='store_true',
                        help='use exclusive readout (default: shared-slots)')
    parser.add_argument('--identity', default=str(HERE / 'runtime-identity.json'))
    parser.add_argument('--tokenizer-dir', help='forward a tokenizer override to JevBench')
    parser.add_argument('--base', default='http://127.0.0.1:8100')
    parser.add_argument('--skip-instrument', action='store_true')
    parser.add_argument('--limit', type=int, default=0,
                        help='items per stage/probe (smoke test); 0 = full stages, 24 probe items')
    parser.add_argument('--out-prefix', default='',
                        help='write outputs and run-status.json here instead of the sweep root')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if not args.confirm_window:
        raise SystemExit('refusing to start without --confirm-window')
    identity_path = Path(args.identity).resolve()
    if not identity_path.is_file():
        raise SystemExit('run python -m jev.identity first: %s is missing' % identity_path)
    prefix = Path(args.out_prefix).resolve() if args.out_prefix else HERE
    prefix.mkdir(parents=True, exist_ok=True)
    status_path = prefix / 'run-status.json'
    # A resumed sweep must still take NEW repeat samples, never reuse old probes.
    probe_root = prefix / 'probes' / str(time.time_ns())
    started = time.monotonic()
    log = []
    probe_reference = None

    def record(entry):
        log.append(entry)
        status_path.write_text(json.dumps({
            'started_local': datetime.fromtimestamp(
                time.time() - (time.monotonic() - started)).strftime('%Y-%m-%d %H:%M:%S'),
            'deadline': args.deadline, 'identity': str(identity_path),
            'readout_scheduling': 'exclusive' if args.exclusive else 'shared-slots',
            'elapsed_s': round(time.monotonic() - started, 1), 'stages': log,
        }, ensure_ascii=False, indent=2) + '\n')

    def finish(code, reason=None):
        record({'stage': 'all', 'finished': code == 0, 'returncode': code,
                'reason': reason, 'minutes': round((time.monotonic() - started) / 60, 1)})
        return code

    def launch(name, argv, limit=None):
        command = [PY, '-m'] + list(argv)
        command += ['--identity', str(identity_path), '--base', args.base, '--confirm-window']
        count = args.limit if limit is None else limit
        if count:
            command += ['--limit', str(count)]
        if args.deadline:
            command += ['--deadline', args.deadline]
        if args.exclusive:
            command.append('--exclusive')
        if argv[0] == 'jev.eval_benchmark' and args.tokenizer_dir:
            command += ['--tokenizer-dir', str(Path(args.tokenizer_dir).resolve())]
        print('STAGE %s: %s' % (name, ' '.join(command)), flush=True)
        try:
            process = subprocess.run(command, cwd=str(HERE.parent))
            code = process.returncode
            record({'stage': name, 'returncode': code, 'command': command})
        except OSError as exc:
            code = 2
            record({'stage': name, 'returncode': code, 'command': command,
                    'error': type(exc).__name__ + ': ' + str(exc)})
        return code

    def phase(name):
        nonlocal probe_reference
        if args.skip_instrument:
            return 0
        target = probe_root / ('repeat-' + name)
        code = launch('instrument-' + name,
                      ['jev.eval_replay', '--split', 'dev', '--stride', '83',
                       '--out', str(target)], limit=min(args.limit, 24) if args.limit else 24)
        if code:
            return code
        try:
            rows_path = Path(str(target) + '.jsonl')
            rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
            if not rows or any(row['status'] != 'ok' for row in rows):
                raise ValueError('repeat probe has no valid complete sample')
            observed = {row['uid']: (row['gold'], row['pred_new']) for row in rows}
            if len(observed) != len(rows):
                raise ValueError('duplicate repeat probe uid')
            if probe_reference is None:
                probe_reference = observed
            if observed.keys() != probe_reference.keys():
                raise ValueError('repeat probe selected different items')
            changes = sum(observed[uid] != probe_reference[uid] for uid in observed)
            record({'stage': 'instrument-' + name + '-comparison',
                    'rows': str(rows_path), 'n': len(rows),
                    'decision_changes_vs_start': changes, 'passed': changes == 0})
            return 2 if changes else 0
        except (OSError, ValueError, KeyError) as exc:
            record({'stage': 'instrument-' + name + '-comparison', 'passed': False,
                    'error': type(exc).__name__ + ': ' + str(exc)})
            return 2

    if past(args.deadline):
        return finish(2, 'deadline reached before start')
    code = phase('start')
    if code:
        return finish(code, 'start probe failed')
    for name, stage in STAGES:
        if past(args.deadline):
            record({'stage': name, 'skipped': 'deadline reached'})
            return finish(2, 'deadline reached')
        argv = list(stage)
        if name.startswith('jevbench'):
            index = argv.index('--out') + 1
            argv[index] = str(prefix / argv[index])
        else:
            argv += ['--out', str(prefix / ('rows-' + name))]
        code = launch(name, argv)
        if code:
            return finish(code, name + ' failed')
        if name == 'jevbench-label':
            code = phase('mid')
            if code:
                return finish(code, 'mid probe failed')
    if past(args.deadline):
        return finish(2, 'deadline reached before end probe')
    code = phase('end')
    return finish(code, 'end probe failed' if code else None)


if __name__ == '__main__':
    raise SystemExit(main())
