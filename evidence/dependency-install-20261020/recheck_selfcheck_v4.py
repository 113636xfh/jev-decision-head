"""Two serial canonical reruns to characterize, not hide, the observed native fault."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--env-root', type=Path, required=True, help='Previously installed Windows venv directory')
parser.add_argument('--output-dir', type=Path, required=True, help='New diagnostic output directory')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
PY = args.env_root.resolve() / 'Scripts/python.exe'
env = os.environ.copy()
env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
trials = []
for number in [1, 2]:
    command = [str(PY), '-m', 'jev.selfcheck']
    name = 'selfcheck-canonical-recheck-v4-%d.log' % number
    with (OUT / name).open('x', encoding='utf-8') as sink:
        sink.write('argv: ' + json.dumps(command) + '\n')
        sink.flush()
        try:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=sink,
                                  stderr=subprocess.STDOUT, timeout=120).returncode
            error = None
        except Exception as exc:
            code, error = -1, repr(exc)
    trials.append({'number': number, 'returncode': code,
                   'returncode_hex': hex(code) if code >= 0 else None,
                   'error': error, 'log': name})
with (OUT / 'selfcheck-canonical-rechecks-v4.json').open('x', encoding='utf-8') as sink:
    json.dump({'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'trials': trials, 'overall_quality_gate': 'FAIL',
               'reason': 'The earlier canonical 0xc0000005 failure remains unexplained; successful reruns do not erase it.',
               'boundary': 'No runtime/selfcheck edits, installation changes, GPU or production calls.'}, sink, indent=2)
    sink.write('\n')
