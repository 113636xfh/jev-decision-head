"""Capture an unbuffered native-fault trace; no selfcheck/runtime edits or workaround."""
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
command = [str(PY), '-X', 'faulthandler', '-u', '-m', 'jev.selfcheck']
with (OUT / 'selfcheck-native-probe-v3.log').open('x', encoding='utf-8') as sink:
    sink.write('argv: ' + json.dumps(command) + '\n')
    sink.flush()
    try:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=sink,
                              stderr=subprocess.STDOUT, timeout=120).returncode
        error = None
    except Exception as exc:
        code, error = -1, repr(exc)
with (OUT / 'selfcheck-native-probe-v3.json').open('x', encoding='utf-8') as sink:
    json.dump({'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'argv': command, 'returncode': code, 'returncode_hex': hex(code) if code >= 0 else None,
               'error': error, 'quality_gate': 'PASS' if code == 0 else 'FAIL',
               'boundary': 'Same environment and code; only unbuffered output/faulthandler added. No runtime or selfcheck edits.'}, sink, indent=2)
    sink.write('\n')
