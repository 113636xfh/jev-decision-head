"""Reproducible local clean-install gate. No network/runtime inference is performed."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--env-root', type=Path, required=True, help='New Windows venv directory')
parser.add_argument('--output-dir', type=Path, required=True, help='Evidence output directory, not the frozen release evidence')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
ENV = args.env_root.resolve()
PY = ENV / 'Scripts/python.exe'
MODULES = ['predict', 'cascade', 'identity', 'fingerprint_model', 'domain',
           'eval_replay', 'eval_benchmark', 'probe_drift', 'check_baseline',
           'train', 'train_distilled', 'tune_policy', 'analyze_flips',
           'finalize_rows', 'run_all']
result = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'platform': platform.platform(), 'bootstrap_python': sys.version,
          'venv': str(ENV), 'system_site_packages': False,
          'scope': 'Windows x86-64 CPython 3.12 CPU only; no GPU, server, tokenizer pack, or production requests',
          'requirements_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in ['requirements.txt', 'requirements-cpu.txt']},
          'steps': [], 'quality_gate': 'FAIL'}
env = os.environ.copy()
env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
           PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1')
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)

def save():
    (OUT / 'install-gate.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

def run(name, command, timeout=1200):
    started = time.monotonic()
    with (OUT / (name + '.log')).open('w', encoding='utf-8') as log:
        log.write('argv: ' + json.dumps([str(x) for x in command]) + '\n')
        log.flush()
        try:
            proc = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
        except Exception as exc:
            log.write('\n' + repr(exc) + '\n')
            code = -1
    result['steps'].append({'name': name, 'returncode': code,
                            'elapsed_s': round(time.monotonic() - started, 3),
                            'log': name + '.log'})
    save()
    return code == 0

try:
    save()
    if ENV.exists():
        raise RuntimeError('refusing to reuse an existing environment: ' + str(ENV))
    if not run('01-create-venv', [sys.executable, '-m', 'venv', ENV]):
        raise RuntimeError('venv creation failed')
    cfg = (ENV / 'pyvenv.cfg').read_text()
    (OUT / 'pyvenv.cfg.txt').write_text(cfg, encoding='utf-8')
    if 'include-system-site-packages = false' not in cfg:
        raise RuntimeError('venv unexpectedly inherits system packages')
    run('02-clean-inventory', [PY, '-m', 'pip', 'list', '--format=json'])
    if not run('03-pip-bootstrap', [PY, '-m', 'pip', 'install', '--upgrade', 'pip']):
        raise RuntimeError('pip bootstrap failed')
    if not run('04-requirements-install', [PY, '-m', 'pip', 'install', '--no-cache-dir',
               '--report', OUT / 'pip-install-report.json', '-r', ROOT / 'requirements-cpu.txt']):
        raise RuntimeError('requirements installation failed')
    install = json.loads((OUT / 'pip-install-report.json').read_text())
    torch_entries = [i for i in install['install'] if i['metadata']['name'].lower() == 'torch']
    if len(torch_entries) != 1 or not torch_entries[0]['download_info']['url'].startswith('https://download.pytorch.org/whl/cpu/'):
        raise RuntimeError('torch wheel source is not the official CPU index')
    if any(i['metadata']['name'].lower().startswith(('nvidia-', 'cuda-')) for i in install['install']):
        raise RuntimeError('unexpected CUDA dependency')
    checks = [run('05-pip-check', [PY, '-m', 'pip', 'check']),
              run('06-freeze', [PY, '-m', 'pip', 'freeze', '--all']),
              run('07-platform', [PY, '-c',
                   "import json,sys,site,torch,numpy,tokenizers,jinja2; print(json.dumps({'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'user_site_enabled':site.ENABLE_USER_SITE,'sys_path':sys.path,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'numpy':numpy.__version__,'tokenizers':tokenizers.__version__,'jinja2':jinja2.__version__},indent=2)); assert sys.prefix != sys.base_prefix; assert not site.ENABLE_USER_SITE; assert torch.version.cuda is None"]),
              run('08-selfcheck', [PY, '-m', 'jev.selfcheck'])]
    for number, module in enumerate(MODULES, 9):
        checks.append(run('%02d-help-%s' % (number, module), [PY, '-m', 'jev.' + module, '--help'], timeout=120))
    result['installation_gate'] = 'PASS'
    result['cli_count'] = len(MODULES)
    result['quality_gate'] = 'PASS' if all(checks) else 'FAIL'
except Exception as exc:
    result['error'] = repr(exc)
    result.setdefault('installation_gate', 'FAIL')
finally:
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
