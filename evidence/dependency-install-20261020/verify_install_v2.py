"""Verify official-index CDN provenance, then audit the already clean-installed venv.

The first install succeeded but its conservative source allowlist rejected PyTorch's
own download-r2.pytorch.org CDN. Preserve that FAIL; bind the exact wheel URL/hash to
the official HTTPS index before permitting a new gate. No packages are installed here.
"""
import argparse
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--env-root', type=Path, required=True, help='Previously installed Windows venv directory')
parser.add_argument('--output-dir', type=Path, required=True, help='Evidence bundle containing install-gate.json and pip-install-report.json')
args = parser.parse_args()
OUT = args.output_dir.resolve()
PY = args.env_root.resolve() / 'Scripts/python.exe'
MODULES = ['predict', 'cascade', 'identity', 'fingerprint_model', 'domain',
           'eval_replay', 'eval_benchmark', 'probe_drift', 'check_baseline',
           'train', 'train_distilled', 'tune_policy', 'analyze_flips', 'finalize_rows', 'run_all']
GATE = OUT / 'verification-gate-v2.json'
if GATE.exists():
    raise SystemExit('refusing to replace existing v2 gate')
result = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'initial_gate': 'install-gate.json', 'initial_gate_preserved': True,
          'installation_repeated': False,
          'venv': str(PY.parent.parent), 'system_site_packages': False,
          'scope': 'Same fresh Windows CPython3.12 CPU venv; metadata source verification and offline checks only',
          'quality_gate': 'FAIL', 'steps': []}
env = os.environ.copy()
env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
           PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1')
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)


def save():
    GATE.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def run(name, command, timeout=120):
    start = time.monotonic()
    with (OUT / (name + '.log')).open('x', encoding='utf-8') as sink:
        sink.write('argv: ' + json.dumps([str(x) for x in command]) + '\n')
        sink.flush()
        try:
            proc = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                                  stdout=sink, stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
        except Exception as exc:
            sink.write(repr(exc) + '\n')
            code = -1
    result['steps'].append({'name': name, 'returncode': code,
                            'elapsed_s': round(time.monotonic() - start, 3), 'log': name + '.log'})
    save()
    return code == 0


try:
    save()
    first = json.loads((OUT / 'install-gate.json').read_text())
    install_step = next(s for s in first['steps'] if s['name'] == '04-requirements-install')
    if install_step['returncode'] != 0:
        raise RuntimeError('initial pip installation did not succeed')
    report = json.loads((OUT / 'pip-install-report.json').read_text())
    torch = [i for i in report['install'] if i['metadata']['name'].lower() == 'torch']
    if len(torch) != 1 or torch[0]['metadata']['version'] != '2.6.0+cpu':
        raise RuntimeError('unexpected torch distribution')
    info = torch[0]['download_info']
    url = urlparse(info['url'])
    if (url.scheme != 'https' or url.hostname not in {'download.pytorch.org', 'download-r2.pytorch.org'}
            or not url.path.startswith('/whl/cpu/')):
        raise RuntimeError('torch source is not an allowed official CPU origin')
    index_url = 'https://download.pytorch.org/whl/cpu/torch/'
    with urlopen(index_url, timeout=90) as response:
        index = response.read()
        final_url = response.url
    with (OUT / 'official-torch-cpu-index-v2.html').open('xb') as sink:
        sink.write(index)
    wanted = unquote(url.path.rsplit('/', 1)[-1])
    matches = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', index.decode('utf-8')):
        link = urlparse(urljoin(index_url, html.unescape(href)))
        if unquote(link.path.rsplit('/', 1)[-1]) == wanted:
            matches.append({'url': link._replace(fragment='').geturl(),
                            'sha256': parse_qs(link.fragment).get('sha256', [''])[0]})
    wheel_hash = info['archive_info']['hashes']['sha256']
    if not any(m['url'] == info['url'] and m['sha256'] == wheel_hash for m in matches):
        raise RuntimeError('exact wheel URL/hash not found in official CPU index')
    if any(i['metadata']['name'].lower().startswith(('nvidia-', 'cuda-')) for i in report['install']):
        raise RuntimeError('CUDA package found in installation report')
    result['official_cpu_provenance'] = {'index_url': index_url, 'index_final_url': final_url,
                                         'index_sha256': hashlib.sha256(index).hexdigest(),
                                         'wheel_url': info['url'], 'wheel_sha256': wheel_hash,
                                         'matching_links': matches}
    cfg = (PY.parent.parent / 'pyvenv.cfg').read_text()
    if 'include-system-site-packages = false' not in cfg:
        raise RuntimeError('venv inherits global packages')
    checks = [run('v2-01-pip-check', [PY, '-m', 'pip', 'check']),
              run('v2-02-freeze', [PY, '-m', 'pip', 'freeze', '--all']),
              run('v2-03-platform', [PY, '-c',
                   "import json,sys,site,torch,numpy,tokenizers,jinja2; print(json.dumps({'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'user_site_enabled':site.ENABLE_USER_SITE,'sys_path':sys.path,'torch':torch.__version__,'torch_file':torch.__file__,'torch_cuda':torch.version.cuda,'numpy':numpy.__version__,'tokenizers':tokenizers.__version__,'jinja2':jinja2.__version__},indent=2)); assert sys.prefix != sys.base_prefix; assert not site.ENABLE_USER_SITE; assert torch.version.cuda is None"]),
              run('v2-04-selfcheck', [PY, '-m', 'jev.selfcheck'])]
    for number, module in enumerate(MODULES, 5):
        checks.append(run('v2-%02d-help-%s' % (number, module), [PY, '-m', 'jev.' + module, '--help']))
    checks.append(run('v2-20-unittest', [PY, '-m', 'unittest', 'discover', '-s', 'tests', '-v']))
    result['installation_gate'] = 'PASS'
    result['cli_count'] = len(MODULES)
    result['quality_gate'] = 'PASS' if all(checks) else 'FAIL'
except Exception as exc:
    result['error'] = repr(exc)
    result.setdefault('installation_gate', 'FAIL')
finally:
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
