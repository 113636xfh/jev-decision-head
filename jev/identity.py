"""Live identity for the abliterated backbone: record which pair actually answers.

Why not the accepted certificate: adaptive-head-v1/compatibility.json binds
engine ce7559e0 / server 3ebd690b / config b6b687b8, i.e. the NON-abliterated backbone.
Production now answers from a different pair (engine c1d3070c / server d85622eb /
config 5440f05c = strata-iq3_s-abliterated.json).  Nothing here requires that certificate;
instead every run records what is really executing and refuses to keep going when it changes.

Recorded on purpose (a mid-run restart or a repointed launcher would otherwise make "no
accuracy drop" an artefact): the SHA-256 of the executing engine read through /proc, the
server.py and running-config SHAs, the model name and pack paths, the tokenizer file hashes,
the /metrics capacity keys, and the head/normalization file identities.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

RUNTIME = Path('/home/xfh/strata/v0.1.44.1')
V3 = Path('/home/xfh/data/jev-v3-fast')
V4 = Path('/home/xfh/data/jev-v4-gpu')
RUN = V4 / 'runs/middle5-s1'
AD = Path('/home/xfh/data/strata-upgrade-20261016/adaptive-head-v1')
BASE = 'http://127.0.0.1:8100'
TAPS = '15,23,29,35,41,-1'
# The frozen serving policy of the accepted head (same numbers the certificate carries).
POLICY = {'threshold': 0.85, 'temperature': 1.1525501023241218, 'guard_band': 0.0}
# /metrics engine keys that define the readout protocol; a change means the answer came from
# a different scheduling/placement than the one recorded.
CAPACITY_KEYS = ('version', 'jev_slot_readout', 'batch_slots', 'batch_groups', 'max_context',
                 'kv_resident', 'expert_slots', 'expert_slots_primary', 'model')
TOKENIZER_FILES = ('chat_template.jinja', 'merges.txt', 'tokenizer.json', 'token_type.json', 'vocab.json')


class IdentityMismatch(ValueError):
    pass


def digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def http_get(path: str, base: str = BASE, timeout: int = 15) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as handle:
        return json.load(handle)


def live_processes(base: str = BASE) -> dict:
    """Server pid, engine pids and the --config path, read from /proc so a rename cannot hide them."""
    server, engines = None, []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            cmdline = (entry / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
        except OSError:
            continue
        exe = ''
        try:
            exe = os.readlink(entry / 'exe')
        except OSError:
            pass
        args = [a for a in cmdline if a]
        if any(a.endswith('serve/server.py') for a in args):
            config = args[args.index('--config') + 1] if '--config' in args else None
            server = {'pid': int(entry.name), 'exe': exe, 'config_path': config, 'args': args}
        elif exe.endswith('/engine-cuda12/strata') or exe.endswith('/engine-cuda12/strata-vision'):
            engines.append({'pid': int(entry.name), 'exe': exe})
    if server is None:
        raise IdentityMismatch('no serve/server.py process is running; refusing to start')
    native = [e for e in engines if str(RUNTIME) in e['exe'] and e['exe'].endswith('/engine-cuda12/strata')]
    vision = [e for e in engines if str(RUNTIME) in e['exe'] and e['exe'].endswith('/strata-vision')]
    foreign = [e for e in engines if str(RUNTIME) not in e['exe']]
    if not native:
        raise IdentityMismatch('no engine executing from %s' % RUNTIME)
    if foreign:
        raise IdentityMismatch('foreign backbone processes: %s' % foreign)
    return {'server': server, 'engines': native, 'vision': vision, 'foreign': foreign}


def capture(base: str = BASE) -> dict:
    procs = live_processes(base)
    config_path = Path(procs['server']['config_path'])
    cfg = json.loads(config_path.read_text())
    health = http_get('/health', base)
    metrics = http_get('/metrics', base)
    engine_metrics = {k: metrics['engine'].get(k) for k in CAPACITY_KEYS}
    tokenizer_dir = Path(cfg['tokenizer'])
    tokenizer = {name: digest(tokenizer_dir / name) for name in TOKENIZER_FILES
                 if (tokenizer_dir / name).is_file()}
    pack = cfg['args'][cfg['args'].index('--pack') + 1] if '--pack' in cfg['args'] else None
    native = cfg['args'][cfg['args'].index('--native') + 1] if '--native' in cfg['args'] else None
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                              '--format=csv,noheader'], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()
    except Exception as exc:                                          # recorded, never fatal
        gpu = ['nvidia-smi failed: %s' % exc]
    load = []
    for _ in range(3):
        live = http_get('/metrics', base).get('live', {})
        load.append({'state': live.get('state'), 'running': live.get('running'),
                     'queued': live.get('queued'), 'phase': live.get('phase')})
        time.sleep(2)
    return {
        'schema': 'jev-abliterated-live-identity-v1',
        'captured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'base': base,
        'server_pid': procs['server']['pid'],
        'engine_pids': [e['pid'] for e in procs['engines']],
        'vision_pids': [e['pid'] for e in procs['vision']],
        'engine_exe_sha256': [digest(Path('/proc') / str(e['pid']) / 'exe') for e in procs['engines']],
        'runtime_files': {
            'serve/server.py': digest(RUNTIME / 'serve/server.py'),
            'engine-cuda12/strata': digest(RUNTIME / 'engine-cuda12/strata'),
            'engine-cuda12/strata-vision': digest(RUNTIME / 'engine-cuda12/strata-vision'),
        },
        'running_config': {'path': str(config_path), 'sha256': digest(config_path),
                           'model_name': cfg.get('model_name'), 'pack': pack, 'native_gguf': native,
                           'tokenizer_dir': str(tokenizer_dir)},
        'tokenizer_sha256': tokenizer,
        'health': health,
        'engine_metrics': engine_metrics,
        'load_samples': load,
        'gpu_snapshot': gpu,
        'head_files': {'best.pt': digest(RUN / 'best.pt'), 'norm.npz': digest(RUN / 'norm.npz'),
                       'manifest.json': digest(RUN / 'manifest.json')},
        'readout_builder_sha256': digest(V3 / 'readout.py'),
        'taps': TAPS, 'pools': 'last,mean',
        'policy': POLICY,
        'old_certificate': {'path': str(AD / 'compatibility.json'), 'sha256': digest(AD / 'compatibility.json'),
                            'binds': {k: v for k, v in json.loads((AD / 'compatibility.json')
                                                                  .read_text())['runtime'].items()},
                            'covers_live_pair': False},
        'watched_files': [str(RUNTIME / 'serve/server.py'), str(config_path),
                          str(RUNTIME / 'engine-cuda12/strata'), str(RUN / 'best.pt'),
                          str(RUN / 'norm.npz')],
    }


def _stats(paths):
    return [(p, os.stat(p).st_ino, os.stat(p).st_size, os.stat(p).st_mtime_ns)
            for p in paths if Path(p).exists()]


def verify(identity: dict, base: str = BASE) -> dict:
    """Re-read the live system and refuse when the answering pair is not the recorded one."""
    procs = live_processes(base)
    problems = []
    config_path = Path(procs['server']['config_path'])
    if digest(config_path) != identity['running_config']['sha256']:
        problems.append('running config changed')
    if digest(RUNTIME / 'serve/server.py') != identity['runtime_files']['serve/server.py']:
        problems.append('server.py changed')
    if digest(RUNTIME / 'engine-cuda12/strata') != identity['runtime_files']['engine-cuda12/strata']:
        problems.append('engine binary changed')
    exe_shas = [digest(Path('/proc') / str(e['pid']) / 'exe') for e in procs['engines']]
    if exe_shas != identity['engine_exe_sha256']:
        problems.append('executing engine differs from the recorded one')
    cfg = json.loads(config_path.read_text())
    if cfg.get('model_name') != identity['running_config']['model_name']:
        problems.append('model name changed')
    tokenizer_dir = Path(cfg['tokenizer'])
    live_tokenizer = {name: digest(tokenizer_dir / name) for name in identity['tokenizer_sha256']
                      if (tokenizer_dir / name).is_file()}
    if live_tokenizer != identity['tokenizer_sha256']:
        problems.append('tokenizer pack changed')
    metrics = http_get('/metrics', base)
    engine_metrics = {k: metrics['engine'].get(k) for k in CAPACITY_KEYS}
    if engine_metrics != identity['engine_metrics']:
        problems.append('capacity/placement changed: %s -> %s' % (identity['engine_metrics'],
                                                                  engine_metrics))
    for path, expected in identity['head_files'].items():
        target = RUN / path
        if digest(target) != expected:
            problems.append('head file changed: ' + path)
    if problems:
        raise IdentityMismatch('; '.join(problems))
    return {'engine_pids': [e['pid'] for e in procs['engines']],
            'live': metrics.get('live'), 'stats': _stats(identity['watched_files'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--verify', metavar='IDENTITY', help='re-verify an existing identity file')
    parser.add_argument('--force', action='store_true',
                        help='re-record over an existing file, archiving it as *.previous-<utc> first')
    args = parser.parse_args()
    target = Path(args.out)
    if args.verify:
        identity = json.loads(Path(args.verify).read_text())
        state = verify(identity, args.base)
        print(json.dumps({'verified': True, 'live': state['live']}, ensure_ascii=False))
        return 0
    identity = capture(args.base)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not args.force:
            raise SystemExit('identity file exists; use --force to archive it and re-record: '
                             + str(target))
        # The sweep binds every stage to ONE identity file, so re-recording must never silently
        # clobber the file a finished stage was measured against: move the old one aside first.
        archive = target.with_name('%s.previous-%s' % (target.name, time.strftime('%Y%m%d-%H%M%S', time.gmtime())))
        target.replace(archive)
    target.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'written': str(target), 'model': identity['running_config']['model_name'],
                      'config_sha256': identity['running_config']['sha256'][:12],
                      'engine_exe_sha256': [s[:12] for s in identity['engine_exe_sha256']],
                      'engine_metrics': identity['engine_metrics']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
