import os as _os
JEV_DATA = _os.environ.get('JEV_DATA', _os.path.expanduser('~/jev-data'))
STRATA = _os.environ.get('STRATA_ROOT', _os.path.expanduser('~/strata'))
JEV_PY = _os.environ.get('JEV_PY', 'python3')
"""Hash immutable packed weights once, under a bounded read-only background job."""
import argparse
import json
from pathlib import Path
from .common import atomic_json, digest

def weight_files(config):
    cfg = json.loads(Path(config).read_text())
    args = cfg['args']
    result = []
    for flag in ('--native', '--ple-gguf', '--expert-profile'):
        if flag in args:
            result.append(Path(args[args.index(flag) + 1]))
    if '--pack' in args:
        root = Path(args[args.index('--pack') + 1])
        result.extend(root / name for name in ('dense.bin', 'index.txt'))
    if len(result) < 4 or not all(p.is_file() for p in result):
        raise ValueError('model weight identity cannot be resolved')
    return sorted(set(result))

def verify_manifest(manifest, config):
    value = json.loads(Path(manifest).read_text())
    if value['model_config_sha256'] != digest(config) or value['mode'] != 'full-sha256':
        raise ValueError('model fingerprint/config mismatch')
    expected = {str(p) for p in weight_files(config)}
    if set(value['files']) != expected:
        raise ValueError('incomplete model fingerprint')
    for name, record in value['files'].items():
        st = Path(name).stat()
        if st.st_size != record['bytes'] or st.st_mtime_ns != record['mtime_ns']:
            raise ValueError('weights changed since fingerprint; recompute full hashes')
    return value

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Strata config whose artifacts are being identified')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        verify_manifest(out, args.config)
        print('cached fingerprint verified')
    else:
        files = {}
        for p in weight_files(args.config):
            before = p.stat()
            sha = digest(p)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('weights changed during hashing')
            files[str(p)] = {'sha256': sha, 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}
            print('hashed', p.name, after.st_size, flush=True)
        atomic_json(out, {'mode': 'full-sha256', 'model_config_sha256': digest(args.config), 'files': files})
