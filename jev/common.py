"""Shared, versioned contracts. Importing this module requires only stdlib."""
import hashlib
import json
import os
from pathlib import Path

SCHEMA = 'jev-v2.1'
PROMPT_VERSION = 'lossless-state-final-assistant-token-v2'

def volunteer_for_oom():
    """Protect the engine by making our bounded data/training jobs expendable."""
    try:
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('800')
    except OSError:
        pass

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def identity(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def source_hashes():
    root = Path(os.path.abspath(__file__)).parent
    return {p.name: digest(p) for p in sorted(root.glob('*.py'))}

def release_pages(path):
    # mmap pages are reclaimable, but proactive advice prevents sequential scans
    # from consuming the engine's remaining cgroup/RAM budget.
    if hasattr(os, 'posix_fadvise'):
        with open(path, 'rb') as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
