"""Derive decision-only evidence from existing full dev output; never infer/re-extract.

This is NOT the missing original 2026-10-20 live smoke's per-item output.
The source is the recorded full dev replay from 2026-10-18.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True, help='Directory containing the recorded full-dev archive')
parser.add_argument('--output-dir', type=Path, required=True, help='New derived evidence output directory')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_DIR = args.source_dir.resolve()
ARCHIVE = SOURCE_DIR / 'evidence-abliterated-eval-20261018.tar.gz'
FIELDS = ['uid', 'split', 'status', 'n_options', 'gold', 'pred_old',
          'correct_old', 'confidence_old']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exclusive_json(path, obj):
    with path.open('x', encoding='utf-8') as sink:
        sink.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    catalog = []
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        matches = [m for m in archive.getmembers() if m.name == 'rows-dev.jsonl']
        for member in matches:
            content = archive.extractfile(member).read()
            catalog.append({'name': member.name, 'offset': member.offset,
                            'bytes': len(content), 'sha256': sha(content)})
        if not matches:
            raise RuntimeError('full dev member missing')
        # tarfile name lookup selects the last occurrence. Record all copies and refuse divergence.
        if len({r['sha256'] for r in catalog}) != 1:
            raise RuntimeError('duplicate archive members differ; manual review required')
        raw = archive.extractfile(matches[-1]).read()
        smoke_members = []
        for m in archive.getmembers():
            if m.isfile() and ('smoke' in m.name.lower() or m.name.endswith('rows-dev-summary.json')):
                data = archive.extractfile(m).read()
                smoke_members.append({'name': m.name, 'bytes': len(data), 'sha256': sha(data)})
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records) != 1996 or len({r['uid'] for r in records}) != 1996:
        raise RuntimeError('source is not a unique full 1996-row dev file')
    first = records[:20]
    if any(r['status'] != 'ok' or r['split'] != 'dev' for r in first):
        raise RuntimeError('first twenty contain failed or non-dev records')
    projected = [{key: row[key] for key in FIELDS} for row in first]
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in projected).encode('utf-8')
    target = OUT / 'rows-smoke20-derived-archive-v2.jsonl'
    with target.open('xb') as sink:
        sink.write(data)
    exclusive_json(OUT / 'rows-smoke20-derived-provenance-v2.json', {
        'evidence_kind': 'derived-archived-decision-replay',
        'original_smoke20_rows_recovered': False,
        'not_original_smoke': 'Do not combine these rows with the existing smoke20 summary, identity, latency, or live accuracy claims.',
        'source_archive_path': str(ARCHIVE), 'source_archive_sha256': sha(ARCHIVE.read_bytes()),
        'source_member': 'rows-dev.jsonl', 'source_member_sha256': sha(raw),
        'source_member_bytes': len(raw), 'source_rows': len(records),
        'source_member_copies': catalog,
        'selection': 'First 20 nonblank records, original file order; no filtering or reordering.',
        'transformation': 'Copy only archived decision fields verbatim. No prediction, feature extraction, model load, live request or synthetic values.',
        'copied_fields': FIELDS, 'derived_file': target.name, 'derived_sha256': sha(data),
        'derived_rows': len(projected), 'uids': [r['uid'] for r in projected],
        'archived_correct': sum(r['correct_old'] for r in projected),
        'search_scope': ['Main workspace work/jev-abliterated-eval-20261018 direct files and tar members',
                         'Read-only recursive <BUILD_ROOT> search on e5 for rows-smoke20* and smoke20-check*'],
        'search_log': 'recovery-search-server-v2.log',
        'other_smoke_archive_members': smoke_members,
        'boundary': 'Validates stored archived predictions against baseline only; does not recover the missing original live smoke or establish new online compatibility.'})
    exclusive_json(OUT / 'recovery-local-catalog-v2.json', {
        'source_directory': str(SOURCE_DIR),
        'direct_files': [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(SOURCE_DIR.iterdir()) if p.is_file()],
        'original_smoke20_named_member_found': False,
        'archive_smoke_members': smoke_members})


if __name__ == '__main__':
    main()
