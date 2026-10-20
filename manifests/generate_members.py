"""Streaming, stdlib-only member exporter. Reads prepared JSONL, never activations.

Usage: python manifests/generate_members.py --input data/train-16000.jsonl > train.members.jsonl
Use --summary for a small stdout summary instead; --input - reads stdin.
Canonical JSON is Python json.dumps(ensure_ascii=False, sort_keys=True,
separators=(',', ':'), allow_nan=False), encoded UTF-8, without a newline.
This is a specified Python JSON protocol, NOT RFC 8785/JCS.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def member(row):
    required = ('uid', 'group_id', 'source', 'split', 'source_split', 'metadata', 'state',
                'question', 'kind', 'original_options', 'original_target', 'options',
                'target', 'option_permutation', 'component', 'raw_sha256')
    if any(k not in row for k in required):
        raise ValueError('missing original prepared field')
    meta = row['metadata']
    order = row['option_permutation']
    if sorted(order) != list(range(len(row['options']))):
        raise ValueError('invalid option permutation')
    if row['options'] != [row['original_options'][i] for i in order]:
        raise ValueError('options do not follow permutation')
    if row['target'] != [row['original_target'][i] for i in order]:
        raise ValueError('target does not follow permutation')
    if json.loads(row['text']) != row['state']:
        raise ValueError('state round trip failed')
    for key in ('uid', 'group_id', 'source', 'split', 'source_split', 'component', 'raw_sha256'):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError('invalid member identifier')
    # Keep exact original values; id is an explicit alias for the prepared uid.
    out = {k: row[k] for k in ('uid', 'group_id', 'source', 'split', 'source_split', 'component', 'kind', 'raw_sha256')}
    out.update({'id': row['uid'], 'schema': 'jev-release-member-v1',
                'content_sha256': identity({k: row[k] for k in ('state', 'question', 'kind', 'original_options', 'original_target')}),
                'prepared_record_sha256': identity(row),
                'metadata_sha256': identity(meta),
                'state_question_sha256': identity([row['state'], row['question']]),
                'state_sha256': identity(row['state']),
                'original_options_sha256': identity(row['original_options']),
                'original_target_sha256': identity(row['original_target']),
                'original_target': row['original_target'], 'target': row['target'],
                'option_permutation': order,
                'gold_original_index': row['original_options'].index(row['gold']),
                'gold_sha256': identity(row['gold']),
                'option_permutation_sha256': identity(order)})
    for key in ('family', 'domain', 'scenario_family', 'template_id', 'semantic_context_sha256'):
        # These are categorical/hash fields, not task bodies. Missing != null.
        if key in meta:
            out[key] = meta[key]
    if 'source_instance_id' in meta:
        out['source_instance_id_sha256'] = identity(meta['source_instance_id'])
    provenance = meta.get('provenance') or {}
    if 'split_policy' in provenance:
        out['provenance_split_policy'] = provenance['split_policy']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()
    stream = sys.stdin.buffer if args.input == '-' else open(args.input, 'rb')
    file_hash = hashlib.sha256()
    member_hash = hashlib.sha256()
    n = 0
    counters = {k: Counter() for k in ('source', 'split', 'source_split', 'family', 'domain', 'provenance_split_policy')}
    try:
        for line in stream:
            file_hash.update(line)
            if not line.strip():
                raise ValueError('blank input line')
            row = member(json.loads(line))
            encoded = (canonical(row) + '\n').encode('utf-8')
            member_hash.update(encoded)
            n += 1
            for key, counter in counters.items():
                counter[canonical(row[key]) if key in row else '<MISSING>'] += 1
            if not args.summary:
                sys.stdout.buffer.write(encoded)
    finally:
        if args.input != '-':
            stream.close()
    if args.summary:
        print(json.dumps({'rows': n, 'input_file_sha256': file_hash.hexdigest(),
                          'members_file_sha256': member_hash.hexdigest(),
                          'counts': {k: dict(v) for k, v in counters.items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
