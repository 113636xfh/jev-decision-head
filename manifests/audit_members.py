"""Recompute public member counts, categorical and exact-hash intersections.

python manifests/audit_members.py > audit.json
No task bodies or activations are read. Memory is bounded by 23,718 small member
records plus sets of their hashes (budget up to 384 MiB on CPython).
"""
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIMENSIONS = ('id', 'group_id', 'component', 'source', 'family', 'domain',
              'scenario_family', 'template_id', 'semantic_context_sha256',
              'source_instance_id_sha256', 'state_sha256', 'state_question_sha256',
              'content_sha256', 'raw_sha256', 'prepared_record_sha256')


def key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def audit():
    rows, sets, splits, value_counts = {}, {}, {}, {}
    for split in ('train', 'dev', 'test', 'ood', 'cal'):
        paths = sorted((ROOT / 'splits' / split).glob('members-*.jsonl'))
        assert paths, 'missing member shards'
        rows[split] = []
        digest, member_bytes, shard_info = hashlib.sha256(), 0, []
        for path in paths:
            raw = path.read_bytes()
            digest.update(raw)
            member_bytes += len(raw)
            shard_info.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
                               'rows': len(raw.splitlines()), 'sha256': hashlib.sha256(raw).hexdigest()})
            rows[split].extend(json.loads(line) for line in raw.splitlines())
        value_counts[split] = {d: Counter(key(r[d]) for r in rows[split] if d in r and r[d] is not None and r[d] != '') for d in DIMENSIONS}
        sets[split] = {d: set(counts) for d, counts in value_counts[split].items()}
        assert len(sets[split]['id']) == len(rows[split]), 'duplicate id'
        for r in rows[split]:
            value_hash = lambda value: hashlib.sha256(key(value).encode('utf-8')).hexdigest()
            assert r['original_target_sha256'] == value_hash(r['original_target'])
            assert r['option_permutation_sha256'] == value_hash(r['option_permutation'])
            assert r['target'] == [r['original_target'][i] for i in r['option_permutation']]
            assert r['id'] == r['uid']
            assert r['schema'] == 'jev-release-member-v1'
            assert r['split'] == ('calibration' if split == 'cal' else split)
        splits[split] = {'rows': len(rows[split]), 'members_file_sha256': digest.hexdigest(),
                         'members_file_bytes': member_bytes, 'shards': shard_info,
                         'hard_gold': {'not_first_original_argmax': sum(r['gold_original_index'] != max(range(len(r['original_target'])), key=lambda i: r['original_target'][i]) for r in rows[split]),
                                       'outside_original_target_maxima': sum(r['original_target'][r['gold_original_index']] != max(r['original_target']) for r in rows[split])},
                         'unique_nonempty': {d: len(v) for d, v in sets[split].items()},
                         'missing_or_null': {d: sum(d not in r or r[d] is None or r[d] == '' for r in rows[split]) for d in DIMENSIONS},
                         'counts': {d: dict(sorted(Counter(key(r[d]) if d in r else '<MISSING>' for r in rows[split]).items()))
                                    for d in ('source', 'source_split', 'split', 'family', 'domain', 'provenance_split_policy')}}
    intersections = {}
    for a, b in combinations(rows, 2):
        dims = {}
        for d in DIMENSIONS:
            overlap = sets[a][d] & sets[b][d]
            dims[d] = {'distinct_intersection': len(overlap),
                       a + '_rows_with_shared_value': sum(value_counts[a][d][v] for v in overlap),
                       b + '_rows_with_shared_value': sum(value_counts[b][d][v] for v in overlap)}
            if d in ('source', 'family', 'domain', 'scenario_family', 'template_id'):
                dims[d]['shared_values'] = [json.loads(v) for v in sorted(overlap)]
        intersections[a + '__' + b] = dims
    strict = ('id', 'group_id', 'component', 'semantic_context_sha256', 'state_question_sha256', 'content_sha256', 'raw_sha256')
    return {'schema': 'jev-release-members-audit-v1', 'total_rows': sum(len(v) for v in rows.values()),
            'splits': splits, 'pairwise_intersections': intersections,
            'zero_cross_split_on_checked_keys': {d: all(v[d]['distinct_intersection'] == 0 for v in intersections.values()) for d in strict},
            'limits': ['Counts use exact strings/JSON values and hashes, not semantic similarity.',
                       'Missing/null/empty categorical values are excluded from intersections and counted separately.',
                       'Hash commitments permit authorized holders to recompute content checks, not recover withheld bodies.',
                       'These subsets do not prove complete upstream-split or foundation-pretraining disjointness.']}


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
