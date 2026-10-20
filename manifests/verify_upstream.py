"""Verify selected member commitments against a cached upstream .jsonl.gz.

python manifests/verify_upstream.py --raw upstream/train.jsonl.gz \
  --members manifests/splits/train > train-origin.json
With --remote HOST, raw path is read-only on that SSH host; only the members and
this source are sent on stdin/command. No file is written remotely. Stdlib only.
Raw records are streamed, never retained. Budget up to 256 MiB for train.
"""
import argparse
import collections
import gzip
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def prepare_row(row, m):
    order = list(range(len(row['options'])))
    random.Random(identity([20261012, row['id']])).shuffle(order)
    assert order == m['option_permutation'], 'seeded permutation mismatch'
    assert row['target'] == m['original_target'], 'original target mismatch'
    assert [row['target'][i] for i in order] == m['target'], 'prepared target mismatch'
    options = [row['options'][i] for i in order]
    return {'schema': 'jev-v2.1', 'uid': row['id'], 'group_id': row['group_id'],
            'component': m['component'], 'source': row['source'], 'source_split': row['split'],
            'split': m['split'], 'metadata': row['metadata'], 'kind': row['kind'],
            'state': row['state'], 'text': canonical(row['state']), 'question': row['question'],
            'options': options, 'crit': {x: x for x in options},
            'target': [row['target'][i] for i in order],
            'gold': row['options'][m['gold_original_index']],
            'original_options': row['options'], 'original_target': row['target'],
            'option_permutation': order, 'raw_sha256': identity(row)}


def member_lines(path):
    path = Path(path)
    paths = sorted(path.glob('members-*.jsonl')) if path.is_dir() else [path]
    if not paths:
        raise ValueError('no member shards')
    for filename in paths:
        with filename.open('rb') as stream:
            yield from stream


def verify(raw, member_stream, prepared_output=None):
    wanted = {}
    member_digest = hashlib.sha256()
    for line in member_stream:
        member_digest.update(line)
        row = json.loads(line)
        if row['id'] in wanted:
            raise ValueError('duplicate requested id')
        wanted[row['id']] = row
    compressed = hashlib.sha256()
    with open(raw, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            compressed.update(chunk)
    uncompressed = hashlib.sha256()
    found, found_order, licenses, total = set(), [], collections.Counter(), 0
    prepared_hash = hashlib.sha256()
    with gzip.open(raw, 'rb') as f:
        for line in f:
            uncompressed.update(line)
            row = json.loads(line)
            total += 1
            if row['id'] not in wanted:
                continue
            if row['id'] in found:
                raise ValueError('duplicate upstream id')
            m = wanted[row['id']]
            assert m['raw_sha256'] == identity(row), 'raw canonical hash mismatch'
            for k in ('group_id', 'source', 'kind'):
                assert m[k] == row[k], k + ' mismatch'
            assert m['source_split'] == row['split'], 'source split mismatch'
            assert m['metadata_sha256'] == identity(row['metadata']), 'metadata mismatch'
            content = {k: row[k] for k in ('state', 'question', 'kind')}
            content.update(original_options=row['options'], original_target=row['target'])
            assert m['content_sha256'] == identity(content), 'content mismatch'
            prepared = prepare_row(row, m)
            assert m['gold_sha256'] == identity(prepared['gold']), 'frozen hard gold mismatch'
            assert m['prepared_record_sha256'] == identity(prepared), 'prepared reconstruction mismatch'
            encoded = (canonical(prepared) + '\n').encode('utf-8')
            prepared_hash.update(encoded)
            if prepared_output:
                prepared_output.write(encoded)
            found.add(row['id'])
            found_order.append(row['id'])
            provenance = row['metadata'].get('provenance') or {}
            licenses[str(provenance.get('license', '<MISSING>'))] += 1
    assert len(found) == len(wanted), 'unmatched members'
    assert found_order == list(wanted), 'source/member ordering mismatch'
    return {'schema': 'jev-release-upstream-match-v1', 'upstream_rows': total,
            'matched_members': len(found), 'unmatched_members': len(wanted) - len(found),
            'members_file_sha256': member_digest.hexdigest(),
            'reconstructed_prepared_file_sha256': prepared_hash.hexdigest(),
            'raw_gzip_sha256': compressed.hexdigest(), 'raw_uncompressed_sha256': uncompressed.hexdigest(),
            'selected_row_provenance_licenses': dict(sorted(licenses.items())),
            'checked': ['raw_sha256 over complete upstream record', 'id', 'group_id', 'source',
                        'source_split', 'kind', 'complete metadata hash', 'content hash including original options/targets',
                        'exact prepared-row reconstruction including option/target permutation', 'member/source ordering']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--members')
    ap.add_argument('--remote')
    ap.add_argument('--prepared-out', help='optional NEW local file; a failed run can leave a partial file')
    args = ap.parse_args()
    if args.remote:
        if args.prepared_out:
            ap.error('--prepared-out is local-only; remote verification is read-only')
        if not args.members:
            ap.error('--remote requires --members')
        import base64
        import shlex
        bootstrap = 'import sys,base64;exec(base64.b64decode(sys.stdin.buffer.readline()))'
        command = 'python3 -c ' + shlex.quote(bootstrap) + ' --raw ' + shlex.quote(args.raw)
        source = base64.b64encode(Path(__file__).read_bytes()) + b'\n'
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', args.remote, command],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            process.stdin.write(source)
            for line in member_lines(args.members):
                process.stdin.write(line)
            process.stdin.close()
            result = process.stdout.read()
            if process.wait() != 0:
                raise RuntimeError('read-only upstream verifier failed; no receipt updated')
            sys.stdout.buffer.write(result)
        finally:
            if process.poll() is None:
                process.terminate()
    else:
        stream = member_lines(args.members) if args.members else sys.stdin.buffer
        output = open(args.prepared_out, 'xb') if args.prepared_out else None
        try:
            print(json.dumps(verify(args.raw, stream, output), sort_keys=True, indent=2, ensure_ascii=False))
        finally:
            if output:
                output.close()
            if args.members:
                stream.close()


if __name__ == '__main__':
    main()
