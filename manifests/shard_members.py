"""Split member JSONL into <=512-row shards, each <=1,500,000 bytes.

python manifests/shard_members.py --input train.members.jsonl --out manifests/splits/train
Input '-' reads stdin. Output directory must not exist. Concatenating shards in
lexical filename order recovers the exact input bytes, including LF delimiters.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = sys.stdin.buffer if args.input == '-' else open(args.input, 'rb')
    stream = None
    n = size = index = 0
    total = hashlib.sha256()
    try:
        for line in source:
            json.loads(line)
            if not line.endswith(b'\n') or len(line) > 1500000:
                raise ValueError('invalid/oversized member line')
            if stream is None or n >= 512 or size + len(line) > 1500000:
                if stream:
                    stream.close()
                stream = (out / ('members-%05d.jsonl' % index)).open('xb')
                index += 1
                n = size = 0
            stream.write(line)
            total.update(line)
            n += 1
            size += len(line)
    finally:
        if stream:
            stream.close()
        if args.input != '-':
            source.close()
    print(json.dumps({'shards': index, 'concatenated_sha256': total.hexdigest()}))


if __name__ == '__main__':
    main()
