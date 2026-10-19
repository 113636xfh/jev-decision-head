"""Serial, atomic, resumable extraction. No corpus or vectors loaded wholesale."""
import argparse
import fcntl
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
from common import PROMPT_VERSION, SCHEMA, atomic_json, digest, identity, source_hashes, volunteer_for_oom
from readout import build, layers_of, validate_response
from fingerprint_model import verify_manifest

def tokenizer():
    # Reuse existing production dependencies; do not install anything.
    sys.path.append('/home/xfh/strata/v0.1.41/.venv/lib/python3.12/site-packages')
    sys.path.insert(0, '/home/xfh/strata/v0.1.41')
    sys.path.insert(0, '/home/xfh/strata/v0.1.41/tools')
    sys.path.insert(0, '/home/xfh/strata/v0.1.41/serve')
    import strata_tokenizer as st
    from serve.frontend import ChatTemplate
    root = Path('/strata/data/packs/iq3_s/tokenizer')
    vocab = json.loads((root / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = st.Tokenizer(tokens, (root / 'merges.txt').read_text().split('\n'), json.loads((root / 'token_type.json').read_text()))
    return tok, ChatTemplate(root / 'chat_template.jinja'), {p.name: digest(p) for p in root.iterdir() if p.is_file()}

def provenance(args, token_hashes):
    return {'schema': SCHEMA, 'prompt_version': PROMPT_VERSION, 'input_sha256': digest(args.items),
            'layers': layers_of(args.layers), 'pools': ['span_last', 'span_mean'], 'tokenizer': token_hashes,
            'server_sha256': digest(args.server_file), 'engine_sha256': digest(args.engine_file),
            'model_config_sha256': digest(args.model_config),
            'model_artifacts': verify_manifest(args.model_identity, args.model_config) if args.model_identity else
                               {str(Path(p).absolute()): digest(p) for p in args.model_artifact},
            'code': source_hashes(), 'base': args.base, 'hc': 4, 'n_embd': 2560,
            'compute_backend': 'batched-prefill-one-shot-r2'}

def main():
    volunteer_for_oom()
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='15,29,-1')
    ap.add_argument('--base', default='http://127.0.0.1:8100')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--server-file', default='/home/xfh/strata/v0.1.41/serve/server.py')
    ap.add_argument('--engine-file', default='/home/xfh/strata/v0.1.41/engine-cuda12/strata')
    ap.add_argument('--model-config', default='/home/xfh/strata/v0.1.41/strata-iq3_s.json')
    ap.add_argument('--model-artifact', action='append', default=[], help='small smoke identity files only; formal runs require --model-identity')
    ap.add_argument('--model-identity', help='full weight hash manifest from fingerprint_model.py')
    ap.add_argument('--acceptance', default='/home/xfh/data/jev-v3-fast/evidence/online-acceptance.json')
    ap.add_argument('--max-prompt-tokens', type=int, default=4096,
                    help='conservative until repaired engine passes long-prompt acceptance; never truncate')
    ap.add_argument('--allow-legacy-smoke', action='store_true', help='explicitly permit <=4 short smoke items on 0006; never formal data')
    args = ap.parse_args()
    if args.allow_legacy_smoke and (not 0 < args.limit <= 4 or args.max_prompt_tokens > 4096):
        raise ValueError('legacy engine is allowed only for <=4 short smoke items')
    if not args.model_identity and (not args.allow_legacy_smoke or not args.model_artifact):
        raise ValueError('formal extraction requires a full model weight fingerprint')
    # One advisory lock for every v2 extractor on this host, not one per output.
    lock = open('/tmp/jev-v2-extraction.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, tpl, token_hashes = tokenizer()
    contract = provenance(args, token_hashes)
    contract['max_prompt_tokens'] = args.max_prompt_tokens
    contract['legacy_smoke'] = args.allow_legacy_smoke
    if not args.allow_legacy_smoke:
        accepted = json.loads(Path(args.acceptance).read_text())
        expected_hashes = {k: contract[k] for k in ('server_sha256', 'engine_sha256', 'model_config_sha256')}
        if not accepted.get('passed') or accepted.get('installed_hashes') != expected_hashes:
            raise ValueError('installed server/engine/config have no matching passed online acceptance')
        contract['acceptance_sha256'] = digest(args.acceptance)
    path = out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('resume provenance changed; use a new output directory')
    else:
        atomic_json(path, contract)
    done = 0
    with open(args.items, encoding='utf-8') as rows:
        for line in rows:
            item = json.loads(line)
            if item.get('schema') != SCHEMA or item['split'] == 'quarantine':
                raise ValueError('incompatible/quarantined item')
            name = identity(item['uid'])
            target = out / (name + '.npz')
            record_path = out / (name + '.json')
            item_sha = identity(item)
            if target.exists() and record_path.exists():
                rec = json.loads(record_path.read_text())
                if rec['item_sha256'] != item_sha or rec['npz_sha256'] != digest(target):
                    raise ValueError('resume item/shard hash mismatch')
                done += 1
            else:
                req = build(item, tok, tpl, args.layers, args.max_prompt_tokens)
                vectors, t0 = {}, time.monotonic()
                for offset in range(0, len(req['latent_spans']), 16):
                    chunk = {**req, 'latent_spans': req['latent_spans'][offset:offset + 16], 'latent_backend': 'fast'}
                    request = urllib.request.Request(args.base + '/v1/latent', data=json.dumps(chunk).encode(),
                                                     headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        blob = json.load(response)
                    if not args.allow_legacy_smoke and blob.get('contract_version') != 'jev-readout-v2':
                        raise ValueError('formal extraction requires accepted repaired server/engine pair')
                    if blob.get('readout_backend') != 'batched-prefill':
                        raise ValueError('fast backend required; refuse silent serial fallback')
                    arrays = validate_response(blob, chunk)
                    for (s, layer, pool), data in arrays.items():
                        v = np.frombuffer(data, dtype='<f4').astype(np.float16)
                        if not np.isfinite(v).all():
                            raise ValueError('FP16 archival overflow')
                        vectors[f'span{s + offset}.L{layer}.{pool}'] = v
                tmp = target.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.savez(f, latent_ids=np.asarray(req['latent_ids'], dtype=np.int32),
                             latent_spans=np.asarray(req['latent_spans'], dtype=np.int64), **vectors)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
                rec = {'item': item, 'item_sha256': item_sha, 'npz_sha256': digest(target),
                       'prompt_sha256': identity(req), 'prompt_tokens': len(req['latent_ids']),
                       'spans': req['latent_spans'], 'prompt_version': PROMPT_VERSION,
                       'elapsed_ms': (time.monotonic() - t0) * 1000}
                atomic_json(record_path, rec)
                done += 1
                print(json.dumps({'completed': done, 'split': item['split'], 'tokens': rec['prompt_tokens'], 'ms': round(rec['elapsed_ms'])}), flush=True)
            if args.limit and done >= args.limit:
                break

if __name__ == '__main__':
    main()
