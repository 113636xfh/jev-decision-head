"""Decision CLI: head-only or head-then-decode cascade, live or from cached archives.

Examples (paths are relative to the release root; `--identity` is produced on the
serving host by `python -m jev.identity --out identity/runtime-identity.json`):

  python -m jev.predict --identity identity/runtime-identity.json \
      --tokenizer-dir /path/to/served/tokenizer \
      --state '{"asset":"X","position":10}' --question 'Sell now?' \
      --options 'yes:no' --out decision.json

  python -m jev.cascade --identity identity/runtime-identity.json \
      --tokenizer-dir ... --items items.jsonl --out predictions.jsonl

The client runs against any patched Strata server.  Record which server you run
against with `python -m jev.identity`; the environment we trained and tested on
is pinned in `REPRODUCE.md`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .cascade import Policy, decide, summarize
from .common import atomic_json, canonical, digest, identity as item_identity, volunteer_for_oom
from .domain import DecodeClient
from .fingerprint_model import verify_manifest
from .runtime import MANIFEST, WEIGHTS, HeadRuntime, load_json

DEFAULT_POLICY = Path(__file__).resolve().parent.parent / 'policies/dev-thresholds.json'


def parse_options(values):
    options = []
    for value in values:
        options.extend(part for part in value.split('\t') if part.strip() != '')
    if len(options) < 2:
        raise ValueError('at least two options are required')
    return options


def build_item(args):
    options = parse_options(args.options)
    state = json.loads(args.state) if args.state.strip().startswith(('{', '[')) else args.state
    crit = {}
    if args.criteria:
        pairs = [c.split('=', 1) for c in args.criteria]
        if len(pairs) != len(options):
            raise ValueError('--criteria must give one description per option')
        if any(pair[0] != opt for opt, pair in zip(options, pairs)):
            raise ValueError('--criteria option names must match --options in order')
        crit = {opt: pair[1] for opt, pair in zip(options, pairs)}
    else:
        crit = {opt: opt for opt in options}
    return {'uid': 'adhoc-%s' % item_identity({'state': state, 'question': args.question,
                                               'options': options})[:16],
            'group_id': 'adhoc', 'split': 'unlabeled', 'text': canonical(state),
            'question': args.question, 'options': options, 'crit': crit,
            'metadata': {'supervision_origin': 'none'}}


def cached_provider(shards, layers):
    from . import train as trainer
    records, _ = trainer.collect(shards, layers)
    index = {r['item']['uid']: r for r in records}

    def provider(item):
        record = index[item['uid']]
        if item_identity(item) != record['item_sha256']:
            raise ValueError('cached item changed; never relabel a feature prompt')
        return trainer.load(record, layers)
    return provider


def runtime_from_args(args):
    return HeadRuntime(load_json(args.identity), base=args.base, device=args.device,
                       tokenizer_dir=args.tokenizer_dir, weights_dir=args.weights)


def policy_from_args(args):
    if args.policy:
        return Policy(**json.loads(Path(args.policy).read_text(encoding='utf-8'))['policy'])
    if DEFAULT_POLICY.is_file():
        return Policy(**json.loads(DEFAULT_POLICY.read_text(encoding='utf-8'))['policy'])
    return Policy(args.min_confidence, args.min_margin, args.max_normalized_entropy)


def main(default_route='cascade'):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--identity', required=True, help='identity JSON recorded on the serving host')
    ap.add_argument('--base', default='http://127.0.0.1:8100')
    ap.add_argument('--weights', default=str(WEIGHTS), help='directory with best.pt and norm.npz')
    ap.add_argument('--tokenizer-dir', help='served model tokenizer directory (tokenizer.json + chat_template.jinja)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--route', choices=['head', 'cascade'], default=default_route)
    ap.add_argument('--model', help='served model name for the decode fallback (default: from identity)')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--items', help='JSONL of prepared decision items')
    ap.add_argument('--state', help='single item: state (JSON or free text)')
    ap.add_argument('--question', help='single item: question')
    ap.add_argument('--options', nargs='+', help='single item: option texts')
    ap.add_argument('--criteria', nargs='+', help='single item: OPTION=description pairs')
    ap.add_argument('--cached-shards', nargs='+', help='replay archived feature shards instead of live readout')
    ap.add_argument('--policy', help='policy JSON selected on dev only')
    ap.add_argument('--min-confidence', type=float, default=0.8)
    ap.add_argument('--min-margin', type=float, default=0.2)
    ap.add_argument('--max-normalized-entropy', type=float, default=0.55)
    ap.add_argument('--model-identity', help='optional backbone artifact identity for a strict check')
    ap.add_argument('--model-config', help='optional backbone config used by the serving host')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    volunteer_for_oom()
    torch.set_num_threads(2)
    if args.items and (args.state or args.question or args.options):
        raise ValueError('--items cannot be combined with --state/--question/--options')
    if not args.items and not (args.state and args.question and args.options):
        raise ValueError('give --items, or --state --question --options together')
    runtime = runtime_from_args(args)
    if args.model_identity and args.model_config:
        artifacts = verify_manifest(args.model_identity, args.model_config)
        if artifacts != json.loads(Path(MANIFEST).read_text(encoding='utf-8'))['extraction'].get('model_artifacts'):
            raise ValueError('current backbone artifacts differ from the trained head manifest')

    if args.cached_shards:
        provider = cached_provider(args.cached_shards, runtime.geometry['layers'])

        def head_predictor(item):
            raw = provider(item)
            probs = runtime.probabilities(raw, len(item['options']))
            return probs
    else:
        def head_predictor(item):
            return runtime.predict(item)['probabilities']

    policy = policy_from_args(args)
    model = args.model or runtime.identity.get('running_config', {}).get('model_name')
    decode = DecodeClient(args.base, model, args.max_tokens) if args.route == 'cascade' else None

    if args.items:
        items = [json.loads(line) for line in Path(args.items).read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        items = [build_item(args)]

    out = Path(args.out)
    if out.exists():
        raise ValueError('output exists: %s (refusing to mix runs)' % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open('w', encoding='utf-8') as handle:
        for item in items:
            if args.route == 'head':
                probs = head_predictor(item)
                assessment = policy.assess(probs)
                row = {'route': 'head', 'choice': item['options'][int(np.argmax(probs))],
                       'head_probabilities': probs, 'assessment': assessment}
            else:
                row = decide(item, head_predictor, decode, policy)
            row.update(uid=item.get('uid'), split=item.get('split'), group_id=item.get('group_id'),
                       options=item['options'],
                       independent_gold=item.get('gold')
                       if item.get('metadata', {}).get('supervision_origin') == 'provided' else None)
            rows.append(row)
            handle.write(canonical(row) + '\n')
            handle.flush()
    atomic_json(out.with_suffix('.summary.json'),
                {'route': args.route, 'policy': vars(policy),
                 'thresholds_status': 'dev-selected default; select on dev only',
                 'head_geometry': runtime.geometry, 'metrics': summarize(rows)})
    print(json.dumps({'items': len(rows),
                      'metrics': summarize(rows)}, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
