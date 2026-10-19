"""Bounded per-batch NPZ loading, train-only normalization and independent test."""
import argparse
import collections
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from common import SCHEMA, PROMPT_VERSION, atomic_json, digest, source_hashes, release_pages, volunteer_for_oom
from head import Head
from readout import layers_of

POOLS = ('span_last', 'span_mean')

def collect(paths, layers):
    records, ids, components, contracts = [], set(), {}, []
    for directory in paths:
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['schema'] != SCHEMA or manifest['prompt_version'] != PROMPT_VERSION:
            raise ValueError('legacy extraction contract')
        if manifest.get('legacy_smoke'):
            raise ValueError('legacy smoke vectors cannot enter formal training')
        if not set(layers).issubset(manifest['layers']):
            raise ValueError('requested layer not extracted')
        comparable = {k: manifest[k] for k in ('prompt_version', 'layers', 'pools', 'tokenizer', 'server_sha256', 'engine_sha256', 'model_config_sha256', 'model_artifacts', 'hc', 'n_embd', 'compute_backend')}
        if contracts and comparable != contracts[0]:
            raise ValueError('mixed model/prompt/extraction contracts')
        contracts.append(comparable)
        for path in sorted(directory.glob('*.json')):
            if path.name == 'manifest.json':
                continue
            rec = json.loads(path.read_text())
            item = rec['item']
            if item['uid'] in ids:
                raise ValueError('duplicate training/evaluation uid')
            ids.add(item['uid'])
            split = item['split']
            if split not in ('train', 'dev', 'test', 'ood', 'calibration'):
                raise ValueError('invalid split')
            comp = item['component']
            if comp in components and components[comp] != split:
                raise ValueError('component crosses train/eval split')
            components[comp] = split
            rec['path'] = str(path.with_suffix('.npz'))
            if digest(rec['path']) != rec['npz_sha256']:
                raise ValueError('corrupt shard')
            records.append(rec)
    if not records:
        raise ValueError('no shards')
    return records, contracts[0]

def load(rec, layers):
    k = len(rec['item']['options'])
    with np.load(rec['path'], allow_pickle=False) as z:
        x = np.stack([np.stack([np.stack([z[f'span{s}.L{l}.{p}'] for l in layers]) for p in POOLS]) for s in range(k + 2)])
    if x.dtype != np.float16 or not np.isfinite(x).all():
        raise ValueError('invalid archived features')
    release_pages(rec['path'])
    return x

def fit_norm(records, layers):
    n, mu, m2 = 0, None, None
    for rec in records:
        x = load(rec, layers).astype(np.float64)
        if mu is None:
            mu, m2 = np.zeros(x.shape[1:], np.float64), np.zeros(x.shape[1:], np.float64)
        count, mean = len(x), x.mean(0)
        delta = mean - mu
        m2 += ((x - mean) ** 2).sum(0) + delta ** 2 * n * count / (n + count)
        mu += delta * count / (n + count)
        n += count
    sd = np.sqrt(m2 / n)
    sd = np.maximum(sd, 1e-3)
    return mu.astype(np.float32), sd.astype(np.float32)

def preprocess(x, norm):
    # Same rounding on CPU and CUDA: raw FP16 -> FP32 zscore -> FP16 -> FP32.
    result = ((x.astype(np.float32) - norm[0]) / norm[1]).astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError('normalized FP16 overflow')
    return result.astype(np.float32)

def batch(records, layers, norm, device):
    arrays = [preprocess(load(rec, layers), norm) for rec in records]
    max_s, shape = max(map(len, arrays)), arrays[0].shape[1:]
    x = np.zeros((len(records), max_s) + shape, np.float32)
    target = np.zeros((len(records), max_s - 2), np.float32)
    gold, counts = [], []
    for i, (rec, a) in enumerate(zip(records, arrays)):
        item = rec['item']
        k = len(item['options'])
        x[i, :len(a)] = a
        target[i, :k] = item['target']
        gold.append(item['options'].index(item['gold']))
        counts.append(k)
    return tuple(torch.as_tensor(v, device=device) for v in (x, counts, target, gold))

def loss_parts(logits, target):
    valid = torch.isfinite(logits)
    logp = logits.log_softmax(-1).masked_fill(~valid, 0)
    nll = -(target * logp).sum(-1)
    brier = (logits.softmax(-1) - target).square().sum(-1)
    return nll, brier

def evaluate(head, records, layers, norm, device, batch_size):
    head.eval()
    correct, nll, brier, predictions, families = 0, 0., 0., [], collections.defaultdict(lambda: [0, 0])
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            x, counts, targets, gold = batch(rows, layers, norm, device)
            logits = head(x, counts)
            losses, bs = loss_parts(logits, targets)
            probs, pred = logits.softmax(-1), logits.argmax(-1)
            correct += int((pred == gold).sum())
            nll += float(losses.sum())
            brier += float(bs.sum())
            for i, rec in enumerate(rows):
                item = rec['item']
                family = str(item['metadata'].get('family', '?'))
                ok = int(pred[i]) == int(gold[i])
                families[family][0] += int(ok)
                families[family][1] += 1
                predictions.append({'uid': item['uid'], 'split': item['split'], 'gold': int(gold[i]), 'pred': int(pred[i]),
                                    'probabilities': probs[i, :int(counts[i])].cpu().tolist()})
    n = len(records)
    return {'n': n, 'correct': correct, 'accuracy': correct / n, 'nll': nll / n, 'brier': brier / n,
            'families': dict(families)}, predictions

def main():
    volunteer_for_oom()
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='15,29,-1')
    ap.add_argument('--blocks', type=int, choices=(2, 4, 8), default=2)
    ap.add_argument('--d', type=int, choices=(512, 1024), default=512)
    ap.add_argument('--fusion', choices=('concat', 'weighted'), default='concat')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--brier', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--train-subset', help='prepared train-N.jsonl; keeps complete original components')
    args = ap.parse_args()
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('CUDA_VISIBLE_DEVICES must be exactly 3')
    if min(args.batch, args.epochs, args.patience) < 1 or args.lr <= 0 or args.brier < 0:
        raise ValueError('invalid training schedule')
    out = Path(args.out)
    if out.exists():
        raise ValueError('output exists; use a new directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    layers = layers_of(args.layers)
    records, contract = collect(args.shards, layers)
    splits = {s: [r for r in records if r['item']['split'] == s] for s in ('train', 'dev', 'test', 'ood', 'calibration')}
    if args.train_subset:
        with open(args.train_subset, encoding='utf-8') as f:
            subset = {json.loads(line)['uid'] for line in f}
        available = {r['item']['uid'] for r in splits['train']}
        if not subset.issubset(available):
            raise ValueError('train subset has unextracted items')
        selected = [r for r in splits['train'] if r['item']['uid'] in subset]
        selected_components = {r['item']['component'] for r in selected}
        if any(r['item']['component'] in selected_components and r['item']['uid'] not in subset for r in splits['train']):
            raise ValueError('subset splits an original group')
        splits['train'] = selected
    if any(not splits[s] for s in ('train', 'dev', 'test')):
        raise ValueError('independent train/dev/test are required')
    out.mkdir(parents=True)
    manifest = {'schema': SCHEMA, 'args': vars(args), 'extraction': contract, 'code': source_hashes(),
                'shards': {r['item']['uid']: r['npz_sha256'] for r in records},
                'split_counts': {s: len(rs) for s, rs in splits.items()},
                'train_subset_sha256': digest(args.train_subset) if args.train_subset else None,
                'normalization': 'train spans only; FP16->FP32 zscore->FP16->FP32',
                'checkpoint_selection': 'dev NLL; test/ood evaluated only after selection'}
    atomic_json(out / 'manifest.json', manifest)
    norm = fit_norm(splits['train'], layers)
    np.savez(out / 'norm.npz', mu=norm[0], sd=norm[1])
    head = Head(len(layers), hc=contract['hc'], n_embd=contract['n_embd'], d=args.d, blocks=args.blocks,
                dropout=args.dropout, fusion=args.fusion).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    best, stale, history = float('inf'), 0, []
    for epoch in range(args.epochs):
        head.train()
        order = list(splits['train'])
        random.shuffle(order)
        t0 = time.monotonic()
        for start in range(0, len(order), args.batch):
            x, counts, target, _ = batch(order[start:start + args.batch], layers, norm, args.device)
            nll, brier = loss_parts(head(x, counts), target)
            loss = (nll + args.brier * brier).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        metrics, _ = evaluate(head, splits['dev'], layers, norm, args.device, args.batch)
        history.append({'epoch': epoch + 1, 'dev': metrics, 'seconds': time.monotonic() - t0})
        print(json.dumps(history[-1]), flush=True)
        if metrics['nll'] < best:
            best, stale = metrics['nll'], 0
            torch.save(head.state_dict(), out / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    head.load_state_dict(torch.load(out / 'best.pt', map_location=args.device, weights_only=True))
    report = {'history': history, 'parameters': sum(p.numel() for p in head.parameters()), 'final': {}}
    with (out / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for split in ('dev', 'test', 'ood', 'calibration'):
            if splits[split]:
                metrics, predictions = evaluate(head, splits[split], layers, norm, args.device, args.batch)
                report['final'][split] = metrics
                for pred in predictions:
                    f.write(json.dumps(pred) + '\n')
    report['files'] = {p.name: digest(p) for p in (out / 'best.pt', out / 'norm.npz', out / 'predictions.jsonl')}
    atomic_json(out / 'results.json', report)

if __name__ == '__main__':
    main()
