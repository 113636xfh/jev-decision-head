"""Offline self-check: runs with no server, no GPU, and no training workspace.

  python -m jev.selfcheck

It proves the release is self-contained at the level that can be proven offline:
every module imports, the published weights load and produce a valid distribution,
the cascade policy behaves as documented, and the public benchmark snapshot parses.
It does not prove online numerical compatibility - that needs the patched server
and an identity recorded by `jev.identity`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
MODULES = ['common', 'readout', 'head', 'gpu_data', 'domain', 'cascade', 'identity',
           'fingerprint_model', 'runtime', 'head_loader', 'predict',
           'cascade', 'jevbench', 'upstream_scoring', 'check_baseline', 'eval_replay',
           'eval_benchmark', 'probe_drift', 'train', 'tune_policy', 'analyze_flips',
           'finalize_rows', 'run_all']


def step(name, ok, detail=''):
    print('%-4s %s%s' % ('ok' if ok else 'FAIL', name, ': ' + detail if detail else ''), flush=True)
    if not ok:
        raise SystemExit('self-check failed at: ' + name)


def main():
    import importlib
    torch.set_num_threads(2)

    for name in MODULES:
        importlib.import_module('jev.' + name)
    step('imports', True, '%d modules import without the training workspace' % len(set(MODULES)))

    # extract.py deliberately needs the served model's own tokenizer stack and is not
    # importable outside the Strata source tree; say so instead of pretending.
    try:
        importlib.import_module('jev.extract')
        note = 'extract imported (Strata tree present)'
    except ModuleNotFoundError as exc:
        note = 'extract needs the Strata source tree (%s) - expected off that host' % exc.name
    step('extract boundary', True, note)

    from .head import Head
    from .runtime import head_geometry, load_head
    geometry = head_geometry()
    head, mu, sd, geometry = load_head(device='cpu')
    parameters = sum(p.numel() for p in head.parameters())
    step('weights load', True, '%d parameters, geometry %s' % (parameters, {k: geometry[k]
                                                                            for k in ('layers', 'blocks', 'd', 'fusion')}))
    step('parameter count', parameters == 13194368, 'expected 13,194,368, got %d' % parameters)
    step('norm shapes', tuple(mu.shape) == (2, len(geometry['layers']), 4 * geometry['n_embd']),
         'mu/sd %s = pools x layers x (hc*n_embd)' % (tuple(mu.shape),))

    k = 3
    raw = np.zeros((1, k + 2, 2, len(geometry['layers']), 4 * 2560), np.float16)
    raw[:] = np.float16(0.01)
    from .gpu_data import normalize_gpu
    with torch.inference_mode():
        normalized = normalize_gpu(torch.from_numpy(raw), mu, sd).float()
        logits = head(normalized, torch.tensor([k]))[0]
        probs = torch.softmax(logits, -1).tolist()
    step('forward pass', len(probs) == k and abs(sum(probs) - 1) < 1e-5 and all(p > 0 for p in probs),
         'k=%d probabilities %s' % (k, [round(p, 4) for p in probs]))

    from .cascade import Policy, decide
    policy_path = RELEASE / 'policies/dev-thresholds.json'
    published = json.loads(policy_path.read_text(encoding='utf-8'))
    policy = Policy(**published['policy'])
    step('published policy', set(published['policy']) >= {'min_confidence', 'min_margin'},
         'selected on dev only: %s' % published['policy'])
    item = {'uid': 'x', 'group_id': 'g', 'split': 'dev', 'text': 's', 'question': 'q',
            'options': ['a', 'b', 'c'], 'crit': {'a': 'a', 'b': 'b', 'c': 'c'},
            'metadata': {'supervision_origin': 'provided'}, 'gold': 'b'}
    confident = decide(item, lambda _: [0.05, 0.90, 0.05], lambda _: {'choice': 'c'}, policy)
    uncertain = decide(item, lambda _: [0.34, 0.33, 0.33], lambda _: {'choice': 'b'}, policy)
    step('cascade routes', confident['route'] == 'head' and uncertain['route'] == 'decode',
         'confident->%s, uncertain->%s' % (confident['route'], uncertain['route']))

    from .jevbench import load_tasks, build_item, score_one
    tasks = load_tasks()
    criterion = build_item(tasks[0], 'criterion_as_option')
    label = build_item(tasks[0], 'label_as_option')
    scored = score_one({lab: 1.0 / len(label['options']) for lab in label['options']},
                       [str(x) for x in tasks[0]['labels']], tasks[0]['expected'],
                       tasks[0]['question']['type'])
    step('benchmark snapshot', len(tasks) == 231 and len(criterion['options']) >= 2,
         '231 public decisions, both renderings build, scorer returns %s' % sorted(scored))

    baseline = RELEASE / 'baselines/middle5-s1.dev-predictions.jsonl'
    rows = [json.loads(l) for l in baseline.read_text(encoding='utf-8').splitlines() if l.strip()]
    step('published baseline', len(rows) == 1996, '%d dev rows for check_baseline.py' % len(rows))

    print(json.dumps({'selfcheck': 'passed', 'parameters': parameters,
                      'note': 'offline only; online use needs the patched server and jev.identity'},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
