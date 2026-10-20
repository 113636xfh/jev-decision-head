"""JevBench-Open helpers vendored into the release.

The 231 public decisions are frozen in `benchmarks/jevbench-public231/tasks.jsonl`
(sha256 recorded in that directory's manifest); scoring uses the frozen upstream
scorer in `jev/upstream_scoring.py`, so the rules are not re-implemented here.
There is no sealed split in this release and no claim of a leaderboard position.
"""
import json
from pathlib import Path

import numpy as np

from .common import canonical          # noqa: E402
from .upstream_scoring import score_task   # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = HERE.parent / 'benchmarks/jevbench-public231/tasks.jsonl'
CALIBRATION_TEMPERATURE = 1.1525501023241218


def softmax(logits):
    shifted = logits - np.max(logits)
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def criteria_for(task):
    """[(label, criterion_text)] in the benchmark's original label order."""
    question = task['question']
    labels = [str(x) for x in task['labels']]
    criteria = question.get('criteria', {})
    if question['type'] == 'noul':
        alias = {'no': 'false', 'yes': 'true', 'false': 'false', 'true': 'true'}
        return [(lab, text(criteria.get(alias.get(lab.lower(), lab), criteria.get(lab, lab)))) for lab in labels]
    if isinstance(criteria, list):
        if len(criteria) != len(labels):
            raise ValueError('score criteria/labels length mismatch')
        return [(lab, text(criteria[i])) for i, lab in enumerate(labels)]
    return [(lab, text(criteria.get(lab, lab))) for lab in labels]


def build_item(task, mode):
    """Map one benchmark task to the frozen decision-item protocol.

    `criterion_as_option` puts each criterion in the option slot; `label_as_option`
    keeps the original labels and attaches the criterion as its description.  Both
    renderings are reported separately; neither puts gold in the prompt.
    """
    pairs = criteria_for(task)
    if mode == 'criterion_as_option':
        options = [crit for _, crit in pairs]
        crit_map = {crit: crit for crit in options}
    elif mode == 'label_as_option':
        options = [lab for lab, _ in pairs]
        crit_map = {lab: crit for lab, crit in pairs}
    else:
        raise ValueError('unknown mode')
    if len(set(options)) != len(options):
        raise ValueError('duplicate option text after mapping: ' + task['id'])
    return {'id': task['id'], 'text': canonical(task['state']), 'question': text(task['question']['instructions']),
            'options': options, 'crit': crit_map}


def score_one(probabilities, labels, expected, qtype):
    """Frozen upstream scoring: strict sum tolerance 1e-3, renormalization band 2e-2."""
    from types import SimpleNamespace
    task = SimpleNamespace(labels=labels, expected=expected, question={'type': qtype})
    outcome = score_task({k: float(v) for k, v in probabilities.items()}, task)
    return {k: outcome.get(k) for k in ('valid', 'strict_valid', 'renormalized', 'correct', 'predicted')}


def ece_brier(records, key):
    """Top-label ECE (10 equal-width bins) and Brier over a probability map."""
    bins, brier = {}, []
    for row in records:
        probs = row[key]
        confidence = max(probs.values())
        correct = 1.0 if row['predicted'] == row['gold_label'] else 0.0
        bins.setdefault(min(9, int(confidence * 10)), []).append((confidence, correct))
        brier.append(sum((v - (1.0 if lab == row['gold_label'] else 0.0)) ** 2 for lab, v in probs.items()))
    ece = 0.0
    for values in bins.values():
        confidence = sum(c for c, _ in values) / len(values)
        accuracy = sum(a for _, a in values) / len(values)
        ece += abs(confidence - accuracy) * len(values) / len(records)
    return ece, sum(brier) / len(brier)


def subset_metrics(records, subset, key):
    rows = [r for r in records if r['id'] in subset]
    valid = [r for r in rows if r['valid']]
    correct = sum(1 for r in valid if r['correct'])
    ece, brier = ece_brier(valid, key) if valid else (None, None)
    head_ms = [r['head_ms'] for r in valid]
    path_ms = [r['head_path_ms'] for r in valid]
    return {'planned': len(rows), 'valid': len(valid), 'errors': len(rows) - len(valid),
            'correct': correct,
            'accuracy_all_planned': correct / len(rows) if rows else None,
            'accuracy_valid': correct / len(valid) if valid else None,
            'top_label_ece': ece, 'brier': brier,
            'head_ms_p50': float(np.percentile(head_ms, 50)) if head_ms else None,
            'feature_plus_head_p50_ms': float(np.percentile(path_ms, 50)) if path_ms else None,
            'feature_plus_head_p95_ms': float(np.percentile(path_ms, 95)) if path_ms else None}


def load_tasks(path=TASKS):
    tasks = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(tasks) != 231:
        raise ValueError('expected the 231 public JevBench decisions, got %d' % len(tasks))
    return tasks
