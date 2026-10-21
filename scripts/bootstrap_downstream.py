#!/usr/bin/env python3
"""Recompute the frozen four-task comparison from public item correctness.

No model, dataset text, GPU, or private runtime artifacts are required. This
checks the sampling statistics only; use review_downstream_core.py to audit
raw candidate scores against item metrics.
"""
import argparse
import json
from pathlib import Path
import numpy as np

TASKS = ['piqa', 'sciq', 'winogrande', 'openbookqa']
MODELS = ['P1', 'P2', 'P4', 'P8']
COUNTS = dict(piqa=1838, sciq=1000, winogrande=1267, openbookqa=500)


def compute(data):
    assert data['task_order'] == TASKS and data['models'] == MODELS
    rng = np.random.default_rng(20261021)
    bootstrap = np.zeros((3, 2000))
    task_means = []
    for name in TASKS:
        task = data['tasks'][name]
        n = COUNTS[name]
        assert len(task['doc_ids']) == len(set(task['doc_ids'])) == n
        assert len(task['gold']) == n
        correct = np.asarray([task['correctness'][m] for m in MODELS])
        assert correct.shape == (4, n) and np.isin(correct, [0, 1]).all()
        task_means.append(correct.mean(axis=1))
        indices = rng.integers(0, n, size=(2000, n))
        for j in range(3):
            bootstrap[j] += (correct[j + 1] - correct[0])[indices].mean(axis=1) / 4
    means = np.mean(task_means, axis=0)
    comparisons = {
        m + '-P1': {
            'delta_percentage_points': float((means[j + 1] - means[0]) * 100),
            'ci95_percentage_points': np.percentile(bootstrap[j] * 100, [2.5, 97.5]).tolist(),
        } for j, m in enumerate(MODELS[1:])
    }
    primary = comparisons['P4-P1']
    return {
        'macro_accuracy': dict(zip(MODELS, map(float, means))),
        'macro_comparisons': comparisons,
        'positive_criterion_met': primary['delta_percentage_points'] >= 1
                                  and primary['ci95_percentage_points'][0] > 0,
        'reps': 2000, 'seed': 20261021,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired_outcomes', type=Path)
    parser.add_argument('--check-against', type=Path, help='Independent supervisor report')
    args = parser.parse_args()
    result = compute(json.loads(args.paired_outcomes.read_text()))
    if args.check_against:
        audit = json.loads(args.check_against.read_text())
        assert result['macro_accuracy'] == audit['macro_accuracy']
        assert result['macro_comparisons'] == audit['macro_comparisons']
        assert result['positive_criterion_met'] == audit['primary']['positive_criterion_met']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
