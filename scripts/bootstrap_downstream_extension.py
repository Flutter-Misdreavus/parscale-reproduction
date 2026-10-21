#!/usr/bin/env python3
"""Recompute prospective extension and descriptive General from correctness only."""
import argparse
import json
from pathlib import Path
import numpy as np

MODELS = ['P1', 'P2', 'P4', 'P8']
CORE = ['piqa', 'sciq', 'winogrande', 'openbookqa']
REST = ['arc_easy', 'arc_challenge', 'hellaswag']
COUNTS = dict(piqa=1838, sciq=1000, winogrande=1267, openbookqa=500,
              arc_easy=2376, arc_challenge=1172, hellaswag=10042)
WEIGHTS = dict(arc_easy=0.25, arc_challenge=0.25, hellaswag=0.5)
GENERAL = {**{t: 1/6 for t in CORE}, 'arc_easy': 1/12,
           'arc_challenge': 1/12, 'hellaswag': 1/6}


def validate(data, tasks):
    assert data['task_order'] == tasks and data['models'] == MODELS
    arrays = {}
    for name in tasks:
        task = data['tasks'][name]
        assert len(task['doc_ids']) == len(set(task['doc_ids'])) == COUNTS[name]
        assert task['metric'] == ('acc' if name == 'winogrande' else 'acc_norm')
        values = np.asarray([task['correctness'][m] for m in MODELS])
        assert values.shape == (4, COUNTS[name]) and np.isin(values, [0, 1]).all()
        arrays[name] = values.astype(np.int8)
    return arrays


def summarize(arrays, weights):
    assert abs(sum(weights.values()) - 1) < 1e-12
    rng = np.random.default_rng(20261022)
    means = np.zeros(4)
    boot = np.zeros((3, 2000))
    per_task = {}
    for name, weight in weights.items():
        values = arrays[name]
        n = values.shape[1]
        assert n > 0
        means += weight * values.mean(axis=1)
        estimates = np.empty((3, 2000))
        differences = values[1:] - values[0]
        # Bound memory; the same item indices are shared across all models.
        for offset in range(0, 2000, 50):
            indices = rng.integers(0, n, size=(min(50, 2000-offset), n))
            estimates[:, offset:offset+indices.shape[0]] = differences[:, indices].mean(axis=2)
        boot += weight * estimates
        per_task[name] = {
            'accuracy': dict(zip(MODELS, map(float, values.mean(axis=1)))),
            'comparisons': {
                m+'-P1': {'delta_percentage_points': float(differences[j].mean()*100),
                         'ci95_percentage_points': np.percentile(estimates[j]*100, [2.5, 97.5]).tolist()}
                for j, m in enumerate(MODELS[1:])},
            'n': n,
        }
    comparisons = {
        m+'-P1': {'delta_percentage_points': float((means[j+1]-means[0])*100),
                 'ci95_percentage_points': np.percentile(boot[j]*100, [2.5, 97.5]).tolist()}
        for j, m in enumerate(MODELS[1:])}
    return {'weighted_accuracy': dict(zip(MODELS, map(float, means))),
            'comparisons': comparisons, 'weights': weights, 'per_task': per_task}


def compute(rest, core):
    arrays = validate(rest, REST)
    primary = summarize(arrays, WEIGHTS)
    comparison = primary['comparisons']['P4-P1']
    primary['positive_criterion_met'] = comparison['delta_percentage_points'] >= 1 and comparison['ci95_percentage_points'][0] > 0
    overlap = rest['arc_easy_train_overlap_doc_ids']
    assert len(overlap) == len(set(overlap)) == 2
    ids = rest['tasks']['arc_easy']['doc_ids']
    assert set(overlap).issubset(ids)
    keep = np.asarray([did not in overlap for did in ids])
    filtered = {**arrays, 'arc_easy': arrays['arc_easy'][:, keep]}
    sensitivity = summarize(filtered, WEIGHTS)
    all_arrays = {**validate(core, CORE), **arrays}
    general = summarize(all_arrays, GENERAL)
    general['interpretation'] = 'Expanded descriptive analysis designed after core results; does not replace the frozen core primary.'
    return {'extension': primary,
            'extension_excluding_two_arc_easy_train_overlaps': sensitivity,
            'general_descriptive': general,
            'bootstrap': {'seed': 20261022, 'reps': 2000,
                          'uncertainty': 'Evaluation items only; no independent training seeds'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rest_outcomes', type=Path)
    parser.add_argument('core_outcomes', type=Path)
    parser.add_argument('--check-against', type=Path)
    args = parser.parse_args()
    result = compute(json.loads(args.rest_outcomes.read_text()), json.loads(args.core_outcomes.read_text()))
    if args.check_against:
        assert result == json.loads(args.check_against.read_text())['statistics']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
