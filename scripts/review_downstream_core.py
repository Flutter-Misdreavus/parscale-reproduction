#!/usr/bin/env python3
"""Independent CPU reconstruction of committed downstream outcomes and paired CIs."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

TASKS = ['piqa', 'sciq', 'winogrande', 'openbookqa']
MODELS = ['P1', 'P2', 'P4', 'P8']
COUNTS = dict(piqa=1838, sciq=1000, winogrande=1267, openbookqa=500)
REQUESTS = dict(piqa=3676, sciq=4000, winogrande=2534, openbookqa=2000)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=2000)
    a = ap.parse_args(); root = a.run_dir
    if a.reps != 2000:
        raise ValueError('The frozen confirmatory analysis requires 2000 resamples')
    result = json.loads((root / 'full-results.json').read_text())
    assert result['status'] == 'SUCCEEDED', result['status']
    assert len(result['completed_pairs']) == 16
    checks, accuracy, ids, golds, outcomes = [], {}, {}, {}, {}
    request_hashes = {}
    for name in TASKS:
        accuracy[name] = {}
        outcomes[name] = {}
        for model in MODELS:
            pair = model + '-' + name
            path = root / 'outcomes' / (pair + '.json')
            candpath = root / 'candidates' / (pair + '.jsonl')
            recorded = result['models'][model]['tasks'][name]
            assert sha(path) == recorded['outcomes_sha256']
            assert sha(candpath) == recorded['candidates_sha256']
            rows = sorted(json.loads(path.read_text()), key=lambda r: r['doc_id'])
            assert len(rows) == COUNTS[name]
            assert len({r['doc_id'] for r in rows}) == len(rows)
            candidate = {}
            stream_hashes = {}
            with candpath.open() as f:
                for line in f:
                    c = json.loads(line); key = (c['doc_id'], c['choice'])
                    assert key not in candidate
                    candidate[key] = c['logprob']
                    stream_hashes[key] = (c['context_sha256'], c['continuation_sha256'])
            assert len(candidate) == REQUESTS[name]
            if name in request_hashes:
                assert stream_hashes == request_hashes[name]
            else:
                request_hashes[name] = stream_hashes
            docids = [r['doc_id'] for r in rows]
            gs = [r['gold'] for r in rows]
            if name in ids:
                assert docids == ids[name] and gs == golds[name]
            ids[name], golds[name] = docids, gs
            collected = {'acc': [], 'acc_norm': []}
            for r in rows:
                scores = np.asarray(r['scores'], dtype=float)
                lens = np.asarray(r['normalization_lengths'], dtype=int)
                assert len(scores) == len(lens) and len(scores) >= 2
                assert np.isfinite(scores).all() and np.all(lens > 0)
                assert 0 <= r['gold'] < len(scores)
                for i, score in enumerate(scores):
                    assert score == candidate[(r['doc_id'], i)]
                acc = float(int(np.argmax(scores)) == r['gold'])
                assert acc == r['acc']
                collected['acc'].append(acc)
                if 'acc_norm' in r:
                    normalized = float(int(np.argmax(scores/lens)) == r['gold'])
                    assert normalized == r['acc_norm']
                    collected['acc_norm'].append(normalized)
            metric = 'acc_norm' if collected['acc_norm'] else 'acc'
            assert metric == recorded['primary_metric']
            for m, values in collected.items():
                if values:
                    assert abs(float(np.mean(values)) - recorded['metrics'][m]) < 1e-12
            values = np.asarray(collected[metric])
            outcomes[name][model] = values
            accuracy[name][model] = {'metric': metric, 'accuracy': float(values.mean()),
                                     'correct': int(values.sum()), 'n': len(values)}
            checks.append({'pair': pair, 'all_candidate_scores_matched': True,
                           'identical_request_hashes_across_models': True,
                           'all_item_metrics_reconstructed': True,
                           'outcomes_sha256': sha(path), 'candidates_sha256': sha(candpath)})
    rng = np.random.default_rng(20261021)
    macros = {m: float(np.mean([accuracy[t][m]['accuracy'] for t in TASKS])) for m in MODELS}
    bootstrap_macro = np.zeros((3, a.reps))
    task_comparisons = {}
    for name in TASKS:
        n = COUNTS[name]
        indices = rng.integers(0, n, size=(a.reps, n))
        task_comparisons[name] = {}
        for j, model in enumerate(MODELS[1:]):
            diffs = outcomes[name][model] - outcomes[name]['P1']
            estimates = diffs[indices].mean(axis=1)
            bootstrap_macro[j] += estimates / len(TASKS)
            bounds = np.percentile(estimates*100, [2.5, 97.5])
            task_comparisons[name][model+'-P1'] = {
                'delta_percentage_points': float(diffs.mean()*100),
                'ci95_percentage_points': bounds.tolist()}
    comparisons = {}
    for j, model in enumerate(MODELS[1:]):
        comparisons[model+'-P1'] = {
            'delta_percentage_points': (macros[model]-macros['P1'])*100,
            'ci95_percentage_points': np.percentile(bootstrap_macro[j]*100,[2.5,97.5]).tolist()}
    primary = comparisons['P4-P1']
    report = {'run_id': result['run_id'], 'status': 'ACCEPTED',
              'actual_run_commit': result['run_git']['head_commit'],
              'analysis': {'script_sha256': sha(Path(__file__)),
                           'numpy_version': np.__version__,
                           'source_commit': subprocess.check_output(
                               ['git', 'rev-parse', 'HEAD'],
                               cwd=Path(__file__).resolve().parents[1], text=True).strip()},
              'full_results_sha256': sha(root/'full-results.json'),
              'checks': checks, 'accuracy': accuracy,
              'macro_accuracy': macros, 'macro_comparisons': comparisons,
              'task_comparisons': task_comparisons,
              'primary': {'comparison': 'P4-P1', 'tasks': TASKS,
                          'positive_criterion_met': bool(primary['delta_percentage_points'] >= 1 and primary['ci95_percentage_points'][0] > 0)},
              'bootstrap': {'reps': a.reps, 'seed': 20261021,
                            'method': 'Paired by document within task; independent task resampling; equal task weights; percentile95%CI',
                            'scope': 'Evaluation-item uncertainty only; no independent training seeds'},
              'resources': {'scoring_wall_seconds': result['scoring_wall_seconds'],
                            'peak_allocated_gib': max(result['models'][m]['tasks'][t]['peak_memory_gib']['allocated'] for m in MODELS for t in TASKS),
                            'peak_reserved_gib': max(result['models'][m]['tasks'][t]['peak_memory_gib']['reserved'] for m in MODELS for t in TASKS)},
              'limitations': ['Only four core tasks; not the paper six-group General average.',
                              'HellaSwag and ARC deferred by pretest cost projection.',
                              'One official pretrained checkpoint per P, no training replication.',
                              'Pretraining overlap unknown; shared GPU timings descriptive.',
                              'No full-test FP32 sensitivity; small training/synthetic smoke only.']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    # Compact per-item correctness only: permits public CI recomputation without text.
    compact = {'run_id': result['run_id'], 'task_order': TASKS, 'models': MODELS,
               'tasks': {t: {'doc_ids': ids[t],
                             'metric': accuracy[t]['P1']['metric'],
                             'correctness': {m: outcomes[t][m].astype(int).tolist() for m in MODELS}}
                         for t in TASKS}}
    (a.output.parent/'paired-outcomes.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    print(json.dumps({'macro_accuracy': macros, 'comparisons': comparisons, 'criterion': report['primary'],
                      'resources': report['resources']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
