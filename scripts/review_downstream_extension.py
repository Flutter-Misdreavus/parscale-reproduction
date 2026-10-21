#!/usr/bin/env python3
"""Independent CPU reconstruction of committed downstream outcomes and paired CIs."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from bootstrap_downstream_extension import compute

TASKS = ['arc_easy', 'arc_challenge', 'hellaswag']
MODELS = ['P1', 'P2', 'P4', 'P8']
COUNTS = dict(arc_easy=2376, arc_challenge=1172, hellaswag=10042)
REQUESTS = dict(arc_easy=9501, arc_challenge=4687, hellaswag=40168)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--core-outcomes', type=Path, required=True)
    ap.add_argument('--reps', type=int, default=2000)
    a = ap.parse_args(); root = a.run_dir
    if a.reps != 2000:
        raise ValueError('The frozen confirmatory analysis requires 2000 resamples')
    result = json.loads((root / 'full-results.json').read_text())
    assert result['status'] == 'SUCCEEDED', result['status']
    assert (root/'raw/full.run.sha').read_text().strip() == result['run_git']['head_commit']
    assert set(result['completed_pairs']) == {m+'-'+t for m in MODELS for t in TASKS}
    assert len(result['completed_pairs']) == 12
    verification = json.loads((root/'request-verification.json').read_text())
    assert set(verification) == set(TASKS)
    assert all(len(v) == 8 and all(v.values()) for v in verification.values())
    sources = json.loads((root/'sources-recheck.json').read_text())
    assert set(sources) == set(TASKS) and all(all(v.values()) for v in sources.values())
    overlap = json.loads((root/'split-id-check.json').read_text())
    assert {t: overlap[t]['train_eval_exact_case_overlap'] for t in TASKS} == dict(arc_easy=2, arc_challenge=0, hellaswag=0)
    assert all(overlap[t]['duplicate_eval_exact_cases'] == 0 for t in TASKS)
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
    compact = {'run_id': result['run_id'], 'task_order': TASKS, 'models': MODELS,
               'tasks': {t: {'doc_ids': ids[t], 'metric': accuracy[t]['P1']['metric'],
                             'correctness': {m: outcomes[t][m].astype(int).tolist() for m in MODELS}}
                         for t in TASKS},
               'arc_easy_train_overlap_doc_ids': overlap['arc_easy']['train_overlap_eval_row_indices']}
    core = json.loads(a.core_outcomes.read_text())
    statistics = compute(compact, core)
    report = {'run_id': result['run_id'], 'status': 'ACCEPTED',
              'actual_run_commit': result['run_git']['head_commit'],
              'analysis': {'script_sha256': sha(Path(__file__)),
                           'bootstrap_script_sha256': sha(Path(__file__).with_name('bootstrap_downstream_extension.py')),
                           'numpy_version': np.__version__,
                           'source_commit': subprocess.check_output(
                               ['git', 'rev-parse', 'HEAD'],
                               cwd=Path(__file__).resolve().parents[1], text=True).strip()},
              'full_results_sha256': sha(root/'full-results.json'),
              'checks': checks, 'accuracy': accuracy,
              'statistics': statistics,
              'primary': {'comparison': 'P4-P1', 'tasks': TASKS,
                          'weights': {'arc_easy': 0.25, 'arc_challenge': 0.25, 'hellaswag': 0.5},
                          'positive_criterion_met': statistics['extension']['positive_criterion_met']},
              'bootstrap': {'reps': a.reps, 'seed': 20261022,
                            'method': 'Paired by document within task; independent task resampling; frozen group weights; percentile95%CI',
                            'scope': 'Evaluation-item uncertainty only; no independent training seeds'},
              'known_overlap': overlap,
              'core_correctness_sha256': sha(a.core_outcomes),
              'resources': {'scoring_wall_seconds': result['scoring_wall_seconds'],
                            'peak_allocated_gib': max(result['models'][m]['tasks'][t]['peak_memory_gib']['allocated'] for m in MODELS for t in TASKS),
                            'peak_reserved_gib': max(result['models'][m]['tasks'][t]['peak_memory_gib']['reserved'] for m in MODELS for t in TASKS)},
              'limitations': ['Prior core-four primary remains inconclusive; never replaced by expanded analysis.',
                              'Combined paper six-group General is an expanded descriptive analysis designed after core results.',
                              'ARC-Easy has two exact question/choices overlaps with train; official primary retained plus frozen sensitivity excluding them.',
                              'One official pretrained checkpoint per P, no training replication.',
                              'Pretraining overlap unknown; shared GPU timings descriptive.',
                              'No full-test FP32 sensitivity; small training/synthetic smoke only.']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    # Correctness only; labels, prompts and raw candidate files remain private.
    (a.output.parent/'paired-outcomes.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    print(json.dumps({'statistics': statistics, 'primary': report['primary'],
                      'resources': report['resources']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
