#!/usr/bin/env python3
"""Bounded, resumable-by-evidence downstream scoring after supervisor preflight.

Uses the frozen lm-eval task instances and float32 log-softmax adapter validated
by downstream_evaluation.py. Each candidate chunk is written durably; no raw
benchmark text is written into the shareable candidate/outcome files.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import downstream_evaluation as d

CORE = ['piqa', 'sciq', 'winogrande', 'openbookqa']
EXTRA = ['hellaswag', 'arc_easy', 'arc_challenge']

class BudgetReached(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    task = json.loads((run / 'task.json').read_text())
    if not task.get('full_test_authorized_by_supervisor'):
        raise RuntimeError('Supervisor authorization missing')
    frozen = d.ROOT / 'experiments' / run.name / 'task.json'
    if frozen.read_bytes() != (run / 'task.json').read_bytes():
        raise RuntimeError('Frozen task mismatch')
    state = d.git_state()
    if state['status_porcelain']:
        raise RuntimeError('Execution requires a clean committed tree')
    for name, sha in task['prepared_artifact_sha256'].items():
        if d.sha256_file(run / name) != sha:
            raise RuntimeError('Prepared evidence hash mismatch: ' + name)
    for name, sha in task['validated_source_sha256'].items():
        if d.sha256_file(d.ROOT / name) != sha:
            raise RuntimeError('Validated source changed: ' + name)
    smoke = json.loads((run / 'scorer-validation.json').read_text())
    if smoke['status'] != 'SUCCEEDED' or not all(smoke['gate'].values()):
        raise RuntimeError('Scorer validation not accepted')
    selected = task['evaluation_tasks']
    if not set(selected) <= set(CORE + EXTRA) or not set(CORE) <= set(selected):
        raise RuntimeError('Unexpected evaluation scope')
    phases = [[t for t in group if t in selected] for group in [CORE, EXTRA]]
    models = task['evaluation_models']
    if models != [1, 2, 4, 8]:
        raise RuntimeError('Expected four frozen checkpoints')
    data = json.loads((run / 'data-manifest.json').read_text())
    d.enforce_determinism()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(d.model_path(1)), local_files_only=True,
                                       trust_remote_code=False)
    # Rehash all weights and reject pending P8 before allocating models.
    manifest = d.model_manifest(run / 'weight-recheck')
    if manifest.get('pending_models'):
        raise RuntimeError('Model verification incomplete')
    tasks = d.make_tasks(run, 'audit')
    verification = {}
    for name in selected:
        got = d.collect_task_requests(name, tasks[name], tok)
        old = data['tasks'][name]
        keys = ['documents', 'requests', 'input_tokens', 'request_stream_sha256',
                'gold_index_sha256', 'num_fewshot', 'eval_split', 'requests_truncated']
        verification[name] = {k: got[k] == old[k] for k in keys}
        if not all(verification[name].values()) or got['gold_invalid']:
            raise RuntimeError('Frozen request/target mismatch: ' + name)
    d.atomic_json(run / 'request-verification.json', verification)
    # No automatic overwrite/restart of previous GPU output.
    lock = run / 'full-once.lock'
    lock.mkdir()
    (lock / 'run.sha').write_text(state['head_commit'] + '\n')
    result = {'run_id': run.name, 'status': 'RUNNING', 'run_git': state,
              'started_at': d.utcnow(), 'models': {}, 'completed_pairs': [],
              'request_set_sha256': data['request_set_sha256'],
              'precision': 'BF16 weights, float32 log-softmax and sum',
              'context_max': d.CONTEXT_MAX, 'batch_size': 1,
              'limitations': ['Shared GPU timings are descriptive.',
                             'One pretrained checkpoint per P; no training seed replication.',
                             'Left truncation statistics are in the frozen data manifest.']}
    d.atomic_json(run / 'full-results.json', result)
    gpu_phase_start = time.monotonic()
    seconds = min(float(task['scoring_wall_seconds']), float(task['budget']['gpu_hours']) * 3600)
    deadline = gpu_phase_start + seconds
    FloatHFLM = d.make_float_hflm()

    class BoundedHFLM(FloatHFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            if time.monotonic() >= deadline - 10:
                raise BudgetReached('Scoring wall cap reached before next forward')
            out = super()._model_call(inps, attn_mask, labels)
            if not torch.isfinite(out).all():
                raise RuntimeError('Non-finite logits')
            return out

    try:
        for phase in phases:
            if not phase:
                continue
            for p in models:
                if time.monotonic() >= deadline - 30:
                    raise BudgetReached('Insufficient time to load next checkpoint')
                adm = d.gpu_admission()
                if not adm['passed']:
                    raise RuntimeError('GPU memory admission failed')
                d.cap_gpu_memory()
                torch.cuda.reset_peak_memory_stats()
                model, load_seconds = d.load_model(p)
                lm = BoundedHFLM(pretrained=model, tokenizer=tok, batch_size=1,
                                 max_length=d.CONTEXT_MAX, truncation=False,
                                 device='cuda', dtype=torch.bfloat16, add_bos_token=False)
                model_rec = result['models'].setdefault(f'P{p}', {'tasks': {}, 'load_seconds': []})
                model_rec['load_seconds'].append(load_seconds)
                try:
                    for name in phase:
                        torch.cuda.reset_peak_memory_stats()
                        begin = time.monotonic()
                        eval_task = tasks[name]
                        instances = eval_task.instances
                        pair = f'P{p}-{name}'
                        directory = run / 'candidates'
                        directory.mkdir(exist_ok=True)
                        answer_map = {}
                        with open(directory / (pair + '.jsonl'), 'x') as fh:
                            for offset in range(0, len(instances), 64):
                                chunk = instances[offset:offset + 64]
                                values = lm.loglikelihood(chunk, disable_tqdm=True)
                                for inst, (ll, greedy) in zip(chunk, values):
                                    if not math.isfinite(ll):
                                        raise RuntimeError('Non-finite candidate score')
                                    key = (inst.doc_id, inst.idx)
                                    if key in answer_map:
                                        raise RuntimeError('Duplicate candidate identity')
                                    answer_map[key] = (float(ll), bool(greedy))
                                    row = {'doc_id': inst.doc_id, 'choice': inst.idx,
                                           'logprob': float(ll), 'is_greedy': bool(greedy),
                                           'context_sha256': d.sha256_bytes(inst.args[0].encode()),
                                           'continuation_sha256': d.sha256_bytes(inst.args[1].encode())}
                                    fh.write(json.dumps(row, sort_keys=True) + '\n')
                                fh.flush()
                                os.fsync(fh.fileno())
                                d.atomic_json(run / 'progress.json', {
                                    'pair': pair, 'completed_requests': len(answer_map),
                                    'total_requests': len(instances), 'at': d.utcnow(),
                                    'scoring_wall_seconds': time.monotonic() - gpu_phase_start,
                                    'completed_pairs': result['completed_pairs']})
                                if offset % 512 == 0:
                                    print(f'{pair}: {len(answer_map)}/{len(instances)} candidates', flush=True)
                        docs = {}
                        by_doc = {}
                        for inst in instances:
                            docs[inst.doc_id] = inst.doc
                            by_doc.setdefault(inst.doc_id, []).append(inst)
                        rows = []
                        for did, group in by_doc.items():
                            group.sort(key=lambda i: i.idx)
                            if [i.idx for i in group] != list(range(len(group))):
                                raise RuntimeError('Candidate indices not contiguous')
                            answers = [answer_map[(did, i.idx)] for i in group]
                            metrics = eval_task.process_results(docs[did], answers)
                            metric_values = {k: float(v) for k, v in metrics.items() if k in ('acc', 'acc_norm')}
                            if not metric_values or any(v not in (0.0, 1.0) for v in metric_values.values()):
                                raise RuntimeError('Invalid item accuracy')
                            scores = [a[0] for a in answers]
                            choices = eval_task.doc_to_choice(docs[did])
                            rows.append({'doc_id': did, 'gold': d.gold_index(eval_task, docs[did]),
                                         'scores': scores, 'normalization_lengths': [len(c) for c in choices],
                                         **metric_values})
                        if len(rows) != data['tasks'][name]['documents']:
                            raise RuntimeError('Incomplete task outcomes')
                        out_dir = run / 'outcomes'
                        out_dir.mkdir(exist_ok=True)
                        d.atomic_json(out_dir / (pair + '.json'), rows)
                        metric_names = rows[0].keys() & {'acc', 'acc_norm'}
                        rec = {'documents': len(rows),
                               'metrics': {m: float(np.mean([r[m] for r in rows])) for m in metric_names},
                               'primary_metric': 'acc_norm' if 'acc_norm' in metric_names else 'acc',
                               'seconds': time.monotonic() - begin,
                               'outcomes_sha256': d.sha256_file(out_dir / (pair + '.json')),
                               'candidates_sha256': d.sha256_file(directory / (pair + '.jsonl')),
                               'peak_memory_gib': {'allocated': torch.cuda.max_memory_allocated()/1024**3,
                                                   'reserved': torch.cuda.max_memory_reserved()/1024**3}}
                        model_rec['tasks'][name] = rec
                        result['completed_pairs'].append(pair)
                        d.atomic_json(run / 'full-results.json', result)
                        print(f'COMPLETE {pair} {rec["metrics"]}', flush=True)
                finally:
                    del lm
                    d.release(model)
        result['status'] = 'SUCCEEDED'
    except BudgetReached as exc:
        result.update(status='BUDGET_STOPPED', error=str(exc))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(finished_at=d.utcnow(), scoring_wall_seconds=time.monotonic() - gpu_phase_start,
                      scoring_wall_limit_seconds=seconds)
        d.atomic_json(run / 'full-results.json', result)
    print(json.dumps({'status': result['status'], 'completed_pairs': result['completed_pairs']}), flush=True)

if __name__ == '__main__':
    main()
