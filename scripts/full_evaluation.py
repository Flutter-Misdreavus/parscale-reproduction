#!/usr/bin/env python3
"""Fixed whole-test teacher-forced comparison; BF16 primary, FP32 sensitivity.

No incremental cache is reused across blocks. The official model still creates
its learned-prefix cache during each forward; do not disable that mechanism.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate
import cache_diagnostic_v2 as diagnostic


def compare(models, seed, reps):
    paired = {key: [(b['nll_sum_nats'], b['token_count_scored']) for b in value['blocks']]
              for key, value in models.items()}
    result = diagnostic.paired_bootstrap(paired, reps=reps, seed=seed)
    primary = result['P2-P1']
    p1 = models['P1']['aggregate']['nll_nats_per_token']
    relative = -primary['observed_delta_nll'] / p1
    return {'pairs': result, 'relative_P2_improvement': relative,
            'prespecified_promising': bool(relative >= .005 and primary['ci95_high'] < 0
                and models['P4']['aggregate']['nll_nats_per_token'] <=
                    models['P2']['aggregate']['nll_nats_per_token']),
            'criterion': 'P2 relative NLL reduction >=0.5%, paired CI upper<0, P4<=P2',
            'unit': 'contiguous text block, not independent training seed'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--reference-validation', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    os.environ['EVAL_RUN_DIR'] = str(run)
    locks = json.loads((ROOT / 'locks.json').read_text())
    frozen_task = json.loads((run / 'task.json').read_text())
    if not args.validation_only and not frozen_task.get('full_test_authorized_by_supervisor'):
        raise RuntimeError('Full evaluation requires an explicit reviewed task')
    if not args.validation_only:
        checked = json.loads((run / 'pretest-validation.json').read_text())
        if (checked.get('status') != 'SUCCEEDED'
                or not checked.get('reference_validation', {}).get('passed')
                or checked['run_git']['head_commit'] != evaluate.git_state()['head_commit']):
            raise RuntimeError('Pretest validation missing or code changed after validation')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    blocks, manifest = evaluate.prepare_data(locks, run, 'smoke' if args.validation_only else 'full')
    diagnostic.atomic_json(run / ('validation-data-manifest.json' if args.validation_only else 'test-data-manifest.json'), manifest)
    expected_split = 'validation' if args.validation_only else 'test'
    if manifest['split'] != expected_split:
        raise RuntimeError('Wrong split')
    paths = evaluate.resolve_model_paths(locks)
    gpu = evaluate.gpu_environment(locks['resource']['gpu_uuid'])
    if not gpu['uuid_matches_lock'] or gpu['device_count_visible'] != 1:
        raise RuntimeError('Wrong GPU binding')
    cap = evaluate.cap_gpu_memory(16)
    result_path = run / ('pretest-validation.json' if args.validation_only else 'full-results.json')
    result = {'run_id': run.name, 'stage': expected_split, 'status': 'RUNNING',
              'started_at': evaluate.utcnow(), 'run_git': evaluate.git_state(),
              'gpu': gpu, 'memory_cap': cap, 'packages': evaluate.package_versions(),
              'primary_precision': 'bfloat16', 'sensitivity_precision': 'float32',
              'tf32_matmul': False, 'blocks': len(blocks),
              'data_token_hash': manifest['token_ids_sha256'],
              'cache_policy': 'new official learned-prefix cache each teacher-forced block; no incremental decoding for NLL',
              'precisions': {}, 'timing': {}, 'admissions': [],
              'limitations': ['Shared GPU timings are descriptive, not isolated performance evidence.',
                             'One checkpoint per P, unknown training seeds; bootstrap describes text-block uncertainty only.',
                             'External WikiText evaluation may overlap unknown pretraining data.',
                             'The original BF16 max-logit cache gate remains failed; not reclassified.']}
    diagnostic.atomic_json(result_path, result)
    for dtype_name, dtype in [('bfloat16', torch.bfloat16), ('float32', torch.float32)]:
        output = result['precisions'][dtype_name] = {'models': {}}
        for key, path in sorted(paths.items()):
            name = f'P{key}'
            admission = evaluate.gpu_admission(locks['resource']['gpu_uuid'], 18432, None)
            admission.update(model=name, dtype=dtype_name)
            result['admissions'].append(admission)
            diagnostic.atomic_json(result_path, result)
            if not admission['passed']:
                raise RuntimeError(f'Insufficient free GPU memory before {name}/{dtype_name}')
            torch.cuda.reset_peak_memory_stats()
            model, load_time = diagnostic.load_model(path, dtype)
            rec = {'parameter_count': sum(p.numel() for p in model.parameters()),
                   'load_seconds': load_time, 'blocks': []}
            output['models'][name] = rec
            try:
                for bi, block in enumerate(blocks):
                    record, ids = evaluate.scored_block(model, block, 'cuda')
                    del ids
                    if abs(record['manual_minus_label']) > 1e-3:
                        raise RuntimeError('Manual/model-label loss mismatch')
                    record.update(block_id=bi, token_ids_sha256=manifest['blocks_used_meta'][bi]['token_ids_sha256'])
                    rec['blocks'].append(record)
                    if bi % 25 == 0 or bi == len(blocks)-1:
                        diagnostic.atomic_json(result_path, result)
                        print(f'{dtype_name} {name} block {bi+1}/{len(blocks)}', flush=True)
                nll, total, count = diagnostic.token_weighted([
                    (b['nll_sum_nats'], b['token_count_scored']) for b in rec['blocks']])
                rec['aggregate'] = {'nll_nats_per_token': nll, 'nll_sum_nats': total,
                                    'token_count_scored': count, 'ppl': math.exp(nll)}
                rec['scoring_peak_memory_gib'] = evaluate.peak_mem_gib()
                rec['score_seconds'] = sum(b['seconds'] for b in rec['blocks'])
                if dtype_name == 'bfloat16' and not args.validation_only:
                    timing = diagnostic.phase_timing(model, blocks[0], name, evaluate.peak_mem_gib())
                    timing['label'] = 'test prompt, shared GPU; functional latency observation only'
                    if not timing['output_length_ok'] or not all(math.isfinite(t) and t > 0 for t in timing['raw_seconds']):
                        raise RuntimeError('Invalid timing output')
                    result['timing'][name] = timing
                diagnostic.atomic_json(result_path, result)
            finally:
                diagnostic.release(model)
                del model
        output['comparison'] = compare(output['models'], 20261021, 2000)
        diagnostic.atomic_json(result_path, result)
    result['precision_sensitivity'] = {
        name: result['precisions']['bfloat16']['models'][name]['aggregate']['nll_nats_per_token'] -
              result['precisions']['float32']['models'][name]['aggregate']['nll_nats_per_token']
        for name in ('P1', 'P2', 'P4')}
    if args.validation_only:
        if args.reference_validation is None:
            raise RuntimeError('Pretest validation must compare against reviewed diagnostic')
        reference = json.loads(args.reference_validation.read_text())
        if reference['status'] != 'complete' or set(reference['models']) != {'P1', 'P2', 'P4'}:
            raise RuntimeError('Incomplete reference validation')
        differences = {}
        for precision, field in [('bfloat16', 'bf16_nll_nats_per_token'), ('float32', 'fp32_nll_nats_per_token')]:
            for name in ('P1', 'P2', 'P4'):
                observed = result['precisions'][precision]['models'][name]['aggregate']['nll_nats_per_token']
                expected = reference['models'][name]['loss_1024']['aggregate'][field]
                diff = observed - expected
                if abs(diff) > 1e-5:
                    raise RuntimeError(f'Pretest NLL mismatch: {precision}/{name} {diff}')
                differences[f'{precision}/{name}'] = diff
        result['reference_validation'] = {'path': str(args.reference_validation),
                                          'nll_differences': differences, 'passed': True, 'tolerance': 1e-5}
    result.update(status='SUCCEEDED', finished_at=evaluate.utcnow(), wall_seconds=time.time()-started)
    diagnostic.atomic_json(result_path, result)
    if not args.validation_only:
        metrics = []
        for precision, values in result['precisions'].items():
            for name, rec in values['models'].items():
                metrics.append({'config': f'{precision}/{name}', 'seed': 42, 'split': 'test',
                                'metric': 'nll_nats_per_token', 'value': rec['aggregate']['nll_nats_per_token'],
                                'unit': 'nats/token'})
        report = {'schema_version': 1, 'run_id': run.name, 'status': 'SUCCEEDED',
                  'summary': 'Completed fixed test BF16 primary and FP32 sensitivity for P1/P2/P4, paired bootstrap and shared-GPU generation timing. Scientific interpretation awaits supervisor review.',
                  'executed_commands': [f'bash scripts/run_full_evaluation.sh {run}'],
                  'metrics': metrics,
                  'artifacts': ['full-results.json', 'test-data-manifest.json', 'pretest-validation.json',
                                'validation-data-manifest.json', 'model-manifest.json', 'raw/full.log'],
                  'deviations': result['limitations'], 'failures': [],
                  'resource_usage': {'evaluation_wall_seconds': result['wall_seconds'],
                                     'gpu_hours_conservative_bound': (result['wall_seconds'] +
                                         json.loads((run/'pretest-validation.json').read_text())['wall_seconds']) / 3600},
                  'next_recommendation': 'Independently recompute block-weighted NLL and paired bootstrap; report limitations.',
                  'version_control': {'repository': 'Flutter-Misdreavus/parscale-reproduction',
                    'visibility': 'private', 'run_commits': [result['run_git']], 'push_status': 'pending'}}
        diagnostic.atomic_json(run / 'report.json', report)
    print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'],
                      'primary': result['precisions']['bfloat16']['comparison']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
