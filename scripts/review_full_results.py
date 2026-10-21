#!/usr/bin/env python3
"""Independent CPU audit of full per-block records (no evaluator imports)."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def same(actual, expected, label, atol=1e-10):
    if not math.isfinite(float(actual)) or not math.isclose(actual, expected, abs_tol=atol, rel_tol=1e-10):
        raise AssertionError((label, actual, expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.run_dir/'full-results.json').read_bytes()
    result = json.loads(raw)
    manifest = json.loads((args.run_dir/'test-data-manifest.json').read_text())
    pretest = json.loads((args.run_dir/'pretest-validation.json').read_text())
    assert result['status'] == pretest['status'] == 'SUCCEEDED'
    assert manifest['split'] == 'test' and pretest['stage'] == 'validation'
    assert pretest['reference_validation']['passed']
    assert result['run_git']['head_commit'] == pretest['run_git']['head_commit']
    assert result['data_token_hash'] == manifest['token_ids_sha256']
    n = manifest['blocks_used']
    expected_counts = np.array([x['token_count']-1 for x in manifest['blocks_used_meta']], dtype=np.int64)
    assert n == len(expected_counts) == manifest['blocks_total']
    review = {'run_id': result['run_id'], 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'status': 'PASSED', 'actual_run_commit': result['run_git']['head_commit'],
              'blocks': n, 'tokens_scored': int(expected_counts.sum()), 'precisions': {},
              'scope': 'Recompute metrics from recorded blocks; not an independent model training/inference rerun.'}
    for precision in ('bfloat16', 'float32'):
        data = result['precisions'][precision]
        assert set(data['models']) == {'P1', 'P2', 'P4'}
        sums, summaries = {}, {}
        for model, rec in data['models'].items():
            blocks = rec['blocks']
            assert len(blocks) == n
            values = []
            for i, block in enumerate(blocks):
                assert block['block_id'] == i and block['token_count_scored'] == int(expected_counts[i])
                assert block['token_ids_sha256'] == manifest['blocks_used_meta'][i]['token_ids_sha256']
                assert block['finite'] and abs(block['manual_minus_label']) <= 1e-3
                same(block['nll_sum_nats']/block['token_count_scored'], block['nll_nats_per_token'], 'block NLL')
                values.append(block['nll_sum_nats'])
            sums[model] = np.array(values, dtype=np.float64)
            nll = math.fsum(values) / int(expected_counts.sum())
            same(nll, rec['aggregate']['nll_nats_per_token'], 'weighted NLL')
            same(math.exp(nll), rec['aggregate']['ppl'], 'PPL')
            summaries[model] = {'nll': nll, 'ppl': math.exp(nll)}
        rng = np.random.default_rng(20261021)
        draws = rng.integers(0, n, size=(2000, n))
        pairs = {}
        for b, a in [('P2', 'P1'), ('P4', 'P1'), ('P4', 'P2')]:
            # Independent delta-numerator form avoids calling worker's ratio/analysis helpers.
            dist = (sums[b][draws].sum(axis=1)-sums[a][draws].sum(axis=1))/expected_counts[draws].sum(axis=1)
            lo, hi = np.percentile(dist, [2.5,97.5])
            recorded = data['comparison']['pairs'][f'{b}-{a}']
            same(float(lo), recorded['ci95_low'], 'CI low')
            same(float(hi), recorded['ci95_high'], 'CI high')
            delta = summaries[b]['nll']-summaries[a]['nll']
            same(delta, recorded['observed_delta_nll'], 'observed effect')
            pairs[f'{b}-{a}'] = {'delta_nll': delta, 'ci95': [float(lo), float(hi)]}
        relative = -pairs['P2-P1']['delta_nll']/summaries['P1']['nll']
        promising = relative >= .005 and pairs['P2-P1']['ci95'][1] < 0 and summaries['P4']['nll'] <= summaries['P2']['nll']
        assert promising == data['comparison']['prespecified_promising']
        review['precisions'][precision] = {'models': summaries, 'pairs': pairs,
            'relative_P2_improvement': relative, 'prespecified_promising': bool(promising)}
    review['timing'] = {}
    for model, timing in result['timing'].items():
        assert timing['output_lengths'] == [160]*13
        values = np.array(timing['raw_seconds'])
        assert len(values) == 10 and np.all(np.isfinite(values)) and np.all(values > 0)
        same(float(np.median(values)), timing['median_seconds'], 'timing median')
        same(float(np.percentile(values, 90)), timing['p90_seconds'], 'timing p90')
        review['timing'][model] = {'median_seconds': float(np.median(values)), 'p90_seconds': float(np.percentile(values,90))}
    assert set(review['timing']) == {'P1','P2','P4'}
    args.out.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    print(json.dumps(review,indent=2))


if __name__ == '__main__':
    main()
