# Downstream core-four result (parscale-20261021-downstream-core)

Status: **SUCCEEDED**; all 16 model-task pairs completed. Actual run SHA: `e2978d61162225cc5f3b82951f1852f94df2e95e`.

Scope: full official held-out PIQA validation (5-shot), SciQ test (3-shot), WinoGrande validation (5-shot), OpenBookQA test (5-shot); batch 1, context 2048, no truncation; 4,605 documents and 12,210 candidate requests per model (48,840 total). This is the pre-registered four-task core subset, **not** the paper six-group General average.

| Model | PIQA acc_norm | SciQ acc_norm | WinoGrande acc | OpenBookQA acc_norm | Equal-weight mean |
|---|---:|---:|---:|---:|---:|
| P1 | 0.649075 | 0.798000 | 0.527230 | 0.268000 | 0.560576 |
| P2 | 0.659412 | 0.812000 | 0.509866 | 0.286000 | 0.566820 |
| P4 | 0.657236 | 0.827000 | 0.524073 | 0.276000 | 0.571077 |
| P8 | 0.663221 | 0.808000 | 0.515391 | 0.278000 | 0.566153 |

Primary endpoint: P4-P1 = **+1.0501 percentage points** (threshold >=1.0 pp: True). The executor did **not** compute the paired stratified 2,000-resample 95% bootstrap CI; therefore positive subset evidence is **not declared** here. The supervisor must run the independent bootstrap audit before applying the decision rule.

Descriptive: P2-P1 = +0.6243 pp; P8-P1 = +0.5577 pp.

Verification: all eight frozen fields matched for all four tasks (`request-verification.json`), all 16 outcome/candidate file hashes match `full-results.json`, and no non-finite values or memory-admission failures occurred. Peak reserved GPU memory was 15.859 GiB under the 16 GiB cap; scoring wall was 2,508.1 s of the 4,800 s cap and wrapper wall 2,625 s of the 5,400 s cap (~0.697 GPU-hours).

Limitations: one checkpoint per P, shared-GPU timing is descriptive, and this four-task mean must not be reported as the paper six-group General average. `progress.json` is the last 64-candidate checkpoint (15 completed pairs); the final `full-results.json` lists all 16.

Git: run executed at `e2978d61162225cc5f3b82951f1852f94df2e95e`; this result summary is committed after the run. Relay push is pending for the local supervisor.
