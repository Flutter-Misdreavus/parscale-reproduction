# parscale-20261021-full — fixed WikiText-2 test evaluation

Date: 2026-10-21 (Asia/Shanghai). Status: completed once; awaiting independent GPT review.
Protocol: `experiments/parscale-20261021-full/protocol-v2.md`. Task: `experiments/parscale-20261021-full/task.json`.

## Execution

- Actual run commit (code + frozen task): `5acb76e310e15f76e4ee8c50adaedd83e7b8aa7a` on `exp/parscale-20261021`.
- Command: `bash scripts/run_full_evaluation.sh runs/parscale-20261021-full`
- The single wrapper first rehashed all three checkpoints, reproduced the v4 validation NLLs, then scored the complete fixed test split in BF16 and FP32. No protocol change after seeing test results; one attempt, no rerun.
- Checkpoints (HF LFS SHA256 verified): P1 `881efa4…`, P2 `4cd35ea…`, P4 `da73617…` at the locked revisions in `locks.json`.
- Data: `Salesforce/wikitext` `wikitext-2-raw-v1` at `b08601e04326c79dfdd32d625aee71d232d685c3`, canonical `'\n\n'.join(text)`.
- Test: 293 contiguous blocks (1024 tokens, final remainder included), 299078 token IDs, 298785 scored targets (one shifted target dropped per block). Identical token IDs and targets for all P and both precisions.
- Pretest: all six validation NLLs matched the reviewed v4 record (max absolute difference `1.19e-7` nats/token, tolerance `1e-5`); `pretest-validation.json` status `SUCCEEDED` under the same commit.
- BF16 primary, FP32 prespecified sensitivity, TF32 matmul disabled, SDPA, batch 1, seed 42. New official learned-prefix cache per block; no incremental decoding or cross-block cache reuse for NLL.

## Test NLL (nats/token, token-weighted over 298785 targets)

| Config | P1 | P2 | P4 |
|---|---:|---:|---:|
| BF16 (primary) | 2.965453 | 2.932469 | 2.899778 |
| FP32 (sensitivity) | 2.965116 | 2.932292 | 2.899535 |

PPL BF16: P1 19.4035, P2 18.7739, P4 18.1701.
PPL FP32: P1 19.3969, P2 18.7706, P4 18.1657.

## Paired block bootstrap (2000 resamples, seed 20261021, 95% percentile CI)

BF16:

| Pair | Delta NLL | 95% CI |
|---|---:|---:|
| P2−P1 | −0.032983 | [−0.036439, −0.029395] |
| P4−P1 | −0.065675 | [−0.069185, −0.062145] |
| P4−P2 | −0.032691 | [−0.036057, −0.029397] |

FP32:

| Pair | Delta NLL | 95% CI |
|---|---:|---:|
| P2−P1 | −0.032824 | [−0.036319, −0.029227] |
| P4−P1 | −0.065581 | [−0.069071, −0.062033] |
| P4−P2 | −0.032757 | [−0.036078, −0.029477] |

## Prespecified decision rule

Rule: promising if BF16 P2 relative NLL reduction ≥ 0.5%, paired CI upper < 0, and P4 ≤ P2.

- BF16 P2 relative reduction: `1.1123%`, CI upper `−0.029395` < 0, P4 (2.899778) ≤ P2 (2.932469) → criterion met.
- FP32 sensitivity: relative reduction `1.1070%`, CI upper `−0.029227` < 0, P4 ≤ P2 → criterion met; the conclusion is not precision-dependent at this scale.
- BF16 minus FP32 NLL: P1 `+0.000337`, P2 `+0.000178`, P4 `+0.000243` nats/token.

Interpretation: at this external corpus/scale and for these fixed official checkpoints, parallel streams 2 and 4 reduce teacher-forced prediction NLL relative to P1, with P4 below P2. This is a checkpoint-comparison pilot, not independent pretraining, a new scaling law, or clean efficiency evidence.

## Generation timing (descriptive only)

First 128 test tokens, exactly 32 greedy new tokens, EOS disabled, 3 warmups + 10 measured repeats, batch 1, shared GPU.

| Model | Median s | p90 s |
|---|---:|---:|
| P1 | 1.221 | 1.576 |
| P2 | 1.267 | 1.285 |
| P4 | 1.296 | 1.316 |

Shared-GPU load makes these timings descriptive; no 22x/6x efficiency claim.

## Resources

- Pretest wall 36.8 s; full test wall 398.4 s; total 435.2 s (0.1209 GPU-hours conservative bound), within the 1740 s / 0.5 GPU-hour task budget.
- Peak allocated 3.825 GiB, peak reserved 4.361 GiB, under the 16 GiB cap; every model admitted with free memory ≥ 18432 MiB. GPU utilization sampled 4–10% (telemetry only).
- No OOM, NaN, tokenizer/hash mismatch, or manual/model-label loss mismatch. One benign tokenizer length warning (whole-corpus tokenization only).

## Limitations

- One checkpoint per P; training seeds unknown. Block bootstrap quantifies text-block uncertainty only, not training-seed uncertainty.
- WikiText-2 may overlap unknown pretraining data; it is an external evaluation corpus, not provably unseen.
- The original BF16 incremental-cache max-logit gate remains failed and is not reclassified; this run uses a separately reviewed teacher-forced path with a new prefix cache per block.
- Shared-GPU timings are not isolated performance evidence.

## Artifacts (outside Git, in `runs/parscale-20261021-full/`)

`report.json`, `full-results.json` (all per-block records), `pretest-validation.json`, `validation-data-manifest.json`, `test-data-manifest.json`, `model-manifest.json`, `raw/full.log`, `full-once.lock/run.sha`.
Sanitized extracts committed here: `metrics.json`, `environment.json`, this summary. Result commit SHA is recorded in `report.json` to avoid self-referential history.
