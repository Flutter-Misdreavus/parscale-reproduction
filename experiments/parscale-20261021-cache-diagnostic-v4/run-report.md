# v4 validation-only FP32/BF16 cache diagnostic

Run: `parscale-20261021-cache-diagnostic-v4`
Date: 2026-10-21 (Asia/Shanghai), execution window ~18:43–18:49 CST
Status: **SUCCEEDED** (diagnostic executed and evidence complete; the scientific
hypothesis is **not** thereby accepted, and the original BF16 gate is unchanged)

## Identity and version control

- Repository: `Flutter-Misdreavus/parscale-reproduction` (private), branch
  `exp/parscale-20261021`.
- Implementation commit: `851ce50` (v4 telemetry-only resource amendment).
- CPU-regression record commit and actual diagnostic run SHA: `c6a28df`.
- The result commit is the commit that adds this summary; its SHA is recorded as
  `head_commit` in `report.json`. `push_status` is `pending` because this run uses
  relay mode: the local GPT pulls the private objects and pushes.
- Before the GPU run the tree was clean at `c6a28df` (`preflight.json.git`).

## Commands actually executed

```bash
# CPU regression, from the remote repository
CUDA_VISIBLE_DEVICES= PYTHONPATH=/path/to/parscale-runtime/pkgs:/path/to/python/site-packages:$PWD \
  python3 scripts/test_diagnostic_cpu.py
# -> 7 tests, 0 failures, OK

# Exactly one GPU diagnostic run
bash scripts/run_diagnostic_v4.sh runs/parscale-20261021-cache-diagnostic-v4
# -> exit 0, run.sha c6a28df, wall 224.48 s

# Post-run CPU-only consistency check (no GPU, no model forward)
CUDA_VISIBLE_DEVICES= PYTHONPATH=...:$PWD:$PWD/scripts \
  python3 scripts/verify_diagnostic_raw_v4.py --run-dir runs/parscale-20261021-cache-diagnostic-v4
# -> PASSED: 18 files / 48 arrays hash-checked; metrics recomputed
```

## What was run

Same locked P1/P2/P4 checkpoints, SDPA, no weight updates, validation split only:
FP32 and BF16 full teacher-forced forward on the first two 1024-token validation
blocks; full vs cached (prefill 1 and 32) logits on the first 64 tokens of those
blocks; and validation-only 128+32 greedy timing repeats. Test split was not read.
GPU was restricted to the locked UUID with a 16 GiB process cap. Admission
observations were one-per-model; utilization (3–19%) was recorded as telemetry
under the v4 amendment, while free memory (18,720–18,798 MiB) stayed above the
18,432 MiB gate and every admission passed. Peak allocated memory was 3.69 GiB.

## Key observations

### Original BF16 <= 0.03 gate is preserved and still fails

All 12 BF16 sequence/prefill paths exceed 0.03 (per-model maxima):

| model | max BF16 full-vs-cached abs logit error | original 0.03 gate |
|---|---:|---|
| P1 | 0.203125 | FAILED |
| P2 | 0.218750 | FAILED |
| P4 | 0.187500 | FAILED |

No new acceptance threshold was selected from these maxima. This run does **not**
redefine, relax, or pass the r3 gate.

### FP32 residual isolates implementation from precision

FP32 full-vs-cached maxima are tiny, confirming the cached implementation is
numerically consistent at FP32:

| model | max FP32 cache residual (abs logit) |
|---|---:|
| P1 | 2.29e-5 |
| P2 | 2.38e-5 |
| P4 | 2.86e-5 |

### BF16 cache discrepancy is the same order as BF16-vs-FP32 error

| model | max BF16 cache error | max BF16 full-vs-FP32 logit error | ratio | max BF16-vs-FP32 target log-prob diff |
|---|---:|---:|---:|---:|
| P1 | 0.203125 | 0.165925 | 1.22 | 0.110337 |
| P2 | 0.218750 | 0.144709 | 1.51 | 0.067593 |
| P4 | 0.187500 | 0.160005 | 1.17 | 0.085376 |

The BF16 cache discrepancy is therefore not an order-of-magnitude anomaly: it is
comparable to the BF16 forward's own deviation from FP32. This is consistent with
a finite-precision explanation, but the two-sequence evidence does not prove a
mechanism.

### Effect on target log probabilities and NLL is small in aggregate

Per-block token-weighted BF16-minus-FP32 NLL (2046 scored tokens per model):

| model | block 0 delta (nats) | block 1 delta (nats) | per-token delta |
|---|---:|---:|---:|
| P1 | -1.002441 | -0.350830 | -0.000661 |
| P2 | +0.130859 | -0.066406 | +0.0000315 |
| P4 | +0.835937 | +0.070068 | +0.000443 |

Over the 63 scored tokens of each 64-token diagnostic sequence, BF16
cached-minus-full NLL spans -0.2239 to +0.0959 nats (max abs 0.00355 nats/token).
Individual target log-probabilities can differ by up to ~0.11, so per-token
predictions are sensitive even though aggregate NLL changes little.

Validation-only paired-block bootstrap (same protocol seed 20261021, 2000
resamples), for context only:

| pair | BF16 observed delta NLL | BF16 95% CI |
|---|---:|---:|
| P2-P1 | -0.070817 | [-0.088328, -0.053307] |
| P4-P1 | -0.091113 | [-0.096108, -0.086119] |
| P4-P2 | -0.020296 | [-0.032812, -0.007780] |

FP32 bootstrap deltas are nearly identical (-0.071510 / -0.092218 / -0.020707).
With only two 1024-token blocks this CI is coarse and conditional on the fixed
checkpoints.

### Timing and evidence checks

- Timing (validation-only, shared GPU): median 1.205 / 1.236 / 1.224 s, p90
  1.232 / 1.271 / 1.261 s for P1/P2/P4; all outputs had length 160. Not an
  isolated performance claim.
- Tokenizer: identical files/vocabulary/encoding across P1/P2/P4 on the full
  validation split (262,337 tokens, text SHA256
  `b44fb967f92b525731216a507d1fbc44b97446b7ef0a2b4d1a893f81f1bbc29e`).
- All logits/log-probs finite; no NaN or OOM. No test split was read.
- Raw-array verification: 18 files and 48 arrays matched the manifest hashes and
  the stored metrics; the largest per-position recomputation difference was
  2.0e-5, attributable to float32 log-softmax implementation differences.
- Analysis fixtures: token-weighting and paired-bootstrap fixtures passed.

## Resource use

- Diagnostic wall: 224.48 s (wrapper ceiling 1740 s; diagnostic timeout 895 s).
- GPU activity estimate: 0.03906 GPU-hours (ceiling 0.25; project ceiling 4).
- Peak allocated/reserved: 3.69 / 4.40 GiB (16 GiB cap).
- Raw diagnostics: 5,134,237,248 bytes (~4.78 GiB), under the 25 GiB new-disk cap.
- Ledger reports 4,912.96 s prior wrapper wall; adding this run gives ~1.427
  project wrapper hours. GPT must reconcile prior-run totals; the per-run GPU
  figure is activity-time, not an independently metered integral.

## Limitations

Two 64-token sequences and two 1024-token blocks; one observation per
model/precision; shared GPU load varied; no repeated runs. The evidence supports
the finite-precision interpretation as the dominant explanation for the observed
BF16 cache discrepancy, but it does not establish a general tolerance, prove the
scientific hypothesis, or justify a full-test run. The original BF16 gate remains
FAILED. GPT review is required before any protocol revision or test-split work.

## Committed artifacts

`preflight.json`, `initial-admission.json`, `data-manifest.json`,
`tokenizer-consistency.json`, `diagnostics.json`, `analysis-checks.json`,
`analysis-supplement.json`, `timing.json`, `environment.json`,
`resource-ledger.json`, `raw-manifest.json`, `raw-consistency.json`,
`requirements-freeze.txt`, `summary.json`, `run-report.md`, `cpu-regression.json`.
Raw logit arrays and execution logs remain in the run directory, outside Git.
