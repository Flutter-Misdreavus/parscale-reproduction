# ParScale smoke stage — parscale-20261021-smoke-r3

Status: **FAILED** (scientific outputs valid; one frozen implementation gate did not pass)

Run window: 2026-10-21 16:21–17:0x CST. One authorized RTX 3090 (GPU-REDACTED), BF16, SDPA, 16 GiB process cap. Smoke only: first two 1024-token WikiText-2 validation blocks. Test split untouched.

## What passed

- All three pinned checkpoints downloaded/resumed and verified byte-for-byte against the Hugging Face LFS SHA256:
  - P1 `0e6632e3…b7eeea`, 2,687,840,768 bytes
  - P2 `c25a184a…be9c01e`, 2,697,379,032 bytes
  - P4 `8efe0426…24c49d`, 2,690,905,312 bytes
- Dataset: `Salesforce/wikitext` `wikitext-2-raw-v1` revision `b08601e…`, validation split. Joined text SHA256 `b44fb967…c29e`; full validation token count 262,337; two smoke blocks of 1,024 tokens. All three tokenizers produced identical probe IDs and tokenizer.json SHA256.
- Tiny random P2 forward/backward: finite loss 4.9014, finite nonzero prefix_k/prefix_v and aggregation gradients.
- Causal invariant: changing future tokens left earlier logits bit-identical (max abs diff 0.0) for P1/P2/P4.
- Manual vs model-label loss: max abs difference 1.10e-7 across all six scored blocks.
- Published-data fits (CPU, pinned `upstream/parametric_fit.py`, 300 starts, best-of-starts):
  - Pile: R2 = 0.998675585, objective 1.813937e-5
  - Stack-v2-Python: R2 = 0.997832701, objective 3.676741e-5

## Smoke NLL (token-weighted, 2 × 1,024-token validation blocks)

| checkpoint | scored tokens | NLL (nats/token) | PPL | peak reserved |
|---|---:|---:|---:|---:|
| P1 | 2,046 | 2.663069590 | 14.34 | 2.86 GiB |
| P2 | 2,046 | 2.592252342 | 13.36 | 2.91 GiB |
| P4 | 2,046 | 2.571956094 | 13.09 | 3.07 GiB |

Paired deltas on these two blocks: P2−P1 = −0.070817 nats/token (−2.66% relative), P4−P1 = −0.091113 (−3.42%). These are smoke-only observations, not the protocol's full-test decision.

## Failed gate: BF16 cached-vs-full logits

Frozen check: full 64-token forward vs 32-token prefill + 32 incremental steps, tolerance ≤ 0.03 absolute for BF16. Actual max absolute logit differences:

| checkpoint | BF16 max diff | BF16 pass | FP32 same-sequence max diff | FP32 pass |
|---|---:|---|---:|---|
| P1 | 0.187500 | no | 0.000022 | yes |
| P2 | 0.140625 | no | 0.000021 | yes |
| P4 | 0.125000 | no | 0.000025 | yes |

Diagnosis (not a gate relaxation): BF16 logits reach |logit| = 21.875 (mean 4.29). A 0.156–0.188 difference is about 1–2 BF16 ulps at that magnitude; an absolute 0.03 tolerance is below one ulp for logits above 8. The same implementation in FP32 agrees to ~2e-5. A 16-token BF16 diagnostic also fails (0.156–0.180), so this is accumulation/rounding in BF16 rather than a length-specific bug. `smoke.json` therefore records `cached_vs_full_within_0.03: false` and status `failed`; no threshold was changed and no retry with altered protocol was made.

## Deviations

- Preflight free memory on GPU 0 was 19.04 GiB (5,078 MiB already held by unrelated processes), just below the protocol's 20 GiB guideline; utilization was 3–10%. Proceeded under the fixed 16 GiB per-process cap; observed task peak was 3.07 GiB reserved. No unrelated process was touched.
- P1's resumed download hit the 1,500 s wrapper timeout with 194 MB remaining; it was resumed once more and completed, then verified by SHA256. P2/P4 completed in the first resume.
- r2 partial files were reused; all final bytes match the pinned hashes, so the mixed r2 download source does not affect content.

## Git / next step

Code and sanitized evidence are committed on `exp/parscale-20261021`; relay push remains pending. GPT should decide whether the BF16 absolute-logit tolerance is appropriate for this 0.7B model (FP32 reference passes by ~3 orders of magnitude) or whether to amend the frozen check. Full test evaluation was **not** run and must not start without a new GPT-approved task.
