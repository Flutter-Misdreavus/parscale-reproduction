# Downstream smoke result (parscale-20261021-downstream-smoke)

Executed commit: `aa8d32efcdb5b3784cb8223823ed2f883e417ae1`
Command: `timeout --signal=TERM --kill-after=5 175 bash scripts/run_downstream_prep.sh runs/parscale-20261021-downstream-smoke smoke`
Single GPU smoke attempt; no full held-out evaluation, no accuracy produced.

## Gates

All nine smoke gates passed (`scorer-validation.json`, status `SUCCEEDED`):

- Max `|adapter - independent manual|` answer logprob difference: **0.0 nats** over
  16 cases (P1/P2/P4/P8, two real train-style cases and two synthetic cases),
  tolerance `1e-5`.
- P1 FP32 adapter-vs-manual difference: **0.0 nats** (tolerance `1e-5`).
- P1 BF16-vs-FP32 fixed-answer sensitivity: **max 0.038949 nats** (tolerance `0.1`).
- GPU seconds used: **21.56 s** (limit 180 s). Wall time of the smoke command: **35 s**
  (task limit 600 s).
- Peak allocated GPU memory: **4.19 GiB** (cap 16 GiB); free-at-admission 19053 MiB
  (minimum 18432 MiB).
- 0-shot train-only task contracts: all 7 tasks loaded, no continuation truncation,
  no special tokens in continuations, gold labels valid.

## Preflight projection (informational)

- All four models over all prepared requests: **28,640 s ≈ 7.96 GPU-hours**.
- Core four tasks only (piqa, sciq, winogrande, openbookqa): **2,426 s ≈ 0.67 GPU-hours**.
- Remaining project cap after this smoke: **7,765 s ≈ 2.16 GPU-hours**
  (14,400 s cap minus 6,613.323 s prior use minus 21.56 s).

Therefore `fits_remaining_project_budget` is **false** for the all-model/all-task
full stage under the existing 4 GPU-hour project cap. This is a supervisor scoping
decision, not a smoke-gate failure; the smoke did not run the full stage.

## Environment

See `environment.json`: Python 3.10.12, torch 2.2.2+cu121, transformers 4.51.3,
lm-eval 0.4.8, datasets 3.5.0, numpy 1.26.3, NVIDIA RTX 3090, driver 570.211.01.
Raw executor log and run SHA remain in the untracked run directory
(`runs/parscale-20261021-downstream-smoke/`).
