# v3 cache/precision diagnostic — BLOCKED at resource admission

Run: `parscale-20261021-cache-diagnostic-v3`  
Parent: `parscale-20261021-cache-diagnostic-v2`  
Status: **BLOCKED** (resource admission; no retry performed)  
Actual run commit: `1f1aeedb5d453a112e94db46fbb6de331f5a7933`  
Branch: `exp/parscale-20261021`

## Command

```
bash scripts/run_diagnostic_once.sh runs/parscale-20261021-cache-diagnostic-v3
```

The runner recorded `run.sha = 1f1aeedb5d453a112e94db46fbb6de331f5a7933` in
`diagnostic-once.lock/run.sha` and executed exactly one bounded subprocess:

```
python3 scripts/cache_diagnostic_v2.py --run-dir <run-dir> --phase all
```

No retry, no second observation after the failed admission, no code or gate change.

## What completed before the stop

| Stage | Result |
|---|---|
| Dataset manifest (validation only) | complete; text `b44fb967...`, 262337 tokens, 257 blocks; two 1024-token blocks used |
| Tokenizer consistency P1/P2/P4 | **passed**; full encodings identical, no mismatches |
| Analysis fixtures (token weighting, paired bootstrap) | **passed** |
| Preflight (GPU identity, 16 GiB cap, disk/budget) | complete; locked UUID matched; RTX 3090, 23.57 GiB |
| P1 FP32 full + cached | complete |
| P1 BF16 full + cached + FP32 reference comparison | complete |
| P2 FP32 full + cached | compute finished; raw arrays written, but no model record/NLL was persisted |
| P2 BF16 | **not started — admission rejected** |
| P4 all paths | not started |
| Timing phase | not started |
| Script `environment.json`, `resource-ledger.json`, `requirements-freeze.txt` | not reached (derived copies assembled here and marked post-hoc) |

## Admission evidence (one observation per attempt)

| Label | Free MiB | Util % | Requirement | Decision |
|---|---:|---:|---|---|
| initial | 19055 | 10.0 | free>=18432, util<=10 | admit |
| P1/float32 | 18798 | 10.0 | same | admit |
| P1/bfloat16 | 18726 | 2.0 | same | admit |
| P2/float32 | 18720 | 4.0 | same | admit |
| **P2/bfloat16** | **18720** | **12.0** | same | **stop** |

Failure timestamp: `2026-10-21T10:29:55.619614+00:00`.
The failed reading was taken after P2/FP32 was released and after the fixed 1.2 s
cooldown required by the amendment. The task forbids a second observation, so it
cannot be determined from this run whether the 12% was external load or a tail
from the just-released local model. It is recorded as a failed admission, not
retried or re-sampled.

## P1 finite-precision results (validation, first two 1024-token blocks)

Cached-path error is `cached - full` on the first 64 tokens of each block,
prefill lengths 1 and 32. All values are absolute logit differences unless
marked "target log-prob".

| Sequence | Path | max | mean | p99 |
|---|---|---:|---:|---:|
| block 0, seq 0 | FP32 full vs cached, prefill 1 | 2.29e-05 | 1.69e-06 | 6.20e-06 |
| block 0, seq 0 | FP32 full vs cached, prefill 32 | 2.29e-05 | 1.72e-06 | 6.68e-06 |
| block 1, seq 1 | FP32 full vs cached, prefill 1 | 2.10e-05 | 1.62e-06 | 6.20e-06 |
| block 1, seq 1 | FP32 full vs cached, prefill 32 | 2.19e-05 | 1.67e-06 | 6.68e-06 |
| block 0, seq 0 | BF16 full vs cached, prefill 1 | **0.203125** | 0.01711 | 0.0625 |
| block 0, seq 0 | BF16 full vs cached, prefill 32 | **0.1875** | 0.01665 | 0.0625 |
| block 1, seq 1 | BF16 full vs cached, prefill 1 | **0.15625** | 0.01485 | 0.0625 |
| block 1, seq 1 | BF16 full vs cached, prefill 32 | **0.16015625** | 0.01489 | 0.0625 |

BF16 versus FP32 reference on the same sequences:

| Sequence | BF16 path | logit max / mean / p99 | target log-prob max / mean / p99 |
|---|---|---:|---:|
| seq 0 | full | 0.157969 / 0.015691 / 0.056765 | 0.110337 / 0.014988 / 0.075261 |
| seq 0 | cached, prefill 1 | 0.186837 / 0.016474 / 0.058586 | 0.066871 / 0.014459 / 0.058057 |
| seq 0 | cached, prefill 32 | 0.147566 / 0.016038 / 0.057718 | 0.061065 / 0.012661 / 0.055931 |
| seq 1 | full | 0.165925 / 0.014949 / 0.053368 | 0.068094 / 0.018588 / 0.058325 |
| seq 1 | cached, prefill 1 | 0.137880 / 0.014463 / 0.050554 | 0.048429 / 0.016739 / 0.046665 |
| seq 1 | cached, prefill 32 | 0.141401 / 0.014774 / 0.052773 | 0.067433 / 0.016022 / 0.057545 |

Two-block teacher-forced loss (FP32 reference, token-weighted over 2046 targets):

| Block | FP32 NLL | BF16 NLL | BF16−FP32 |
|---|---:|---:|---:|
| 0 (1023 tokens) | 2583.4150390625 | 2582.4125976562 | −1.00244140625 nats |
| 1 (1023 tokens) | 2866.5788574219 | 2866.2280273438 | −0.350830078125 nats |
| aggregate | 2.6637311322 nats/token | 2.6630697092 nats/token | **−0.0006614230 nats/token** |

## Gate status (unchanged)

The original r3 BF16 criterion is preserved exactly: maximum absolute
full-vs-cached logit error **<= 0.03**. On the completed P1 evidence the observed
maximum is **0.203125**, so the original gate **still fails**. The FP32 residuals
(~2.3e-05) confirm that the cached path itself is numerically consistent; the
BF16 discrepancy is a finite-precision effect of the BF16 path. This run does
not redefine or pass the gate, and it does not supply P2/P4 BF16 evidence.

## Resources

- Wall time from worker start to stop: ~130.8 s (ceiling 1740 s).
- GPU active time measured for P1: 30.82 s (load+compute, both dtypes).
- P2 FP32 activity bounded by 14.6 s (15.816 s admission-to-admission gap minus
  the fixed 1.2 s cooldown).
- Estimated GPU activity: ~45.4 s ≈ 0.0126 h; conservative wall-clock bound
  0.0363 h (ceiling 0.25 h).
- Peak P1 memory: 3.671 GiB allocated / 4.018 GiB reserved (16 GiB cap).
- Raw diagnostic arrays: ~3.0 GiB under `raw/diagnostics/`; not committed.

## Failures, deviations, missing evidence

- **Failure:** P2/bfloat16 admission: free 18720 MiB passed, utilization 12.0%
  exceeded the 10.0% limit. Per the fixed rule this ended the attempt.
- **Deviations:** none in the scientific protocol, data, seeds, metric, or
  acceptance criterion.
- **Missing:** P2 BF16, P4, timing repeats, and the script's final environment,
  resource-ledger and requirements-freeze outputs. No metric was invented for
  these stages.
- P2 FP32 raw arrays exist and are hashed in `raw-manifest.json`, but the
  per-model record and NLL aggregate were not written before the stop.

## Version control

- Repository: `Flutter-Misdreavus/parscale-reproduction` (private)
- Branch: `exp/parscale-20261021`
- Run commit: `1f1aeedb5d453a112e94db46fbb6de331f5a7933`
- Result/evidence commit: recorded in `report.json`
- Push mode: relay; this worker does not hold a GitHub token. Push status is
  **pending** for the local GPT to pull and push.

## Recommendation for GPT

Decide whether the 12% reading should be treated as external contention or as a
tail from the just-released P2/FP32 model. If the latter is suspected, the
minimal adjustment would be a longer fixed settle interval or a pre-registered
admission rule that excludes the worker's own tail; that is a protocol change
and must not be made by this worker. Otherwise the remaining P2 BF16, P4 and
timing evidence still needs one authorized run.
