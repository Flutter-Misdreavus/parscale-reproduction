# Downstream remaining-three result (parscale-20261021-downstream-rest)

Status: **SUCCEEDED**; completed pairs: **12/12**. Actual run SHA: `34d4a1f9464562d563320c871e1ca1db6f2282d3`.

Scope: full official HellaSwag validation (10-shot), ARC-Easy test and ARC-Challenge test (25-shot); batch 1, context 2048, no truncation; 13,590 documents and 54,356 candidate requests per model (217,424 candidate scores total).

| Model | HellaSwag acc_norm | ARC-Easy acc_norm | ARC-Challenge acc_norm | Extension composite |
|---|---:|---:|---:|---:|
| P1 | 0.356602 | 0.491162 | 0.249147 | 0.363378 |
| P4 | 0.377216 | 0.521886 | 0.261092 | 0.384352 |
| P2 | 0.366461 | 0.507576 | 0.259386 | 0.374971 |
| P8 | 0.385282 | 0.514731 | 0.258532 | 0.385957 |

Primary prospective extension endpoint (point estimate only): P4-P1 = **2.0974 pp**. P2-P1 = 1.1593 pp; P8-P1 = 2.2578 pp. The executor did not compute the paired 2000-replicate bootstrap CI, so positive evidence is **not declared** here; the supervisor applies the >=1 pp and lower>0 rule independently.

ARC-Easy known exact overlaps retained in the official primary set (doc ids [242, 493]); descriptive sensitivity excluding them: P1=0.490733, P2=0.507582, P4=0.521483, P8=0.514743; P4-P1=3.0750 pp. The official primary benchmark is unchanged.

Expanded six-group General (post hoc descriptive, **not** a replacement for the failed core primary test): P1=0.494844, P4=0.508836, P2=0.502870, P8=0.506087.

Verification: all frozen request fields matched for all three tasks; all three local parquet sources rehashed; all 12 candidate/outcome file hashes matched full-results; 217,424 candidate rows and 54,360 outcome rows; non-finite candidate scores=0; invalid outcome rows/counts=0.

Resource: scoring wall 26840.7 s of 30600.0 s (~7.456 GPU-hours); 12/12 pairs. Peaks are in `full-results-summary.json`; the 16 GiB allocator cap was used.

Limitations: one checkpoint per P; shared-GPU timings descriptive; no executor bootstrap; expanded six-group General is post hoc; ARC-Easy two known exact overlaps; HellaSwag ind is not globally unique.

Git: run executed at `34d4a1f9464562d563320c871e1ca1db6f2282d3`; this result summary is committed after the run. Relay push is pending for the local supervisor.
