# Preparation accepted; scoring still gated

The preparation wrapper timed out (event e978936586c2a5c9cebd8356d15cb51e9a690734900ec5f8484c1dbb3eb3b753). Its early BLOCKED report was stale: the complete data/model manifests were subsequently produced before termination. The supervisor retrieved those as supplemental evidence with independent SHA256 receipts rather than changing the failed event.

Accepted preparation: 18,195 held-out documents; 66,566 candidate requests; 49,079,514 input tokens per model; no request truncation; all dataset/task row-hash cross-checks true; all P1/P2/P4/P8 weight SHA256 checks true. Preparation used ecf8383, source later corrected at292376f and438d84d; these later SHAs do not replace the preparation version. P8 was independently rehashed by supervisor against the official pinned Hugging Face API.

No downstream test accuracy or model GPU smoke was produced by those preparation runs. This next stage only validates training/synthetic cases. The caller-reference cleanup was corrected before GPU execution so admission checks are not tripped by retained cached model storage.
