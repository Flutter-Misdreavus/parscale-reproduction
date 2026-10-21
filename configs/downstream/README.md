# Frozen downstream task settings

Seven task YAMLs are byte-for-byte derived from EleutherAI lm-evaluation-harness
`0.4.8` (wheel SHA256 recorded in `requirements/downstream.txt`) with only these
changes: canonical immutable dataset repository IDs, `dataset_kwargs.revision`
pins, `trust_remote_code: false` (parquet datasets need no remote code), and the
task's paper few-shot count set explicitly. Prompt templates, target extraction,
choice order, splits and metrics are unchanged. `preprocess_winogrande.py` and
`utils.py` are unmodified copies of the upstream helpers those YAMLs reference.

Context limit: 2048 tokens (checkpoint `max_position_embeddings`), left
truncation of context only to 2047 tokens so the answer continuation is intact.
No right truncation and no synthetic EOS/BOS insertion.
