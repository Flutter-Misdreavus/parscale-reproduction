# Full-test protocol v2: teacher-forced NLL and precision sensitivity

This protocol is fixed before any test-split scoring. Original PROTOCOL.md, r3 BF16 cache failure and v2/v3 stopped runs remain unchanged.

The original combined smoke gate mixed teacher-forced prediction loss with incremental-cache equivalence. These are different execution paths. v4 completes all three models in FP32/BF16, two validation sequences, two prefill lengths and two full validation blocks. Independent CPU reconstruction checks original arrays, tokens, reference NLLs and cache differences; see v4 supervisor-independent.json. FP32 cache residuals are below3e-5. BF16 max logit differences still exceed0.03; they are not relabeled as passes. Average validation NLL changes with precision are -0.00066,+0.00003,+0.00044nats/token forP1/P2/P4. These observations justify evaluating teacher-forced loss and explicitly checking numerical sensitivity, not treating BF16 incremental outputs as identical to FP32.

## Fixed evaluation

- Preserve model/data/tokenizer revisions, P1 baseline and P2/P4 comparisons, BF16 primary, batch1, context1024, double-newline joining, no added special tokens, nonoverlapping blocks including tail>=2, scoring L-1 shifted targets per block.
- Each forward initializes the official learned-prefix cache. No cache is passed between blocks or incrementally decoded for NLL. Do not disable the official prefix mechanism.
- Run new evaluator on the two previously reviewed validation blocks in both precisions and all three models; require all six NLLs to match v4 within1e-5nats/token under the identical committed implementation. The test entry checks this validation record and commit before accessing test.
- Evaluate the entire fixed test split in BF16 and in FP32 as a prespecified sensitivity check. TF32 matmul disabled. No hyperparameter/model/dataset selection from test.
- Primary result remains BF16 P2−P1 token-weighted NLL. Paired bootstrap:2000 shared block-index resamples, seed20261021, ratio of summed losses to summed token counts, percentile95%CI. Report all three pairs, same analysis forFP32. Preserve criterion: P2 relative decrease>=0.5%, CIupper<0 andP4<=P2. Precision-dependent conclusions are explicitly flagged.
- Each scored block checks finite manual/model-label losses, agreement<=1e-3 and matching token counts. Persist partial results every25blocks. First exception stops and is reported.
- Timing uses first128test tokens, exactly32greedy new tokens,3warmups and10measured repeats with synchronization; EOS consistently disabled. Report raw repeats, median/p90 and peak memory. GPU utilization is telemetry. Shared-load timings and failed BF16 cache equivalence limit interpretation; no claimed clean speedup or paper22x/6x reproduction.

## Resource and evidence

Fixed GPU0UUID,16GiBcap, free>=18432MiB before models; utilization alone does not stop correctness. All work fits systemd1800s/worker1740s, <=0.5GPUh for this task and original project4GPUh/6active hours/21:30CST outer stop. Rehash existing weights, no downloads/training. Completion and small artifacts return through MCP to the current conversation, which is the sole supervisor.

Full test may start only after current supervisor accepts v4 diagnostic evidence and sets ready/full_test_authorized_by_supervisor in the separately committed task. Original failed gate is a recorded limitation, not overwritten. Review per-block metrics and bootstrap independently before scientific interpretation. This is a minimal external-corpus checkpoint comparison plus published-data refit, not independent pretraining or proof of a scaling law. Text-block intervals do not quantify training-seed uncertainty; possible pretraining overlap remains unknown.
