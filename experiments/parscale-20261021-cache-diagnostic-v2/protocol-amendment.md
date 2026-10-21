# Resource admission amendment — cache-diagnostic-v2

Authorized by the user/controller on 2026-10-21. Applies only to run
`parscale-20261021-cache-diagnostic-v2`; previous tasks, r3 failures, and the
original PROTOCOL.md are retained unchanged.

Before each GPU model starts, require free memory >=18GiB (18432MiB), replacing
the previous >=20GiB admission requirement for this run. Still require GPU
utilization <=10%, locked UUID GPU-REDACTED, and
16GiB allocator cap. A failed admission check writes evidence and stops; no
indefinite polling, alternate GPU, or interference with others.

Rationale: r3 BF16 peak reserved was <=3.07GiB and FP32 diagnosis completed.
18GiB free retains 2GiB headroom above the allocator cap. This does not guarantee
absence of shared-GPU interference; record checks and actual memory use.

No scientific acceptance criterion changes. BF16 absolute cache tolerance .03
remains failed for r3; new diagnostics assess precision effects without changing
that outcome. No full test until GPT reviews diagnostics and explicitly freezes
any needed scientific protocol version and new task.

Task ceiling: one systemd service, RuntimeMaxSec1800, KillMode=control-group,
worker1740s, GPU0.25h. Cumulative project ceiling remains6 active hours/4GPUh;
use the earlier remaining limit. Conservative wall-clock outer stop is
2026-10-21T21:30:00+08:00; delivery2026-10-22T12:00:00+08:00. No training, purchases,
new download-preparation retries or changes to checkpoints/dataset/environment.
