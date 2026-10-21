# v3 execution authorization — 2026-10-21

The controller explicitly authorized one new post-fix validation after v2 was
stopped. Reuse supervisor fixes7798c9c and review e620326; no download or training.
Resource admission remains one observation per model: free>=18432MiB,
utilization<=10%, locked GPU UUID and16GiB allocator cap. A failed observation
ends the attempt without polling, altered sampling, or another invocation.

After releasing a model, wait a fixed1.2 seconds before the next first admission
observation, regardless of its result. This avoids measuring the immediately
preceding model's work as external utilization; it is not wait-until-idle logic.
The scientific BF16 .03 threshold and all old failures remain unchanged.

Systemd1800s/KillMode=control-group; wrapper1740s; the single diagnostic command
is bounded to900s including CPU work, conservatively <=0.25GPUh. Project bounds
remain6 active hours/4GPUh and today's21:30CST outer stop. GPT reviews evidence
before any explicit scientific protocol version/full task; DSH does not start it.
