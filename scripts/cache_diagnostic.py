#!/usr/bin/env python3
"""FP32 same-sequence diagnostic for the BF16 cached-vs-full smoke check.

The frozen protocol asks for an FP32 diagnostic when the BF16 cache-equivalence
gate fails, without relaxing the gate.  This script recomputes the exact same
short sequence (first 64 validation tokens, prefill 32 + 32 incremental steps)
for each pinned checkpoint in float32 and records the actual logit error.
It does not modify smoke.json or any scientific threshold.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate  # noqa: E402


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def compare(model, block, device, prefill_len):
    t = torch.tensor(block, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(t)
    with torch.inference_mode():
        full = model(input_ids=t, attention_mask=mask, use_cache=True).logits.float()
        prefill = model(input_ids=t[:, :prefill_len],
                        attention_mask=torch.ones((1, prefill_len), dtype=torch.long, device=device),
                        use_cache=True)
        past = prefill.past_key_values
        prefill_diff = (full[:, :prefill_len] - prefill.logits.float()).abs()
        inc = []
        for i in range(prefill_len, t.shape[1]):
            step = model(input_ids=t[:, i:i + 1],
                         attention_mask=torch.ones((1, i + 1), dtype=torch.long, device=device),
                         past_key_values=past, use_cache=True)
            past = step.past_key_values
            inc.append(step.logits[:, 0].float())
        inc = torch.stack(inc, dim=1)
        diff = (full[:, prefill_len:] - inc).abs()
        per_pos = diff[0].max(dim=-1).values
        out = {
            "sequence_tokens": int(t.shape[1]),
            "prefill_tokens": prefill_len,
            "incremental_tokens": int(t.shape[1] - prefill_len),
            "full_vs_prefill_max_abs_logit_diff": float(prefill_diff.max().cpu()),
            "full_vs_incremental_max_abs_logit_diff": float(diff.max().cpu()),
            "full_vs_incremental_mean_abs_logit_diff": float(diff.mean().cpu()),
            "per_incremental_position_max_abs_diff": [float(v.cpu()) for v in per_pos],
            "tolerance_abs": 0.03,
            "passed_tolerance": bool(float(diff.max().cpu()) <= 0.03),
        }
        del full, prefill, past, inc, diff
    torch.cuda.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--prefill", type=int, default=32)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    os.environ["EVAL_RUN_DIR"] = str(run_dir)
    with open(ROOT / "locks.json", encoding="utf-8") as fh:
        locks = json.load(fh)

    data, manifest = evaluate.prepare_data(locks, run_dir, "smoke")
    block = data[0][: args.length]
    if len(block) < args.length:
        raise RuntimeError(f"block too short: {len(block)} < {args.length}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu = evaluate.gpu_environment(locks["resource"]["gpu_uuid"])
    if not gpu.get("uuid_matches_lock"):
        raise RuntimeError(f"authorized GPU UUID mismatch: {gpu}")
    cap = evaluate.cap_gpu_memory(evaluate.EXPECTED_GPU_GIB)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    paths = evaluate.resolve_model_paths(locks)
    results = {}
    for m in locks["models"]:
        p = m["p"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        from upstream.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM
        model = Qwen2ParScaleForCausalLM.from_pretrained(
            str(paths[p]), torch_dtype=dtype, attn_implementation="sdpa",
            local_files_only=True, low_cpu_mem_usage=True,
        )
        model.eval()
        model.to("cuda")
        loaded = time.time() - t0
        check = compare(model, block, "cuda", args.prefill)
        check["model_id"] = m["id"]
        check["revision"] = m["revision"]
        check["requested_dtype"] = args.dtype
        check["actual_weight_dtype"] = str(next(model.parameters()).dtype)
        check["load_seconds"] = loaded
        check["peak_memory_gib"] = evaluate.peak_mem_gib()
        print(f"[diag] P{p} dtype={check['actual_weight_dtype']} "
              f"full_vs_inc_max={check['full_vs_incremental_max_abs_logit_diff']:.6f} "
              f"passed={check['passed_tolerance']}", flush=True)
        results[f"P{p}"] = check
        del model
        torch.cuda.empty_cache()

    out = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "purpose": "FP32 same-sequence diagnostic for failed BF16 cache gate; protocol gate unchanged",
        "git": evaluate.git_state(),
        "gpu": gpu,
        "memory_cap": cap,
        "data": {
            "block_id": 0,
            "token_ids_sha256": manifest["blocks_used_meta"][0]["token_ids_sha256"],
            "sequence_tokens": args.length,
            "prefill_tokens": args.prefill,
        },
        "results": results,
        "all_within_0.03": all(v["passed_tolerance"] for v in results.values()),
    }
    atomic_json(run_dir / "cache-diagnostic.json", evaluate.jsonable(out))
    print(f"[diag] all_within_0.03={out['all_within_0.03']}", flush=True)
    if not out["all_within_0.03"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
