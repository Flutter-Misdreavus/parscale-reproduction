#!/usr/bin/env python3
"""Validation-only FP32/BF16 cache and precision diagnostic (v2).

Reads the frozen PROTOCOL.md, locks.json and
experiments/<run_id>/protocol-amendment.md.  It does not train, does not touch
the test split, and does not modify or relax the r3 BF16 0.03 gate.  The FP32
run is a separate explicit mode: this script never writes smoke.json.

Phases:
  preflight   identity, GPU/disk/budget snapshot and bounded admission
  tokenizer   full validation-corpus tokenizer consistency across P1/P2/P4
  analysis    CPU bootstrap fixtures (and token-weighted arithmetic)
  diagnostics FP32 and BF16 full/cached logits on two 64-token sequences,
              plus BF16-vs-FP32 loss on the two 1024-token validation blocks
  timing      validation-only 128+32 greedy generation raw repeats
  all         preflight -> tokenizer -> analysis -> diagnostics -> timing

Raw arrays stay under <run_dir>/raw/diagnostics and are hashed in
diagnostics.json; only compact metrics/manifests are copied to the repo.
"""

import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evaluate  # noqa: E402

FREE_MIB_MIN = 18432
UTIL_MAX = 10.0
ADMISSION_SAMPLES = 1


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(evaluate.jsonable(obj), fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def sha256_np(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def bf16_spacing(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("nonfinite reference logit")
    if x == 0:
        return 2.0 ** -133
    exp = math.floor(math.log2(abs(x)))
    return 2.0 ** (exp - 7)


def error_stats(diff: torch.Tensor) -> dict:
    d = diff.detach().abs().float()
    flat = d.reshape(-1).cpu().numpy()
    return {
        "max_abs": float(flat.max()) if flat.size else 0.0,
        "mean_abs": float(flat.mean()) if flat.size else 0.0,
        "p99_abs": float(np.percentile(flat, 99)) if flat.size else 0.0,
        "elements": int(flat.size),
    }


def worst_coord(diff: torch.Tensor, a: torch.Tensor, b: torch.Tensor, vocab: int) -> dict:
    idx = int(diff.detach().abs().reshape(-1).argmax().item())
    pos, vid = divmod(idx, vocab)
    av = float(a[pos, vid].item())
    bv = float(b[pos, vid].item())
    return {
        "position": int(pos),
        "vocab_index": int(vid),
        "path_logit": av,
        "fp32_reference_logit": bv,
        "abs_diff": abs(av - bv),
        "local_bf16_spacing_at_reference": bf16_spacing(bv),
        "abs_diff_in_bf16_spacing": (abs(av - bv) / bf16_spacing(bv)
                                     if bf16_spacing(bv) else None),
    }


def full_logits(model, tokens) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.tensor(np.asarray(tokens), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        logits = out.logits[0].float()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return logits


def cached_logits(model, tokens, prefill_len: int) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.tensor(np.asarray(tokens), dtype=torch.long, device=device).unsqueeze(0)
    length = int(ids.shape[1])
    with torch.inference_mode():
        pre = model(input_ids=ids[:, :prefill_len],
                    attention_mask=torch.ones((1, prefill_len), dtype=torch.long, device=device),
                    use_cache=True)
        past = pre.past_key_values
        parts = [pre.logits[0].float()]
        for i in range(prefill_len, length):
            step = model(input_ids=ids[:, i:i + 1],
                         attention_mask=torch.ones((1, i + 1), dtype=torch.long, device=device),
                         past_key_values=past, use_cache=True)
            past = step.past_key_values
            parts.append(step.logits[0].float())
        out = torch.cat(parts, dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out


def target_logprobs(logits: torch.Tensor, tokens) -> torch.Tensor:
    lp = F.log_softmax(logits[:-1].float(), dim=-1)
    tgt = torch.tensor(np.asarray(tokens)[1:], dtype=torch.long, device=logits.device)
    return lp.gather(1, tgt[:, None]).squeeze(1)


def diff_summary(path_logits, ref_logits, tokens, seq_meta) -> dict:
    diff = path_logits - ref_logits
    st = error_stats(diff)
    vocab = int(ref_logits.shape[1])
    lp_p = target_logprobs(path_logits, tokens)
    lp_r = target_logprobs(ref_logits, tokens)
    dlp = (lp_p - lp_r).abs()
    dvals = dlp.detach().cpu().numpy()
    st.update({
        "target_logprob_abs_diff_max": float(dvals.max()),
        "target_logprob_abs_diff_mean": float(dvals.mean()),
        "target_logprob_abs_diff_p99": float(np.percentile(dvals, 99)),
        "target_logprob_per_position_abs_diff": [float(x) for x in dvals],
        "worst_logit_coordinate": worst_coord(diff, path_logits, ref_logits, vocab),
        "finite": bool(torch.isfinite(path_logits).all().item()
                       and torch.isfinite(ref_logits).all().item()
                       and torch.isfinite(dlp).all().item()),
        "sequence": seq_meta,
    })
    if not st["finite"]:
        raise RuntimeError(f"non-finite logits/logprobs in diff_summary: {seq_meta}")
    return st


def load_model(path, dtype):
    from upstream.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM
    t0 = time.time()
    model = Qwen2ParScaleForCausalLM.from_pretrained(
        str(path), torch_dtype=dtype, attn_implementation="sdpa",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("cuda")
    torch.cuda.synchronize()
    return model, time.time() - t0


def release(model):
    model.to("cpu")  # Caller may retain a reference; actually release GPU parameters.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def admission_record(locks, label: str, checks: list) -> dict:
    rec = evaluate.gpu_admission(locks["resource"]["gpu_uuid"],
                                 free_mib_min=FREE_MIB_MIN,
                                 util_max=UTIL_MAX)
    rec["label"] = label
    rec["created_at"] = utcnow()
    checks.append(rec)
    return rec


# --------------------------------------------------------------------------- #
# preflight / tokenizer / analysis
# --------------------------------------------------------------------------- #
def phase_preflight(locks, run_dir, model_paths):
    started = json.loads((run_dir / "started.json").read_text()) \
        if (run_dir / "started.json").is_file() else {}
    gpu = evaluate.gpu_environment(locks["resource"]["gpu_uuid"])
    cap = evaluate.cap_gpu_memory(evaluate.EXPECTED_GPU_GIB)
    nvidia = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
         "--format=csv,noheader"], text=True, timeout=20)
    disk = shutil.disk_usage("/path/to/data")
    models = {}
    for p, path in sorted(model_paths.items()):
        f = path / "model.safetensors"
        models[f"P{p}"] = {
            "path": str(path),
            "model_safetensors_bytes": f.stat().st_size if f.is_file() else None,
        }
    pre = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": utcnow(),
        "worker_started_at": started.get("started_at"),
        "wrapper_pid": os.getpid(),
        "systemd_unit": os.environ.get("PARSCALE_SYSTEMD_UNIT", "parscale-cache-diagnostic-v2.service"),
        "gpu": gpu,
        "memory_cap": cap,
        "nvidia_smi_all": [ln for ln in nvidia.splitlines() if ln.strip()],
        "admission_requirement": {
            "gpu_uuid": locks["resource"]["gpu_uuid"],
            "free_mib_min": FREE_MIB_MIN,
            "utilization_percent_max": UTIL_MAX,
            "sampling": f"{ADMISSION_SAMPLES} consecutive 1 s samples within <=10 readings",
            "allocator_cap_gib": evaluate.EXPECTED_GPU_GIB,
            "scope": "before every creation of an FP32 or BF16 GPU model",
        },
        "models": models,
        "dataset": locks["dataset"],
        "disk": {"data_root_total_bytes": disk.total, "data_root_used_bytes": disk.used,
                 "data_root_free_bytes": disk.free,
                 "new_disk_budget_gib": 25},
        "budget": {
            "task_wall_seconds_ceiling": 1740,
            "systemd_runtime_max_sec": 1800,
            "task_gpu_hours_ceiling": 0.25,
            "project_active_hours_ceiling": 6,
            "project_gpu_hours_ceiling": 4,
            "known_prior_active_wall_seconds": {
                "parscale-20261021-smoke": 84.136,
                "parscale-20261021-smoke-r2": 1782.092,
                "parscale-20261021-smoke-r3": 2487.346,
                "sum": 4353.574,
            },
            "known_prior_gpu_hours": {
                "parscale-20261021-smoke-r3_reported_estimate": 0.06,
                "note": "Earlier runs were download/preparation or short smoke; not independently metered. This run records per-model active seconds and wall; GPT must reconcile cumulative totals.",
            },
        },
        "packages": evaluate.package_versions(),
        "git": evaluate.git_state(),
        "checks": [],
    }
    atomic_json(run_dir / "preflight.json", pre)
    print(f"[preflight] gpu={gpu.get('device_name')} free={gpu.get('total_memory_gib')} "
          f"uuid_ok={gpu.get('uuid_matches_lock')} disk_free_gib={disk.free/1024**3:.2f}", flush=True)
    if not gpu.get("uuid_matches_lock"):
        raise RuntimeError("locked GPU UUID mismatch")
    return pre


def phase_tokenizer(locks, run_dir, model_paths, data_manifest):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds_lock = locks["dataset"]
    split = ds_lock["smoke_split"]
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "datasets"
    ds = load_dataset(ds_lock["id"], ds_lock["subset"], revision=ds_lock["revision"],
                      split=split, cache_dir=str(cache_dir))
    rows = list(ds["text"])
    text = "\n\n".join(rows)
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    enc, identity = {}, {}
    for p, path in sorted(model_paths.items()):
        tk = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        ids = tk(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        arr = np.asarray(ids, dtype=np.int32)
        vocab = tk.get_vocab()
        vocab_payload = json.dumps(sorted(vocab.items()), ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8")
        enc[p] = arr
        identity[f"P{p}"] = {
            "class": type(tk).__name__,
            "name_or_path": tk.name_or_path,
            "vocab_size": int(tk.vocab_size),
            "len": int(len(tk)),
            "vocab_map_entries": len(vocab),
            "vocab_map_sha256": hashlib.sha256(vocab_payload).hexdigest(),
            "bos_token_id": tk.bos_token_id,
            "eos_token_id": tk.eos_token_id,
            "pad_token_id": tk.pad_token_id,
            "special_tokens": {k: v for k, v in (tk.special_tokens_map or {}).items()},
            "files": evaluate.tokenizer_files(path),
            "validation_token_count": int(arr.size),
            "validation_token_ids_sha256": sha256_np(arr),
        }
        print(f"[tokenizer] P{p} tokens={arr.size} vocab={tk.vocab_size}", flush=True)
    ref = enc[1]
    mismatches = {}
    for p, arr in enc.items():
        if arr.shape != ref.shape or not np.array_equal(arr, ref):
            n = min(arr.size, ref.size)
            first = int(np.argmax(arr[:n] != ref[:n])) if n else 0
            mismatches[f"P{p}"] = {"shape": list(arr.shape), "first_mismatch_index": first}
    files_equal = all(identity[f"P{p}"]["files"] == identity["P1"]["files"] for p in enc)
    vocab_equal = all(identity[f"P{p}"]["vocab_map_sha256"] == identity["P1"]["vocab_map_sha256"] for p in enc)
    special_equal = all(identity[f"P{p}"]["special_tokens"] == identity["P1"]["special_tokens"] for p in enc)
    text_matches_parent = (text_sha == data_manifest.get("text_sha256"))
    passed = (not mismatches) and files_equal and vocab_equal and special_equal and text_matches_parent
    out = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": utcnow(),
        "split": split,
        "dataset_id": ds_lock["id"],
        "dataset_revision": ds_lock["revision"],
        "subset": ds_lock["subset"],
        "join_separator": "\\n\\n",
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "num_rows": len(rows),
        "num_empty_rows": sum(1 for r in rows if r == ""),
        "text_chars": len(text),
        "text_sha256": text_sha,
        "text_matches_parent_manifest": text_matches_parent,
        "identity_per_checkpoint": identity,
        "full_encoding_equal_across_checkpoints": not mismatches,
        "encoding_mismatches": mismatches,
        "tokenizer_files_equal": files_equal,
        "vocab_maps_equal": vocab_equal,
        "special_tokens_equal": special_equal,
        "requirement": "tokenizer files, vocabulary maps and full validation-corpus encodings identical across P1/P2/P4",
        "passed": bool(passed),
    }
    atomic_json(run_dir / "tokenizer-consistency.json", out)
    print(f"[tokenizer] passed={passed} mismatches={mismatches}", flush=True)
    if not passed:
        raise RuntimeError(f"tokenizer consistency failed: {out}")
    return out


def token_weighted(blocks):
    n = sum(float(b[0]) for b in blocks)
    d = sum(int(b[1]) for b in blocks)
    return n / d if d else float("nan"), n, d


def paired_bootstrap(blocks_by_model, reps=2000, seed=20261021):
    from itertools import combinations
    keys = sorted(blocks_by_model)
    n = len(blocks_by_model[keys[0]])
    if not n:
        raise ValueError("no paired blocks")
    counts = [int(b[1]) for b in blocks_by_model[keys[0]]]
    for key in keys:
        if len(blocks_by_model[key]) != n or [int(b[1]) for b in blocks_by_model[key]] != counts:
            raise ValueError("unpaired block counts")
        if any(c <= 0 or not math.isfinite(float(a)) for a,c in blocks_by_model[key]):
            raise ValueError("invalid NLL/count")
    idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    totals = {}
    for key in keys:
        values = np.asarray(blocks_by_model[key], dtype=np.float64)
        totals[key] = values[idx, 0].sum(axis=1) / values[idx, 1].sum(axis=1)
    out = {}
    for a,b in combinations(keys,2):
        d = totals[b] - totals[a]
        out[f"{b}-{a}"] = {
            "observed_delta_nll": token_weighted(blocks_by_model[b])[0]-token_weighted(blocks_by_model[a])[0],
            "bootstrap_mean": float(d.mean()), "ci95_low": float(np.percentile(d,2.5)),
            "ci95_high": float(np.percentile(d,97.5)), "replicates": reps, "seed": seed,
            "method": "shared paired contiguous-block draws; sum NLL/sum tokens per draw"}
    return out


def phase_analysis(run_dir, actual_loss=None):
    unequal = {"P1": [(1.0, 3), (4.0, 7)], "P2": [(0.8, 3), (3.8, 7)]}
    w1, n1, d1 = token_weighted(unequal["P1"])
    mean_of_means = (1.0 / 3 + 4.0 / 7) / 2
    fixtures = {
        "unequal_blocks_token_weighted": {
            "blocks": unequal["P1"],
            "token_weighted": w1,
            "sum_nll": n1,
            "sum_tokens": d1,
            "mean_of_block_means": mean_of_means,
            "detects_mean_of_means_error": abs(w1 - mean_of_means) > 1e-12,
        },
        "paired_effect": {
            "P1": [(10.0, 10), (12.0, 10)],
            "P2": [(9.0, 10), (11.0, 10)],
            "expected_delta": -0.1,
            "bootstrap": paired_bootstrap({"P1": [(10.0, 10), (12.0, 10)],
                                           "P2": [(9.0, 10), (11.0, 10)]}),
        },
    }
    ok = (fixtures["unequal_blocks_token_weighted"]["detects_mean_of_means_error"]
          and abs(fixtures["unequal_blocks_token_weighted"]["token_weighted"] - 0.5) < 1e-12
          and abs(fixtures["paired_effect"]["bootstrap"]["P2-P1"]["ci95_high"] + 0.1) < 1e-9)
    out = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": utcnow(),
        "fixtures": fixtures,
        "fixtures_passed": bool(ok),
        "actual_validation_two_block_bootstrap": None,
        "bootstrap_config": {"replicates": 2000, "seed": 20261021,
                             "unit": "contiguous 1024-token validation blocks",
                             "note": "Only two blocks exist in this validation diagnostic; the CI is correspondingly coarse."},
    }
    if actual_loss:
        out["actual_validation_two_block_bootstrap"] = paired_bootstrap(actual_loss)
        out["actual_block_nll"] = {
            k: [{"nll_sum_nats": float(a), "token_count": int(b)} for a, b in v]
            for k, v in actual_loss.items()
        }
    atomic_json(run_dir / "analysis-checks.json", out)
    print(f"[analysis] fixtures_passed={ok}", flush=True)
    if not ok:
        raise RuntimeError("analysis fixture self-check failed")
    return out


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def raw_manifest_entry(path: Path, run_dir: Path, arrays: dict) -> dict:
    rel = str(path.relative_to(run_dir))
    return {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": evaluate.sha256_file(path),
        "arrays": {k: {"shape": list(np.asarray(v).shape),
                       "dtype": str(np.asarray(v).dtype),
                       "sha256": sha256_np(np.asarray(v))} for k, v in arrays.items()},
    }


def run_dtype(model, dtype_name, blocks, seq_blocks, model_dir: Path,
              fp32_loss_logits=None, fp32_nll=None, fp32_seq_npz=None):
    out = {"dtype": dtype_name, "loss_1024": [], "sequences": {}}
    # 1024-token teacher-forced loss (full forward)
    for bidx, block in enumerate(blocks):
        logits = full_logits(model, block)
        lp = target_logprobs(logits, block)
        nll_sum = float(-lp.sum().item())
        tgt = np.asarray(block)[1:]
        rec = {
            "block_id": int(bidx),
            "token_count": int(len(block)),
            "token_count_scored": int(len(tgt)),
            "target_token_ids_sha256": sha256_np(np.asarray(tgt, dtype="<i4")),
            "nll_sum_nats": nll_sum,
            "nll_nats_per_token": nll_sum / int(len(tgt)),
            "target_logprobs": [float(x) for x in lp.detach().cpu().numpy()],
            "finite": bool(torch.isfinite(lp).all().item()),
        }
        if not rec["finite"]:
            raise RuntimeError(f"non-finite {dtype_name} loss block {bidx}")
        if fp32_loss_logits is not None:
            ref = torch.tensor(fp32_loss_logits[bidx], dtype=torch.float32, device="cuda")
            diff = logits - ref
            st = error_stats(diff)
            vocab = int(ref.shape[1])
            dlp = (lp - target_logprobs(ref, block)).abs().detach().cpu().numpy()
            st.update({
                "target_logprob_abs_diff_max": float(dlp.max()),
                "target_logprob_abs_diff_mean": float(dlp.mean()),
                "target_logprob_abs_diff_p99": float(np.percentile(dlp, 99)),
                "worst_logit_coordinate": worst_coord(diff, logits, ref, vocab),
                "bf16_minus_fp32_nll_nats": nll_sum - fp32_nll[bidx],
            })
            rec["vs_fp32"] = st
        out["loss_1024"].append(rec)
        # Persist only the FP32 reference logits remotely for independent recomputation.
        if dtype_name == "float32":
            np.save(model_dir / f"P_{dtype_name}_loss_b{bidx}.npy", logits.detach().cpu().numpy())
        del logits, lp

    # 64-token full vs cached paths, prefill 1 and 32
    for sid, block in enumerate(seq_blocks):
        tokens = np.asarray(block[:64], dtype=np.int64)
        seq_meta = {
            "sequence_id": int(sid),
            "parent_block_id": int(sid),
            "token_count": 64,
            "token_dtype": "int32",
            "token_endianness": "little",
            "token_ids_sha256_i32le": sha256_np(tokens.astype("<i4")),
            "target_ids_sha256_i32le": sha256_np(tokens[1:].astype("<i4")),
        }
        full = full_logits(model, tokens)
        c1 = cached_logits(model, tokens, 1)
        c32 = cached_logits(model, tokens, 32)
        arrays = {"tokens": tokens.astype(np.int32),
                  "full": full.detach().cpu().numpy().astype(np.float32),
                  "cached_p1": c1.detach().cpu().numpy().astype(np.float32),
                  "cached_p32": c32.detach().cpu().numpy().astype(np.float32)}
        path = model_dir / f"P_{dtype_name}_seq{sid}.npz"
        np.savez(path, **arrays)
        rec = {
            "sequence": seq_meta,
            "prefill_lengths": [1, 32],
            "raw_npz": str(path.relative_to(model_dir.parent.parent)),
            "full_vs_cached_p1": error_stats(c1 - full),
            "full_vs_cached_p32": error_stats(c32 - full),
            "finite": bool(torch.isfinite(full).all().item()
                           and torch.isfinite(c1).all().item()
                           and torch.isfinite(c32).all().item()),
        }
        if not rec["finite"]:
            raise RuntimeError(f"non-finite {dtype_name} sequence {sid}")
        if fp32_seq_npz is not None:
            ref = np.load(fp32_seq_npz[sid])
            for name, tensor in (("full", full), ("cached_p1", c1), ("cached_p32", c32)):
                rec[f"{name}_vs_fp32"] = diff_summary(
                    tensor, torch.tensor(ref[name], dtype=torch.float32, device="cuda"),
                    tokens, seq_meta)
        out["sequences"][str(sid)] = rec
        del full, c1, c32
        torch.cuda.empty_cache()
    return out


def phase_diagnostics(locks, run_dir, model_paths, blocks, seq_blocks, admissions):
    diag_dir = run_dir / "raw" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": utcnow(),
        "purpose": "validation-only BF16/FP32 full vs cached precision diagnostic; r3 0.03 gate preserved unchanged",
        "git": evaluate.git_state(),
        "attention_backend": "sdpa",
        "dtype_paths": ["float32", "bfloat16"],
        "sequences": "first 64 tokens of validation blocks 0 and 1",
        "loss_blocks": "first two 1024-token validation blocks; targets are tokens[1:]",
        "models": {},
        "raw_manifest": [],
        "status": "running",
    }
    atomic_json(run_dir / "diagnostics.json", out)
    actual_loss = {}
    for m in locks["models"]:
        p = int(m["p"])
        key = f"P{p}"
        model_dir = diag_dir / key
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[diag] {key} FP32 start", flush=True)
        adm = admission_record(locks, f"{key}/float32", admissions)
        atomic_json(run_dir / "preflight.json", {**(json.loads((run_dir / "preflight.json").read_text())),
                                                 "checks": admissions, "updated_at": utcnow()})
        if not adm["passed"]:
            out["status"] = "blocked_admission"
            out["blocked_model"] = f"{key}/float32"
            out["blocked_admission"] = adm
            atomic_json(run_dir / "diagnostics.json", out)
            raise RuntimeError(f"admission failed before {key}/float32: {adm}")

        torch.cuda.reset_peak_memory_stats()
        period = {"dtype": "float32"}
        try:
            model, load_s = load_model(model_paths[p], torch.float32)
            if int(model.config.parscale_n) != p:
                raise RuntimeError(f"{key} parscale_n={model.config.parscale_n} != {p}")
            t0 = time.time()
            fp32_loss = run_dtype(model, "float32", blocks, seq_blocks, model_dir)
            fp32_seconds = time.time() - t0
            fp32_peak = evaluate.peak_mem_gib()
            release(model)
            model = None
        except Exception as exc:
            out["status"] = "failed"
            out["failure"] = {"model": f"{key}/float32", "error": repr(exc)}
            atomic_json(run_dir / "diagnostics.json", out)
            raise
        # save FP32 loss logits references and NLL for BF16 comparison
        fp32_logits = [np.load(model_dir / f"P_float32_loss_b{i}.npy") for i in range(len(blocks))]
        fp32_nll = [r["nll_sum_nats"] for r in fp32_loss["loss_1024"]]
        actual_loss[key] = [(r["nll_sum_nats"], r["token_count_scored"]) for r in fp32_loss["loss_1024"]]
        fp32_seq = {sid: model_dir / f"P_float32_seq{sid}.npz" for sid in range(len(seq_blocks))}

        print(f"[diag] {key} BF16 start", flush=True)
        adm_bf = admission_record(locks, f"{key}/bfloat16", admissions)
        atomic_json(run_dir / "preflight.json", {**(json.loads((run_dir / "preflight.json").read_text())),
                                                 "checks": admissions, "updated_at": utcnow()})
        if not adm_bf["passed"]:
            out["status"] = "blocked_admission"
            out["blocked_model"] = f"{key}/bfloat16"
            out["blocked_admission"] = adm_bf
            atomic_json(run_dir / "diagnostics.json", out)
            raise RuntimeError(f"admission failed before {key}/bfloat16: {adm_bf}")

        torch.cuda.reset_peak_memory_stats()
        try:
            model, load_s_bf = load_model(model_paths[p], torch.bfloat16)
            if int(model.config.parscale_n) != p:
                raise RuntimeError(f"{key} bf16 parscale_n={model.config.parscale_n} != {p}")
            t0 = time.time()
            bf16_loss = run_dtype(model, "bfloat16", blocks, seq_blocks, model_dir,
                                  fp32_loss_logits=fp32_logits, fp32_nll=fp32_nll,
                                  fp32_seq_npz=fp32_seq)
            bf16_seconds = time.time() - t0
            bf16_peak = evaluate.peak_mem_gib()
            # paired BF16-minus-FP32 NLL per block and token-weighted aggregate
            paired = []
            for r, nll32 in zip(bf16_loss["loss_1024"], fp32_nll):
                paired.append({
                    "block_id": r["block_id"],
                    "token_count_scored": r["token_count_scored"],
                    "fp32_nll_nats": nll32,
                    "bf16_nll_nats": r["nll_sum_nats"],
                    "bf16_minus_fp32_nll_nats": r["nll_sum_nats"] - nll32,
                })
            tok_total = sum(x["token_count_scored"] for x in fp32_loss["loss_1024"])
            fp32_agg = sum(fp32_nll) / sum(x["token_count_scored"] for x in fp32_loss["loss_1024"])
            bf16_agg = sum(x["nll_sum_nats"] for x in bf16_loss["loss_1024"]) / tok_total
            cfg = model.config
            cfg_keep = {k: getattr(cfg, k, None) for k in [
                "model_type", "vocab_size", "hidden_size", "num_layers",
                "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "max_position_embeddings", "rope_theta", "rms_norm_eps",
                "tie_word_embeddings", "parscale_n", "parscale_n_tokens",
                "parscale_attn_smooth", "use_sliding_window", "sliding_window",
                "max_window_layers", "_attn_implementation",
            ]}
            peak = {"allocated": max(fp32_peak["allocated"], bf16_peak["allocated"]),
                    "reserved": max(fp32_peak["reserved"], bf16_peak["reserved"])}
            rec = {
                "model_id": m["id"],
                "revision": m["revision"],
                "local_path": str(model_paths[p]),
                "config": cfg_keep,
                "parameter_count": int(sum(q.numel() for q in model.parameters())),
                "load_seconds": {"float32": load_s, "bfloat16": load_s_bf},
                "compute_seconds": {"float32": fp32_seconds, "bfloat16": bf16_seconds},
                "loss_1024": {
                    "fp32": fp32_loss["loss_1024"],
                    "bfloat16": bf16_loss["loss_1024"],
                    "paired": paired,
                    "aggregate": {
                        "token_count_scored": int(tok_total),
                        "fp32_nll_nats_per_token": fp32_agg,
                        "bf16_nll_nats_per_token": bf16_agg,
                        "bf16_minus_fp32_nll_nats_per_token": bf16_agg - fp32_agg,
                        "token_weighted": True,
                    },
                },
                "cache_sequences": {
                    "float32": fp32_loss["sequences"],
                    "bfloat16": bf16_loss["sequences"],
                },
                "peak_memory_gib": peak,
                "fp32_peak_memory_gib": fp32_peak,
                "bf16_peak_memory_gib": bf16_peak,
                "admissions": [adm, adm_bf],
            }
            out["models"][key] = rec
            # update raw manifest incrementally (include both dtypes for this model)
            for sid in range(len(seq_blocks)):
                for dt in ("float32", "bfloat16"):
                    path = model_dir / f"P_{dt}_seq{sid}.npz"
                    out["raw_manifest"].append(raw_manifest_entry(
                        path, run_dir, {"tokens": np.load(path)["tokens"],
                                        "full": np.load(path)["full"]}))
            for bidx in range(len(blocks)):
                path = model_dir / f"P_float32_loss_b{bidx}.npy"
                arr = np.load(path)
                out["raw_manifest"].append({
                    "path": str(path.relative_to(run_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": evaluate.sha256_file(path),
                    "arrays": {"full_logits": {"shape": list(arr.shape),
                                               "dtype": str(arr.dtype),
                                               "sha256": sha256_np(arr)}},
                })
            out["status"] = "partial"
            atomic_json(run_dir / "diagnostics.json", out)
            print(f"[diag] {key} fp32_nll={fp32_agg:.9f} bf16_nll={bf16_agg:.9f} "
                  f"delta={bf16_agg-fp32_agg:.3e}", flush=True)
            release(model)
            model = None
        except Exception as exc:
            out["status"] = "failed"
            out["failure"] = {"model": f"{key}/bfloat16", "error": repr(exc)}
            atomic_json(run_dir / "diagnostics.json", out)
            raise
        del fp32_logits
        gc.collect()

    out["status"] = "complete"
    out["created_at_finished"] = utcnow()
    out["bf16_gate_unchanged"] = {
        "tolerance_abs": 0.03,
        "note": "r3 failure preserved; these diagnostics do not redefine or pass the original BF16 gate.",
    }
    out["summary"] = {
        "max_bf16_cache_error_by_model": {
            k: max(max(s["full_vs_cached_p1"]["max_abs"], s["full_vs_cached_p32"]["max_abs"])
                   for s in v["cache_sequences"]["bfloat16"].values())
            for k, v in out["models"].items()
        },
        "max_bf16_vs_fp32_target_logprob_abs_diff": {
            k: max(
                max(v["cache_sequences"]["bfloat16"][sid][p]["target_logprob_abs_diff_max"]
                    for p in ("full_vs_fp32", "cached_p1_vs_fp32", "cached_p32_vs_fp32"))
                for sid in v["cache_sequences"]["bfloat16"]
            ) for k, v in out["models"].items()
        },
    }
    atomic_json(run_dir / "diagnostics.json", out)
    return out, actual_loss


# --------------------------------------------------------------------------- #
# generation timing (validation-only check, not a performance claim)
# --------------------------------------------------------------------------- #
def phase_timing(model, block, model_key, peak_before):
    prompt = np.asarray(block[:128], dtype=np.int64)
    ids = torch.tensor(prompt, dtype=torch.long, device="cuda").unsqueeze(0)
    mask = torch.ones_like(ids)
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    model.config.eos_token_id = None
    raw = []
    lengths = []
    all_repeat_seconds = []
    for r in range(13):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=mask,
                                 max_new_tokens=32, min_new_tokens=32,
                                 do_sample=False, use_cache=True,
                                 eos_token_id=None, pad_token_id=0,
                                 return_dict_in_generate=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lengths.append(int(gen.shape[1]))
        all_repeat_seconds.append(dt)
        if r >= 3:
            raw.append(dt)
        del gen
    peak = evaluate.peak_mem_gib()
    arr = np.asarray(raw, dtype=np.float64)
    return {
        "model": model_key,
        "label": "validation-only functional timing check; NOT a performance or efficiency benchmark",
        "prompt_tokens": 128,
        "new_tokens": 32,
        "greedy": True,
        "eos_stopping_disabled": True,
        "warmup_repeats": 3,
        "measured_repeats": 10,
        "output_lengths": lengths,
        "expected_output_length": 160,
        "output_length_ok": all(x == 160 for x in lengths),
        "raw_seconds": [float(x) for x in raw],
        "all_repeat_seconds": all_repeat_seconds,
        "median_seconds": float(np.median(arr)),
        "p90_seconds": float(np.percentile(arr, 90)),
        "peak_memory_gib": peak,
        "peak_before_timing": peak_before,
    }


def phase_timing_all(locks, run_dir, model_paths, blocks, admissions):
    out = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
           "label": "validation-only functional timing; not a performance claim",
           "models": {}}
    for m in locks["models"]:
        p = int(m["p"])
        key = f"P{p}"
        adm = admission_record(locks, f"{key}/timing_bfloat16", admissions)
        atomic_json(run_dir / "preflight.json", {**(json.loads((run_dir / "preflight.json").read_text())),
                                                 "checks": admissions, "updated_at": utcnow()})
        if not adm["passed"]:
            out["blocked_admission"] = adm
            atomic_json(run_dir / "timing.json", out)
            raise RuntimeError(f"admission failed before {key}/timing: {adm}")
        torch.cuda.reset_peak_memory_stats()
        model, load_s = load_model(model_paths[p], torch.bfloat16)
        try:
            rec = phase_timing(model, blocks[0], key, evaluate.peak_mem_gib())
            rec["load_seconds"] = load_s
            if not rec["output_length_ok"]:
                raise RuntimeError(f"{key} generation length mismatch: {rec['output_lengths']}")
            out["models"][key] = rec
            atomic_json(run_dir / "timing.json", out)
            print(f"[timing] {key} median={rec['median_seconds']:.4f}s p90={rec['p90_seconds']:.4f}s", flush=True)
        finally:
            release(model)
            model = None
    out["status"] = "complete"
    out["created_at_finished"] = utcnow()
    atomic_json(run_dir / "timing.json", out)
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(run_dir, pre, tok, ana, diag, timing, started_wall):
    metrics = []
    for key, rec in diag.get("models", {}).items():
        agg = rec["loss_1024"]["aggregate"]
        metrics.append({"config": key, "seed": 42, "split": "validation",
                        "metric": "bf16_minus_fp32_nll_nats_per_token",
                        "value": agg["bf16_minus_fp32_nll_nats_per_token"],
                        "unit": "nats/token"})
        for sid, s in rec["cache_sequences"]["bfloat16"].items():
            metrics.append({"config": f"{key}-seq{sid}", "seed": 42, "split": "validation",
                            "metric": "max_abs_logit_error_bf16_cached_p1_vs_fp32",
                            "value": s["cached_p1_vs_fp32"]["max_abs"],
                            "unit": "absolute logit difference"})
            metrics.append({"config": f"{key}-seq{sid}", "seed": 42, "split": "validation",
                            "metric": "max_abs_logit_error_bf16_cached_p32_vs_fp32",
                            "value": s["cached_p32_vs_fp32"]["max_abs"],
                            "unit": "absolute logit difference"})
    failures = []
    status = "SUCCEEDED"
    if diag.get("status") != "complete":
        status = "BLOCKED" if str(diag.get("status", "")).startswith("blocked") else "FAILED"
        failures.append(f"diagnostics status={diag.get('status')}: {diag.get('failure') or diag.get('blocked_admission')}")
    report = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": status,
        "summary": (
            "Validation-only finite-precision diagnostic completed for P1/P2/P4: FP32 and BF16 full/cached paths on two 64-token sequences (prefill 1 and 32) and BF16-vs-FP32 loss on the two 1024-token validation blocks. "
            "Original BF16 <=0.03 gate outcome is preserved unchanged and was not relaxed. "
            "Raw logit arrays remain in the run directory with hashes/shapes; compact metrics are in this run."
        ) if status == "SUCCEEDED" else f"Diagnostic did not complete: {failures}",
        "executed_commands": [
            "python3 scripts/cache_diagnostic_v2.py --run-dir runs/parscale-20261021-cache-diagnostic-v2 --phase all",
        ],
        "metrics": metrics,
        "artifacts": ["report.json", "preflight.json", "diagnostics.json",
                      "tokenizer-consistency.json", "analysis-checks.json",
                      "timing.json", "environment.json", "resource-ledger.json"],
        "deviations": [],
        "failures": failures,
        "resource_usage": {},
        "next_recommendation": (
            "GPT should review FP32 residuals, BF16 target-log-probability differences and worst-error coordinates, then decide whether to version an explicit precision/cache protocol. Do not start full test from this run."
        ),
        "version_control": {},
        "diagnostic_conclusion": {
            "bf16_gate_r3_unchanged": "FAILED; max errors recorded in r3 remain the gate result",
            "fp32_residual": "reported per model/sequence/path in diagnostics.json",
            "no_new_acceptance_threshold_selected_from_observed_bf16_maxima": True,
        },
    }
    role = os.environ.get("PARSCALE_RUN_COMMIT_ROLE", "implementation")
    run_sha = evaluate.git_state().get("head_commit")
    report["version_control"] = {
        "repository": "Flutter-Misdreavus/parscale-reproduction",
        "visibility": "private",
        "branch": "exp/parscale-20261021",
        "run_commits": [{"command": "cache-diagnostic-v2 implementation/run", "sha": run_sha,
                         "role": role}],
        "head_commit": run_sha,
        "commits": 1,
        "push_status": "pending",
        "push_error": "relay mode: local GPT pulls private repository objects and pushes; worker does not push.",
    }
    atomic_json(run_dir / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase", default="all",
                    choices=["all", "preflight", "tokenizer", "analysis", "diagnostics", "timing"])
    ap.add_argument("--dtype", default="paired", choices=["paired", "float32", "bfloat16"],
                    help="paired (default) runs both explicit precision paths; single dtype is a separate explicit mode")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_RUN_DIR"] = str(run_dir)
    wall_start = time.time()
    with open(ROOT / "locks.json", encoding="utf-8") as fh:
        locks = json.load(fh)
    model_paths = evaluate.resolve_model_paths(locks)
    print(f"[v2] run={run_dir.name} phase={args.phase} dtype={args.dtype} head={evaluate.git_state()['head_commit']}", flush=True)

    data, manifest = evaluate.prepare_data(locks, run_dir, "smoke")
    atomic_json(run_dir / "data-manifest.json", manifest)
    tok = phase_tokenizer(locks, run_dir, model_paths, manifest)
    ana = phase_analysis(run_dir)
    if args.phase in ("preflight", "tokenizer", "analysis"):
        print("[v2] non-GPU phase done", flush=True)
        return

    initial = evaluate.gpu_admission(locks["resource"]["gpu_uuid"], FREE_MIB_MIN, UTIL_MAX)
    atomic_json(run_dir / "initial-admission.json", initial)
    if not initial["passed"]:
        raise RuntimeError("initial GPU admission failed; no retries")
    pre = phase_preflight(locks, run_dir, model_paths)
    admissions = []
    diag = None
    actual_loss = None
    if args.phase in ("all", "diagnostics"):
        diag, actual_loss = phase_diagnostics(locks, run_dir, model_paths, data,
                                              [data[0], data[1]], admissions)
        ana = phase_analysis(run_dir, actual_loss)
    timing = None
    if args.phase in ("all", "timing"):
        timing = phase_timing_all(locks, run_dir, model_paths, data, admissions)

    env = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": utcnow(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "hostname": platform.node(),
        "packages": evaluate.package_versions(),
        "git": evaluate.git_state(),
        "gpu": evaluate.gpu_environment(locks["resource"]["gpu_uuid"]),
        "memory_cap": evaluate.cap_gpu_memory(evaluate.EXPECTED_GPU_GIB),
        "env": {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_ENDPOINT", "PYTHONPATH"]},
        "attention_backend": "sdpa",
        "tf32": {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)},
        "torch_threads": torch.get_num_threads(),
    }
    atomic_json(run_dir / "environment.json", env)
    freeze = sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
                     if d.metadata.get("Name") and d.version})
    (run_dir / "requirements-freeze.txt").write_text("\n".join(freeze) + "\n", encoding="utf-8")

    wall = time.time() - wall_start
    gpu_seconds = sum(sum(v.get("compute_seconds", {}).values()) + sum(v.get("load_seconds", {}).values())
                      for v in (diag or {}).get("models", {}).values())
    gpu_seconds += sum(v.get("load_seconds", 0) + sum(v.get("all_repeat_seconds", []))
                       for v in (timing or {}).get("models", {}).values())
    ledger = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": utcnow(),
        "wall_seconds_observed": wall,
        "wall_seconds_ceiling": 1740,
        "gpu_active_seconds_observed_estimate": gpu_seconds,
        "gpu_hours_observed_estimate": gpu_seconds / 3600.0,
        "gpu_hours_ceiling": 0.25,
        "method": "sum of recorded per-model load+compute seconds for FP32/BF16 diagnostics; including timing loads and all warmup/measured generation repeats; excludes tokenizer CPU time",
        "peak_memory_gib": {k: v.get("peak_memory_gib") for k, v in (diag or {}).get("models", {}).items()},
        "admission_checks": admissions,
        "uncertainty": "GPU-hour value is a wall-clock activity estimate for this run, not an independently metered integral; prior project totals must be reconciled by GPT.",
        "disk": {"raw_diagnostics_bytes": sum(p.stat().st_size for p in (run_dir / "raw" / "diagnostics").rglob("*") if p.is_file())},
        "project_known_prior_active_wall_seconds": 4353.574,
        "project_active_hours_if_prior_plus_this": (4353.574 + wall) / 3600.0,
    }
    atomic_json(run_dir / "resource-ledger.json", ledger)
    report = write_report(run_dir, pre, tok, ana, diag or {"status": "not_run"}, timing, wall_start)
    print(f"[v2] report status={report['status']} wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
