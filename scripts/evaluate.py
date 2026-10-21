#!/usr/bin/env python3
"""ParScale minimal-reproduction evaluator (stage smoke or full).

Reads the frozen PROTOCOL.md / locks.json.  Smoke stage uses the first two
1024-token validation blocks only; full stage is reserved for a later
GPT-approved task and is implemented with the same scoring code.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_GPU_GIB = 16.0


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def jsonable(obj):
    """Recursively convert numpy / torch scalars to plain Python."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def git_state():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception as exc:  # pragma: no cover - evidence only
            return f"ERROR: {exc}"

    return {
        "head_commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def package_versions():
    wanted = [
        "torch", "transformers", "datasets", "huggingface-hub", "tokenizers",
        "safetensors", "numpy", "scipy", "einops", "pyarrow", "pandas", "regex",
    ]
    out = {}
    for name in wanted:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def gpu_environment(expected_uuid):
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if uuid is not None and not str(uuid).startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    info = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_count_visible": torch.cuda.device_count(),
        "device_name": props.name,
        "device_uuid_torch": uuid,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gib": props.total_memory / (1024 ** 3),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_gpu_uuid": expected_uuid,
    }
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, timeout=30)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        info["nvidia_smi_all"] = lines
        for ln in lines:
            if expected_uuid in ln:
                info["authorized_gpu_nvidia_smi"] = ln
    except Exception as exc:  # pragma: no cover
        info["nvidia_smi_error"] = str(exc)
    info["uuid_matches_lock"] = (
        os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid
        or uuid == expected_uuid
    )
    return info


def cap_gpu_memory(gib=EXPECTED_GPU_GIB):
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, gib * (1024 ** 3) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {"cap_gib": gib, "fraction_of_total": fraction,
            "total_gib": total / (1024 ** 3)}


def peak_mem_gib():
    return {
        "allocated": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "reserved": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }


def gpu_admission(expected_uuid, free_mib_min=18432, util_max=10.0,
                  required_consecutive=2, max_samples=10, interval=1.0):
    """Bounded admission check for a single locked GPU.

    Requires `required_consecutive` consecutive 1 s samples with
    memory.free >= free_mib_min MiB and utilization.gpu <= util_max percent.
    At most max_samples readings (<= ~6 s); this is a bounded gate, not polling.
    """
    samples = []
    streak = 0
    for i in range(max_samples):
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        rec = None
        for line in raw.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if parts and parts[0] == expected_uuid:
                rec = {"uuid": parts[0], "free_mib": int(parts[1]),
                       "utilization_percent": float(parts[2]),
                       "sampled_at": utcnow()}
                break
        if rec is None:
            raise RuntimeError(f"locked GPU {expected_uuid} not found in nvidia-smi: {raw}")
        samples.append(rec)
        if rec["free_mib"] >= free_mib_min and rec["utilization_percent"] <= util_max:
            streak += 1
        else:
            streak = 0
        if streak >= required_consecutive:
            break
        if i < max_samples - 1:
            time.sleep(interval)
    passed = streak >= required_consecutive
    return {
        "passed": passed,
        "requirement": {"free_mib_min": free_mib_min,
                        "utilization_percent_max": util_max,
                        "required_consecutive_samples": required_consecutive,
                        "max_samples": max_samples,
                        "sample_interval_seconds": interval},
        "samples": samples,
        "consecutive_pass_samples": streak,
        "decision": "admit" if passed else "stop: admission not met",
    }


# --------------------------------------------------------------------------- #
# data and tokenizer
# --------------------------------------------------------------------------- #
def tokenizer_files(path: Path):
    names = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "added_tokens.json", "special_tokens_map.json",
             "tokenizer.model"]
    out = {}
    for name in names:
        p = path / name
        if p.is_file():
            out[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def prepare_data(locks, run_dir, stage):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds_lock = locks["dataset"]
    model_paths = resolve_model_paths(locks)
    p1_path = model_paths[1]
    split = ds_lock["smoke_split"] if stage == "smoke" else ds_lock["evaluation_split"]

    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "datasets"
    t0 = time.time()
    ds = load_dataset(
        ds_lock["id"], ds_lock["subset"], revision=ds_lock["revision"],
        split=split, cache_dir=str(cache_dir),
    )
    load_seconds = time.time() - t0
    rows = list(ds["text"])
    text = "\n\n".join(rows)

    tok = AutoTokenizer.from_pretrained(str(p1_path), local_files_only=True,
                                       trust_remote_code=False)
    t0 = time.time()
    ids = tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    tokenize_seconds = time.time() - t0
    ids = np.asarray(ids, dtype=np.int32)

    ctx = locks["evaluation"]["context_length"]
    blocks = [ids[i:i + ctx] for i in range(0, len(ids), ctx) if len(ids[i:i + ctx]) >= 2]
    n_used = 2 if stage == "smoke" else len(blocks)
    used = blocks[:n_used]

    probe = "ParScale reproduction: causal scoring check. 你好，世界！\n"
    tok_probe_ids = tok(probe, add_special_tokens=False)["input_ids"]

    tokenizer_identity = {}
    for p, path in sorted(model_paths.items()):
        tk = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                           trust_remote_code=False)
        probe_ids = tk(probe, add_special_tokens=False)["input_ids"]
        tokenizer_identity[f"P{p}"] = {
            "class": type(tk).__name__,
            "name_or_path": tk.name_or_path,
            "vocab_size": tk.vocab_size,
            "len": len(tk),
            "special_tokens": {k: v for k, v in (tk.special_tokens_map or {}).items()},
            "probe_ids_sha256": sha256_bytes(
                np.asarray(probe_ids, dtype=np.int32).tobytes()),
            "probe_ids_match_P1": probe_ids == tok_probe_ids,
            "files": tokenizer_files(path),
        }

    block_meta = []
    for i, b in enumerate(used):
        block_meta.append({
            "block_id": i,
            "token_count": int(len(b)),
            "first_token_id": int(b[0]),
            "last_token_id": int(b[-1]),
            "token_ids_sha256": sha256_bytes(b.astype("<i4").tobytes()),
        })

    manifest = {
        "dataset_id": ds_lock["id"],
        "dataset_revision": ds_lock["revision"],
        "subset": ds_lock["subset"],
        "split": split,
        "num_rows": len(rows),
        "num_empty_rows": sum(1 for r in rows if r == ""),
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "text_chars": len(text),
        "join_separator": "\\n\\n",
        "tokenizer": {
            "source": f"ParScale/ParScale-0.7B-P1-Pile@{locks['models'][0]['revision']}",
            "add_special_tokens": False,
            "probe_text": probe,
            "probe_ids": tok_probe_ids,
            "probe_ids_sha256": sha256_bytes(
                np.asarray(tok_probe_ids, dtype=np.int32).tobytes()),
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "identity_per_checkpoint": tokenizer_identity,
        },
        "token_count_full_split": int(len(ids)),
        "token_ids_sha256": sha256_bytes(ids.astype("<i4").tobytes()),
        "context_length": ctx,
        "blocks_total": len(blocks),
        "blocks_used": len(used),
        "blocks_used_meta": block_meta,
        "block_definition": "contiguous non-overlapping token blocks of context_length; final remainder >=2 tokens included; smoke stage uses first two validation blocks",
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "timing": {"load_dataset_seconds": load_seconds,
                   "tokenize_seconds": tokenize_seconds},
    }
    return used, manifest


def resolve_model_paths(locks):
    from huggingface_hub import snapshot_download

    cached_json = None
    run_dir = os.environ.get("EVAL_RUN_DIR")
    if run_dir:
        candidate = Path(run_dir) / "raw" / "downloads" / "model_paths.json"
        if candidate.is_file():
            cached_json = json.loads(candidate.read_text())
    paths = {}
    for m in locks["models"]:
        p = m["p"]
        if cached_json and str(p) in cached_json:
            paths[p] = Path(cached_json[str(p)])
        else:
            paths[p] = Path(snapshot_download(
                repo_id=m["id"], revision=m["revision"], local_files_only=True))
    return paths


def model_file_manifest(path: Path):
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name != ".gitattributes":
            rel = str(p.relative_to(path))
            out[rel] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p) if p.stat().st_size < 2 * 1024 ** 3 else "skipped_large",
            }
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def scored_block(model, block, device):
    import torch.nn.functional as F

    ids = torch.tensor(block, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(ids)
    t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, use_cache=True, labels=ids)
        logits = out.logits[0, :-1].float()
        targets = ids[0, 1:]
        nll_sum = F.cross_entropy(logits, targets, reduction="sum")
        label_loss = float(out.loss.detach().float().cpu())
        del out, logits
    torch.cuda.synchronize()
    n_tokens = int(ids.shape[1] - 1)
    nll = float(nll_sum.detach().float().cpu()) / n_tokens
    return {
        "token_count_scored": n_tokens,
        "nll_sum_nats": float(nll_sum.detach().float().cpu()),
        "nll_nats_per_token": nll,
        "ppl": math.exp(nll) if nll < 700 else float("inf"),
        "label_loss_mean": label_loss,
        "manual_minus_label": nll - label_loss,
        "finite": bool(torch.isfinite(nll_sum).item() and math.isfinite(label_loss)),
        "seconds": time.time() - t0,
    }
    if not rec["finite"]:
        raise RuntimeError(f"non-finite NLL/loss at block: {rec}")
    if int(ids.shape[1] - 1) != int(targets.numel()):
        raise RuntimeError("target/token count mismatch in scored_block")
    return rec, ids


def causal_check(model, block, device):
    """Changing future tokens must not change earlier logits."""
    t = torch.tensor(block[:64], dtype=torch.long, device=device).unsqueeze(0)
    g = torch.Generator(device=device)
    g.manual_seed(42)
    future = torch.randint(0, int(model.config.vocab_size), (16,),
                           generator=g, device=device)
    t2 = t.clone()
    t2[0, 48:64] = future
    with torch.inference_mode():
        a = model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=True).logits.float()
        b = model(input_ids=t2, attention_mask=torch.ones_like(t2), use_cache=True).logits.float()
        diff = (a[:, :32] - b[:, :32]).abs()
        max_diff = float(diff.max().cpu())
        del a, b
    torch.cuda.synchronize()
    return {
        "checked_tokens_compared": 32,
        "changed_positions": [48, 64],
        "max_abs_logit_diff": max_diff,
        "tolerance_abs": 1e-6,
        "passed": max_diff <= 1e-6,
    }


def cache_check(model, block, device):
    """Full 64-token forward vs 32-token prefill + 32 incremental steps."""
    t = torch.tensor(block[:64], dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        full = model(input_ids=t, attention_mask=torch.ones_like(t),
                     use_cache=True).logits.float()
        prefill = model(input_ids=t[:, :32], attention_mask=torch.ones((1, 32), dtype=torch.long, device=device),
                        use_cache=True)
        past = prefill.past_key_values
        inc = []
        for i in range(32, 64):
            step = model(input_ids=t[:, i:i + 1],
                         attention_mask=torch.ones((1, i + 1), dtype=torch.long, device=device),
                         past_key_values=past, use_cache=True)
            past = step.past_key_values
            inc.append(step.logits[:, 0].float())
        inc = torch.stack(inc, dim=1)
        diff = (full[:, 32:64] - inc).abs()
        max_diff = float(diff.max().cpu())
        mean_diff = float(diff.mean().cpu())
        del full, prefill, past, inc, diff
    torch.cuda.synchronize()
    return {
        "prefill_tokens": 32,
        "incremental_tokens": 32,
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "tolerance_abs": 0.03,
        "passed": max_diff <= 0.03,
    }


def tiny_backward_check(device, seed=42):
    """Tiny random P2 model: finite loss and non-zero finite prefix/aggregation grads."""
    from upstream.configuration_qwen2_parscale import Qwen2ParScaleConfig
    from upstream.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2ParScaleConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, parscale_n=2, parscale_n_tokens=8,
        parscale_attn_smooth=0.01, tie_word_embeddings=True,
    )
    cfg._attn_implementation = "sdpa"
    model = Qwen2ParScaleForCausalLM(cfg)
    # Upstream creates prefix Parameters with torch.empty; initialize only this
    # synthetic check. Real pretrained checkpoint parameters remain untouched.
    for name, param in model.named_parameters():
        if name.endswith(("prefix_k", "prefix_v")):
            torch.nn.init.normal_(param, mean=0.0, std=cfg.initializer_range)
    model = model.to(device)
    model.train()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    ids = torch.randint(0, 128, (1, 16), generator=g, device=device)
    out = model(input_ids=ids, labels=ids, use_cache=False)
    loss = out.loss
    loss.backward()
    params = {
        "prefix_k": model.model.layers[0].self_attn.prefix_k,
        "prefix_v": model.model.layers[0].self_attn.prefix_v,
        "aggregate_linear0_weight": model.model.aggregate_layer[0].weight,
        "aggregate_linear1_weight": model.model.aggregate_layer[2].weight,
    }
    grads = {}
    all_ok = math.isfinite(float(loss.detach().cpu()))
    for name, p in params.items():
        gr = p.grad
        if gr is None:
            grads[name] = {"present": False, "finite": False, "norm": None}
            all_ok = False
            continue
        finite = bool(torch.isfinite(gr).all().item())
        norm = float(gr.detach().float().norm().cpu())
        grads[name] = {"present": True, "finite": finite, "norm": norm,
                       "nonzero": norm > 0}
        all_ok = all_ok and finite and norm > 0
    result = {
        "seed": seed,
        "config": {k: getattr(cfg, k) for k in [
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "parscale_n",
            "parscale_n_tokens", "parscale_attn_smooth"]},
        "loss": float(loss.detach().cpu()),
        "loss_finite": math.isfinite(float(loss.detach().cpu())),
        "grads": grads,
        "passed": bool(all_ok),
    }
    del model, out, loss, ids
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "full"], required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVAL_RUN_DIR"] = str(run_dir)
    started = utcnow()
    t_start = time.time()

    with open(ROOT / "locks.json", encoding="utf-8") as fh:
        locks = json.load(fh)
    run_id = locks.get("run_id") or run_dir.name

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu = gpu_environment(locks["resource"]["gpu_uuid"])
    if not gpu.get("uuid_matches_lock"):
        raise RuntimeError(f"authorized GPU UUID mismatch: {gpu}")
    mem_cap = cap_gpu_memory(EXPECTED_GPU_GIB)

    print(f"[eval] stage={args.stage} run_id={run_id} gpu={gpu['device_name']}", flush=True)
    data, manifest = prepare_data(locks, run_dir, args.stage)
    atomic_json(run_dir / "data-manifest.json", manifest)
    print(f"[eval] data: {len(data)} blocks used, tokenizer probe ok="
          f"{all(v['probe_ids_match_P1'] for v in manifest['tokenizer']['identity_per_checkpoint'].values())}",
          flush=True)

    tiny = tiny_backward_check("cuda")
    print(f"[eval] tiny backward passed={tiny['passed']}", flush=True)

    model_paths = resolve_model_paths(locks)
    results = {}
    checks = {}
    admissions = []
    for m in locks["models"]:
        p = m["p"]
        path = model_paths[p]
        adm = gpu_admission(locks["resource"]["gpu_uuid"], free_mib_min=18432,
                            util_max=10.0, required_consecutive=2)
        adm["model"] = f"P{p}"
        admissions.append(adm)
        atomic_json(run_dir / "preflight.json", {
            "run_id": run_id, "schema_version": 1, "created_at": utcnow(),
            "requirement": {"free_mib_min": 18432, "utilization_percent_max": 10.0,
                            "sampling": "2 consecutive 1 s samples within <=10 readings"},
            "checks": admissions,
        })
        print(f"[eval] P{p} admission passed={adm['passed']}", flush=True)
        if not adm["passed"]:
            raise RuntimeError(f"admission failed before P{p}: {adm}")
        print(f"[eval] P{p} load {path}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        from upstream.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM
        model = Qwen2ParScaleForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to("cuda")
        load_seconds = time.time() - t0
        assert int(model.config.parscale_n) == p, (model.config.parscale_n, p)

        blocks = []
        first_ids = None
        t0 = time.time()
        for bidx, block in enumerate(data):
            rec, ids = scored_block(model, block, "cuda")
            rec["block_id"] = bidx
            blocks.append(rec)
            if first_ids is None:
                first_ids = ids
            print(f"[eval] P{p} block {bidx}: nll={rec['nll_nats_per_token']:.6f} "
                  f"label_diff={rec['manual_minus_label']:.2e}", flush=True)
        score_seconds = time.time() - t0

        c = {
            "causal": causal_check(model, data[0], "cuda"),
            "cache_equivalence": cache_check(model, data[0], "cuda"),
        }
        print(f"[eval] P{p} causal={c['causal']['passed']} "
              f"cache={c['cache_equivalence']['passed']} "
              f"max_cache_diff={c['cache_equivalence']['max_abs_logit_diff']:.5f}", flush=True)

        cfg = model.config
        cfg_keep = {k: getattr(cfg, k, None) for k in [
            "model_type", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "max_position_embeddings", "rope_theta",
            "rms_norm_eps", "tie_word_embeddings", "parscale_n",
            "parscale_n_tokens", "parscale_attn_smooth",
            "use_sliding_window", "sliding_window", "max_window_layers",
            "_attn_implementation",
        ]}
        n_params = sum(q.numel() for q in model.parameters())
        total_tokens = sum(b["token_count_scored"] for b in blocks)
        total_nll = sum(b["nll_sum_nats"] for b in blocks)
        results[f"P{p}"] = {
            "model_id": m["id"],
            "revision": m["revision"],
            "local_snapshot": str(path),
            "dtype": "bfloat16",
            "attention": cfg._attn_implementation,
            "config": cfg_keep,
            "parameter_count": int(n_params),
            "blocks": blocks,
            "aggregate": {
                "token_count_scored": total_tokens,
                "nll_sum_nats": total_nll,
                "nll_nats_per_token": total_nll / total_tokens,
                "ppl": math.exp(total_nll / total_tokens),
                "token_weighted": True,
                "num_blocks": len(blocks),
            },
            "checks": c,
            "timing": {"load_seconds": load_seconds, "score_seconds": score_seconds},
            "peak_memory_gib": peak_mem_gib(),
            "file_manifest": model_file_manifest(path),
        }
        checks[f"P{p}"] = c
        atomic_json(run_dir / "smoke.partial.json", {
            "run_id": run_id, "schema_version": 1, "stage": args.stage,
            "status": "partial", "updated_at": utcnow(),
            "models": results, "checks": checks, "admissions": admissions,
        })
        if not c["causal"]["passed"]:
            raise RuntimeError(f"causal invariant failed for P{p}: {c['causal']}")
        del model
        torch.cuda.empty_cache()

    finite_all = all(b["finite"] for r in results.values() for b in r["blocks"])
    label_ok = all(abs(b["manual_minus_label"]) <= 1e-3
                   for r in results.values() for b in r["blocks"])
    causal_ok = all(c["causal"]["passed"] for c in checks.values())
    cache_ok = all(c["cache_equivalence"]["passed"] for c in checks.values())
    tokenizer_ok = all(v["probe_ids_match_P1"]
                       for v in manifest["tokenizer"]["identity_per_checkpoint"].values())

    # token-weighted NLL deltas (pilot direction), paired blocks
    deltas = {}
    if "P1" in results:
        p1 = results["P1"]["aggregate"]["nll_nats_per_token"]
        for key in ("P2", "P4"):
            if key in results:
                v = results[key]["aggregate"]["nll_nats_per_token"]
                deltas[f"{key}-P1"] = {"delta_nll": v - p1,
                                       "relative_decrease": (p1 - v) / p1}

    status = "passed" if (finite_all and label_ok and causal_ok and cache_ok
                          and tiny["passed"] and tokenizer_ok) else "failed"
    summary = {
        "run_id": run_id,
        "stage": args.stage,
        "status": status,
        "started_at": started,
        "finished_at": utcnow(),
        "wall_seconds": time.time() - t_start,
        "git": git_state(),
        "gpu": gpu,
        "memory_cap": mem_cap,
        "tiny_backward_check": tiny,
        "models": results,
        "checks": {
            "all_scored_nll_finite": finite_all,
            "manual_vs_label_within_1e-3": label_ok,
            "causal_invariant": causal_ok,
            "cached_vs_full_within_0.03": cache_ok,
            "tiny_backward_grads": tiny["passed"],
            "tokenizer_identical_probe": tokenizer_ok,
        },
        "nll_deltas": deltas,
        "notes": [
            "Smoke stage scores only the first two 1024-token validation blocks; test split untouched.",
            "NLL is token-weighted sum over blocks, not a mean of block means.",
            "Peak memory is per-model process peak since the last reset.",
        ],
    }
    atomic_json(run_dir / "smoke.json", jsonable(summary))

    env = {
        "run_id": run_id,
        "stage": args.stage,
        "created_at": utcnow(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "hostname": platform.node(),
        "packages": package_versions(),
        "git": git_state(),
        "gpu": gpu,
        "memory_cap": mem_cap,
        "env": {k: os.environ.get(k) for k in [
            "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_ENDPOINT", "PYTHONPATH",
            "PIP_CACHE_DIR"]},
        "resource": locks["resource"],
        "evaluation": locks["evaluation"],
    }
    atomic_json(run_dir / "environment.json", jsonable(env))
    freeze = sorted({f"{d.metadata['Name']}=={d.version}"
                     for d in md.distributions()
                     if d.metadata.get("Name") and d.version})
    (run_dir / "requirements-freeze.txt").write_text(
        "\n".join(freeze) + "\n", encoding="utf-8")

    print(f"[eval] status={status} wall={summary['wall_seconds']:.1f}s", flush=True)
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
