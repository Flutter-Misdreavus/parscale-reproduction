#!/usr/bin/env python3
"""Downstream multiple-choice preparation, scorer validation and evaluation.

Pinned to lm-evaluation-harness 0.4.8 and the task YAMLs in
configs/downstream/tasks.  This file deliberately keeps the preparation,
GPU-smoke and full-evaluation stages in one reproducible entrypoint so that the
full stage can reuse exactly the same request builder and scorer.

No held-out test accuracy is produced in `prepare` or `smoke`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASKS = OrderedDict([
    ("piqa",          {"repo": "ybisk/piqa",         "config": None,                  "revision": "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0", "shots": 5,  "split": "validation"}),
    ("sciq",          {"repo": "allenai/sciq",       "config": None,                  "revision": "2c94ad3e1aafab77146f384e23536f97a4849815", "shots": 3,  "split": "test"}),
    ("winogrande",    {"repo": "allenai/winogrande", "config": "winogrande_xl",       "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "shots": 5,  "split": "validation"}),
    ("openbookqa",    {"repo": "allenai/openbookqa", "config": "main",                "revision": "388097ea7776314e93a529163e0fea805b8a6454", "shots": 5,  "split": "test"}),
    ("hellaswag",     {"repo": "Rowan/hellaswag",    "config": None,                  "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "shots": 10, "split": "validation"}),
    ("arc_easy",      {"repo": "allenai/ai2_arc",    "config": "ARC-Easy",            "revision": "210d026faf9955653af8916fad021475a3f00453", "shots": 25, "split": "test"}),
    ("arc_challenge", {"repo": "allenai/ai2_arc",    "config": "ARC-Challenge",       "revision": "210d026faf9955653af8916fad021475a3f00453", "shots": 25, "split": "test"}),
])

MODELS = OrderedDict([
    (1, {"repo": "ParScale/ParScale-0.7B-P1-Pile", "revision": "881efa413e390400d5121468f259c6180073c721", "expected_bytes": 2687840768, "expected_sha256": "0e6632e3e64c7125b67f0189d7c0d22d8fce73f1545c9ba37a3b683c0db7eeea"}),
    (2, {"repo": "ParScale/ParScale-0.7B-P2-Pile", "revision": "4cd35ea21ca4a721b69bc9854cbd397605e3c146", "expected_bytes": 2697379032, "expected_sha256": "c25a184a1240c3361311d58f2512a4856cb810b7c01d3960b4bd95c97be9c01e"}),
    (4, {"repo": "ParScale/ParScale-0.7B-P4-Pile", "revision": "da736174c40a0d6f2eca0b6daf0ee7916dffa362", "expected_bytes": 2706905312, "expected_sha256": "8efe0426c53bdb8af3fe2d44847bc5533e05374fd32ef2bedaa14b274924c49d"}),
    (8, {"repo": "ParScale/ParScale-0.7B-P8-Pile", "revision": "2cf8614c5ac7b0edab5912ed1d07aecc3b800f53", "expected_bytes": 2725957880, "expected_sha256": "f0ff8a090b2f885b990be940f24a470208891368da8f0750b4eb198a78acab0a"}),
])

CONTEXT_MAX = 2048
FEWSHOT_SEED = 20261021
TASK_YAML_DIR = ROOT / "configs" / "downstream" / "tasks"
# Tasks whose pinned Hub revision only publishes parquet files are materialised
# outside Git and rendered into a local `parquet` loader config at runtime.
LOCAL_PARQUET = {
    "openbookqa": {"train": "openbookqa/train.parquet",
                    "validation": "openbookqa/validation.parquet",
                    "test": "openbookqa/test.parquet"},
    "hellaswag": {"train": "hellaswag/train.parquet",
                   "validation": "hellaswag/validation.parquet",
                   "test": "hellaswag/test.parquet"},
    "arc_easy": {"train": "arc_easy/train.parquet",
                  "validation": "arc_easy/validation.parquet",
                  "test": "arc_easy/test.parquet"},
    "arc_challenge": {"train": "arc_challenge/train.parquet",
                       "validation": "arc_challenge/validation.parquet",
                       "test": "arc_challenge/test.parquet"},
}
GPU_SECONDS_LIMIT = 180.0
FREE_MIB_MIN = 18432
UTIL_MAX = 10.0
MEMORY_CAP_GIB = 16.0


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def git_state():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception as exc:
            return f"ERROR: {exc}"
    return {"head_commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": run("status", "--porcelain")}


def package_versions():
    wanted = ["torch", "transformers", "datasets", "huggingface-hub", "tokenizers",
              "safetensors", "numpy", "scipy", "einops", "lm-eval", "accelerate",
              "sacrebleu", "scikit-learn", "sqlitedict", "rouge-score", "pytablewriter"]
    out = {}
    for name in wanted:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def environment(run_dir):
    from transformers import AutoTokenizer
    p1 = model_path(1)
    tok = AutoTokenizer.from_pretrained(str(p1), local_files_only=True, trust_remote_code=False)
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        uuid = getattr(props, "uuid", None)
        if uuid is not None and not str(uuid).startswith("GPU-"):
            uuid = f"GPU-{uuid}"
        gpu = {"name": props.name, "uuid": str(uuid), "total_mib": props.total_memory // (1024**2),
               "cuda_visible_devices_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
               "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    return {
        "schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": package_versions(),
        "gpu_authorized_uuid_matches_env": bool(gpu and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu["uuid"]),
        "gpu": gpu,
        "tokenizer_sha256": sha256_file(p1 / "tokenizer.json"),
        "context_max": CONTEXT_MAX, "fewshot_seed": FEWSHOT_SEED,
        "determinism": {"cuda_matmul_allow_tf32": False, "torch_deterministic": False,
                        "batch_size": 1, "padding": "none", "dtype": "bfloat16"},
        "notes": ["Shared GPU; timings are descriptive, not isolated benchmark evidence.",
                  "No credentials or executor logs are included."],
    }


def model_path(p):
    rt = os.environ.get("PARSCALE_RUNTIME_DIR")
    if not rt:
        raise RuntimeError("PARSCALE_RUNTIME_DIR is required")
    return Path(rt) / "models" / f"P{p}"


def model_manifest(run_dir, verify_hashes=True):
    out = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
           "context_max": CONTEXT_MAX, "models": {}}
    for p, spec in MODELS.items():
        path = model_path(p)
        weight = path / "model.safetensors"
        if not weight.is_file():
            raise RuntimeError(f"P{p} model.safetensors missing")
        got_bytes = weight.stat().st_size
        got_sha = sha256_file(weight) if verify_hashes else None
        files = {}
        for name in ["config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "model.safetensors"]:
            f = path / name
            if f.is_file():
                files[name] = {"bytes": f.stat().st_size,
                               "sha256": got_sha if name == "model.safetensors" else sha256_file(f)}
        cfg = json.loads((path / "config.json").read_text())
        rec = {"repo_id": spec["repo"], "revision": spec["revision"],
               "weight_bytes": got_bytes, "weight_sha256": got_sha,
               "expected_bytes": spec["expected_bytes"], "expected_sha256": spec["expected_sha256"],
               "bytes_match": got_bytes == spec["expected_bytes"],
               "sha256_match": (got_sha == spec["expected_sha256"]) if got_sha else None,
               "parscale_n": cfg.get("parscale_n"), "parscale_n_tokens": cfg.get("parscale_n_tokens"),
               "parscale_attn_smooth": cfg.get("parscale_attn_smooth"),
               "max_position_embeddings": cfg.get("max_position_embeddings"),
               "files": files}
        if not rec["bytes_match"] or rec["sha256_match"] is False:
            raise RuntimeError(f"P{p} model hash/size mismatch")
        out["models"][f"P{p}"] = rec
    atomic_json(run_dir / "model-manifest.json", out)
    return out


# --------------------------------------------------------------------------- #
# tasks / data
# --------------------------------------------------------------------------- #
def _runtime_task_dir(run_dir, mode):
    """Render the frozen YAMLs, substituting local pinned parquet where needed."""
    include = run_dir / "raw" / (f"{mode}-tasks")
    include.mkdir(parents=True, exist_ok=True)
    rt = Path(os.environ["PARSCALE_RUNTIME_DIR"])
    for name in TASKS:
        text = (TASK_YAML_DIR / f"{name}.yaml").read_text()
        if mode == "smoke":
            text = re.sub(r"^num_fewshot:\s*\d+\s*$", "num_fewshot: 0", text, flags=re.M)
            text = re.sub(r"^test_split:\s*.*$", "test_split: train", text, flags=re.M)
            text = re.sub(r"^validation_split:\s*.*$", "validation_split: null", text, flags=re.M)
        if name in LOCAL_PARQUET:
            files = {split: str(rt / "downstream-data" / rel)
                     for split, rel in LOCAL_PARQUET[name].items()}
            text = re.sub(r"^dataset_path:.*$", "dataset_path: parquet", text, flags=re.M)
            text = re.sub(r"^dataset_name:.*$", "dataset_name: null", text, flags=re.M)
            text = re.sub(r"^dataset_kwargs:\n(?:  .*\n?)*",
                          "dataset_kwargs: " + json.dumps({"data_files": files}) + "\n",
                          text, flags=re.M)
        (include / f"{name}.yaml").write_text(text)
    for helper in ["preprocess_winogrande.py", "utils.py"]:
        shutil.copy2(TASK_YAML_DIR / helper, include / helper)
    return include


def make_tasks(run_dir, mode):
    """mode=audit uses frozen YAML; mode=smoke uses 0-shot train-only copies."""
    from lm_eval.tasks import TaskManager
    include = _runtime_task_dir(run_dir, mode)
    tm = TaskManager(include_path=str(include))
    tasks = tm.load_task_or_group(list(TASKS))
    for name, task in tasks.items():
        task.set_fewshot_seed(FEWSHOT_SEED)
    return tasks


def encode_pair(tok, context, continuation):
    """Exact causal-LM branch of lm_eval.api.model.LM._encode_pair."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok.encode(context + continuation, add_special_tokens=False)
    ctx = tok.encode(context, add_special_tokens=False)
    return ctx, whole[len(ctx):]


def gold_index(task, doc):
    if task.multiple_input:
        gold = task.doc_to_text(doc)
    else:
        gold = task.doc_to_target(doc)
    choices = task.doc_to_choice(doc)
    if isinstance(gold, int):
        return gold if 0 <= gold < len(choices) else -100
    if isinstance(gold, str):
        return choices.index(gold) if gold in choices else -100
    return -100


def request_record(inst, tok):
    context, continuation = inst.args
    ctx, cont = encode_pair(tok, context, continuation)
    total = len(ctx) + len(cont)
    input_tokens = min(total, CONTEXT_MAX + 1) - 1
    left_truncated = max(0, total - (CONTEXT_MAX + 1))
    return {
        "doc_id": inst.doc_id, "choice": inst.idx,
        "context_sha256": sha256_bytes(context.encode()),
        "continuation_sha256": sha256_bytes(continuation.encode()),
        "context_tokens": len(ctx), "continuation_tokens": len(cont),
        "total_tokens": total, "input_tokens": input_tokens,
        "left_truncated_tokens": left_truncated,
        "scored_tokens": min(len(cont), input_tokens),
        "special_tokens_used": False,
        "context": context, "continuation": continuation,
    }


def prepare(run_dir):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    started = time.time()
    raw = run_dir / "raw"
    (raw / "requests").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(model_path(1)), local_files_only=True,
                                        trust_remote_code=False)
    tasks = make_tasks(run_dir, "audit")
    dataset_manifest = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
                        "framework": {"name": "EleutherAI lm-evaluation-harness", "version": "0.4.8"},
                        "fewshot_seed": FEWSHOT_SEED, "context_max": CONTEXT_MAX,
                        "truncation": "left context only; continuation intact",
                        "datasets": {}, "tasks": {}}
    all_stat = {}
    for name, spec in TASKS.items():
        task = tasks[name]
        t0 = time.time()
        if name in LOCAL_PARQUET:
            local_files = {split: str(Path(os.environ["PARSCALE_RUNTIME_DIR"]) /
                                      "downstream-data" / rel)
                           for split, rel in LOCAL_PARQUET[name].items()}
            ds = load_dataset("parquet", data_files=local_files)
        else:
            ds = load_dataset(spec["repo"], spec["config"], revision=spec["revision"],
                              cache_dir=str(Path(os.environ["HF_HOME"]) / "datasets"))
        splits = {}
        for split_name in sorted(ds.keys()):
            d = ds[split_name]
            h = hashlib.sha256()
            for row in d:
                h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            splits[split_name] = {"rows": len(d), "content_sha256": h.hexdigest(),
                                  "fingerprint": d._fingerprint}
        local = {}
        for split, rel in LOCAL_PARQUET.get(name, {}).items():
            f = Path(os.environ["PARSCALE_RUNTIME_DIR"]) / "downstream-data" / rel
            local[split] = {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
        dataset_manifest["datasets"][name] = {"repo_id": spec["repo"], "config": spec["config"],
                                              "revision": spec["revision"], "splits": splits,
                                              "local_parquet": local}
        task.build_all_requests(limit=None)
        records = []
        with gzip.open(raw / "requests" / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for inst in task.instances:
                rec = request_record(inst, tok)
                fh.write(json.dumps({k: rec[k] for k in ("doc_id", "choice", "context", "continuation")},
                                    ensure_ascii=False) + "\n")
                records.append({k: rec[k] for k in rec if k not in ("context", "continuation")})
        # gold and alignment / prompt-label checks
        docs = {inst.doc_id: inst.doc for inst in task.instances}
        golds = {did: gold_index(task, doc) for did, doc in docs.items()}
        by_doc = {}
        for rec in records:
            by_doc.setdefault(rec["doc_id"], []).append(rec)
        multi = common = 0
        aligned = 0
        for did, recs in by_doc.items():
            recs.sort(key=lambda r: r["choice"])
            conts = None
            # recover continuations from raw file records is unnecessary: check distinct hashes
            hashes = [r["continuation_sha256"] for r in recs]
            if len(set(hashes)) == len(hashes):
                aligned += 1
            if any(r["continuation_tokens"] > 1 for r in recs):
                multi += 1
            prefixes = set()
            for r in recs:
                prefixes.add(r["continuation_sha256"][:12])
            # common-prefix is checked on text below for a compact sample
        # read back a small sample for string prefix checks (raw stays outside Git)
        sample_prefix = 0
        with gzip.open(raw / "requests" / f"{name}.jsonl.gz", "rt", encoding="utf-8") as fh:
            first = [json.loads(next(fh)) for _ in range(min(16, len(records)))]
        for r in first:
            if any(other["continuation"].startswith(r["continuation"]) and
                   other["choice"] != r["choice"] and len(r["continuation"]) < len(other["continuation"])
                   for other in first):
                sample_prefix += 1
        req_tokens = sum(r["input_tokens"] for r in records)
        trunc = sum(1 for r in records if r["left_truncated_tokens"] > 0)
        maxlen = max(r["total_tokens"] for r in records)
        all_stat[name] = {"documents": len(by_doc), "requests": len(records),
                          "input_tokens": req_tokens, "requests_truncated": trunc,
                          "max_total_tokens": maxlen,
                          "mean_input_tokens": req_tokens / max(1, len(records)),
                          "gold_valid": sum(1 for v in golds.values() if v >= 0),
                          "gold_invalid": sum(1 for v in golds.values() if v < 0),
                          "distinct_choice_docs": aligned,
                          "multi_token_choice_docs": multi,
                          "common_prefix_sample_pairs": sample_prefix,
                          "num_fewshot": task.config.num_fewshot,
                          "eval_split": spec["split"],
                          "build_seconds": time.time() - t0}
        # exact gold vector hash, no dataset text
        gold_vec = [[did, golds[did]] for did in sorted(golds)]
        all_stat[name]["gold_index_sha256"] = sha256_bytes(json.dumps(gold_vec).encode())
        print(f"[prepare] {name}: {len(by_doc)} docs, {len(records)} requests, "
              f"{req_tokens} input tokens, {trunc} truncated", flush=True)
    dataset_manifest["tasks"] = all_stat
    dataset_manifest["totals"] = {
        "documents": sum(v["documents"] for v in all_stat.values()),
        "requests": sum(v["requests"] for v in all_stat.values()),
        "input_tokens": sum(v["input_tokens"] for v in all_stat.values()),
        "requests_truncated": sum(v["requests_truncated"] for v in all_stat.values()),
        "prepare_seconds": time.time() - started,
    }
    dataset_manifest["request_set_sha256"] = sha256_bytes(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk in
                        ("documents", "requests", "input_tokens", "gold_index_sha256")}
                    for k, v in all_stat.items()}, sort_keys=True).encode())
    atomic_json(run_dir / "data-manifest.json", dataset_manifest)
    model_manifest(run_dir)
    atomic_json(run_dir / "environment.json", environment(run_dir))
    print(f"[prepare] done in {time.time()-started:.1f}s", flush=True)
    return dataset_manifest


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def gpu_admission():
    expected = os.environ["CUDA_VISIBLE_DEVICES"]
    raw = subprocess.check_output(["nvidia-smi", "-i", expected,
                                   "--query-gpu=uuid,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    parts = [x.strip() for x in raw.strip().split(",")]
    rec = {"uuid": parts[0], "free_mib": int(parts[1]),
           "utilization_percent": float(parts[2]), "sampled_at": utcnow()}
    rec["passed"] = (rec["uuid"] == expected and rec["free_mib"] >= FREE_MIB_MIN
                     and rec["utilization_percent"] <= UTIL_MAX)
    return rec


def cap_gpu_memory():
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, MEMORY_CAP_GIB * 1024**3 / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return {"cap_gib": MEMORY_CAP_GIB, "fraction": frac}


def load_model(p, dtype=torch.bfloat16):
    from upstream.modeling_qwen2_parscale import Qwen2ParScaleForCausalLM
    t0 = time.time()
    model = Qwen2ParScaleForCausalLM.from_pretrained(
        str(model_path(p)), torch_dtype=dtype, attn_implementation="sdpa",
        local_files_only=True, low_cpu_mem_usage=True)
    model.eval().to("cuda")
    torch.cuda.synchronize()
    return model, time.time() - t0


def release(model):
    model.to("cpu")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def manual_loglikelihood(model, tok, context, continuation):
    ctx, cont = encode_pair(tok, context, continuation)
    ids_list = (ctx + cont)[-(CONTEXT_MAX + 1):][:-1]
    ids = torch.tensor([ids_list], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits = model(input_ids=ids).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = ids[:, 1:]
    per = logp[:, :-1, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    n = min(len(cont), per.shape[1])
    return float(per[0, -n:].sum()), {k: len(v) for k, v in (("context", ctx), ("continuation", cont))}, ids.shape[1]


def benchmark(model, tok, samples, repeats=2):
    rows = []
    for label, text in samples:
        ids = tok.encode(text, add_special_tokens=False)[:CONTEXT_MAX]
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        for _ in range(1):
            with torch.inference_mode():
                _ = model(input_ids=inp)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                _ = model(input_ids=inp)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        rows.append({"label": label, "tokens": len(ids), "seconds": times,
                     "median_seconds": float(np.median(times))})
    return rows


def smoke(run_dir):
    from transformers import AutoTokenizer
    from lm_eval.models.huggingface import HFLM
    started = time.time()
    run_dir = Path(run_dir)
    data = json.loads((run_dir / "data-manifest.json").read_text())
    tasks = make_tasks(run_dir, "smoke")
    tok = AutoTokenizer.from_pretrained(str(model_path(1)), local_files_only=True,
                                        trust_remote_code=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    gpu = gpu_admission()
    if not gpu["passed"]:
        raise RuntimeError(f"GPU admission failed: {gpu}")
    cap = cap_gpu_memory()
    validation = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
                  "gpu": gpu, "memory_cap": cap, "batch_size": 1, "padding": "none",
                  "checks": {}, "models": {}}
    # Fixed smoke requests: first 4 training documents per task, 0-shot.
    inst_by_task = {}
    for name, task in tasks.items():
        task.build_all_requests(limit=4)
        inst_by_task[name] = task.instances
    # scorer/material checks on CPU-side metadata
    for name, insts in inst_by_task.items():
        recs = [request_record(i, tok) for i in insts]
        task = tasks[name]
        docs = {i.doc_id: i.doc for i in insts}
        golds = {d: gold_index(task, doc) for d, doc in docs.items()}
        checks = {"requests": len(recs), "documents": len(docs),
                  "gold_valid": all(v >= 0 for v in golds.values()),
                  "all_continuations_nonempty": all(r["continuation_tokens"] > 0 for r in recs),
                  "no_right_truncation": True,
                  "special_tokens_used": False,
                  "max_total_tokens": max(r["total_tokens"] for r in recs),
                  "eval_split": "train (smoke only)"}
        validation["checks"][name] = checks
    # model loop
    summaries = {}
    manual_rows = []
    benchmark_rows = {}
    for p in MODELS:
        adm = gpu_admission()
        if not adm["passed"]:
            raise RuntimeError(f"GPU admission failed before P{p}: {adm}")
        torch.cuda.reset_peak_memory_stats()
        model, load_s = load_model(p)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=CONTEXT_MAX,
                  truncation=False, device="cuda", dtype=torch.bfloat16, add_bos_token=False)
        # adapter vs manual on two fixed requests (one piqa, one arc_challenge)
        chosen = []
        if inst_by_task["piqa"]:
            chosen.append(next((i for i in inst_by_task["piqa"] if i.idx == 0), inst_by_task["piqa"][0]))
        if inst_by_task["arc_challenge"]:
            chosen.append(next((i for i in inst_by_task["arc_challenge"] if i.idx == 0), inst_by_task["arc_challenge"][0]))
        adapter = lm.loglikelihood(chosen)
        manual = []
        for inst in chosen:
            ctx, cont = inst.args
            val, lens, inp_len = manual_loglikelihood(model, tok, ctx, cont)
            manual.append({"value": val, "lens": lens, "input_tokens": inp_len,
                           "context_sha256": sha256_bytes(ctx.encode()),
                           "continuation_sha256": sha256_bytes(cont.encode())})
        diffs = [abs(a[0] - m["value"]) for a, m in zip(adapter, manual)]
        # benchmark fixed synthetic/training-derived inputs at two lengths
        base = "The quick brown fox jumps over the lazy dog. " * 120
        samples = [("short_128", base[:512]), ("long_2048", base)]
        b = benchmark(model, tok, samples)
        benchmark_rows[f"P{p}"] = b
        peak = {"allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        summaries[f"P{p}"] = {"load_seconds": load_s, "peak_memory_gib": peak,
                              "adapter_loglikelihood": [float(a[0]) for a in adapter],
                              "manual_loglikelihood": [m["value"] for m in manual],
                              "abs_diff": diffs, "max_abs_diff": max(diffs),
                              "manual_details": manual,
                              "benchmark": b}
        manual_rows.append({"model": f"P{p}", "max_abs_diff": max(diffs)})
        release(model)
        print(f"[smoke] P{p} load={load_s:.1f}s peak={peak} max_manual_diff={max(diffs):.5f}", flush=True)
    # candidate alignment / common-prefix checks on smoke requests
    align = {}
    for name, insts in inst_by_task.items():
        by_doc = {}
        for i in insts:
            by_doc.setdefault(i.doc_id, []).append(i)
        multi = sum(1 for group in by_doc.values()
                    if any(len(encode_pair(tok, *i.args)[1]) > 1 for i in group))
        common = 0
        for group in by_doc.values():
            conts = [i.args[1] for i in group]
            for a in conts:
                if any(b != a and b.startswith(a) and len(a) < len(b) for b in conts):
                    common += 1
        align[name] = {"docs": len(by_doc), "multi_token_answer_docs": multi,
                       "common_prefix_answer_pairs": common}
    validation["candidate_alignment"] = align
    validation["checks"]["manual_adapter_agreement"] = {
        "tolerance": 0.05, "max_abs_diff": max(r["max_abs_diff"] for r in manual_rows),
        "passed": all(r["max_abs_diff"] <= 0.05 for r in manual_rows), "by_model": manual_rows}
    validation["checks"]["gpu_seconds_used"] = time.time() - started
    validation["models"] = summaries
    validation["status"] = "SUCCEEDED"
    atomic_json(run_dir / "scorer-validation.json", validation)

    # Preflight projection from measured benchmark medians.
    preflight = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
                 "context_max": CONTEXT_MAX, "batch_size": 1,
                 "gpu": gpu, "memory_cap": cap, "smoke_gpu_seconds": validation["checks"]["gpu_seconds_used"],
                 "data": {"requests": data["totals"]["requests"],
                          "input_tokens": data["totals"]["input_tokens"],
                          "requests_truncated": data["totals"]["requests_truncated"],
                          "by_task": {k: {"documents": v["documents"], "requests": v["requests"],
                                          "input_tokens": v["input_tokens"],
                                          "requests_truncated": v["requests_truncated"],
                                          "max_total_tokens": v["max_total_tokens"]}
                                      for k, v in data["tasks"].items()}},
                 "models": {}, "projection": {}}
    for p, rec in summaries.items():
        b = rec["benchmark"]
        xs = np.array([r["tokens"] for r in b], dtype=float)
        ys = np.array([r["median_seconds"] for r in b], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        r2 = 1 - ((ys - pred) ** 2).sum() / max(1e-12, ((ys - ys.mean()) ** 2).sum())
        total_s = sum(slope * v["input_tokens"] + intercept * v["requests"] for v in data["tasks"].values())
        preflight["models"][p] = {"load_seconds": rec["load_seconds"],
                                  "peak_memory_gib": rec["peak_memory_gib"],
                                  "latency_rows": b,
                                  "fit_seconds_per_token": float(slope),
                                  "fit_intercept_seconds": float(intercept), "fit_r2": float(r2)}
        preflight["projection"][p] = {"estimated_seconds": float(total_s),
                                      "estimated_gpu_hours": float(total_s / 3600),
                                      "scope": "all 7 tasks, all held-out requests, batch=1"}
    preflight["projection"]["all_models_seconds"] = float(sum(v["estimated_seconds"] for v in preflight["projection"].values()))
    preflight["projection"]["all_models_gpu_hours"] = preflight["projection"]["all_models_seconds"] / 3600
    preflight["projection"]["assumption"] = ("Linear fit to measured 512/2048-token latency on the shared GPU; "
                                             "excludes model load and few-shot construction. Full test is NOT run in prep.")
    preflight["status"] = "SUCCEEDED"
    atomic_json(run_dir / "preflight.json", preflight)
    print(f"[smoke] completed in {time.time()-started:.1f}s", flush=True)
    return validation, preflight


# --------------------------------------------------------------------------- #
# full stage (guarded)
# --------------------------------------------------------------------------- #
def evaluate(run_dir):
    run_dir = Path(run_dir)
    task = json.loads((run_dir / "task.json").read_text())
    if not task.get("full_test_authorized_by_supervisor"):
        raise RuntimeError("Full evaluation requires an explicit supervisor-approved task")
    from transformers import AutoTokenizer
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
    data = json.loads((run_dir / "data-manifest.json").read_text())
    state = git_state()
    results = {"schema_version": 1, "run_id": run_dir.name, "stage": "full",
               "status": "RUNNING", "started_at": utcnow(), "run_git": state,
               "config_sha256": sha256_file(ROOT / "configs" / "downstream" / "tasks" / "piqa.yaml"),
               "request_set_sha256": data.get("request_set_sha256"), "models": {}}
    atomic_json(run_dir / "full-results.json", results)
    tok = AutoTokenizer.from_pretrained(str(model_path(1)), local_files_only=True,
                                        trust_remote_code=False)
    for p in MODELS:
        torch.cuda.reset_peak_memory_stats()
        model, load_s = load_model(p)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=CONTEXT_MAX,
                  truncation=False, device="cuda", dtype=torch.bfloat16, add_bos_token=False)
        out = simple_evaluate(model=lm, tasks=list(TASKS),
                              num_fewshot=None, batch_size=1, device="cuda",
                              log_samples=True, bootstrap_iters=0,
                              random_seed=FEWSHOT_SEED, numpy_random_seed=FEWSHOT_SEED,
                              torch_random_seed=FEWSHOT_SEED, fewshot_random_seed=FEWSHOT_SEED,
                              task_manager=make_tasks(run_dir, "audit") and None)
        results["models"][f"P{p}"] = {"load_seconds": load_s,
                                      "peak_memory_gib": {"allocated": torch.cuda.max_memory_allocated()/1024**3,
                                                          "reserved": torch.cuda.max_memory_reserved()/1024**3},
                                      "results": out["results"], "samples": out["samples"]}
        atomic_json(run_dir / "full-results.json", results)
        release(model)
    results["status"] = "SUCCEEDED"
    results["finished_at"] = utcnow()
    atomic_json(run_dir / "full-results.json", results)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["prepare", "smoke", "evaluate"])
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "task.json").is_file():
        raise RuntimeError("run task.json missing")
    if (run_dir / "task.json").read_bytes() != (ROOT / "experiments" / run_dir.name / "task.json").read_bytes():
        raise RuntimeError("run task.json differs from committed task.json")
    if args.mode == "prepare":
        prepare(run_dir)
    elif args.mode == "smoke":
        smoke(run_dir)
    else:
        evaluate(run_dir)


if __name__ == "__main__":
    main()
