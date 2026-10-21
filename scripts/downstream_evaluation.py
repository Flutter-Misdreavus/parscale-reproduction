#!/usr/bin/env python3
"""Downstream multiple-choice preparation, scorer validation and evaluation.

Pinned to lm-evaluation-harness 0.4.8 and the task YAMLs in
configs/downstream/tasks.  This file keeps preparation, GPU smoke and the
guarded full stage in one reproducible entrypoint so that the full stage can
reuse exactly the same request builder and scorer.

No held-out test accuracy is produced in `prepare` or `smoke`.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import random
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
MEMORY_CAP_GIB = 16.0
MANUAL_TOL = 1e-5
FP32_TOL = 0.1
# Conservative project ledger from the task controls: 6613.323s used of 4h.
PROJECT_GPU_CAP_SECONDS = 4.0 * 3600.0
PROJECT_GPU_USED_SECONDS = 6613.323


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


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def enforce_determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(FEWSHOT_SEED)
    np.random.seed(FEWSHOT_SEED)
    random.seed(FEWSHOT_SEED)
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 8))
    except Exception:
        pass


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


def nvidia_smi_gpu():
    """Read the authorized GPU through nvidia-smi, not torch's optional uuid."""
    expected = os.environ.get("PARSCALE_GPU_UUID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    target = visible or expected
    if not target:
        return {"error": "no CUDA_VISIBLE_DEVICES/PARSCALE_GPU_UUID", "uuid_matches": False}
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", target,
             "--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        parts = [x.strip() for x in raw.strip().splitlines()[0].split(",")]
        rec = {"name": parts[1], "total_mib": int(float(parts[2])),
               "free_mib": int(float(parts[3])),
               "utilization_percent": float(parts[4]),
               "uuid_matches": parts[0] == expected and parts[0] == visible,
               "sampled_at": utcnow()}
        return rec
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "uuid_matches": False}


def environment(run_dir):
    from transformers import AutoTokenizer
    p1 = model_path(1)
    tok = AutoTokenizer.from_pretrained(str(p1), local_files_only=True, trust_remote_code=False)
    gpu = nvidia_smi_gpu() if torch.cuda.is_available() else None
    return {
        "schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu,
        "tokenizer_sha256": sha256_file(p1 / "tokenizer.json"),
        "context_max": CONTEXT_MAX, "fewshot_seed": FEWSHOT_SEED,
        "determinism": {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "torch_manual_seed": FEWSHOT_SEED, "numpy_seed": FEWSHOT_SEED,
                        "torch_deterministic": bool(torch.backends.cudnn.deterministic),
                        "batch_size": 1, "padding": "none", "dtype": "bfloat16"},
        "notes": ["Shared GPU; timings are descriptive, not isolated benchmark evidence.",
                  "No credentials, executor logs or GPU UUIDs are included."],
    }


def model_path(p):
    rt = os.environ.get("PARSCALE_RUNTIME_DIR")
    if not rt:
        raise RuntimeError("PARSCALE_RUNTIME_DIR is required")
    return Path(rt) / "models" / f"P{p}"


def _file_record(path, weight_sha=None):
    rec = {"bytes": path.stat().st_size}
    rec["sha256"] = weight_sha if weight_sha is not None else sha256_file(path)
    return rec


def model_manifest(run_dir, verify_hashes=True):
    out = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
           "context_max": CONTEXT_MAX, "models": {}, "available_models": [], "pending_models": []}
    for p, spec in MODELS.items():
        path = model_path(p)
        weight = path / "model.safetensors"
        rec = {"repo_id": spec["repo"], "revision": spec["revision"],
               "expected_bytes": spec["expected_bytes"], "expected_sha256": spec["expected_sha256"],
               "files": {}}
        for name in ["config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json"]:
            f = path / name
            if f.is_file():
                rec["files"][name] = _file_record(f)
        if not weight.is_file():
            rec.update({"status": "pending", "actual_bytes": 0,
                        "remaining_bytes": spec["expected_bytes"]})
        else:
            got_bytes = weight.stat().st_size
            rec["actual_bytes"] = got_bytes
            rec["remaining_bytes"] = max(0, spec["expected_bytes"] - got_bytes)
            complete = got_bytes == spec["expected_bytes"]
            if complete:
                got_sha = sha256_file(weight) if verify_hashes else None
                rec["actual_sha256"] = got_sha
                rec["bytes_match"] = True
                rec["sha256_match"] = (got_sha == spec["expected_sha256"]) if got_sha else None
                if got_sha and got_sha != spec["expected_sha256"]:
                    raise RuntimeError(f"P{p} model SHA256 mismatch")
                rec["status"] = "verified"
                rec["files"]["model.safetensors"] = {"bytes": got_bytes, "sha256": got_sha}
            elif p == 8:
                # Partial evidence is preserved; never treat stat size as verification.
                rec["status"] = "pending"
                rec["aria2_control_present"] = (path / "model.safetensors.aria2").is_file()
            else:
                raise RuntimeError(f"P{p} model.safetensors incomplete: {got_bytes}/{spec['expected_bytes']}")
        cfg_file = path / "config.json"
        if cfg_file.is_file():
            cfg = json.loads(cfg_file.read_text())
            rec.update({"parscale_n": cfg.get("parscale_n"),
                        "parscale_n_tokens": cfg.get("parscale_n_tokens"),
                        "parscale_attn_smooth": cfg.get("parscale_attn_smooth"),
                        "max_position_embeddings": cfg.get("max_position_embeddings")})
        out["models"][f"P{p}"] = rec
        (out["available_models"] if rec["status"] == "verified" else out["pending_models"]).append(f"P{p}")
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


def make_task_manager(run_dir, mode):
    from lm_eval.tasks import TaskManager
    include = _runtime_task_dir(run_dir, mode)
    return TaskManager(include_path=str(include))


def make_tasks(run_dir, mode):
    """mode=audit uses frozen YAML; mode=smoke uses 0-shot train-only copies."""
    tm = make_task_manager(run_dir, mode)
    loaded = tm.load_task_or_group(list(TASKS))
    tasks = OrderedDict((name, loaded[name]) for name in TASKS if name in loaded)
    expected = {name: (0 if mode == "smoke" else spec["shots"]) for name, spec in TASKS.items()}
    for name, task in tasks.items():
        task.set_fewshot_seed(FEWSHOT_SEED)
        got = task.config.num_fewshot
        if got != expected[name]:
            raise RuntimeError(f"{name} num_fewshot mismatch: {got} != {expected[name]}")
    if len(tasks) != len(TASKS):
        raise RuntimeError(f"loaded {len(tasks)} tasks, expected {len(TASKS)}")
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
    if len(ctx) == 0:
        raise RuntimeError(f"{inst.doc_id}: empty context encoding")
    if len(cont) == 0:
        raise RuntimeError(f"{inst.doc_id}: empty continuation encoding")
    if len(cont) > CONTEXT_MAX:
        raise RuntimeError(f"{inst.doc_id}: continuation {len(cont)} > context max {CONTEXT_MAX}")
    total = len(ctx) + len(cont)
    input_tokens = min(total, CONTEXT_MAX + 1) - 1
    left_truncated = max(0, total - (CONTEXT_MAX + 1))
    scored = min(len(cont), input_tokens)
    if scored < len(cont):
        raise RuntimeError(f"{inst.doc_id}: continuation would be truncated")
    return {
        "doc_id": inst.doc_id, "choice": inst.idx,
        "context_sha256": sha256_bytes(context.encode()),
        "continuation_sha256": sha256_bytes(continuation.encode()),
        "context_tokens": len(ctx), "continuation_tokens": len(cont),
        "total_tokens": total, "input_tokens": input_tokens,
        "left_truncated_tokens": left_truncated,
        "scored_tokens": scored,
        "special_tokens_used": any(t in tok.all_special_ids for t in cont),
        "context": context, "continuation": continuation,
    }


def _row_hash(dataset):
    h = hashlib.sha256()
    for row in dataset:
        h.update(canonical_bytes(row))
        h.update(b"\n")
    return h.hexdigest()


def _dataset_hashes(ds):
    out = {}
    for split in sorted(ds.keys()):
        d = ds[split]
        out[split] = {"rows": len(d), "content_sha256": _row_hash(d),
                      "fingerprint": getattr(d, "_fingerprint", None)}
    return out


def _parquet_check(path):
    rec = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    with open(path, "rb") as fh:
        head = fh.read(4)
        fh.seek(-4, os.SEEK_END)
        tail = fh.read(4)
    rec["head"] = head.decode("latin-1")
    rec["tail"] = tail.decode("latin-1")
    rec["magic_ok"] = head == b"PAR1" and tail == b"PAR1"
    if not rec["magic_ok"]:
        raise RuntimeError(f"not a parquet file: {path.name}")
    return rec


def collect_task_requests(name, task, tok, write_path=None):
    """Build all requests once and return exact stream/gold evidence."""
    spec = TASKS[name]
    task.build_all_requests(limit=None)
    instances = task.instances
    docs = {inst.doc_id: inst.doc for inst in instances}
    h = hashlib.sha256()
    fh = gzip.open(write_path, "wt", encoding="utf-8") if write_path else None
    requests = 0
    input_tokens = 0
    truncated = 0
    max_total = 0
    special = 0
    per_doc = OrderedDict()
    try:
        for inst in instances:
            rec = request_record(inst, tok)
            line = canonical_bytes({k: rec[k] for k in
                                    ("doc_id", "choice", "context", "continuation")})
            h.update(line)
            h.update(b"\n")
            if fh:
                fh.write(line.decode() + "\n")
            requests += 1
            input_tokens += rec["input_tokens"]
            truncated += int(rec["left_truncated_tokens"] > 0)
            special += int(rec["special_tokens_used"])
            max_total = max(max_total, rec["total_tokens"])
            info = per_doc.setdefault(rec["doc_id"], {"conts": [], "cont_ids": set(),
                                                      "multi": False, "trunc": [],
                                                      "ctx": rec["context"]})
            info["conts"].append(rec["continuation"])
            info["cont_ids"].add(tuple(encode_pair(tok, rec["context"], rec["continuation"])[1]))
            info["multi"] = info["multi"] or rec["continuation_tokens"] > 1
            if rec["left_truncated_tokens"] > 0:
                info["trunc"].append((rec["choice"], rec["left_truncated_tokens"],
                                      rec["context_tokens"], rec["continuation_tokens"]))
    finally:
        if fh:
            fh.close()

    golds = {did: gold_index(task, doc) for did, doc in docs.items()}
    gold_vec = [[str(did), golds[did]] for did in sorted(golds, key=str)]
    distinct = multi = common = 0
    for did, info in per_doc.items():
        if len(info["cont_ids"]) == len(info["conts"]):
            distinct += 1
        if info["multi"]:
            multi += 1
        conts = info["conts"]
        for a in conts:
            if any(b != a and b.startswith(a) and len(a) < len(b) for b in conts):
                common += 1

    # Exact disclosure of few-shot token loss for truncated requests.
    fewshot_audit = {"docs_with_left_truncation": 0, "docs_losing_fewshot_tokens": 0,
                     "docs_losing_query_tokens": 0, "fewshot_tokens_lost": 0,
                     "query_tokens_lost": 0, "unverified_prefix": 0}
    if spec["shots"] > 0:
        prefix_fn = getattr(task, "doc_to_prefix", None)
        for did, info in per_doc.items():
            if not info["trunc"]:
                continue
            fewshot_audit["docs_with_left_truncation"] += 1
            doc = docs[did]
            try:
                gen_prefix = prefix_fn(doc) if prefix_fn else None
                zero_ctx = task.fewshot_context(doc, 0, gen_prefix=gen_prefix)
                zero_n = len(encode_pair(tok, zero_ctx, "x")[0])
                if not info["ctx"].endswith(zero_ctx):
                    fewshot_audit["unverified_prefix"] += 1
                    continue
                prefix_n = max(0, len(encode_pair(tok, info["ctx"], "x")[0]) - zero_n)
            except Exception:
                fewshot_audit["unverified_prefix"] += 1
                continue
            for _choice, left, _ctx_n, _cont_n in info["trunc"]:
                demo_lost = min(left, prefix_n)
                query_lost = max(0, left - prefix_n)
                fewshot_audit["fewshot_tokens_lost"] += demo_lost
                fewshot_audit["query_tokens_lost"] += query_lost
                fewshot_audit["docs_losing_fewshot_tokens"] += int(demo_lost > 0)
                fewshot_audit["docs_losing_query_tokens"] += int(query_lost > 0)

    return {
        "documents": len(per_doc), "requests": requests, "input_tokens": input_tokens,
        "requests_truncated": truncated, "max_total_tokens": max_total,
        "mean_input_tokens": input_tokens / max(1, requests),
        "gold_valid": sum(1 for v in golds.values() if v >= 0),
        "gold_invalid": sum(1 for v in golds.values() if v < 0),
        "gold_index_sha256": sha256_bytes(canonical_bytes(gold_vec)),
        "distinct_choice_docs": distinct, "multi_token_choice_docs": multi,
        "common_prefix_answer_pairs": common,
        "special_tokens_in_continuation": special,
        "num_fewshot": task.config.num_fewshot,
        "eval_split": spec["split"],
        "request_stream_sha256": h.hexdigest(),
        "fewshot_truncation_audit": fewshot_audit,
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
                        "truncation": "left context only; continuation must remain intact",
                        "datasets": {}, "tasks": {}}
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
        independent = _dataset_hashes(ds)
        task_rows = _dataset_hashes(task.dataset) if hasattr(task, "dataset") else {}
        comparisons = {}
        for split in sorted(set(independent) | set(task_rows)):
            a = independent.get(split)
            b = task_rows.get(split)
            comparisons[split] = {
                "independent_rows": a["rows"] if a else None,
                "task_rows": b["rows"] if b else None,
                "rows_match": bool(a and b and a["rows"] == b["rows"]),
                "content_sha256_match": bool(a and b and a["content_sha256"] == b["content_sha256"]),
                "independent_content_sha256": a["content_sha256"] if a else None,
                "task_content_sha256": b["content_sha256"] if b else None,
            }
        local = {}
        for split, rel in LOCAL_PARQUET.get(name, {}).items():
            local[split] = _parquet_check(Path(os.environ["PARSCALE_RUNTIME_DIR"]) /
                                          "downstream-data" / rel)
        dataset_manifest["datasets"][name] = {
            "repo_id": spec["repo"], "config": spec["config"], "revision": spec["revision"],
            "splits": independent, "task_dataset_splits": task_rows,
            "independent_vs_task": comparisons,
            "cross_check_passed": all(v["rows_match"] and v["content_sha256_match"]
                                      for v in comparisons.values()),
            "local_parquet": local,
        }
        stat = collect_task_requests(name, task, tok, raw / "requests" / f"{name}.jsonl.gz")
        stat["build_seconds"] = time.time() - t0
        dataset_manifest["tasks"][name] = stat
        print(f"[prepare] {name}: {stat['documents']} docs, {stat['requests']} requests, "
              f"{stat['input_tokens']} input tokens, {stat['requests_truncated']} truncated", flush=True)
    dataset_manifest["totals"] = {
        "documents": sum(v["documents"] for v in dataset_manifest["tasks"].values()),
        "requests": sum(v["requests"] for v in dataset_manifest["tasks"].values()),
        "input_tokens": sum(v["input_tokens"] for v in dataset_manifest["tasks"].values()),
        "requests_truncated": sum(v["requests_truncated"] for v in dataset_manifest["tasks"].values()),
        "fewshot_tokens_lost": sum(v["fewshot_truncation_audit"]["fewshot_tokens_lost"]
                                   for v in dataset_manifest["tasks"].values()),
        "prepare_seconds": time.time() - started,
    }
    dataset_manifest["request_set_sha256"] = sha256_bytes(canonical_bytes({
        k: {kk: v[kk] for kk in ("documents", "requests", "input_tokens",
                                 "request_stream_sha256", "gold_index_sha256",
                                 "num_fewshot", "eval_split")}
        for k, v in dataset_manifest["tasks"].items()}))
    dataset_manifest["data_cross_checks_passed"] = all(
        v["cross_check_passed"] for v in dataset_manifest["datasets"].values())
    atomic_json(run_dir / "data-manifest.json", dataset_manifest)
    manifest = model_manifest(run_dir)
    atomic_json(run_dir / "environment.json", environment(run_dir))
    print(f"[prepare] done in {time.time()-started:.1f}s; "
          f"models={manifest['available_models']} pending={manifest['pending_models']}", flush=True)
    return dataset_manifest


def verify_frozen_requests(run_dir, data):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path(1)), local_files_only=True,
                                        trust_remote_code=False)
    tasks = make_tasks(run_dir, "audit")
    checks = {}
    for name in TASKS:
        got = collect_task_requests(name, tasks[name], tok, None)
        frozen = data["tasks"][name]
        keys = ("documents", "requests", "input_tokens", "request_stream_sha256",
                "gold_index_sha256", "max_total_tokens", "requests_truncated",
                "num_fewshot", "eval_split")
        diff = {k: {"frozen": frozen.get(k), "rebuilt": got.get(k)}
                for k in keys if frozen.get(k) != got.get(k)}
        checks[name] = {"passed": not diff, "mismatches": diff}
    return {"schema_version": 1, "created_at": utcnow(), "checks": checks,
            "passed": all(v["passed"] for v in checks.values())}


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def gpu_admission():
    rec = nvidia_smi_gpu()
    rec["free_mib_min"] = FREE_MIB_MIN
    rec["passed"] = bool(rec.get("uuid_matches") and rec.get("free_mib", 0) >= FREE_MIB_MIN)
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


def release(model, lm=None):
    if lm is not None:
        try:
            lm.model = None
        except Exception:
            pass
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def manual_loglikelihood(model, tok, context, continuation):
    """Independent full teacher-forced scoring including the final cont token."""
    ctx, cont = encode_pair(tok, context, continuation)
    if len(cont) == 0:
        raise RuntimeError("empty continuation")
    if len(cont) > CONTEXT_MAX:
        raise RuntimeError("continuation exceeds context max")
    full = (ctx + cont)[-(CONTEXT_MAX + 1):]
    inp = full[:-1]
    if not inp:
        raise RuntimeError("empty model input")
    ids = torch.tensor([inp], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits = model(input_ids=ids).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = torch.tensor(full[-len(cont):], dtype=torch.long, device="cuda")
    per = logp[0, -len(cont):, :].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(per).all():
        raise RuntimeError("non-finite manual logprob")
    return float(per.sum()), {"context": len(ctx), "continuation": len(cont),
                              "input_tokens": len(inp)}


def make_float_hflm():
    from lm_eval.models.huggingface import HFLM

    class _FloatHFLM(HFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            # Fresh prefix is rebuilt inside the model on every forward; this
            # subclass only changes the dtype of the returned logits.
            return super()._model_call(inps, attn_mask, labels).float()

    return _FloatHFLM


def make_checkpoint_hflm(sink_path):
    from lm_eval.models.huggingface import HFLM

    class _CheckpointHFLM(HFLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sink_path = Path(sink_path)
            self.current_task = None
            self.gpu_seconds_measured = 0.0

        def _model_call(self, inps, attn_mask=None, labels=None):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = super()._model_call(inps, attn_mask, labels).float()
            end.record()
            end.synchronize()
            self.gpu_seconds_measured += start.elapsed_time(end) / 1000.0
            return out

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
            res = super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm,
                                                override_bs=override_bs)
            self.sink_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sink_path, "a", encoding="utf-8") as fh:
                for req, out in zip(requests, res):
                    context, continuation = req[0]
                    fh.write(json.dumps({
                        "task": self.current_task,
                        "context_sha256": sha256_bytes(context.encode()),
                        "continuation_sha256": sha256_bytes(continuation.encode()),
                        "logprob": float(out[0]), "is_greedy": bool(out[1]),
                    }, sort_keys=True) + "\n")
                fh.flush()
            return res

    return _CheckpointHFLM


def _token_ids_for_length(tok, n):
    base = "The quick brown fox jumps over the lazy dog near the river bank. " * 420
    ids = tok.encode(base, add_special_tokens=False)
    if len(ids) < n:
        raise RuntimeError(f"base text tokenizes to {len(ids)}, need {n}")
    return ids[:n]


def benchmark(model, tok, repeats=2):
    """Exact-token-length timing including the candidate softmax scoring cost."""
    rows = []
    for n in (128, 512, 2048):
        ids = _token_ids_for_length(tok, n)
        inp = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            _ = model(input_ids=inp)
        torch.cuda.synchronize()
        times, forward_times, score_times = [], [], []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                logits = model(input_ids=inp).logits
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.inference_mode():
                logp = F.log_softmax(logits.float(), dim=-1)
                _ = logp[0, -min(16, n):, :].gather(
                    -1, torch.zeros((min(16, n), 1), dtype=torch.long, device="cuda"))
            torch.cuda.synchronize()
            t2 = time.time()
            del logits, logp
            times.append(t2 - t0)
            forward_times.append(t1 - t0)
            score_times.append(t2 - t1)
        rows.append({"label": f"exact_{n}", "tokens": n,
                     "seconds": times, "median_seconds": float(np.median(times)),
                     "forward_seconds": forward_times, "score_softmax_seconds": score_times})
        del inp
    return rows


def smoke(run_dir):
    from transformers import AutoTokenizer
    started = time.time()
    run_dir = Path(run_dir)
    enforce_determinism()
    data = json.loads((run_dir / "data-manifest.json").read_text())
    model_manifest_data = json.loads((run_dir / "model-manifest.json").read_text())
    available = [int(k[1:]) for k in model_manifest_data["available_models"]]
    required = {1, 2, 4}
    if not required.issubset(available):
        raise RuntimeError(f"required models unavailable: {sorted(required - set(available))}")
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
                  "checks": {}, "task_checks": {}, "models": {}, "manual_cases": []}
    task_instances = {}
    for name, task in tasks.items():
        task.build_all_requests(limit=4)
        task_instances[name] = list(task.instances)
        docs = {i.doc_id: i.doc for i in task.instances}
        recs = [request_record(i, tok) for i in task.instances]
        validation["task_checks"][name] = {
            "requests": len(recs), "documents": len(docs),
            "num_fewshot": task.config.num_fewshot,
            "eval_split": "train (smoke only)",
            "gold_valid": all(gold_index(task, d) >= 0 for d in docs.values()),
            "all_continuations_nonempty": all(r["continuation_tokens"] > 0 for r in recs),
            "no_continuation_truncation": all(r["scored_tokens"] == r["continuation_tokens"]
                                              for r in recs),
            "no_special_tokens_in_continuation": not any(r["special_tokens_used"] for r in recs),
            "continuation_within_context": all(r["continuation_tokens"] <= CONTEXT_MAX for r in recs),
            "max_total_tokens": max(r["total_tokens"] for r in recs),
        }

    cases = []
    for name in ("piqa", "arc_challenge"):
        insts = task_instances[name]
        if insts:
            inst = next((i for i in insts if i.idx == 0), insts[0])
            cases.append({"name": f"real_{name}", "context": inst.args[0],
                          "continuation": inst.args[1], "source": "held-out-style smoke train doc"})
    cases.append({"name": "synthetic_multi_token", "source": "synthetic",
                  "context": "Question: Which city is the capital of France?\nAnswer:",
                  "continuation": "Paris, France"})
    cases.append({"name": "synthetic_common_prefix", "source": "synthetic",
                  "context": "Question: Which city is the capital of France?\nAnswer:",
                  "continuation": "Paris"})

    FloatHFLM = make_float_hflm()
    gpu_start = time.time()
    manual_by_case = {}
    models = {}
    for p in available:
        adm = gpu_admission()
        if not adm["passed"]:
            raise RuntimeError(f"GPU admission failed before P{p}: {adm}")
        torch.cuda.reset_peak_memory_stats()
        model, load_s = load_model(p)
        lm = FloatHFLM(pretrained=model, tokenizer=tok, batch_size=1, max_length=CONTEXT_MAX,
                       truncation=False, device="cuda", dtype=torch.bfloat16,
                       add_bos_token=False)
        case_rows = []
        for case in cases:
            adapter = lm.loglikelihood([(case["context"], case["continuation"])])
            manual, lens = manual_loglikelihood(model, tok, case["context"], case["continuation"])
            if p == available[0]:
                manual_by_case[case["name"]] = manual
            row = {"model": f"P{p}", "case": case["name"], "source": case["source"],
                   "adapter": float(adapter[0][0]), "manual": float(manual),
                   "abs_diff": abs(float(adapter[0][0]) - float(manual)),
                   "is_greedy": bool(adapter[0][1]), "lens": lens,
                   "context_sha256": sha256_bytes(case["context"].encode()),
                   "continuation_sha256": sha256_bytes(case["continuation"].encode())}
            case_rows.append(row)
            validation["manual_cases"].append(row)
        b = benchmark(model, tok, repeats=2)
        peak = {"allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        models[f"P{p}"] = {"load_seconds": load_s, "peak_memory_gib": peak,
                           "cases": case_rows, "benchmark": b,
                           "max_abs_diff": max(r["abs_diff"] for r in case_rows)}
        del lm
        release(model)
        print(f"[smoke] P{p} load={load_s:.1f}s peak={peak['allocated_gib']:.2f}GiB "
              f"max_manual_diff={models[f'P{p}']['max_abs_diff']:.2e}", flush=True)

    # FP32 sensitivity on the same small fixed cases with P1.
    torch.cuda.reset_peak_memory_stats()
    model32, load32 = load_model(1, dtype=torch.float32)
    lm32 = FloatHFLM(pretrained=model32, tokenizer=tok, batch_size=1, max_length=CONTEXT_MAX,
                     truncation=False, device="cuda", dtype=torch.float32, add_bos_token=False)
    fp32_rows = []
    for case in cases[:2]:
        adapter32 = lm32.loglikelihood([(case["context"], case["continuation"])])
        manual32, lens32 = manual_loglikelihood(model32, tok, case["context"], case["continuation"])
        fp32_rows.append({"case": case["name"], "manual_fp32": float(manual32),
                          "adapter_fp32": float(adapter32[0][0]),
                          "adapter_manual_abs_diff": abs(float(adapter32[0][0]) - float(manual32)),
                          "bf16_vs_fp32_abs_diff": abs(manual_by_case.get(case["name"], manual32) - float(manual32)),
                          "lens": lens32})
    fp32_peak = {"allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                 "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
    del lm32
    release(model32)
    gpu_seconds = time.time() - gpu_start
    validation["models"] = models
    validation["fp32_sensitivity"] = {"model": "P1", "load_seconds": load32,
                                      "peak_memory_gib": fp32_peak, "cases": fp32_rows}

    # Candidate alignment: real smoke requests plus synthetic multi/common cases.
    align = {}
    for name, insts in task_instances.items():
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
    synthetic = next(r for r in validation["manual_cases"]
                     if r["case"] == "synthetic_common_prefix")
    multi_case = next(r for r in validation["manual_cases"]
                      if r["case"] == "synthetic_multi_token")
    alignment_ok = (synthetic["abs_diff"] <= MANUAL_TOL
                    and multi_case["abs_diff"] <= MANUAL_TOL
                    and multi_case["lens"]["continuation"] > 1
                    and synthetic["lens"]["continuation"] >= 1)
    validation["candidate_alignment"] = {"tasks": align, "synthetic_ok": alignment_ok}

    finite = all(math.isfinite(r["adapter"]) and math.isfinite(r["manual"])
                 for r in validation["manual_cases"])
    finite = finite and all(math.isfinite(r["adapter_fp32"]) and math.isfinite(r["manual_fp32"])
                            for r in fp32_rows)
    max_manual = max(r["abs_diff"] for r in validation["manual_cases"])
    max_fp32_adapter = max(r["adapter_manual_abs_diff"] for r in fp32_rows)
    max_fp32_delta = max(r["bf16_vs_fp32_abs_diff"] for r in fp32_rows)
    gate = {
        "gpu_admission": all(gpu_admission()["passed"] for _ in range(1)),
        "models_verified": required.issubset(set(available)),
        "manual_adapter_agreement_1e-5": max_manual <= MANUAL_TOL,
        "fp32_adapter_agreement_1e-5": max_fp32_adapter <= MANUAL_TOL,
        "fp32_sensitivity_0.1": max_fp32_delta <= FP32_TOL,
        "candidate_alignment": alignment_ok,
        "task_contract": all(all(v for k, v in t.items() if isinstance(v, bool))
                             for t in validation["task_checks"].values()),
        "gpu_seconds_within_180": gpu_seconds <= GPU_SECONDS_LIMIT,
        "finite_values": finite,
    }
    validation["checks"] = {
        "manual_adapter_agreement": {"tolerance": MANUAL_TOL, "max_abs_diff": max_manual,
                                     "passed": max_manual <= MANUAL_TOL},
        "fp32_adapter_agreement": {"tolerance": MANUAL_TOL, "max_abs_diff": max_fp32_adapter,
                                   "passed": max_fp32_adapter <= MANUAL_TOL},
        "fp32_sensitivity": {"tolerance": FP32_TOL, "max_bf16_vs_fp32_abs_diff": max_fp32_delta,
                             "passed": max_fp32_delta <= FP32_TOL},
        "gpu_seconds_used": gpu_seconds,
        "gpu_seconds_limit": GPU_SECONDS_LIMIT,
        "pending_models": model_manifest_data["pending_models"],
    }
    validation["gate"] = gate
    validation["status"] = "SUCCEEDED" if all(gate.values()) else "FAILED"
    atomic_json(run_dir / "scorer-validation.json", validation)

    # Preflight projection from exact-length benchmark medians.
    preflight = {"schema_version": 1, "run_id": run_dir.name, "created_at": utcnow(),
                 "context_max": CONTEXT_MAX, "batch_size": 1,
                 "gpu": gpu, "memory_cap": cap,
                 "smoke_gpu_seconds": gpu_seconds,
                 "pending_models": model_manifest_data["pending_models"],
                 "data": {"requests": data["totals"]["requests"],
                          "input_tokens": data["totals"]["input_tokens"],
                          "requests_truncated": data["totals"]["requests_truncated"],
                          "by_task": {k: {"documents": v["documents"], "requests": v["requests"],
                                          "input_tokens": v["input_tokens"],
                                          "requests_truncated": v["requests_truncated"],
                                          "max_total_tokens": v["max_total_tokens"]}
                                      for k, v in data["tasks"].items()}},
                 "models": {}, "projection": {}}
    remaining = PROJECT_GPU_CAP_SECONDS - PROJECT_GPU_USED_SECONDS - gpu_seconds
    preflight["project_budget"] = {"project_cap_seconds": PROJECT_GPU_CAP_SECONDS,
                                   "prior_used_seconds": PROJECT_GPU_USED_SECONDS,
                                   "remaining_after_smoke_seconds": remaining,
                                   "source": "task controls; conservative prior usage"}
    for p, rec in models.items():
        b = rec["benchmark"]
        xs = np.array([r["tokens"] for r in b], dtype=float)
        ys = np.array([r["median_seconds"] for r in b], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        r2 = 1 - ((ys - pred) ** 2).sum() / max(1e-12, ((ys - ys.mean()) ** 2).sum())
        total_s = sum(slope * v["input_tokens"] + intercept * v["requests"]
                      for v in data["tasks"].values()) + rec["load_seconds"]
        preflight["models"][p] = {"load_seconds": rec["load_seconds"],
                                  "peak_memory_gib": rec["peak_memory_gib"],
                                  "latency_rows": b,
                                  "fit_seconds_per_token": float(slope),
                                  "fit_intercept_seconds": float(intercept),
                                  "fit_r2": float(r2),
                                  "estimated_seconds": float(total_s),
                                  "estimated_gpu_hours": float(total_s / 3600)}
    all_s = sum(v["estimated_seconds"] for v in preflight["models"].values())
    core_names = ["piqa", "sciq", "winogrande", "openbookqa"]
    core_s = sum(
        sum(preflight["models"][p]["fit_seconds_per_token"] * data["tasks"][n]["input_tokens"]
            + preflight["models"][p]["fit_intercept_seconds"] * data["tasks"][n]["requests"]
            for n in core_names) + preflight["models"][p]["load_seconds"]
        for p in preflight["models"])
    preflight["projection"] = {
        "all_models_seconds": float(all_s),
        "all_models_gpu_hours": float(all_s / 3600),
        "core_four_tasks_seconds": float(core_s),
        "core_four_tasks_gpu_hours": float(core_s / 3600),
        "fits_remaining_project_budget": bool(all_s <= remaining),
        "scope": "batch=1, all held-out requests for the models verified above; "
                 "excludes later analysis and any retries",
        "assumption": "Linear fit to exact 128/512/2048-token latency including "
                      "float32 log-softmax scoring cost; shared GPU, descriptive only.",
    }
    preflight["status"] = validation["status"]
    atomic_json(run_dir / "preflight.json", preflight)
    print(f"[smoke] {validation['status']} gpu={gpu_seconds:.1f}s "
          f"projected_all={all_s:.1f}s", flush=True)
    if validation["status"] != "SUCCEEDED":
        raise RuntimeError(f"smoke gates failed: {json.dumps(gate, sort_keys=True)}")
    return validation, preflight


# --------------------------------------------------------------------------- #
# full stage (guarded, not executed during preparation)
# --------------------------------------------------------------------------- #
def _sanitize_sample(s):
    keep = ("doc_id", "target", "filtered_resps", "acc", "acc_norm", "exact_match")
    return {k: s.get(k) for k in keep if k in s}


def evaluate(run_dir):
    from transformers import AutoTokenizer
    from lm_eval.evaluator import evaluate as lm_evaluate
    run_dir = Path(run_dir)
    enforce_determinism()
    task = json.loads((run_dir / "task.json").read_text())
    if not task.get("full_test_authorized_by_supervisor"):
        raise RuntimeError("Full evaluation requires an explicit supervisor-approved task")
    data = json.loads((run_dir / "data-manifest.json").read_text())
    manifest = json.loads((run_dir / "model-manifest.json").read_text())
    if manifest.get("pending_models"):
        raise RuntimeError(f"full stage blocked by pending models: {manifest['pending_models']}")
    budget = task.get("budget", {})
    gpu_limit = float(budget.get("gpu_hours", 0)) * 3600.0
    wall_limit = float(budget.get("wall_seconds", 0))
    started = time.time()
    results = {"schema_version": 1, "run_id": run_dir.name, "stage": "full",
               "status": "RUNNING", "started_at": utcnow(), "run_git": git_state(),
               "request_set_sha256": data.get("request_set_sha256"),
               "request_verification": None, "models": {}}
    atomic_json(run_dir / "full-results.json", results)
    verification = verify_frozen_requests(run_dir, data)
    results["request_verification"] = verification
    atomic_json(run_dir / "full-results.json", results)
    if not verification["passed"]:
        results["status"] = "FAILED"
        results["error"] = "frozen request stream mismatch"
        atomic_json(run_dir / "full-results.json", results)
        raise RuntimeError("frozen request stream mismatch; refusing to score")

    tok = AutoTokenizer.from_pretrained(str(model_path(1)), local_files_only=True,
                                        trust_remote_code=False)
    CheckpointHFLM = make_checkpoint_hflm(run_dir / "raw" / "full-per-item.jsonl")
    try:
        for p in [1, 2, 4, 8]:
            adm = gpu_admission()
            if not adm["passed"]:
                raise RuntimeError(f"GPU admission failed before P{p}: {adm}")
            cap = cap_gpu_memory()
            torch.cuda.reset_peak_memory_stats()
            model, load_s = load_model(p)
            lm = CheckpointHFLM(pretrained=model, tokenizer=tok, batch_size=1,
                                max_length=CONTEXT_MAX, truncation=False, device="cuda",
                                dtype=torch.bfloat16, add_bos_token=False)
            results["models"][f"P{p}"] = {"load_seconds": load_s, "memory_cap": cap,
                                          "tasks": {}}
            eval_tasks = make_tasks(run_dir, "audit")
            for name in TASKS:
                if gpu_limit and lm.gpu_seconds_measured > gpu_limit:
                    raise RuntimeError(f"GPU budget exhausted before P{p}/{name}")
                if wall_limit and time.time() - started > wall_limit:
                    raise RuntimeError(f"wall budget exhausted before P{p}/{name}")
                eval_tasks[name].set_fewshot_seed(FEWSHOT_SEED)
                lm.current_task = name
                out = lm_evaluate(lm=lm, task_dict={name: eval_tasks[name]}, limit=None,
                                  bootstrap_iters=0, log_samples=True)
                raw_path = run_dir / "raw" / "full-samples" / f"P{p}-{name}.json"
                atomic_json(raw_path, out.get("samples", {}))
                sanitized = [_sanitize_sample(s) for s in out.get("samples", {}).get(name, [])]
                results["models"][f"P{p}"]["tasks"][name] = {
                    "results": out["results"][name], "samples": sanitized,
                    "gpu_seconds": lm.gpu_seconds_measured,
                }
                results["models"][f"P{p}"]["peak_memory_gib"] = {
                    "allocated": torch.cuda.max_memory_allocated() / 1024**3,
                    "reserved": torch.cuda.max_memory_reserved() / 1024**3}
                atomic_json(run_dir / "full-results.json", results)
            del lm
            release(model)
        results["status"] = "SUCCEEDED"
        results["finished_at"] = utcnow()
        results["resource_usage"] = {"wall_seconds": time.time() - started,
                                     "gpu_seconds": sum(
                                         v["tasks"][n]["gpu_seconds"]
                                         for v in results["models"].values()
                                         for n in v["tasks"])}
        atomic_json(run_dir / "full-results.json", results)
        return results
    except Exception as exc:
        results["status"] = "FAILED"
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["finished_at"] = utcnow()
        atomic_json(run_dir / "full-results.json", results)
        raise


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
    enforce_determinism()
    if args.mode == "prepare":
        prepare(run_dir)
    elif args.mode == "smoke":
        smoke(run_dir)
    else:
        evaluate(run_dir)


if __name__ == "__main__":
    main()
