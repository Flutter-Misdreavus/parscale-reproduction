#!/usr/bin/env python3
"""CPU-only consistency check of v4 diagnostic raw arrays.

Recomputes the stored error statistics from the saved FP32/BF16 arrays and
validates the validation token/target hashes.  It never loads a model on GPU
and never writes outside the run directory.  This is an analysis check added
after the single GPU run; it does not change any scientific threshold.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import evaluate  # noqa: E402


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(diff):
    d = np.abs(np.asarray(diff, dtype=np.float32)).reshape(-1)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "p99_abs": float(np.percentile(d, 99)), "elements": int(d.size)}


def target_logprobs(logits, tokens):
    x = np.asarray(logits[:-1], dtype=np.float32)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    lp = x - m - np.log(e.sum(axis=-1, keepdims=True))
    tgt = np.asarray(tokens[1:], dtype=np.int64)
    return lp[np.arange(lp.shape[0]), tgt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    os.environ["EVAL_RUN_DIR"] = str(run_dir)
    diag = json.loads((run_dir / "diagnostics.json").read_text())
    locks = json.loads((ROOT / "locks.json").read_text())
    data, manifest = evaluate.prepare_data(locks, run_dir, "smoke")
    dm = json.loads((run_dir / "data-manifest.json").read_text())

    failures = []
    for b, meta in zip(data, dm["blocks_used_meta"]):
        if sha_arr(np.asarray(b, dtype="<i4")) != meta["token_ids_sha256"]:
            failures.append(f"block {meta['block_id']} token hash mismatch")
    if manifest["text_sha256"] != dm["text_sha256"]:
        failures.append("validation text hash mismatch")
    if manifest["token_ids_sha256"] != dm["token_ids_sha256"]:
        failures.append("full token ids hash mismatch")

    stored = {e["path"]: e for e in diag["raw_manifest"]}
    files = {"checked": 0, "mismatches": []}
    arrays = {"checked": 0, "mismatches": []}
    recomputed, maxdev = {}, {}
    for key, rec in diag["models"].items():
        recomputed[key] = {"sequences": {}, "loss": {}}
        for sid in ("0", "1"):
            block = np.asarray(data[int(sid)][:64], dtype=np.int64)
            meta = rec["cache_sequences"]["float32"][sid]["sequence"]
            if sha_arr(block.astype("<i4")) != meta["token_ids_sha256_i32le"]:
                failures.append(f"{key} seq{sid} token hash mismatch")
            z32 = None
            for dt in ("float32", "bfloat16"):
                path = run_dir / f"raw/diagnostics/{key}/P_{dt}_seq{sid}.npz"
                rel = str(path.relative_to(run_dir))
                ent = stored.get(rel)
                if not ent:
                    failures.append(f"{rel} absent from diagnostics manifest")
                    continue
                files["checked"] += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != ent["sha256"]:
                    files["mismatches"].append({"path": rel})
                z = np.load(path)
                for name in ("tokens", "full", "cached_p1", "cached_p32"):
                    if name not in z.files:
                        failures.append(f"{rel} missing {name}")
                        continue
                    arrays["checked"] += 1
                    if name in ent["arrays"] and sha_arr(z[name]) != ent["arrays"][name]["sha256"]:
                        arrays["mismatches"].append({"path": rel, "array": name})
                if dt == "float32":
                    z32 = z
                    out = {}
                    for name, other in (("full_vs_cached_p1", z["cached_p1"] - z["full"]),
                                        ("full_vs_cached_p32", z["cached_p32"] - z["full"])):
                        d = stats(other)
                        ref = rec["cache_sequences"][dt][sid][name]
                        for k in ("max_abs", "mean_abs", "p99_abs"):
                            if abs(d[k] - ref[k]) > 1e-6:
                                failures.append(f"{key} seq{sid} FP32 {name} {k}")
                        out[name] = d
                else:
                    if z32 is None:
                        failures.append(f"{key} seq{sid} missing FP32 reference")
                        continue
                    ref32 = z32["full"]
                    out = {}
                    for pname, arr in (("full", z["full"]),
                                       ("cached_p1", z["cached_p1"]),
                                       ("cached_p32", z["cached_p32"])):
                        ref = rec["cache_sequences"][dt][sid][pname + "_vs_fp32"]
                        d = stats(arr - ref32)
                        out[pname + "_vs_fp32"] = d
                        for k in ("max_abs", "mean_abs", "p99_abs"):
                            if abs(d[k] - ref[k]) > 1e-4:
                                failures.append(f"{key} seq{sid} {pname} {k}")
                        dlp = np.abs(target_logprobs(arr, block) -
                                     target_logprobs(ref32, block))
                        vals = {"max": float(dlp.max()), "mean": float(dlp.mean()),
                                "p99": float(np.percentile(dlp, 99))}
                        for k, v in vals.items():
                            if abs(v - ref["target_logprob_abs_diff_" + k]) > 2e-5:
                                failures.append(f"{key} seq{sid} {pname} target_logprob {k}")
                        per = np.asarray(ref["target_logprob_per_position_abs_diff"],
                                         dtype=np.float32)
                        if per.size != dlp.size:
                            failures.append(f"{key} seq{sid} {pname} per-position length")
                        else:
                            dev = float(np.max(np.abs(dlp - per)))
                            maxdev[f"{key}/seq{sid}/{pname}/per_position"] = dev
                            if dev > 5e-5:
                                failures.append(f"{key} seq{sid} {pname} per-position {dev}")
                    for name, arr in (("p1", z["cached_p1"]), ("p32", z["cached_p32"])):
                        d = stats(arr - z["full"])
                        ref = rec["cache_sequences"][dt][sid]["full_vs_cached_" + name]
                        for k in ("max_abs", "mean_abs", "p99_abs"):
                            if abs(d[k] - ref[k]) > 1e-4:
                                failures.append(f"{key} seq{sid} BF16 cached_{name} {k}")
                        out["full_vs_cached_" + name] = d
                recomputed[key]["sequences"][f"{sid}/{dt}"] = out

        for bidx in (0, 1):
            block = np.asarray(data[bidx], dtype=np.int64)
            tgt = block[1:]
            ref = rec["loss_1024"]["fp32"][bidx]
            if sha_arr(tgt.astype("<i4")) != ref["target_token_ids_sha256"]:
                failures.append(f"{key} block{bidx} target hash mismatch")
            path = run_dir / f"raw/diagnostics/{key}/P_float32_loss_b{bidx}.npy"
            rel = str(path.relative_to(run_dir))
            ent = stored.get(rel)
            files["checked"] += 1
            if not ent or hashlib.sha256(path.read_bytes()).hexdigest() != ent["sha256"]:
                files["mismatches"].append({"path": rel})
            logits = np.load(path, mmap_mode="r")
            lp = target_logprobs(logits, block)
            nll = float(-lp.sum())
            dpos = float(np.max(np.abs(lp - np.asarray(ref["target_logprobs"],
                                                       dtype=np.float32))))
            maxdev[f"{key}/block{bidx}/fp32_per_position"] = dpos
            if abs(nll - ref["nll_sum_nats"]) > 0.01:
                failures.append(f"{key} block{bidx} FP32 NLL")
            if dpos > 1e-5:
                failures.append(f"{key} block{bidx} FP32 per-position {dpos}")
            bf = rec["loss_1024"]["bfloat16"][bidx]
            bf_nll = -float(np.sum(np.asarray(bf["target_logprobs"], dtype=np.float64)))
            if abs(bf_nll - bf["nll_sum_nats"]) > 0.01:
                failures.append(f"{key} block{bidx} BF16 stored-logprob NLL")
            recomputed[key]["loss"][f"block{bidx}"] = {
                "fp32_nll_recomputed": nll,
                "fp32_nll_stored": ref["nll_sum_nats"],
                "fp32_target_logprob_max_abs_diff": dpos,
                "bf16_nll_from_stored_logprobs": bf_nll,
                "bf16_nll_stored": bf["nll_sum_nats"],
                "bf16_minus_fp32_recomputed": bf_nll - nll,
            }

    out = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": evaluate.utcnow(),
        "status": "PASSED" if not failures and not files["mismatches"] and not arrays["mismatches"] else "FAILED",
        "purpose": "CPU-only recomputation from saved raw arrays and validation tokens; no GPU model rerun.",
        "data_checks": {
            "text_sha256_match": manifest["text_sha256"] == dm["text_sha256"],
            "token_ids_sha256_match": manifest["token_ids_sha256"] == dm["token_ids_sha256"],
            "blocks_used": 2,
            "block_hashes_match": not any("block" in f and "hash" in f for f in failures),
        },
        "file_hashes": files,
        "array_hashes": arrays,
        "metric_recompute": recomputed,
        "max_abs_deviations": maxdev,
        "tolerances": {"error_stats_abs": 1e-4, "target_logprob_abs": 2e-5,
                       "fp32_cache_abs": 1e-6, "nll_nats": 0.01,
                       "per_position_logprob_abs": 5e-5},
        "failures": failures,
    }
    path = run_dir / "raw-consistency.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print(f"status={out['status']} failures={len(failures)} "
          f"files={files['checked']} arrays={arrays['checked']} "
          f"max_per_position_dev={max(maxdev.values()) if maxdev else 0:.3e}")


if __name__ == "__main__":
    main()
