#!/usr/bin/env python3
"""Verify the pinned checkpoint files against their HF LFS SHA256 values.

Reads configs/model-sources.json and the run's raw/downloads/model_paths.json,
hashes each model.safetensors and the tokenizer/config files, and writes a
sanitized model-manifest.json into the run directory.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()

    sources = json.loads((ROOT / "configs" / "model-sources.json").read_text())
    paths = json.loads((run_dir / "raw" / "downloads" / "model_paths.json").read_text())

    models = {}
    all_ok = True
    for key, spec in sorted(sources["models"].items(), key=lambda kv: int(kv[0])):
        d = Path(paths[key])
        f = d / spec["file"]
        t0 = time.time()
        actual_bytes = f.stat().st_size
        actual_sha = sha256_file(f) if actual_bytes == spec["bytes"] else None
        size_ok = actual_bytes == spec["bytes"]
        sha_ok = actual_sha == spec["sha256"]
        small = {}
        for p in sorted(d.iterdir()):
            if p.is_file() and p.name != spec["file"] and not p.name.endswith(".aria2"):
                small[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        ok = bool(size_ok and sha_ok)
        all_ok = all_ok and ok
        models[f"P{key}"] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "url": spec["url"],
            "local_dir": str(d),
            "expected_bytes": spec["bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual_sha,
            "size_ok": size_ok,
            "sha256_ok": sha_ok,
            "verified": ok,
            "other_files": small,
            "hash_seconds": time.time() - t0,
        }
        print(f"[verify] P{key} size_ok={size_ok} sha_ok={sha_ok}", flush=True)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "all_verified": all_ok,
        "source": sources["source"],
        "models": models,
    }
    atomic_json(run_dir / "model-manifest.json", manifest)
    print(f"[verify] all_verified={all_ok}", flush=True)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
