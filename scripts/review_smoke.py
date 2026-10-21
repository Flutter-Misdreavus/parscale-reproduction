#!/usr/bin/env python3
"""Independent, standard-library-only audit of a received smoke event.

Reads artifacts, never imports worker/model code, and never runs inference.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return literal(node.left) * literal(node.right)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(x) for x in node.elts]
    return ast.literal_eval(node)


def close(a, b):
    assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--received", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    event = read(args.received / "completion.json")
    index = read(args.received / "local-artifacts.json")
    task = read(args.task)
    assert event["event_id"] == index["event_id"]
    assert event["event_id"] == digest({k: v for k, v in event.items() if k != "event_id"})
    assert event["task"] == task and event["task_sha256"] == digest(task)
    assert event["run_id"] == task["run_id"]
    entries = {e["path"]: e for e in event["artifacts"]}
    assert set(entries) == {e["path"] for e in index["artifacts"]}
    verified = []
    for item in index["artifacts"]:
        path = Path(item["local_path"]).resolve()
        assert path == (args.received / "artifacts" / item["path"]).resolve()
        assert path.is_relative_to((args.received / "artifacts").resolve())
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        assert actual == item["sha256"] == entries[item["path"]]["sha256"]
        assert len(data) == item["bytes"] == entries[item["path"]]["bytes"]
        verified.append(item["path"])
    artifacts = args.received / "artifacts"
    report = read(artifacts / "report.json")
    assert report == event["report"]
    smoke = read(artifacts / "smoke.json")
    manifest = read(artifacts / "data-manifest.json")
    assert manifest["split"] == "validation" and manifest["blocks_used"] == 2
    recomputed = {}
    for name, model in smoke["models"].items():
        assert [b["block_id"] for b in model["blocks"]] == [0, 1]
        for b, meta in zip(model["blocks"], manifest["blocks_used_meta"]):
            assert b["token_count_scored"] == meta["token_count"] - 1 == 1023
            close(b["nll_sum_nats"] / b["token_count_scored"], b["nll_nats_per_token"])
        total = sum(b["nll_sum_nats"] for b in model["blocks"])
        count = sum(b["token_count_scored"] for b in model["blocks"])
        nll = total / count
        close(nll, model["aggregate"]["nll_nats_per_token"])
        close(math.exp(nll), model["aggregate"]["ppl"])
        recomputed[name] = {"nll_sum": total, "tokens": count, "nll": nll,
                            "ppl": math.exp(nll)}
    deltas = {f"{a}-{b}": recomputed[a]["nll"] - recomputed[b]["nll"]
              for a, b in [("P2", "P1"), ("P4", "P1"), ("P4", "P2")]}
    fits = read(artifacts / "fit-results.json")
    source = root / "upstream/parametric_fit.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fits["source"]["sha256"]
    tables = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in {"params", "p", "stack_loss", "pile_loss"}:
                tables[t.id] = [x for row in literal(node.value) for x in row]
    fit_checks = {}
    for name, value in fits["tables"].items():
        raw, f = value["raw"], value["fit"]
        assert raw["params_non_embedding"] == tables["params"] and raw["p"] == tables["p"]
        assert raw["loss"] == tables["pile_loss" if name == "pile" else "stack_loss"]
        pred = [f["E"] + (f["A_code_billions"] * 1e9 / (n * (math.log(p) * f["k"] + 1))) ** f["alpha"]
                for n, p in zip(raw["params_non_embedding"], raw["p"])]
        y = raw["loss"]
        mean = sum(y) / len(y)
        r2 = 1 - sum((a-b)**2 for a,b in zip(y,pred)) / sum((a-mean)**2 for a in y)
        errors = [abs(math.log(a)-math.log(b)) for a,b in zip(y,pred)]
        objective = sum(e*e/2 if e <= .001 else .001*(e-.0005) for e in errors)
        close(r2, f["r2"])
        close(objective, f["objective_huber_log_delta"])
        for a,b in zip(pred, f["predicted_loss"]):
            close(a,b)
        fit_checks[name] = {"r2": r2, "objective": objective, "source_table_matches": True,
                            "best_optimizer_success_reported": f["success"], "optimizer_rerun": False}
    expected = read(root / "configs/model-sources.json")["models"]
    model_manifest = read(artifacts / "model-manifest.json")
    for p, spec in expected.items():
        actual = model_manifest["models"][f"P{p}"]
        assert actual["actual_sha256"] == spec["sha256"] and actual["actual_bytes"] == spec["bytes"]
        assert actual["revision"] == spec["revision"]
    tokenizers = manifest["tokenizer"]["identity_per_checkpoint"]
    assert all(t["files"] == tokenizers["P1"]["files"] for t in tokenizers.values())
    fp32 = read(artifacts / "cache-diagnostic-fp32-l64.json")
    cache = {}
    for name, r in fp32["results"].items():
        close(max(r["per_incremental_position_max_abs_diff"]), r["full_vs_incremental_max_abs_logit_diff"])
        cache[name] = {"bf16_max": smoke["models"][name]["checks"]["cache_equivalence"]["max_abs_logit_diff"],
                       "fp32_max": r["full_vs_incremental_max_abs_logit_diff"]}
    result = {"run_id": task["run_id"], "event_id": event["event_id"],
              "report_sha256": entries["report.json"]["sha256"],
              "audit_run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "task_and_event_identity_verified": True, "verified_artifacts": verified,
              "nll_recomputed_from_blocks": recomputed, "nll_deltas": deltas,
              "fit_recomputed_from_published_tables_and_reported_parameters": fit_checks,
              "tokenizer_file_hashes_identical": True,
              "weight_manifest_matches_pinned_expected_hashes": True,
              "cache_summary_cross_checks": cache,
              "limitations": ["No inference rerun; per-token logits and targets unavailable in received artifacts.",
                              "Full-corpus tokenizer equivalence was not enforced by worker.",
                              "Resource usage estimate is not a measured GPU activity integral.",
                              "FP32 sequence hash labels the parent 1024-token block, not the actual 64-token input."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
