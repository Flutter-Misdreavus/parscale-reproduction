#!/usr/bin/env python3
"""Recompute the published ParScale scaling-law fits (Eq. 5), CPU only.

The 24 published points per corpus are extracted with ast.literal_eval from the
pinned upstream/parametric_fit.py; no manual transcription.  The authors'
multistart L-BFGS-B grid, bounds and log-space Huber delta=0.001 are preserved.
Unlike the upstream script, the reported parameters/objective belong to the
best of all starts, not to the last optimizer call.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "parametric_fit.py"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _safe_literal(node):
    """ast.literal_eval plus list repetition, used by the published ``p`` table.

    The upstream file writes ``[1] * 6``; that is still source-derived literal
    data (no manual transcription), so evaluate only the narrow, safe subset:
    constants, unary minus, list/tuple displays and list*int repetition.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_literal(v) for v in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_literal(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = _safe_literal(node.left), _safe_literal(node.right)
        if isinstance(left, list) and isinstance(right, int):
            return left * right
        if isinstance(left, int) and isinstance(right, list):
            return right * left
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left * right
        raise ValueError(f"unsupported multiplication at {ast.dump(node)}")
    return ast.literal_eval(node)


def extract_literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"params", "p", "stack_loss", "pile_loss"}
    found = {}
    # The published tables live inside ``if __name__ == "__main__":``, so walk
    # the whole module tree instead of only top-level statements.  Only simple
    # literal assignments to the four known names are considered.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = _safe_literal(node.value)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"missing literals in {path}: {sorted(missing)}")
    return found


def huber_loss(y_true, y_pred, delta=0.001):
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    squared = np.square(error) / 2
    linear = delta * (np.abs(error) - delta / 2)
    return np.where(is_small, squared, linear).sum()


def pred_loss(params, param_list, p_list):
    E, A, alpha, k = params
    # upstream code: A is a coefficient in billions, multiplied by 1e9 to give
    # absolute (non-embedding) parameter units inside the formula.
    return E + (A * 1e9 / (param_list * (np.log(p_list) * k + 1))) ** alpha


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def fit_table(param_list, p_list, loss_list, label):
    param_list = np.asarray(param_list, dtype=float).reshape(-1)
    p_list = np.asarray(p_list, dtype=float).reshape(-1)
    loss_list = np.asarray(loss_list, dtype=float).reshape(-1)

    def objective(params):
        pred = pred_loss(params, param_list, p_list)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e300
        val = huber_loss(np.log(loss_list), np.log(pred), delta=0.001)
        return float(val) if math.isfinite(float(val)) else 1e300

    starts = [
        (E0, A0, alpha0, k0)
        for E0 in (-1.0, -0.5, 0.0)
        for A0 in (-4.0, -2.0, 0.0, 2.0, 4.0)
        for alpha0 in (0.0, 0.5, 1.0, 1.5, 2.0)
        for k0 in (0.2, 0.4, 0.6, 0.8)
    ]
    bounds = [(1e-8, None), (1e-8, None), (1e-8, None), (1e-8, None)]
    best = None
    success_count = 0
    objectives = []
    for idx, (E0, A0, alpha0, k0) in enumerate(starts):
        initial = [math.exp(E0), math.exp(A0), alpha0, k0]
        res = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
        fun = float(res.fun)
        objectives.append(fun)
        success_count += int(bool(res.success))
        if best is None or fun < best["fun"]:
            best = {
                "start_index": idx,
                "initial": initial,
                "fun": fun,
                "x": [float(v) for v in res.x],
                "success": bool(res.success),
                "status": int(res.status),
                "message": str(res.message),
                "nit": int(res.nit),
                "nfev": int(res.nfev),
            }

    params = np.asarray(best["x"], dtype=float)
    pred = pred_loss(params, param_list, p_list)
    residuals = pred - loss_list
    r2 = r2_score_np(loss_list, pred)
    out = {
        "table": label,
        "num_points": int(len(param_list)),
        "raw": {
            "params_non_embedding": param_list.tolist(),
            "p": p_list.tolist(),
            "loss": loss_list.tolist(),
        },
        "fit": {
            "E": float(best["x"][0]),
            "A_code_billions": float(best["x"][1]),
            "A_absolute_parameters": float(best["x"][1] * 1e9),
            "alpha": float(best["x"][2]),
            "k": float(best["x"][3]),
            "objective_huber_log_delta": best["fun"],
            "r2": float(r2),
            "residuals_pred_minus_true": residuals.tolist(),
            "predicted_loss": pred.tolist(),
            "best_start_index": best["start_index"],
            "best_start_initial": best["initial"],
            "success": best["success"],
            "status": best["status"],
            "message": best["message"],
            "nit": best["nit"],
            "nfev": best["nfev"],
            "n_starts": len(starts),
            "n_starts_success": success_count,
            "min_start_objective": min(objectives),
            "max_start_objective": max(objectives),
        },
        "r2_sanity_pass": bool(r2 >= 0.99),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--upstream", default=str(UPSTREAM))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    run_id = args.run_id or out_path.parent.name
    upstream = Path(args.upstream).resolve()
    t0 = time.time()

    literals = extract_literals(upstream)
    params = literals["params"]
    p_list = literals["p"]
    stack = literals["stack_loss"]
    pile = literals["pile_loss"]
    if not (len(params) == len(p_list) == len(stack) == len(pile) == 4):
        raise RuntimeError("unexpected published table shape")

    tables = {}
    for label, losses in (("stack-v2-python", stack), ("pile", pile)):
        # flatten the 4x6 published grid (P in {1,2,4,8} x 6 model sizes)
        flat_params = [v for row in params for v in row]
        flat_p = [v for row in p_list for v in row]
        flat_loss = [v for row in losses for v in row]
        if not (len(flat_params) == len(flat_p) == len(flat_loss) == 24):
            raise RuntimeError(f"{label}: expected 24 points, got "
                               f"{len(flat_params)}/{len(flat_p)}/{len(flat_loss)}")
        tables[label] = fit_table(flat_params, flat_p, flat_loss, label)

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": utcnow(),
        "wall_seconds": time.time() - t0,
        "source": {
            "path": str(upstream.relative_to(ROOT)) if upstream.is_relative_to(ROOT) else str(upstream),
            "sha256": sha256_file(upstream),
            "method": "ast.literal_eval with safe list-repeat evaluation",
        },
        "method": {
            "formula": "E + (A_abs / (params_non_embedding * (log(P)*k + 1)))**alpha",
            "A_units": "upstream A is in billions; A_absolute_parameters = A_code_billions * 1e9",
            "loss": "log-space Huber, delta=0.001, sum over points",
            "optimizer": "scipy.optimize.minimize L-BFGS-B",
            "starts": "E0 in exp([-1,-0.5,0]); A0 in exp([-4,-2,0,2,4]); alpha0 in [0,.5,1,1.5,2]; k0 in [.2,.4,.6,.8]",
            "bounds": "all four parameters (1e-8, None)",
            "best_selection": "minimum objective over all starts (not last optimizer call)",
            "r2_definition": "1 - SS_res/SS_tot on raw loss values",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "tables": tables,
        "both_r2_pass": all(t["r2_sanity_pass"] for t in tables.values()),
    }
    atomic_json(out_path, result)
    print(json.dumps({k: {kk: vv for kk, vv in v["fit"].items()
                          if kk in ("E", "A_code_billions", "alpha", "k", "r2", "objective_huber_log_delta", "success")}
                      for k, v in tables.items()}, indent=2))
    print(f"[fit] both_r2_pass={result['both_r2_pass']} wall={result['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
