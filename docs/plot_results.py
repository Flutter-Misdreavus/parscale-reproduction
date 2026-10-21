#!/usr/bin/env python3
"""Render the README figure from reviewed metrics; no model inference."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/parscale-20261021-full/supervisor-independent.json"
data = json.loads(SOURCE.read_text())
assert data["status"] == "PASSED"
OUT = ROOT / "docs/assets"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": "#15263d", "axes.labelcolor": "#40516a",
    "xtick.color": "#52647a", "ytick.color": "#52647a",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "svg.fonttype": "none", "svg.hashsalt": "parscale-reviewed-results",
})
fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.4))
fig.patch.set_facecolor("#f7f9fc")
fig.subplots_adjust(left=.075, right=.97, bottom=.24, top=.72, wspace=.38)
fig.text(.055, .92, "ParScale / Minimal reproduction", fontsize=22, weight="bold")
fig.text(.055, .85, "Official ~0.7B checkpoints on WikiText-2  |  298,785 scored tokens per model", color="#52647a")
colors = {"bfloat16": "#087f8c", "float32": "#7360bb"}
labels = {"bfloat16": "BF16 · primary", "float32": "FP32 · sensitivity"}
models = ["P1", "P2", "P4"]
for ax in (left, right):
    ax.set_facecolor("#f7f9fc")
    ax.tick_params(length=0, pad=9)
    ax.set_axisbelow(True)
left.grid(axis="y", color="#dfe6ee", linewidth=.8)
right.grid(axis="x", color="#dfe6ee", linewidth=.8)

for precision in colors:
    values = [data["precisions"][precision]["models"][m]["ppl"] for m in models]
    left.plot(range(3), values, color=colors[precision], linewidth=2,
              linestyle="-" if precision == "bfloat16" else "--",
              marker="o" if precision == "bfloat16" else "s",
              markersize=8 if precision == "bfloat16" else 5,
              markerfacecolor=colors[precision] if precision == "bfloat16" else "#f7f9fc",
              label=labels[precision], zorder=3)
for i, model in enumerate(models):
    value = data["precisions"]["bfloat16"]["models"][model]["ppl"]
    left.annotate(f"{value:.3f}", (i, value), xytext=(0, 15),
                  textcoords="offset points", ha="center", fontsize=11, weight="bold")
left.set_title("Prediction quality", loc="left", fontsize=13, weight="bold", pad=20)
left.set_xticks(range(3), models)
left.set_xlim(-.25, 2.25)
left.set_ylim(17.95, 19.8)
left.set_yticks([18, 18.5, 19, 19.5])
left.set_ylabel("Perplexity · lower is better")
left.set_xlabel("Parallel streams")

pairs = ["P2-P1", "P4-P1", "P4-P2"]
for precision, offset in [("bfloat16", .11), ("float32", -.11)]:
    records = [data["precisions"][precision]["pairs"][p] for p in pairs]
    means = np.array([r["delta_nll"] for r in records])
    errors = np.array([[r["delta_nll"]-r["ci95"][0] for r in records],
                       [r["ci95"][1]-r["delta_nll"] for r in records]])
    right.errorbar(means, np.array([2, 1, 0])+offset, xerr=errors,
                   fmt="o" if precision == "bfloat16" else "s", markersize=5,
                   color=colors[precision], capsize=4, elinewidth=2, zorder=3)
right.axvline(0, color="#8b99aa", linestyle="--", linewidth=1)
right.set_title("Loss differences with 95% intervals", loc="left", fontsize=13, weight="bold", pad=20)
right.set_yticks([2, 1, 0], ["P2 − P1", "P4 − P1", "P4 − P2"])
right.set_ylim(-.55, 2.55)
right.set_xlim(-.078, .007)
right.set_xticks([-.06, -.04, -.02, 0], ["−0.06", "−0.04", "−0.02", "0"])
right.set_xlabel("Δ NLL (nats/token) · left of zero favors higher P")
handles, names = left.get_legend_handles_labels()
fig.legend(handles, names, loc="lower left", bbox_to_anchor=(.055, .065),
           ncol=2, frameon=False, fontsize=10)
fig.text(.055, .04, "Paired text-block bootstrap · 2,000 resamples · Training-seed uncertainty is not measured.",
         fontsize=9, color="#52647a")
for extension in ("svg", "png"):
    fig.savefig(OUT / f"results.{extension}", dpi=180, facecolor=fig.get_facecolor(),
                metadata={"Date": None} if extension == "svg" else {})
    if extension == "svg":
        path = OUT / "results.svg"
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
print(f"Rendered {OUT / 'results.svg'} and PNG from {SOURCE.relative_to(ROOT)}")
