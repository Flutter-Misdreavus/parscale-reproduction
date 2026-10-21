#!/usr/bin/env python3
"""Plot audited downstream results; no inference and no hand-entered scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/parscale-20261021-downstream-core/supervisor-independent.json'
data = json.loads(SOURCE.read_text())
assert data['status'] == 'ACCEPTED'
OUT = ROOT / 'docs/assets'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': '#15263d', 'axes.labelcolor': '#40516a',
    'xtick.color': '#52647a', 'ytick.color': '#52647a',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'svg.fonttype': 'none', 'svg.hashsalt': 'parscale-downstream-reviewed',
})
fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.5))
fig.patch.set_facecolor('#f7f9fc')
fig.subplots_adjust(left=.075, right=.97, bottom=.24, top=.72, wspace=.42)
fig.text(.055, .92, 'ParScale / Downstream evaluation', fontsize=22, weight='bold')
fig.text(.055, .85, 'Official ~0.7B checkpoints  |  Four complete tasks  |  4,605 items per model', color='#52647a')
for ax in (left, right):
    ax.set_facecolor('#f7f9fc')
    ax.tick_params(length=0, pad=9)
    ax.set_axisbelow(True)
left.grid(axis='y', color='#dfe6ee', linewidth=.8)
right.grid(axis='x', color='#dfe6ee', linewidth=.8)
models = ['P1', 'P2', 'P4', 'P8']
values = np.array([data['macro_accuracy'][m] * 100 for m in models])
left.plot(range(4), values, color='#087f8c', marker='o', linewidth=2.3, markersize=8)
for i, value in enumerate(values):
    left.annotate(f'{value:.2f}%', (i, value), xytext=(0, 14),
                  textcoords='offset points', ha='center', fontsize=11, weight='bold')
left.set_xticks(range(4), models)
left.set_xlim(-.3, 3.3)
span = max(float(np.ptp(values)), .8)
left.set_ylim(values.min()-.3*span, values.max()+.55*span)
left.set_title('Equal-weight mean of four tasks', loc='left', fontsize=13, weight='bold', pad=20)
left.set_ylabel('Accuracy (%) · higher is better')
left.set_xlabel('Parallel streams')
comparisons = [data['macro_comparisons'][m+'-P1'] for m in models[1:]]
means = np.array([x['delta_percentage_points'] for x in comparisons])
ci = np.array([x['ci95_percentage_points'] for x in comparisons])
for j, (mean, bounds) in enumerate(zip(means, ci)):
    color = '#087f8c' if j == 1 else '#8798b2'
    right.errorbar(mean, 2-j, xerr=[[mean-bounds[0]], [bounds[1]-mean]],
                   fmt='o', color=color, markersize=8, capsize=5, elinewidth=2)
right.axvline(0, color='#8b99aa', linestyle='--', linewidth=1)
right.set_yticks([2, 1, 0], ['P2 − P1', 'P4 − P1 · primary', 'P8 − P1'])
right.set_ylim(-.6, 2.6)
right.set_xlim(min(0, ci.min())-.25, max(1, ci.max())+.25)
right.set_title('Paired differences with 95% intervals', loc='left', fontsize=13, weight='bold', pad=20)
right.set_xlabel('Accuracy difference (percentage points)')
fig.text(.055, .105, 'PIQA · SciQ · WinoGrande · OpenBookQA  |  Normalized accuracy when available', color='#52647a', fontsize=10)
fig.text(.055, .065, 'Paired, task-stratified bootstrap · 2,000 resamples · Equal task weights', color='#52647a', fontsize=9)
fig.text(.055, .03, 'Four-task subset, not the paper General average. Intervals exclude training-seed uncertainty.', color='#52647a', fontsize=9)
for extension in ('svg', 'png'):
    path = OUT / f'downstream.{extension}'
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(),
                metadata={'Date': None} if extension == 'svg' else {})
    if extension == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plt.close(fig)
print('Rendered downstream.svg and downstream.png from audited metrics')
