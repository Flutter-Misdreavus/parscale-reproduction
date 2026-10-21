#!/usr/bin/env python3
"""Render the complete downstream figure only from accepted independent audits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
core = json.loads((ROOT/'experiments/parscale-20261021-downstream-core/supervisor-independent.json').read_text())
rest = json.loads((ROOT/'experiments/parscale-20261021-downstream-rest/supervisor-independent.json').read_text())
assert core['status'] == rest['status'] == 'ACCEPTED'
assert len(core['checks']) == 16 and len(rest['checks']) == 12
stats = rest['statistics']
models = ['P1', 'P2', 'P4', 'P8']
values = np.array([stats['general_descriptive']['weighted_accuracy'][m]*100 for m in models])
comparisons = [core['macro_comparisons']['P4-P1'],
               stats['extension']['comparisons']['P4-P1'],
               stats['general_descriptive']['comparisons']['P4-P1']]
means = np.array([c['delta_percentage_points'] for c in comparisons])
intervals = np.array([c['ci95_percentage_points'] for c in comparisons])
assert np.isfinite(values).all() and np.isfinite(intervals).all()
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': '#15263d', 'axes.labelcolor': '#40516a',
    'xtick.color': '#52647a', 'ytick.color': '#52647a',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'svg.fonttype': 'none', 'svg.hashsalt': 'parscale-complete-downstream-reviewed',
})
fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.8))
fig.patch.set_facecolor('#f7f9fc')
fig.subplots_adjust(left=.075, right=.965, bottom=.25, top=.72, wspace=.65)
fig.text(.055, .92, 'ParScale / Complete downstream evaluation', fontsize=22, weight='bold')
fig.text(.055, .85, 'Official ~0.7B checkpoints  |  Seven complete tasks  |  18,195 items per model', color='#52647a')
for ax in (left, right):
    ax.set_facecolor('#f7f9fc')
    ax.tick_params(length=0, pad=9)
    ax.set_axisbelow(True)
left.grid(axis='y', color='#dfe6ee', linewidth=.8)
right.grid(axis='x', color='#dfe6ee', linewidth=.8)
left.plot(range(4), values, color='#087f8c', marker='o', linewidth=2.3, markersize=8)
for i, value in enumerate(values):
    left.annotate(f'{value:.2f}%', (i, value), xytext=(0, 14),
                  textcoords='offset points', ha='center', fontsize=11, weight='bold')
left.set_xticks(range(4), models)
left.set_xlim(-.3, 3.3)
span = max(float(np.ptp(values)), .8)
left.set_ylim(values.min()-.3*span, values.max()+.55*span)
left.set_title('Paper six-group General · descriptive', loc='left', fontsize=12, weight='bold', pad=20)
left.set_ylabel('Accuracy (%) · higher is better')
left.set_xlabel('Parallel streams')
for j, (mean, bounds) in enumerate(zip(means, intervals)):
    color = ['#8798b2', '#087f8c', '#a07845'][j]
    # Draw interval and point separately: percentile intervals need not enclose the point estimate.
    right.hlines(2-j, bounds[0], bounds[1], color=color, linewidth=2)
    right.plot(bounds, [2-j, 2-j], '|', color=color, markersize=10)
    right.plot(mean, 2-j, 'o', color=color, markersize=8)
right.axvline(0, color='#8b99aa', linestyle='--', linewidth=1)
right.set_yticks([2, 1, 0], ['Core four · primary', 'New tasks · primary', 'General · descriptive'])
right.set_ylim(-.6, 2.6)
right.set_xlim(min(0, intervals.min(), means.min())-.25, max(1, intervals.max(), means.max())+.25)
right.set_title('P4 − P1 with paired 95% intervals', loc='left', fontsize=12, weight='bold', pad=20)
right.set_xlabel('Accuracy difference (percentage points)')
fig.text(.055, .12, 'General averages six groups; ARC-Easy and ARC-Challenge together form one group.', color='#52647a', fontsize=10)
fig.text(.055, .075, 'New-task primary: 50% HellaSwag + 25% ARC-Easy + 25% ARC-Challenge. Prior core test is preserved.', color='#52647a', fontsize=9)
fig.text(.055, .03, 'General was added after core results and is descriptive. Intervals exclude training-seed uncertainty.', color='#52647a', fontsize=9)
out = ROOT/'docs/assets'
out.mkdir(parents=True, exist_ok=True)
for extension in ('svg', 'png'):
    path = out/f'downstream-complete.{extension}'
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor(),
                metadata={'Date': None} if extension == 'svg' else {})
    if extension == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plt.close(fig)
print('Rendered complete downstream figure from accepted audits')
