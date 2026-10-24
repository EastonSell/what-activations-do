"""Replot all recorded A-arm curves; no training, inference, or data selection."""
import argparse
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'frozen-head-mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
root = args.package
rows = json.loads((root / 'receipts/heldout-curves.json').read_text())
rows = [r for r in rows if r['arm'] == 'A']
seeds = [101, 211, 307, 401, 503]
ranks = list(range(33))
families = [('pca', 'Leading PCA', '#1f3763'), ('rank', 'Rank-random', '#8a5a38'),
            ('energy', 'Equal-energy rotation', '#008879')]
def group(family, rank, seed=None):
    return [r for r in rows if r['condition'].split('-')[0] == family and
            r['rank'] == rank and (seed is None or r['seed'] == seed)]
assert len(rows) == 5445
for family, _, _ in families:
    for k in ranks:
        assert len(group(family, k)) == (5 if family == 'pca' else 80)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
for ax, metric, scale, title, ylabel in zip(
        axes, ['accuracy', 'ce'], [100, 1], ['Accuracy', 'Cross-entropy'], ['Percent correct', 'Nats']):
    ax.axvspan(1, 8, color='#e8edf5', zorder=0)
    for seed in seeds:
        ax.plot(ranks, [scale * group('pca', k, seed)[0][metric] for k in ranks],
                color='#1f3763', alpha=.22, lw=.8)
    for family, label, color in families:
        ax.plot(ranks, [scale * np.mean([r[metric] for r in group(family, k)]) for k in ranks],
                color=color, lw=2.5, label=label)
    ax.set(xlim=(0, 32), xticks=[0, 4, 8, 12, 16, 20, 24, 28, 32],
           xlabel='Rank k', ylabel=ylabel, title=title)
    ax.grid(axis='y', color='#e6e6e6', lw=.7)
    ax.set_axisbelow(True)
axes[0].set_ylim(0, 100)
axes[1].set_ylim(bottom=0)
axes[0].legend(loc='lower left', frameon=False, fontsize=9)
fig.suptitle('Actual trained heads: PCA removal hurts more than equal-energy rotations',
             fontsize=13, fontweight='bold', y=.98)
fig.text(.5, .015, 'Five fixed A seeds, 360 held-out cases; 16 draws per control. Shading: frozen primary ranks 1–8. Thin lines: PCA seeds.',
         ha='center', fontsize=9, color='#444444')
fig.tight_layout(rect=(0, .05, 1, .93))
for ext in ['png', 'svg']:
    fig.savefig(root / f'report/frozen-head-curves.{ext}', dpi=180)
plt.close(fig)

summary = {'arm': 'A', 'all_ranks': [], 'primary_range': {},
           'note': 'Presentation aggregation from archived curve rows; no producer run.'}
for k in ranks:
    item = {'k': k}
    for family, _, _ in families:
        item[family] = {metric: float(np.mean([r[metric] for r in group(family, k)]))
                        for metric in ['accuracy', 'ce', 'norm']}
    summary['all_ranks'].append(item)
assert [item['k'] for item in summary['all_ranks']] == list(range(33))
for family, _, _ in families:
    subset = [r for r in rows if r['condition'].split('-')[0] == family and 1 <= r['rank'] <= 8]
    summary['primary_range'][family] = {metric: float(np.mean([r[metric] for r in subset]))
                                        for metric in ['accuracy', 'ce', 'norm']}
(root / 'report/presentation-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
