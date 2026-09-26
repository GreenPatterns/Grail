"""
Figure: complete-cohort single-edge attack over ten independently trained models (Reviewer 2),
drawn from results/unified/cohort_summary.json (code/analyze_cohort.py).

  a, b : flip rate by the target's distance to the 0.5 gate (clean margin R - 0.5), OTC, Alpha;
         optimized vs random source (colour), propagation-only (solid, filled) vs naive total
         effect (dashed, hollow); bars are seed-clustered bootstrap 95% intervals
  c    : the propagation-only shift one edge achieves (median |dR_prop|, pooled over models)
         against the shift needed to cross the gate (gray: each margin band's range)
  d    : propagation-only flip rate by the target's number of raters, mean +- std over models
Colours: the dataviz skill's validated 2-slot palette (blue, orange); datasets in c and d are
told apart by marker shape and line style, never by colour alone. TrueType fonts only.

Run: python code/experiments/make_cohort_figure.py
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, '..', '..', 'results', 'unified', 'cohort_summary.json')
OUT = os.path.join(HERE, '..', '..', 'figures', 'fig_cohort.pdf')
INK, INK2, MUTED, GRID, BAND = '#1f1f1f', '#4d4d4d', '#8a8a8a', '#e8e8e8', '#dcdcdc'
COL = {'opt': '#2a78d6', 'random': '#eb6834'}
DS = {'otc': dict(name='Bitcoin-OTC', marker='o', ls='-'), 'alpha': dict(name='Bitcoin-Alpha', marker='s', ls=(0, (2.5, 1.5))),
      'edg60000': dict(name='Epinions (dated)', marker='^', ls=(0, (0.8, 1.2)))}
BIN_LABEL = {'<.05': '<.05', '.05-.1': '.05–.1', '.1-.2': '.1–.2', '.2-.3': '.2–.3', '>=.3': '≥.3'}
BIN_RANGE = {'<.05': (0, .05), '.05-.1': (.05, .10), '.1-.2': (.10, .20), '.2-.3': (.20, .30), '>=.3': (.30, .50)}
DEG_LABEL = {'2-3': '2–3', '4-5': '4–5', '6-10': '6–10', '11-30': '11–30', '>=31': '≥31'}
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 6.5, 'axes.titlesize': 7, 'axes.labelsize': 6.5,
    'xtick.labelsize': 5.8, 'ytick.labelsize': 6, 'axes.linewidth': 0.5, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 2,
    'ytick.major.size': 2, 'legend.fontsize': 6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 300,
})


def _style(ax, pct=True):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if pct:
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])


def _title(ax, letter, text):
    ax.set_title(f"$\\bf{{{letter}}}$  {text}", loc='left', pad=3, color=INK)


def _errline(ax, xx, m, lo, hi, color, ls, marker, filled, lw=1.0):
    m, lo, hi = map(np.asarray, (m, lo, hi))
    ax.plot(xx, m, color=color, lw=lw, ls=ls, zorder=3)
    ax.errorbar(xx, m, yerr=[m - lo, hi - m], fmt='none', ecolor=color, elinewidth=0.6, capsize=1.2,
                alpha=0.7, zorder=2)
    ax.plot(xx, m, ls='none', marker=marker, ms=2.8, color=color, mfc=color if filled else 'white',
            mec='white' if filled else color, mew=0.35 if filled else 0.6, zorder=4)


def panel_margin(ax, r, letter, name):
    bins = list(r['cells']['opt_prop']['by_margin'])
    for src, dx in (('opt', -0.08), ('random', 0.08)):
        for eff, ls, filled in (('prop', '-', True), ('total', (0, (2.5, 1.5)), False)):
            cells = [r['cells'][f'{src}_{eff}']['by_margin'][b] for b in bins]
            _errline(ax, np.arange(len(bins)) + dx, [c['mean'] for c in cells], [c['ci'][0] for c in cells],
                     [c['ci'][1] for c in cells], COL[src], ls, 'o', filled, lw=1.0 if eff == 'prop' else 0.8)
    n = [int(round(r['cells']['opt_prop']['by_margin'][b]['n_per_seed_mean'])) for b in bins]
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{BIN_LABEL[b]}\n({k})" for b, k in zip(bins, n)])
    ax.set_xlabel('clean margin $R-0.5$')
    _title(ax, letter, name)
    _style(ax)


def panel_shift(ax, S):
    bins = list(BIN_RANGE)
    for j, b in enumerate(bins):
        lo, hi = BIN_RANGE[b]
        ax.add_patch(plt.Rectangle((j - 0.32, lo), 0.64, hi - lo, color=BAND, lw=0, zorder=1))
    for ds, dx in (('otc', -0.15), ('alpha', 0.0), ('edg60000', 0.15)):
        if ds not in S:
            continue
        sh = S[ds]['shift']['by_margin']
        for src in ('opt', 'random'):
            med = [-sh[b][src]['median'] for b in bins]
            ax.plot(np.arange(len(bins)) + dx, med, color=COL[src], lw=0.9, ls=DS[ds]['ls'], zorder=3)
            ax.plot(np.arange(len(bins)) + dx, med, ls='none', marker=DS[ds]['marker'], ms=2.8, color=COL[src],
                    mec='white', mew=0.35, zorder=4)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([BIN_LABEL[b] for b in bins], fontsize=5.0)
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    ax.set_ylim(-0.03, 0.52)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel('clean margin $R-0.5$')
    ax.set_ylabel('prop.-only drop $-\\Delta R$ (median)')
    ax.text(len(bins) - 0.55, 0.49, 'gray: drop needed\nto cross the gate', ha='right', va='top', fontsize=5.4,
            color=INK2)
    _title(ax, 'c', 'shift achieved vs. needed')
    _style(ax, pct=False)


def panel_raters(ax, S):
    dbins = None
    for ds, dx in (('otc', -0.15), ('alpha', 0.0), ('edg60000', 0.15)):
        if ds not in S:
            continue
        r = S[ds]['by_indegree']
        dbins = list(r)
        for src in ('opt', 'random'):
            m = [r[b][f'{src}_prop']['mean'] for b in dbins]
            sd = [r[b][f'{src}_prop']['std'] for b in dbins]
            _errline(ax, np.arange(len(dbins)) + dx, m, np.subtract(m, sd), np.add(m, sd), COL[src], DS[ds]['ls'],
                     DS[ds]['marker'], True, lw=0.9)
    ax.set_xticks(range(len(dbins)))
    ax.set_xticklabels([DEG_LABEL[b] for b in dbins], fontsize=5.4)
    ax.set_xlabel('raters of the target')
    ax.set_ylim(-1, 42)
    ax.set_yticks([0, 10, 20, 30, 40])
    _title(ax, 'd', 'prop.-only flip by raters')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    S = json.load(open(SUMMARY))
    fig, axes = plt.subplots(1, 4, figsize=(7.85, 1.45), gridspec_kw=dict(wspace=0.36, width_ratios=[1, 1, 1.05, 1]))
    for j, ds in enumerate(('otc', 'alpha')):
        if ds in S:
            panel_margin(axes[j], S[ds], 'ab'[j], DS[ds]['name'])
    axes[0].set_ylabel('single-edge flip rate (%)')
    panel_shift(axes[2], S)
    panel_raters(axes[3], S)
    # One legend over each pair of panels, and every entry is a series exactly as drawn there
    # (colour = source, line and marker = estimand in a and b, dataset in c and d).
    dash = (0, (2.5, 1.5))
    LEG_NAME = {'otc': 'OTC', 'alpha': 'Alpha', 'edg60000': 'Epinions (dated)'}
    ab = [Line2D([], [], color=COL['opt'], lw=1.0, marker='o', ms=2.8, mec='white', mew=0.35,
                 label='optimized, propagation-only'),
          Line2D([], [], color=COL['opt'], lw=0.8, ls=dash, marker='o', ms=2.8, mfc='white', mec=COL['opt'],
                 mew=0.6, label='optimized, naive total'),
          Line2D([], [], color=COL['random'], lw=1.0, marker='o', ms=2.8, mec='white', mew=0.35,
                 label='random, propagation-only'),
          Line2D([], [], color=COL['random'], lw=0.8, ls=dash, marker='o', ms=2.8, mfc='white', mec=COL['random'],
                 mew=0.6, label='random, naive total')]
    cd = [Line2D([], [], color=COL[src], lw=0.9, ls=DS[ds]['ls'], marker=DS[ds]['marker'], ms=2.8, mec='white',
                 mew=0.35, label=f"{LEG_NAME[ds]}, {word}")
          for ds in ('otc', 'alpha', 'edg60000') if ds in S for src, word in (('opt', 'optimized'), ('random', 'random'))]
    pos = [ax.get_position() for ax in axes]
    y = max(p.y1 for p in pos) + 0.12
    fig.legend(handles=ab, loc='lower center', ncol=2, bbox_to_anchor=((pos[0].x0 + pos[1].x1) / 2, y),
               handletextpad=0.35, columnspacing=1.2, labelspacing=0.25, handlelength=2.4)
    fig.legend(handles=cd, loc='lower center', ncol=3, bbox_to_anchor=((pos[2].x0 + pos[3].x1) / 2, y),
               handletextpad=0.35, columnspacing=1.0, labelspacing=0.25, handlelength=2.4)
    fig.savefig(OUT, bbox_inches='tight', pad_inches=0.01)
    if os.environ.get('GRAIL_FIG_PREVIEW'):      # PNG preview only on request, so figures/ stays clean
        fig.savefig(OUT.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.01, dpi=300)
    print('wrote', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
