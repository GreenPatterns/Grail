"""
Figures of the DSN version of the paper, drawn from the saved JSONs with the panel functions of
make_evidence_figure.py and make_cohort_figure.py. The DSN paper uses one term per concept, so
this script relabels the panels (total / propagation-only, in / out, threshold, optimized) and
writes the three evidence panels as separate single-column figures:

  fig_placement.pdf  label sensitivity by the snapshot that holds the attack edge   (Fig. 3)
  fig_cohort.pdf     complete cohort over ten runs                                   (Fig. 4)
  fig_functions.pdf  reputation functions on the common target set                  (Fig. S1 of the supplement)
  fig_sources.pdf    source-selection strategies, in-window and out-of-window       (Fig. S2 of the supplement)

Fig. 3 labels the snapshots s0..s6 and out, and writes the small values of its lower axis as
1e-7; Fig. 4 angles the margin-bin labels of panel c. In both, every text, tick labels included,
is raised to a minimum size at the figure's own width (7.4 pt in Fig. 3, 6.6 pt in Fig. 4), which
prints at about 6.9 and 7.1 pt in the paper. Fig. S1 and S2 keep the sizes they are drawn with.

Run: python code/experiments/make_dsn_figures.py [output_dir]   (no GPU; default: figures/dsn)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

import make_cohort_figure as C
import make_evidence_figure as E

COLUMN_WIDTH_IN = 3.45
# Old label fragment -> DSN term. Applied to every text object after a panel is drawn.
RELABEL = {
    'past the\nwindow': 'out-of-\nwindow',
    'label-$\\Delta$ (log scale)': 'label sensitivity $\\Lambda$',
    'signed GNN': 'signed snapshot GNN',
    'gray: drop needed\nto cross the gate': 'gray: drop needed to\ncross the threshold',
    'clean margin $R-0.5$': 'clean margin $R-\\tau$',
    'naive total': 'total',
    'single-edge flip rate, common targets (%)': 'single-edge flip rate on the common set (%)',
}
E.DS['edg60000'] = (E.DS['edg60000'][0], E.DS['edg60000'][1], 'Epinions-D')
E.DS['epn30000'] = (E.DS['epn30000'][0], E.DS['epn30000'][1], 'Epinions-S')
E.SHORT.update({'edg60000': 'Epinions-D', 'epn30000': 'Epinions-S'})
E.ROWS_B = [('counterfactual', 'Optimized, 60f'), ('expert', 'Heuristic (model-free), 0')] + E.ROWS_B[2:]
E.ROWS_C = [('gdte_mean_total', 'GDTE, total'), ('gdte_mean_prop', 'GDTE, propagation-only'),
            ('gdte_median_prop', 'GDTE median, prop.-only'), ('fg', 'Fairness and Goodness')] + E.ROWS_C[4:]
C.DS['edg60000']['name'] = 'Epinions-D'


def relabel(fig) -> None:
    """Replace the old label fragments in every text object of the figure."""
    for text in fig.findobj(Text):
        new = text.get_text()
        for old, term in RELABEL.items():
            new = new.replace(old, term)
        if new != text.get_text():
            text.set_text(new)


def dataset_legend(fig, keys, y: float) -> None:
    handles = [Line2D([], [], ls='none', marker=E.DS[k][1], ms=3.2, color=E.DS[k][0], mec='white',
                      mew=0.35, label=E.DS[k][2]) for k in keys]
    fig.legend(handles=handles, loc='upper center', ncol=len(keys), bbox_to_anchor=(0.55, y),
               handletextpad=0.2, columnspacing=1.0)


def enforce_print_size(fig, min_font: float, min_marker: float) -> None:
    """Raise every text and marker below the given sizes, so the figure stays readable when
    printed at its size in the paper (the venue asks for no on-screen magnification)."""
    fig.canvas.draw()  # tick labels exist only after a draw
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontsize() < min_font:
            text.set_fontsize(min_font)
    for line in fig.findobj(Line2D):
        if line.get_marker() not in (None, '', 'None', ' ') and line.get_markersize() < min_marker:
            line.set_markersize(min_marker)


def save(fig, out_dir: str, name: str, min_font: float = 0.0, min_marker: float = 0.0) -> None:
    relabel(fig)
    if min_font or min_marker:
        enforce_print_size(fig, min_font, min_marker)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.01, metadata={'CreationDate': None})
    plt.close(fig)
    print('wrote', os.path.abspath(path))


def fig_placement(out_dir: str) -> None:
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 1.75))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.7, 1], hspace=0.12, left=0.14, right=0.99,
                            top=0.88, bottom=0.2)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    E.panel_placement(top, bottom)
    top.set_title('', loc='left')
    bottom.set_xticklabels([f's{i}' for i in range(7)] + ['out'])
    from matplotlib.ticker import FuncFormatter
    for ax in (top, bottom):
        ax.tick_params(labelsize=7.5)
    # Plain '1e-7' labels: superscript exponents would print below 5 pt.
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0e}'.replace('e-0', 'e-')))
    bottom.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    dataset_legend(fig, list(E.DS), 1.0)
    save(fig, out_dir, 'fig_placement.pdf', min_font=7.4, min_marker=3.2)


def fig_sources(out_dir: str) -> None:
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 1.75))
    ax = fig.add_axes([0.33, 0.14, 0.66, 0.85])
    E.panel_selection(ax)
    ax.set_title('', loc='left')
    ax.set_xticklabels(['in', 'out'] * len(E.COLS_B), fontsize=5.5)
    ax.set_xlabel("% of the optimized strategy's in-window $\\Delta R_{\\mathrm{tot}}$ ($B{=}5$)")
    save(fig, out_dir, 'fig_sources.pdf')


def fig_functions(out_dir: str) -> None:
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, 1.7))
    ax = fig.add_axes([0.33, 0.2, 0.65, 0.68])
    shown = E.panel_functions(ax)
    ax.set_title('', loc='left')
    dataset_legend(fig, shown, 1.0)
    save(fig, out_dir, 'fig_functions.pdf')


def fig_cohort(out_dir: str) -> None:
    """Same layout as make_cohort_figure.main, with the DSN terms in the legends."""
    S = json.load(open(C.SUMMARY))
    fig, axes = plt.subplots(1, 4, figsize=(7.85, 1.45),
                             gridspec_kw=dict(wspace=0.36, width_ratios=[1, 1, 1.05, 1]))
    for j, ds in enumerate(('otc', 'alpha')):
        C.panel_margin(axes[j], S[ds], 'ab'[j], C.DS[ds]['name'])
    axes[0].set_ylabel('single-edge flip rate (%)')
    C.panel_shift(axes[2], S)
    for note in axes[2].texts:      # keep the note clear of the last gray bar
        note.set_position((3.6, 0.49))
    C.panel_raters(axes[3], S)
    dash = (0, (2.5, 1.5))
    names = {'otc': 'OTC', 'alpha': 'Alpha', 'edg60000': 'Epinions-D'}
    left = []
    for src, word in (('opt', 'optimized'), ('random', 'random')):
        left.append(Line2D([], [], color=C.COL[src], lw=1.0, marker='o', ms=2.8, mec='white', mew=0.35,
                           label=f'{word}, propagation-only'))
        left.append(Line2D([], [], color=C.COL[src], lw=0.8, ls=dash, marker='o', ms=2.8, mfc='white',
                           mec=C.COL[src], mew=0.6, label=f'{word}, total'))
    right = [Line2D([], [], color=C.COL[src], lw=0.9, ls=C.DS[ds]['ls'], marker=C.DS[ds]['marker'], ms=2.8,
                    mec='white', mew=0.35, label=f'{names[ds]}, {word}')
             for ds in ('otc', 'alpha', 'edg60000') if ds in S
             for src, word in (('opt', 'optimized'), ('random', 'random'))]
    pos = [ax.get_position() for ax in axes]
    y = max(p.y1 for p in pos) + 0.12
    for handles, cols, lo, hi in ((left, 2, pos[0], pos[1]), (right, 3, pos[2], pos[3])):
        fig.legend(handles=handles, loc='lower center', ncol=cols, bbox_to_anchor=((lo.x0 + hi.x1) / 2, y),
                   handletextpad=0.35, columnspacing=1.1, labelspacing=0.25, handlelength=2.4)
    for ax in fig.axes[2:3]:  # panel c's margin-bin labels touch at print size; angle them
        ax.tick_params(axis='x', labelrotation=25)
        plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    save(fig, out_dir, 'fig_cohort.pdf', min_font=6.6, min_marker=3.0)


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'figures', 'dsn')
    os.makedirs(out_dir, exist_ok=True)
    fig_placement(out_dir)
    fig_cohort(out_dir)
    fig_functions(out_dir)
    fig_sources(out_dir)


if __name__ == '__main__':
    main()
