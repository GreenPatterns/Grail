"""
Figure 3 of the AAAI-27 paper (figure*, three panels), drawn only from saved JSONs:

  a  placement signature : label-Delta of one distrust edge vs the snapshot it is injected
                           into (s0..s6) or appended past the window, for GDTE on four
                           datasets, two in-house temporal encoders and the official
                           TrustGuard release on the Bitcoin graphs (p0b_main.json,
                           p0b_edgfull.json, p2_arch_signature.json, p0b_trustguard_{otc,alpha}.json)
  b  selection and placement : each strategy's mean shift at B = 5 as a share of the exact
                           counterfactual's in-window shift, in-window and appended, as an
                           annotated diverging heatmap
                           (p1_sota.json, p1_sota_edgfull.json, p1_sota_epn.json)
  c  reputation functions : single-edge flip rate of every reputation function on the common
                           target set (targets eligible under all functions; same injected
                           edges), worst case over three attackers, exact 95% intervals; hollow
                           markers restrict the set to targets with more than ten raters
                           (common_set_summary.json from code/analyze_common_set.py)

Colour and marker both encode the dataset. Slots follow the dataviz skill's reference
palette: the three temporal graphs take slots 1-3 (blue, orange, aqua), which validate
all-pairs; the undated SNAP Epinions takes slot 4 (yellow) and appears only in the line
panel, where adjacent-pair validation and redundant markers apply. The heatmap uses the
skill's blue <-> red diverging pair with the gray midpoint #f0efec, symmetric about 0.
TrueType fonts only (AAAI rejects Type 3).

Run: python code/experiments/make_evidence_figure.py   (no GPU)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker

from project_paths import RESULTS_DIR

U = RESULTS_DIR / "unified"
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'fig_evidence.pdf')

INK, INK2, MUTED, GRID, SHADE = '#1f1f1f', '#4d4d4d', '#8a8a8a', '#e8e8e8', '#f1f1ef'
DS = {  # colour, marker, label (legend order)
    'otc':      ('#2a78d6', 'o', 'Bitcoin-OTC'),
    'alpha':    ('#eb6834', 's', 'Bitcoin-Alpha'),
    'edg60000': ('#1baf7a', '^', 'Epinions (dated)'),
    'epn30000': ('#eda100', 'D', 'Epinions (SNAP, undated)'),
}
SHORT = {'otc': 'OTC', 'alpha': 'Alpha', 'edg60000': 'Epn (dated)', 'epn30000': 'Epn (SNAP)'}
# diverging blue <-> red, gray midpoint; blue = reputation lowered (damage)
DIVERGING = LinearSegmentedColormap.from_list(
    'grail_div', [(0.0, '#8f2323'), (0.25, '#e34948'), (0.5, '#f0efec'),
                  (0.75, '#3987e5'), (1.0, '#0d366b')])
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 6.5, 'axes.titlesize': 7,
    'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'axes.linewidth': 0.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.5,
    'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'text.color': INK, 'legend.fontsize': 6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 300,
})


def _load(name):
    p = U / name
    return json.load(open(p)) if p.exists() else None


def _style(ax, grid_axis='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _title(ax, letter, text, x=0.0):
    ax.set_title(f"$\\bf{{{letter}}}$  {text}", loc='left', pad=3, color=INK, x=x)


# ── a: placement signature ──────────────────────────────────────────────────
def panel_placement(ax, axb):
    sweeps = []   # (dataset, encoder, {pos: label-delta})
    main = _load('p0b_main.json') or {}
    edt = _load('p0b_edgfull.json') or {}
    for ds, src in (('otc', main), ('alpha', main), ('edg60000', edt), ('epn30000', main)):
        if ds in src:
            sweeps.append((ds, 'GDTE', src[ds]['sweep_label_delta']))
    arch = _load('p2_arch_signature.json') or {}
    for ds in ('otc', 'alpha'):
        for enc in ('SignedSnapshotGNN', 'EvolveGCN'):
            if enc in arch.get(ds, {}):
                sweeps.append((ds, enc, arch[ds][enc]['sweep_label_delta']))
        tg = _load(f'p0b_trustguard_{ds}.json') or {}
        if ds in tg:   # official TrustGuard release (default configuration)
            sweeps.append((ds, 'TrustGuard', {('appended' if k == 'appended' else f's{k}'): v
                                              for k, v in tg[ds]['sweep_label_delta'].items()}))
    ls = {'GDTE': '-', 'SignedSnapshotGNN': (0, (3, 1.2)), 'EvolveGCN': (0, (1, 1)),
          'TrustGuard': (0, (4, 1, 1, 1))}
    # Broken y-axis: the upper axis zooms on the processed snapshots s0..s6 (label-Delta 0.04 to 0.5),
    # the lower axis shows the appended edges (about 1e-8), and dotted connectors cross the break.
    x_app = 8.0
    for a in (ax, axb):
        a.axvspan(6.6, 8.6, color=SHADE, lw=0, zorder=0)
    ax.text(7.6, 0.62, 'past the\nwindow', ha='center', va='top', fontsize=5.5, color=INK2)
    for ds, enc, sw in sweeps:
        c, m, _ = DS[ds]
        ys = [sw[f's{i}'] for i in range(7)]
        y_app = max(sw['appended'], 1e-9)
        lw, ms = (1.0, 2.6) if enc == 'GDTE' else (0.7, 2.0)
        ax.plot(range(7), ys, color=c, lw=lw, ls=ls[enc], zorder=3)
        ax.plot(range(7), ys, ls='none', marker=m, ms=ms, color=c, mec='white', mew=0.35, zorder=4)
        axb.plot([x_app], [y_app], ls='none', marker=m, ms=ms, color=c, mec='white', mew=0.35, zorder=4)
        ax.figure.add_artist(ConnectionPatch(xyA=(6, ys[6]), coordsA=ax.transData, xyB=(x_app, y_app),
                                             coordsB=axb.transData, color=c, lw=0.5, ls=(0, (1, 1.5)),
                                             alpha=0.7, zorder=2))
    ax.set_yscale('log')
    ax.set_ylim(0.03, 0.7)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5])
    ax.set_yticklabels(['0.05', '0.1', '0.2', '0.5'])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axb.set_yscale('log')
    axb.set_ylim(5e-9, 3e-7)
    axb.set_yticks([1e-8, 1e-7])
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for a in (ax, axb):
        a.set_xlim(-0.5, 8.6)
        a.set_xticks(list(range(7)) + [x_app])
        _style(a, 'y')
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)
    axb.spines['top'].set_visible(False)
    axb.set_xticklabels([f'$s_{i}$' for i in range(7)] + ['app.'])
    axb.set_xlabel('snapshot holding the attack edge')
    ax.set_ylabel('label-$\\Delta$ (log scale)')
    ax.yaxis.set_label_coords(-0.2, 0.3)
    # diagonal break marks on the left spine
    hr = ax.get_position().height / axb.get_position().height
    for a, y0, k in ((ax, 0.0, 1.0), (axb, 1.0, hr)):
        a.plot((-0.018, 0.018), (y0 - 0.03 * k, y0 + 0.03 * k), transform=a.transAxes, color=MUTED,
               lw=0.6, clip_on=False)
    handles = [Line2D([], [], color=INK, lw=1.0, ls='-', label='GDTE'),
               Line2D([], [], color=INK, lw=0.7, ls=(0, (3, 1.2)), label='signed GNN'),
               Line2D([], [], color=INK, lw=0.7, ls=(0, (1, 1)), label='EvolveGCN-style'),
               Line2D([], [], color=INK, lw=0.7, ls=(0, (4, 1, 1, 1)), label='TrustGuard')]
    axb.legend(handles=handles, loc='center left', bbox_to_anchor=(0.0, 0.5), ncol=2, handlelength=1.9,
               handletextpad=0.4, borderaxespad=0.1, labelspacing=0.15, columnspacing=0.7, fontsize=5.0)
    _title(ax, 'a', 'Placement signature (broken axis)')


# ── b: selection and placement (heatmap) ────────────────────────────────────
ROWS_B = [('counterfactual', 'Counterfactual, 60f'), ('expert', 'Expert (model-free), 0'),
          ('ig', 'Integrated grad., 16b'), ('batch_trust', 'Batch grad., 1b'),
          ('greedy', 'Greedy grad., 5b'), ('prbcd', 'PRBCD$^\\dagger$, 30b'),
          ('influence', 'Influence edit$^\\dagger$, 2b'),
          ('node_injection', 'Node injection$^\\dagger$, 15f'), ('random', 'Random, 0')]
COLS_B = ('otc', 'alpha', 'edg60000', 'epn30000')
SHARE_LIM = 125.0


def _p1_results():
    res = {}
    for fn in ('p1_sota.json', 'p1_sota_epn.json', 'p1_sota_edgfull.json'):
        d = _load(fn) or {}
        res.update({k: v for k, v in d.items() if k in DS})
    return res


def share_matrix(res, cols=COLS_B):
    """Rows x (dataset, placement) shares (%) of the counterfactual's in-window shift."""
    mat = np.full((len(ROWS_B), 2 * len(cols)), np.nan)
    for j, ds in enumerate(cols):
        if ds not in res:
            continue
        v = res[ds]
        cf = v['mean_dr']['counterfactual']
        for i, (key, _) in enumerate(ROWS_B):
            mat[i, 2 * j] = 100 * v['mean_dr'][key] / cf
            mat[i, 2 * j + 1] = 100 * v['mean_dr_appended'][key] / cf
    return mat


def _text_color(rgba):
    r, g, b = rgba[:3]
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def panel_selection(ax):
    res = _p1_results()
    mat = share_matrix(res)
    gap = 0.55
    xs = [2 * j * 1.0 + j * gap + k for j in range(len(COLS_B)) for k in (0, 1)]
    for i in range(len(ROWS_B)):
        for c, x in enumerate(xs):
            val = mat[i, c]
            if val != val:
                continue
            rgba = DIVERGING(0.5 + 0.5 * np.clip(val, -SHARE_LIM, SHARE_LIM) / SHARE_LIM)
            ax.add_patch(plt.Rectangle((x - 0.5, i - 0.5), 1.0, 1.0, facecolor=rgba,
                                       edgecolor='white', lw=0.6))
            txt = f"{val:.0f}"
            ax.text(x, i, '0' if txt == '-0' else txt, ha='center', va='center', fontsize=5.4,
                    color=_text_color(rgba))
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(len(ROWS_B) - 0.5, -2.1)
    ax.set_yticks(range(len(ROWS_B)))
    ax.set_yticklabels([lab for _, lab in ROWS_B])
    ax.set_xticks(xs)
    ax.set_xticklabels(['in', 'app.'] * len(COLS_B), fontsize=5.5)
    ax.tick_params(axis='both', length=0, pad=1.5)
    for j, ds in enumerate(COLS_B):
        xc = (xs[2 * j] + xs[2 * j + 1]) / 2
        ax.text(xc, -0.98, SHORT[ds], ha='center', va='bottom', fontsize=5.6, color=INK2)
        ax.plot([xs[2 * j] - 0.45, xs[2 * j + 1] + 0.45], [-0.7, -0.7], color=DS[ds][0], lw=1.4,
                solid_capstyle='butt')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("% of the counterfactual's in-window $\\Delta R$ ($B{=}5$)")
    _title(ax, 'b', 'Selection and placement')
    return [ds for ds in COLS_B if ds in res]


# ── c: reputation functions on a common target set ─────────────────────────
ROWS_C = [('gdte_mean_total', 'GDTE, naive total'), ('gdte_mean_prop', 'GDTE, prop.-only'),
          ('gdte_median_prop', 'GDTE median, prop.'), ('fg', 'Fairness–Goodness'),
          ('wilson', 'Wilson lower bound'), ('beta', 'Beta mean'), ('fraction', 'Trust fraction')]
RUNS_C = ('otc', 'alpha')
SPLIT_C = 10


def panel_functions(ax):
    """Single-edge flip rate of every function on the targets eligible under all of them
    (common_set_summary.json), worst of three attackers: filled = all common targets,
    hollow = those with more than SPLIT_C raters; bars are exact 95% intervals."""
    S = _load('common_set_summary.json') or {}
    order = []
    n_gnn = 3
    ypos = {i: len(ROWS_C) - 1 - i + (0.6 if i < n_gnn else 0) for i in range(len(ROWS_C))}
    runs = [ds for ds in RUNS_C if ds in S]
    offs = {ds: dy for ds, dy in zip(runs, np.linspace(0.2, -0.2, len(runs)) if len(runs) > 1 else [0.0])}
    for ds in runs:
        order.append(ds)
        c, m, _ = DS[ds]
        r = S[ds]
        for i, (func, _) in enumerate(ROWS_C):
            y = ypos[i] + offs[ds]
            cell = r['per_function'][func]['B1']
            lo, hi = cell['worst_ci']
            ax.plot([lo, hi], [y, y], color=c, lw=0.8, alpha=0.55, solid_capstyle='round', zorder=2)
            ax.plot(cell['worst_pct'], y, ls='none', marker=m, ms=2.6, color=c, mec='white', mew=0.35, zorder=4)
            hi_cell = r['common_by_raters'][f'>{SPLIT_C}'][f'{func}|B1']
            ax.plot(hi_cell['pct'], y, ls='none', marker=m, ms=2.6, mfc='white', mec=c, mew=0.6, zorder=3)
    sep = (ypos[n_gnn - 1] + ypos[n_gnn]) / 2
    ax.axhline(sep, color=GRID, lw=0.8)
    ax.text(99, ypos[0] + 0.5, 'GNN reputation', ha='right', va='bottom', fontsize=5.5, color=INK2)
    ax.text(99, ypos[n_gnn] + 0.38, 'non-GNN reputation', ha='right', va='bottom', fontsize=5.5, color=INK2)
    ax.set_yticks([ypos[i] for i in range(len(ROWS_C))])
    ax.set_yticklabels([lab for _, lab in ROWS_C])
    ax.set_xlim(-3, 101)
    ax.set_ylim(-0.6, ypos[0] + 1.15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('single-edge flip rate, common targets (%)')
    ax.tick_params(axis='y', length=0, pad=2)
    key = [Line2D([], [], ls='none', marker='o', ms=2.6, color=INK2, mec='white', mew=0.35, label='all common targets'),
           Line2D([], [], ls='none', marker='o', ms=2.6, mfc='white', mec=INK2, mew=0.6, label=f'>{SPLIT_C} raters')]
    ax.legend(handles=key, loc='lower right', fontsize=5.4, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    _style(ax, 'x')
    _title(ax, 'c', 'Same targets and edges, other functions')
    return order


def main():
    fig = plt.figure(figsize=(7.0, 1.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 0.98], wspace=0.6,
                          left=0.052, right=0.995, top=0.8, bottom=0.2)
    ga = gs[0].subgridspec(2, 1, height_ratios=[2.7, 1], hspace=0.12)
    ax_top = fig.add_subplot(ga[0])
    panel_placement(ax_top, fig.add_subplot(ga[1], sharex=ax_top))
    shown_b = panel_selection(fig.add_subplot(gs[1]))
    shown_c = panel_functions(fig.add_subplot(gs[2]))
    handles = [Line2D([], [], ls='none', marker=DS[ds][1], ms=3.2, color=DS[ds][0], mec='white',
                      mew=0.35, label=DS[ds][2]) for ds in DS]
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.53, 1.0),
               handletextpad=0.3, columnspacing=1.6)
    for name, shown, want in (('b', shown_b, COLS_B), ('c', shown_c, list(RUNS_C))):
        missing = [ds for ds in want if ds not in shown]
        if missing:
            print(f"WARNING panel {name} lacks {missing}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight', pad_inches=0.01)
    if os.environ.get('GRAIL_FIG_PREVIEW'):      # PNG preview only on request, so figures/ stays clean
        fig.savefig(OUT.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.01, dpi=300)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == '__main__':
    main()
