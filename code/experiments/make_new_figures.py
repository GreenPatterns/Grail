"""Generate the figures for the placement-hazard paper.

fig_placement : single-edge ΔR, appended (silent no-op) vs processed, 3 datasets.
fig_flip_budget : threshold-flip rate vs budget by stratum (OTC), optimized sources.
fig_trivial : ΔR by source-selection strategy at processed placement, 3 datasets.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_paths import RESULTS_DIR, IMAGES_DIR

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
                     # embed TrueType (Type42), never Type3 bitmap fonts: Type3
                     # renders as letter-spaced/dropped glyphs in some PDF viewers
                     # (the "M e a n R" artifact) and USENIX disallows it.
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'dejavuserif'})
U = RESULTS_DIR / "unified"
os.makedirs(IMAGES_DIR, exist_ok=True)


def _load(name, default=None):
    p = U / name
    return json.load(open(p)) if p.exists() else default


# ── Fig 0: snapshot-window placement schematic ───────────────────────
def fig_schematic():
    K, T = 10, 7
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    bw = 1.0
    for i in range(K):
        processed = i < T
        ax.add_patch(plt.Rectangle((i, 0), bw * 0.92, 1.0,
                     facecolor='#cfe8ef' if processed else '#f2f2f2',
                     edgecolor='#457B9D' if processed else '#bbbbbb', lw=1.0))
        ax.text(i + 0.46, 0.5, f'$s_{{{i}}}$', ha='center', va='center', fontsize=9,
                color='#1D3557' if processed else '#999999')
    # processed-window brace + read-out marker
    ax.annotate('', xy=(0, 1.25), xytext=(T - 0.08, 1.25),
                arrowprops=dict(arrowstyle='<->', color='#457B9D', lw=1.2))
    ax.text((T - 0.08) / 2, 1.42, r'processed window ($T_{\mathrm{train}}{=}7$)',
            ha='center', fontsize=9, color='#1D3557')
    ax.text(T - 1 + 0.46, -0.42, 'read-out\nstep', ha='center', fontsize=8, color='#1D3557')
    ax.annotate('', xy=(T - 1 + 0.46, -0.02), xytext=(T - 1 + 0.46, -0.3),
                arrowprops=dict(arrowstyle='->', color='#1D3557', lw=1.0))
    # correct placement (inside window)
    ax.annotate('correct: edge processed', xy=(5.46, 1.02), xytext=(2.0, 2.05),
                fontsize=8.5, color='#7a1d25',
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.4))
    # appended placement (outside window) -> snapshot 9
    ax.annotate('appended: silent no-op', xy=(9.46, 1.02), xytext=(6.4, 2.05),
                fontsize=8.5, color='#555555',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.4))
    ax.set_xlim(-0.3, K + 0.3); ax.set_ylim(-0.7, 2.4)
    ax.axis('off')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(IMAGES_DIR / f'fig_schematic.{ext}')
    plt.close(); print('wrote fig_schematic')


# ── Fig 1: appended vs processed (single edge) ───────────────────────
def fig_placement():
    """Bar chart of mean ΔR at appended vs processed placement, per dataset.

    All values loaded from JSON (paper's run_p0_* artifacts) so the figure
    tracks any rerun:
      - p0_linchpin.json:  per-target means over 20 targets, otc + alpha
      - p0b_main.json:     Epinions sweep (no linchpin run for epn);
                           "processed" is the mean of s0..s6, the natural
                           position for a fresh attack edge
    """
    ds = ['BTC-OTC', 'BTC-Alpha', 'Epinions']
    linch = _load('p0_linchpin.json', {})
    p0b = _load('p0b_main.json', {})
    appended = []
    processed = []
    # otc
    if 'otc' in linch:
        appended.append(linch['otc']['appended_mean_dr'])
        processed.append(linch['otc']['processed_mean_dr'])
    else:
        appended.append(float('nan')); processed.append(float('nan'))
    # alpha
    if 'alpha' in linch:
        appended.append(linch['alpha']['appended_mean_dr'])
        processed.append(linch['alpha']['processed_mean_dr'])
    else:
        appended.append(float('nan')); processed.append(float('nan'))
    # epn: take the mean of s0..s6 as the processed value (paper convention)
    if 'epn30000' in p0b:
        sweep = p0b['epn30000']['sweep_dr']
        keys = [k for k in sweep if k.startswith('s')]
        processed_epn = float(np.mean([sweep[k] for k in keys]))
        appended.append(p0b['epn30000']['sweep_dr']['appended'])
        processed.append(processed_epn)
    else:
        appended.append(float('nan')); processed.append(float('nan'))
    x = np.arange(len(ds)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(x - w/2, appended, w, label='Appended (silent no-op)',
           color='#A8DADC', edgecolor='#457B9D')
    ax.bar(x + w/2, processed, w, label='Processed (correct)',
           color='#E63946', edgecolor='#7a1d25')
    ax.axhline(0, color='k', lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(ds)
    ax.set_ylabel(r'Mean $\Delta R$ (one distrust edge)')
    ax.set_title('Same edge, two placements')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(IMAGES_DIR / f'fig_placement.{ext}')
    plt.close(); print(f'wrote fig_placement  (app={appended}, proc={processed})')


# ── Fig 2: flip-rate vs budget by stratum (OTC) ──────────────────────
def fig_flip_budget():
    """Threshold-flip rate (%) vs budget by reputation stratum, BTC-OTC.

    Loaded from p0d_otcalpha.json (optimized sources, n=25/stratum).
    flip_rate_optimized is in [0,1] in the JSON; converted to percent here.
    """
    p0d = _load('p0d_otcalpha.json', {})
    B = [1, 3, 5, 10]
    cells = p0d.get('otc', {})
    flips = {}
    for stratum, label in [('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')]:
        rates = []
        for b in B:
            e = cells.get(f'{stratum}_B{b}')
            if e is None or 'flip_rate_optimized' not in e:
                rates.append(float('nan'))
            else:
                rates.append(100.0 * e['flip_rate_optimized'])
        flips[label] = rates
    col = {'Low': '#E63946', 'Moderate': '#F4A261', 'High': '#457B9D'}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, v in flips.items():
        ax.plot(B, v, marker='o', label=k, color=col[k], lw=2, ms=5)
    ax.set_xlabel('Attack budget (edges)')
    ax.set_ylabel('Threshold-flip rate (\\%)')
    ax.set_title('Victims pushed below trust gate (BTC-OTC)')
    ax.set_ylim(-3, 105); ax.set_xticks(B)
    ax.legend(frameon=False, fontsize=8, title='Stratum')
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(IMAGES_DIR / f'fig_flip_budget.{ext}')
    plt.close(); print(f'wrote fig_flip_budget  (flips={flips})')


# ── Fig 3: ΔR by strategy at processed placement ─────────────────────
def fig_trivial():
    p1 = _load('p1_efficient.json', {})
    epn = _load('p1_efficient_epn.json', {})
    data = {}
    if p1:
        data.update(p1)
    if epn:
        data.update(epn)
    order = ['counterfactual', 'expert', 'ig', 'batch_trust', 'greedy', 'random']
    labels = ['Counterf.', 'Expert', 'IntGrad', 'Batch', 'Greedy', 'Random']
    dss = [d for d in ['otc', 'alpha', 'epn30000'] if d in data]
    disp = {'otc': 'BTC-OTC', 'alpha': 'BTC-Alpha', 'epn30000': 'Epinions'}
    if not dss:
        # fallback to captured numbers
        data = {'otc': {'mean_dr': dict(counterfactual=-0.248, expert=-0.240, ig=-0.234,
                        batch_trust=-0.219, greedy=-0.219, random=-0.230)},
                'alpha': {'mean_dr': dict(counterfactual=-0.160, expert=-0.163, ig=-0.137,
                          batch_trust=-0.134, greedy=-0.135, random=-0.119)},
                'epn30000': {'mean_dr': dict(counterfactual=-0.127, expert=-0.123, ig=-0.120,
                             batch_trust=-0.120, greedy=-0.120, random=-0.100)}}
        dss = ['otc', 'alpha', 'epn30000']
    x = np.arange(len(order)); w = 0.26
    cols = ['#1D3557', '#457B9D', '#A8DADC']
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, ds in enumerate(dss):
        vals = [abs(data[ds]['mean_dr'].get(m, 0)) for m in order]
        ax.bar(x + (i - 1) * w, vals, w, label=disp[ds], color=cols[i % 3],
               edgecolor='w', lw=.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'$|\Delta R|$ (processed, $B{=}5$)')
    ax.set_title('Selection sophistication barely matters')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(IMAGES_DIR / f'fig_trivial.{ext}')
    plt.close(); print('wrote fig_trivial')


# ── Fig 4: rater-history defense (low stratum, B=5 flip rate) ────────
def fig_defense():
    """Rater-history defense vs the B=5 attack (low-stratum flip rate, %).

    Loaded from p3_defense.json. The "fresh" bar maps to low-low-history
    sources (Sybil/dormant) and "estab" maps to low-random (established)
    sources; flip_rate is in [0,1] in the JSON, converted to percent.
    """
    pd = _load('p3_defense.json', {})
    groups = ['OTC\nfresh', 'OTC\nestab.', 'Alpha\nfresh', 'Alpha\nestab.']
    # (label, dataset, source-class)
    sources = [
        ('otc',  'low_low_history_B5'),   # OTC fresh
        ('otc',  'low_random_B5'),         # OTC estab.
        ('alpha', 'low_low_history_B5'),   # Alpha fresh
        ('alpha', 'low_random_B5'),        # Alpha estab.
    ]
    mean, rw = [], []
    for ds, key in sources:
        cell = pd.get(ds, {}).get(key, {})
        m = cell.get('mean', {}).get('flip_rate')
        w = cell.get('rater_weighted', {}).get('flip_rate')
        mean.append(100.0 * m if m is not None else float('nan'))
        rw.append(100.0 * w if w is not None else float('nan'))
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.bar(x - w/2, mean, w, label='unweighted mean', color='#E63946', edgecolor='w')
    ax.bar(x + w/2, rw, w, label='rater-history weighted', color='#457B9D', edgecolor='w')
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel('low-stratum flip rate (\\%)')
    ax.set_ylim(0, 108)
    ax.set_title('Rater-history weighting vs.\\ the $B{=}5$ attack')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(IMAGES_DIR / f'fig_defense.{ext}')
    plt.close(); print(f'wrote fig_defense  (mean={mean}, rw={rw})')


# ── Fig 5: single-edge flip rate vs clean-margin band ────────────────
def fig_margin():
    """Single-edge flip rate (%) by clean-margin band above the 0.5 gate.

    Loaded from p3_margin.json. Bins in the JSON (0.00-0.05, 0.05-0.15,
    0.15-0.30, 0.30-1.00) map 1:1 to the paper's display labels; flip
    rates are in [0,1] in the JSON, converted to percent here.
    """
    pm = _load('p3_margin.json', {})
    bands = ['$<$.05', '.05–.15', '.15–.30', '$>$.30']
    keys = ['0.00-0.05', '0.05-0.15', '0.15-0.30', '0.30-1.00']
    otc = pm.get('otc', {}).get('by_margin', {})
    alp = pm.get('alpha', {}).get('by_margin', {})
    def _row(d, budget):
        out = []
        for k in keys:
            cell = d.get(k, {})
            v = cell.get(f'flip_{budget}')
            out.append(100.0 * v if v is not None else float('nan'))
        return out
    otc_b1 = _row(otc, 'B1'); otc_b5 = _row(otc, 'B5')
    alp_b1 = _row(alp, 'B1'); alp_b5 = _row(alp, 'B5')
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.plot(x, otc_b1, '-o', color='#E63946', lw=2, ms=5, label='OTC $B{=}1$')
    ax.plot(x, otc_b5, '--o', color='#E63946', lw=1.5, ms=4, alpha=.6, label='OTC $B{=}5$')
    ax.plot(x, alp_b1, '-s', color='#1D3557', lw=2, ms=5, label='Alpha $B{=}1$')
    ax.plot(x, alp_b5, '--s', color='#1D3557', lw=1.5, ms=4, alpha=.6, label='Alpha $B{=}5$')
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_xlabel('clean margin above the $0.5$ gate')
    ax.set_ylabel('flip rate (\\%)'); ax.set_ylim(-5, 108)
    ax.set_title('Flips concentrate near the gate; budget extends reach')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(IMAGES_DIR / f'fig_margin.{ext}')
    plt.close(); print(f'wrote fig_margin  (otc_b1={otc_b1}, alp_b1={alp_b1})')


if __name__ == '__main__':
    fig_margin()
    fig_defense()
    fig_schematic()
    fig_placement()
    fig_flip_budget()
    fig_trivial()
    print('figures ->', IMAGES_DIR)
