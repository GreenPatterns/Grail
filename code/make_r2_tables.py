"""Print the LaTeX rows of the Reviewer-2 tables from the result JSONs, so every number in them
can be regenerated.

  cohort   : complete-cohort single-edge flip rates over ten models (cohort_summary.json)
  official : placement test on official releases (TrustGuard, EvolveGCN-O/H, DySAT, SignedGCN)
  common   : reputation functions on the common target set (common_set_summary.json)

Run: python3 code/make_r2_tables.py [cohort|official|common ...]      (default: all)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, '..', 'results', 'unified')
NAME = {'otc': 'OTC', 'alpha': 'Alpha', 'edg60000': 'Epn-d'}


def _load(name):
    path = os.path.join(RDIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


def _pm(c):
    return f"${c['mean']:.0f}{{\\pm}}{c['std']:.0f}$"


def cohort():
    S = _load('cohort_summary.json')
    print('% --- Table: complete cohort (mean +- std over models, one edge)')
    for ds in [d for d in ('otc', 'alpha', 'edg60000') if d in S]:
        r = S[ds]
        ng = r['near_gate']
        for src, label in (('random', 'Random'), ('opt', 'Optimized')):
            cells = [ng[f'{src}_total'], ng[f'{src}_prop'], r['cells'][f'{src}_total'], r['cells'][f'{src}_prop']]
            first = f"\\multirow{{2}}{{*}}{{{NAME[ds]}}}" if src == 'random' else ''
            print(f" {first} & {label} & " + ' & '.join(_pm(c) for c in cells) + '\\\\')
        n_all = sum(r['n_eligible'].values())
        print(f"% {ds}: models {r['n_seeds']}, low-reputation pairs {ng['n_pooled']}, all pairs {n_all}, "
              f"sign tests total>prop {ng['total_above_prop_opt']}, opt>random {ng['opt_above_random_prop']}")
        print(f"% {ds}: seed-bootstrap CIs (low, opt prop) {ng['opt_prop']['ci']}, (low, random prop) {ng['random_prop']['ci']}")


def cohort_per_model():
    """Supplement table: one row per (dataset, model)."""
    import glob
    print('% --- Table S: complete cohort per model: seed, AUC, eligible, low-rep n, low-rep flips (opt prop, opt naive, rand prop, rand naive), all-eligible opt prop')
    for ds in ('otc', 'alpha', 'edg60000'):
        paths = sorted(glob.glob(os.path.join(RDIR, 'cohort', f'cohort_{ds}_s*.json')),
                       key=lambda p: int(p.rsplit('_s', 1)[1].split('.')[0]))
        for path in paths:
            d = json.load(open(path))
            rows = d['rows']
            low = [r for r in rows if r['r0'] < 0.6]
            f = lambda sel, src, eff: 100 * sum(r[src][eff] < 0.5 for r in sel) / max(len(sel), 1)
            print(f"{NAME[ds]} & {d['seed']} & {d['perf']['AUC']:.3f} & {len(rows):,} & {len(low)} & "
                  f"{f(low, 'opt', 'prop'):.0f} & {f(low, 'opt', 'total'):.0f} & {f(low, 'random', 'prop'):.0f} & "
                  f"{f(low, 'random', 'total'):.0f} & {f(rows, 'opt', 'prop'):.1f}\\\\".replace(',', '{,}'))


def _fmt_small(x):
    """Exact zeros print as 0; tiny floats in scientific notation."""
    if x == 0:
        return '$0$'
    if abs(x) < 1e-3:
        m, e = f"{x:.0e}".split('e')
        return f"${m}{{\\times}}10^{{{int(e)}}}$"
    return f"${x:.2f}$"


def _signed(x):
    return f"${x:+.4f}$" if abs(x) < 0.0005 else f"${x:+.3f}$"


def _range_emb(cells):
    lo, hi = f"{min(c['emb_change'] for c in cells):.2f}", f"{max(c['emb_change'] for c in cells):.2f}"
    return f"emb.\\ ${lo}$" if lo == hi else f"emb.\\ ${lo}$--${hi}$"


def _official_rows():
    """One row per (release, dataset): release, DS, held-out score, unprocessed-edge diagnostic,
    processed-edge diagnostic, processed-edge propagation-only shift, flips (optimized, random)."""
    rows = []
    for ds in ('otc', 'alpha'):
        d = _load(f'p0b_trustguard_{ds}.json')
        if d:
            d = d[ds]
            lab = [v for k, v in d['sweep_label_delta'].items() if k != 'appended']
            f = d['flip_low_B1']
            rows.append(('TrustGuard', NAME[ds], f"AUC {d['clean_perf_last_epoch']['AUC']:.2f}",
                         'label-$\\Delta$ ' + _fmt_small(d['sweep_label_delta']['appended']),
                         f"label-$\\Delta$ ${min(lab):.2f}$--${max(lab):.2f}$", '--',
                         f"{f['k_prop']}/{f['n_eligible']}", '--'))
    sg = _load('official_signedgcn.json')
    for ds in ('otc', 'alpha'):
        if sg:
            d = sg[ds]['seed42']
            rows.append(('SignedGCN', NAME[ds], f"AUC {d['auc']:.2f}", 'label-$\\Delta$ ' + _fmt_small(d['label_delta_notenc']),
                         f"label-$\\Delta$ ${d['label_delta_inenc']:.2f}$", '--',
                         f"{round(d['flip_prop_pct'] * d['n_eligible'] / 100)}/{d['n_eligible']}", '--'))
    for model, label in (('egcn_o', 'EvolveGCN-O'), ('egcn_h', 'EvolveGCN-H')):
        for ds in ('otc', 'alpha'):
            d = _load(f'evolvegcn_placement_{model}_{ds}.json')
            if not d:
                continue
            d = d['bitcoinotc' if ds == 'otc' else 'bitcoinalpha']
            sw = d['sweep']; T = d['T']
            past, last = sw[str(d['past_window_step'])], sw[str(T)]
            earlier = max(sw[str(p)]['emb_change'] for p in range(d['window'][0], T))
            f = d['flip_low_B1']; h = d['held_out']
            rows.append((label, NAME[ds], f"F1 {h['test_microavg_f1']:.2f}", 'emb.\\ ' + _fmt_small(past['emb_change']),
                         f"emb.\\ ${last['emb_change']:.2f}$ (earlier " + _fmt_small(earlier) + ")", f"${last['dR_prop']:+.3f}$",
                         f"{f['k_prop']}/{f['n']}", f"{f['random_k_prop']}/{f['n']}"))
    for ds in ('otc', 'alpha'):
        d = _load(f'dysat_placement_{ds}.json')
        if not d:
            continue
        d = d[ds]; sw = d['sweep']; h = d['held_out']; t = d['time_steps']
        past = sw[str(t - 1)]
        proc = [sw[str(p)] for p in range(0, t - 1)]
        f = d['flip_low_B1']
        rows.append(('DySAT', NAME[ds], f"AUC {h['test_auc_had']:.2f}", 'emb.\\ ' + _fmt_small(past['emb_change']),
                     _range_emb(proc),
                     f"${sw[str(t - 2)]['dR_prop']:+.3f}$", f"{f['k_prop']}/{f['n']}", f"{f['random_k_prop']}/{f['n']}"))
    return rows


def official_compact():
    """Main-paper table: Release | DS | Held-out | Unprocessed | Processed | Flip (optimized, random)."""
    print('% --- main-paper official-release table rows')
    groups = {}
    for r in _official_rows():
        groups.setdefault(r[0], []).append(r)
    blind = {'EvolveGCN-O', 'EvolveGCN-H', 'DySAT'}
    for g, rows in groups.items():
        for j, r in enumerate(rows):
            unproc = r[3].split(' ', 1)[1] if ' ' in r[3] else r[3]
            proc = r[4].split(' ', 1)[1].split(' (earlier')[0]
            flip = r[6] + (f" ({r[7].split('/')[0]})" if r[7] != '--' else '')
            first = (f"\\multirow{{{len(rows)}}}{{*}}{{{g}}}" if len(rows) > 1 else g) if j == 0 else ''
            print(f" {first} & {r[1]} & {r[2].replace('AUC 0.', 'AUC .').replace('F1 0.', 'F1 .')} & {unproc} & {proc} & {flip}\\\\")
        if g != list(groups)[-1]:
            print('\\midrule')


def official_supplement():
    """Supplement table for the label-blind releases: selected epoch, test score, unprocessed edge,
    earlier window steps, last processed placement, flips (optimized / random, prop.-only / naive)."""
    print('% --- supplement: label-blind official releases')
    out = []
    for model, label in (('egcn_o', 'EvolveGCN-O'), ('egcn_h', 'EvolveGCN-H')):
        for ds in ('otc', 'alpha'):
            d = _load(f'evolvegcn_placement_{model}_{ds}.json')
            if not d:
                continue
            d = d['bitcoinotc' if ds == 'otc' else 'bitcoinalpha']
            sw, T, h, f = d['sweep'], d['T'], d['held_out'], d['flip_low_B1']
            past, last = sw[str(d['past_window_step'])], sw[str(T)]
            earlier = [sw[str(p)] for p in range(d['window'][0], T)]
            flips = (f"{f['k_prop']}/{f['k_total']}", f"{f['random_k_prop']}/{f['random_k_total']}", str(f['n'])) if f['n'] else ('--', '--', '0')
            out.append((label, NAME[ds], str(h['selected_epoch']), f"F1 {h['test_microavg_f1']:.2f}",
                         _fmt_small(past['emb_change']), _fmt_small(max(c['emb_change'] for c in earlier)),
                         f"${last['emb_change']:.2f}$", _signed(last['dR_prop']), *flips))
    for ds in ('otc', 'alpha'):
        d = _load(f'dysat_placement_{ds}.json')
        if not d:
            continue
        d = d[ds]; sw, h, f, t = d['sweep'], d['held_out'], d['flip_low_B1'], d['time_steps']
        past, last = sw[str(t - 1)], sw[str(t - 2)]
        earlier = [sw[str(p)] for p in range(0, t - 2)]
        out.append(('DySAT', NAME[ds], str(h['selected_epoch']), f"AUC {h['test_auc_had']:.2f}",
                    _fmt_small(past['emb_change']), f"${max(c['emb_change'] for c in earlier):.2f}$",
                    f"${last['emb_change']:.2f}$", _signed(last['dR_prop']),
                    f"{f['k_prop']}/{f['k_total']}", f"{f['random_k_prop']}/{f['random_k_total']}", str(f['n'])))
    for r in out:
        print(' & '.join(r) + ' \\\\')


def official():
    print('% --- Table: official releases: release, DS, held-out, unprocessed diag, processed diag, processed prop.-only dR (most negative), flips opt, flips rand')
    for r in _official_rows():
        print(' & '.join(r) + ' \\\\')


def common():
    S = _load('common_set_summary.json')
    print('% --- Table: common target set (worst attacker), flip % at B=1 / B=5')
    funcs = ['gdte_mean_total', 'gdte_mean_prop', 'gdte_median_prop', 'fg', 'wilson', 'beta', 'fraction']
    for ds, r in S.items():
        print(f"% {ds}: all {r['n_all']} targets, common set {r['n_common']}")
        for f in funcs:
            c = r['per_function'][f]
            br = r['by_raters'][f]
            low = [br[b] for b in ('4-5', '6-10')]
            high = [br[b] for b in ('11-30', '>=31')]
            k_hi = sum(x['flip_B1']['k'] for x in high); n_hi = sum(x['n_eligible'] for x in high)
            print(f"  {f:18s} common B1 {c['B1']['worst_pct']} (strongest per target) B5 {c['B5']['worst_pct']} | "
                  f">10 raters B1 {100 * k_hi / max(n_hi, 1):.1f}% (n={n_hi}) | shift(>=31, worst) {br['>=31'].get('shift_B1_worst_median')} "
                  f"| 4-10 raters B1 {[x['flip_B1']['pct'] for x in low]}")


if __name__ == '__main__':
    which = sys.argv[1:] or ['cohort', 'cohort_per_model', 'official', 'common']
    for w in which:
        {'cohort': cohort, 'cohort_per_model': cohort_per_model, 'official': official,
         'official_compact': official_compact, 'official_supplement': official_supplement, 'common': common}[w]()
