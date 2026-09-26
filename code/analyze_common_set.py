"""Compare reputation functions on a common target set (experiments/run_common_set.py output).

The common set holds the targets whose clean reputation is at least 0.5 under EVERY function,
so each function is judged on the same targets and the same injected edges. For each function
and budget it reports the flip rate against each attacker and against the strongest attacker per
target (the best of the three strategies for that target; one rule throughout), and
the flip rate by the target's clean margin under that function (margin = R - 0.5), which
controls for gate proximity.

Run: python3 code/analyze_common_set.py        (writes results/unified/common_set_summary.json)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, '..', 'results', 'unified')
FUNCS = ['gdte_mean_total', 'gdte_mean_prop', 'gdte_median_total', 'gdte_median_prop',
         'fg', 'wilson', 'beta', 'fraction']
ATTACKERS = ['random', 'gdte_opt', 'fg_opt']
BUDGETS = [1, 5]
MARGIN_BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5001)]


def cp(k, n, alpha=0.05):
    from scipy import stats
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (round(100 * float(lo), 1), round(100 * float(hi), 1))


def _strongest(r, f, B):
    """Attacked reputation under the strongest of the attackers for this target."""
    return min(r['attacked'][f'{a}|B{B}'][f] for a in ATTACKERS)


def summarize(ds):
    path = os.path.join(RDIR, f'common_set_{ds}.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rows = d['rows']
    base = lambda f: 'gdte_mean' if f.startswith('gdte_mean') else ('gdte_median' if f.startswith('gdte_median') else f)
    clean = lambda r, f: r['clean'][f]
    common = [r for r in rows if all(clean(r, f) >= 0.5 for f in FUNCS)]
    out = {'n_all': len(rows), 'n_common': len(common), 'per_function': {}}
    for f in FUNCS:
        own = [r for r in rows if clean(r, f) >= 0.5]
        cell = {'n_own_eligible': len(own)}
        for B in BUDGETS:
            per_att = {}
            for a in ATTACKERS:
                k = sum(r['attacked'][f'{a}|B{B}'][f] < 0.5 for r in common)
                per_att[a] = {'pct': round(100 * k / len(common), 1) if common else None, 'ci': cp(k, len(common))}
            # the strongest attacker per target (the best of the three strategies for each target), the
            # same rule as _by_raters and _common_by_raters, so every reported rate uses one definition
            k = sum(_strongest(r, f, B) < 0.5 for r in common)
            by_m = {}
            for lo, hi in MARGIN_BINS:
                sel = [r for r in common if lo <= clean(r, f) - 0.5 < hi]
                km = sum(_strongest(r, f, B) < 0.5 for r in sel)
                by_m[f'{lo}-{hi:.1f}'] = {'n': len(sel), 'pct': round(100 * km / len(sel), 1) if sel else None,
                                           'ci': cp(km, len(sel))}
            cell[f'B{B}'] = {'per_attacker': per_att, 'best_single_attacker': max(per_att, key=lambda a: per_att[a]['pct'] or 0),
                             'worst_pct': round(100 * k / len(common), 1) if common else None, 'worst_ci': cp(k, len(common)),
                             'by_own_margin': by_m}
        out['per_function'][f] = cell
    out['by_raters'] = _by_raters(rows)
    out['common_by_raters'] = _common_by_raters(common)
    return out


def _common_by_raters(common, split=10):
    """Flip rate on the common set, split at `split` raters (worst attacker, B = 1 and 5)."""
    res = {}
    for name, sel in ((f'<={split}', [r for r in common if r['indeg'] <= split]),
                      (f'>{split}', [r for r in common if r['indeg'] > split])):
        res[name] = {'n': len(sel)}
        for f in FUNCS:
            for B in BUDGETS:
                worst = [_strongest(r, f, B) for r in sel]
                k = sum(x < 0.5 for x in worst)
                res[name][f'{f}|B{B}'] = {'pct': round(100 * k / len(sel), 1) if sel else None, 'k': k, 'ci': cp(k, len(sel))}
    return res


DEG_BINS = [(2, 3), (4, 5), (6, 10), (11, 30), (31, 10 ** 9)]


def _deg_name(lo, hi):
    return f'{lo}-{hi}' if hi < 10 ** 9 else f'>={lo}'


def _by_raters(rows):
    """Per function and number of raters: the flip rate on the function's own eligible
    targets (worst attacker at B = 1 and B = 5) and the median per-edge shift at B = 1, both
    for the random source (the same edge for every function) and for the worst attacker."""
    res = {}
    for f in FUNCS:
        res[f] = {}
        for lo, hi in DEG_BINS:
            sel = [r for r in rows if lo <= r['indeg'] <= hi]
            elig = [r for r in sel if r['clean'][f] >= 0.5]
            cell = {'n_targets': len(sel), 'n_eligible': len(elig)}
            for B in BUDGETS:
                worst = [_strongest(r, f, B) for r in elig]
                k = sum(x < 0.5 for x in worst)
                cell[f'flip_B{B}'] = {'pct': round(100 * k / len(elig), 1) if elig else None, 'k': k, 'ci': cp(k, len(elig))}
            if sel:
                sh_rand = [r['attacked']['random|B1'][f] - r['clean'][f] for r in sel]
                sh_worst = [_strongest(r, f, 1) - r['clean'][f] for r in sel]
                cell['shift_B1_random_median'] = round(float(np.nanmedian(sh_rand)), 4)
                cell['shift_B1_worst_median'] = round(float(np.nanmedian(sh_worst)), 4)
            res[f][_deg_name(lo, hi)] = cell
    return res


def main():
    res = {ds: summarize(ds) for ds in ('otc', 'alpha')}
    res = {k: v for k, v in res.items() if v}
    with open(os.path.join(RDIR, 'common_set_summary.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    for ds, r in res.items():
        print(f"== {ds}: all {r['n_all']} targets, common eligible set {r['n_common']}")
        for part, c in r['common_by_raters'].items():
            print(f"  common set, {part} raters (n={c['n']}): " + ', '.join(f"{f} {c[f + '|B1']['pct']}" for f in FUNCS))
        for f, bins in r['by_raters'].items():
            print(f"  raters {f:18s} " + ' | '.join(
                f"{b}: flip1 {c['flip_B1']['pct']} (n={c['n_eligible']}) shift {c.get('shift_B1_worst_median')}/{c.get('shift_B1_random_median')}"
                for b, c in bins.items()))
        for f, c in r['per_function'].items():
            print(f"  {f:18s} own-eligible {c['n_own_eligible']:5d} | B1 strongest {c['B1']['worst_pct']}% (best single {c['B1']['best_single_attacker']}) "
                  f"by margin {[(k, v['pct']) for k, v in c['B1']['by_own_margin'].items()]} | B5 worst {c['B5']['worst_pct']}%")


if __name__ == '__main__':
    main()
