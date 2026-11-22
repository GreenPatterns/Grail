"""Analyze the complete-cohort, multi-seed single-edge runs (experiments/run_cohort_seeds.py).

For every dataset it reports, per genuine training seed and pooled over seeds:
  * flip rates (%) of the complete eligible cohort (in-degree >= 2, clean R >= 0.5) under one
    distrust edge from a random or an optimized source, total effect and propagation-only;
  * the same by distance to the gate (margin R - 0.5 in [0, .05), [.05, .1), [.1, .2), [.2, .3),
    [.3, .5]);
  * the fixed cohort: targets eligible under every seed, so seeds are compared on one set;
  * seed-level uncertainty: mean and std over seeds, a seed-clustered bootstrap 95% interval
    (seeds resampled, targets kept within their seed), and a two-sided sign test over seeds;
  * the propagation-only shift itself, not thresholded (median, quartiles, share negative), by
    margin and by the number of raters, and the near-gate cohort (clean R in [0.5, 0.6)).

Run: python3 code/analyze_cohort.py            (writes results/unified/cohort_summary.json)
"""
import glob
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, '..', 'results', 'unified')
BINS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.5001)]
BIN_NAMES = ['<.05', '.05-.1', '.1-.2', '.2-.3', '>=.3']
N_BOOT = 2000
RNG = np.random.RandomState(0)


def load(ds):
    runs = {}
    for path in sorted(glob.glob(os.path.join(RDIR, 'cohort', f'cohort_{ds}_s*.json'))):
        d = json.load(open(path))
        runs[d['seed']] = d
    return runs


def flips(rows, src, eff):
    return np.array([r[src][eff] < 0.5 for r in rows], dtype=float)


def sign_test_two_sided(k, n):
    """P-value of k successes out of n under p = 0.5 (exact, two-sided)."""
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def seed_bootstrap(per_seed_rows, src, eff, n_boot=N_BOOT):
    """Seed-clustered bootstrap 95% interval of the pooled flip rate (%): seeds are resampled
    with replacement and each drawn seed contributes all of its targets."""
    hits = np.array([flips(v, src, eff).sum() for v in per_seed_rows.values()], dtype=float)
    ns = np.array([len(v) for v in per_seed_rows.values()], dtype=float)
    if ns.sum() == 0:
        return (float('nan'), float('nan'))
    pick = RNG.randint(0, len(ns), size=(n_boot, len(ns)))
    den = ns[pick].sum(1)
    rates = 100 * hits[pick].sum(1)[den > 0] / den[den > 0]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def summarize(ds):
    runs = load(ds)
    if not runs:
        return None
    rows = {s: d['rows'] for s, d in runs.items()}
    out = {'seeds': sorted(runs), 'n_seeds': len(runs),
           'auc': {s: runs[s]['perf']['AUC'] for s in runs}, 'n_eligible': {s: len(rows[s]) for s in runs}}
    cells = {}
    for src in ('random', 'opt'):
        for eff in ('total', 'prop'):
            key = f'{src}_{eff}'
            per_seed = {s: 100 * flips(rows[s], src, eff).mean() for s in rows}
            by_bin = {}
            for (lo, hi), name in zip(BINS, BIN_NAMES):
                sel = {s: [r for r in rows[s] if lo <= r['r0'] - 0.5 < hi] for s in rows}
                rates = [100 * flips(v, src, eff).mean() for v in sel.values() if v]
                ns = [len(v) for v in sel.values()]
                ci = seed_bootstrap(sel, src, eff)
                by_bin[name] = {'mean': float(np.mean(rates)), 'std': float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                                'n_per_seed_mean': float(np.mean(ns)), 'ci': ci}
            cells[key] = {'per_seed': per_seed, 'mean': float(np.mean(list(per_seed.values()))),
                          'std': float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else 0.0,
                          'ci': seed_bootstrap(rows, src, eff),
                          'by_margin': by_bin}
    out['cells'] = cells
    # by the target's number of raters (the curated strata of Table 2 preferred about 30)
    deg_bins = [(2, 3), (4, 5), (6, 10), (11, 30), (31, 10 ** 9)]
    by_deg = {}
    for lo, hi in deg_bins:
        name = f'{lo}-{hi}' if hi < 10 ** 9 else f'>={lo}'
        sel = {sd: [r for r in rows[sd] if lo <= r['indeg'] <= hi] for sd in rows}
        by_deg[name] = {'n_per_seed_mean': float(np.mean([len(v) for v in sel.values()]))}
        for src in ('random', 'opt'):
            for eff in ('total', 'prop'):
                rates = [100 * flips(v, src, eff).mean() for v in sel.values() if v]
                by_deg[name][f'{src}_{eff}'] = {'mean': float(np.mean(rates)),
                                                'std': float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0}
    out['by_indegree'] = by_deg
    share_low = [np.mean([r['indeg'] <= 3 for r in rows[sd]]) for sd in rows]
    out['share_indegree_le3'] = float(np.mean(share_low))
    # propagation-only shift of the optimized edge, per seed: sign test over seeds
    med = {s: float(np.median([r['opt']['prop'] - r['r0'] for r in rows[s]])) for s in rows}
    k_neg = sum(v < 0 for v in med.values())
    out['opt_prop_median_shift'] = {'per_seed': med, 'k_negative': k_neg, 'n': len(med),
                                    'sign_test_p': sign_test_two_sided(k_neg, len(med))}
    gap = {s: 100 * (flips(rows[s], 'opt', 'prop').mean() - flips(rows[s], 'random', 'prop').mean()) for s in rows}
    k_pos = sum(v > 0 for v in gap.values())
    out['opt_minus_random_prop'] = {'per_seed': gap, 'k_positive': k_pos, 'n': len(gap),
                                    'sign_test_p': sign_test_two_sided(k_pos, len(gap))}
    # fixed cohort: eligible under every seed
    common = set.intersection(*[set(r['t'] for r in rows[s]) for s in rows])
    fixed = {}
    for src in ('random', 'opt'):
        for eff in ('total', 'prop'):
            per = {s: 100 * flips([r for r in rows[s] if r['t'] in common], src, eff).mean() for s in rows}
            fixed[f'{src}_{eff}'] = {'mean': float(np.mean(list(per.values()))),
                                     'std': float(np.std(list(per.values()), ddof=1)) if len(per) > 1 else 0.0}
    near = {s: [r for r in rows[s] if r['t'] in common and r['r0'] - 0.5 < 0.10] for s in rows}
    fixed['n_common'] = len(common)
    fixed['n_common_near_gate_mean'] = float(np.mean([len(v) for v in near.values()]))
    for src in ('random', 'opt'):
        for eff in ('total', 'prop'):
            per = [100 * flips(v, src, eff).mean() for v in near.values() if v]
            fixed[f'near_gate_{src}_{eff}'] = {'mean': float(np.mean(per)), 'std': float(np.std(per, ddof=1)) if len(per) > 1 else 0.0}
    out['fixed_cohort'] = fixed
    pooled = [r for sd in rows for r in rows[sd]]
    out['shift'] = {
        'all': {src: _shift(pooled, src) for src in ('opt', 'random')},
        'by_margin': {name: {src: _shift([r for r in pooled if lo <= r['r0'] - 0.5 < hi], src) for src in ('opt', 'random')}
                      for (lo, hi), name in zip(BINS, BIN_NAMES)},
        'by_indegree': {(f'{lo}-{hi}' if hi < 10 ** 9 else f'>={lo}'):
                        {src: _shift([r for r in pooled if lo <= r['indeg'] <= hi], src) for src in ('opt', 'random')}
                        for lo, hi in deg_bins},
    }
    out['near_gate'] = _near_gate(rows)
    return out


def _shift(sel, src):
    """Propagation-only shift R_prop - R of one edge from `src` (not thresholded)."""
    d = np.array([r[src]['prop'] - r['r0'] for r in sel])
    if len(d) == 0:
        return {'n': 0}
    return {'n': int(len(d)), 'median': float(np.median(d)), 'q25': float(np.percentile(d, 25)),
            'q75': float(np.percentile(d, 75)), 'frac_negative': float(np.mean(d < 0))}


def _near_gate(rows, width=0.10):
    """Targets within `width` of the gate (clean R in [0.5, 0.6), the paper's low stratum)."""
    sel = {sd: [r for r in rows[sd] if r['r0'] - 0.5 < width] for sd in rows}
    out = {'n_per_seed': {sd: len(v) for sd, v in sel.items()}, 'n_pooled': int(sum(len(v) for v in sel.values()))}
    rate = {}
    for src in ('random', 'opt'):
        for eff in ('total', 'prop'):
            per = {sd: 100 * flips(v, src, eff).mean() for sd, v in sel.items() if v}
            k = int(sum(flips(v, src, eff).sum() for v in sel.values()))
            rate[f'{src}_{eff}'] = per
            out[f'{src}_{eff}'] = {'per_seed': per, 'mean': float(np.mean(list(per.values()))),
                                   'std': float(np.std(list(per.values()), ddof=1)) if len(per) > 1 else 0.0,
                                   'pooled_pct': 100 * k / max(out['n_pooled'], 1),
                                   'ci': seed_bootstrap(sel, src, eff)}
    for name, (a, b) in {'total_above_prop_opt': ('opt_total', 'opt_prop'),
                         'opt_above_random_prop': ('opt_prop', 'random_prop')}.items():
        diffs = [rate[a][sd] - rate[b][sd] for sd in rate[a]]
        k_pos, n_nz = sum(x > 0 for x in diffs), sum(x != 0 for x in diffs)
        out[name] = {'k_positive': int(k_pos), 'n_nonzero': int(n_nz), 'sign_test_p': sign_test_two_sided(k_pos, n_nz) if n_nz else 1.0}
    return out


def main():
    res = {ds: summarize(ds) for ds in ('otc', 'alpha', 'edg60000')}
    res = {k: v for k, v in res.items() if v}
    with open(os.path.join(RDIR, 'cohort_summary.json'), 'w') as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    for ds, r in res.items():
        print(f"== {ds}: seeds {r['seeds']} eligible {list(r['n_eligible'].values())}")
        for key, c in r['cells'].items():
            print(f"  {key:12s} all: {c['mean']:5.1f} ± {c['std']:4.1f}  | by margin: " + '  '.join(
                f"{b} {v['mean']:5.1f}±{v['std']:4.1f} (n~{v['n_per_seed_mean']:.0f})" for b, v in c['by_margin'].items()))
        print('  opt prop median shift <0 in', r['opt_prop_median_shift']['k_negative'], 'of', r['opt_prop_median_shift']['n'],
              'seeds, p =', round(r['opt_prop_median_shift']['sign_test_p'], 4))
        print('  fixed cohort n =', r['fixed_cohort']['n_common'], {k: round(v['mean'], 1) for k, v in r['fixed_cohort'].items() if isinstance(v, dict)})
        ng = r['near_gate']
        print('  near gate (R<0.6): pooled n =', ng['n_pooled'], {k: (round(ng[k]['mean'], 1), round(ng[k]['std'], 1), [round(x, 1) for x in ng[k]['ci']])
              for k in ('opt_prop', 'opt_total', 'random_prop', 'random_total')},
              'sign tests', {k: (ng[k]['k_positive'], ng[k]['n_nonzero'], round(ng[k]['sign_test_p'], 4)) for k in ('total_above_prop_opt', 'opt_above_random_prop')})
        print('  shift by margin (opt median):', {k: round(v['opt']['median'], 3) for k, v in r['shift']['by_margin'].items()},
              'share negative (all, opt):', round(r['shift']['all']['opt']['frac_negative'], 4))


if __name__ == '__main__':
    main()
