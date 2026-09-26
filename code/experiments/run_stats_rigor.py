"""
Inferential-statistics pass over the trained models (no retraining).

Emits, from per-target arrays at the optimized in-window placement:
  (A) paired significance + effect size for the headline contrasts, Holm-corrected:
       appended vs in-window ; naive vs leakage-free ; distrust vs trust ;
       leakage-free vs 0 (effect is real) ; appended vs 0 (null control).
  (B) three-way additive decomposition of the naive drop into
       mechanical (dilution) + self-edge conditioning + propagation, with shares.
  (C) distribution of leakage-free dR by stratum: median, IQR, worst case.
  (D) decision cost: mean rank shift and fraction ejected from the top-10% trusted set.
  (E) dose-response OLS of leakage-free dR on budget, clean margin, in-degree.
Also writes two figures: a forest plot of the leakage-free single-edge flip across
seeds and datasets (from multiseed_flip.json), and a violin of leakage-free dR by
stratum.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_stats_rigor.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
RDIR = str(RESULTS_DIR / "unified")
FDIR = os.path.join(str(RESULTS_DIR / "unified"), "figures")
os.makedirs(FDIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'figure.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})  # no Type 3 fonts (AAAI rule)

C_POOL = 50
BUDGETS = [1, 3, 5]


def _cp(k, n, a=0.05):
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else sp.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else sp.beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def _wilcoxon(diffs):
    d = np.asarray([x for x in diffs if x == x])
    nz = d[d != 0]
    if len(nz) < 6:
        return {'n': len(d), 'median': float(np.median(d)) if len(d) else float('nan'),
                'cohen_d': float('nan'), 'p': 1.0}
    try:
        p = float(sp.wilcoxon(nz).pvalue)
    except Exception:
        p = 1.0
    sd = np.std(d, ddof=1)
    return {'n': int(len(d)), 'median': float(np.median(d)),
            'cohen_d': float(np.mean(d) / sd) if sd > 1e-12 else float('nan'), 'p': p}


def _holm(pdict):
    items = sorted(pdict.items(), key=lambda kv: kv[1])
    m = len(items); out = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def _ptrust(orc, z, target, edges, cols):
    s = edges[0, cols]
    feats = torch.cat((z[s], z[target].unsqueeze(0).expand(len(cols), -1)), 1)
    return F.softmax(feats @ orc.model.regression_weights, 1)[:, 0]


def _enc(orc, edges, labels, idx):
    sI = orc.model.index_list
    try:
        orc.model.index_list = idx
        with torch.no_grad():
            return orc._compute_node_embeddings(edges, labels)
    finally:
        orc.model.index_list = sI


def _legit(orc, edges, labels, target, srcs, sign):
    """R0, R_full, R_legit, R_avg(mechanical baseline) at in-window placement."""
    m = orc.model; T = m.args.train_time_slots
    split = m.index_list[T - 1] + 1
    oi = list(m.index_list)
    lab = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
    ne = torch.tensor([list(srcs), [target] * len(srcs)], dtype=torch.long, device=orc.device)
    nl = torch.tensor([lab] * len(srcs), dtype=torch.float, device=orc.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], 1)
    al = torch.cat([labels[:split], nl, labels[split:]], 0)
    ai = [b + len(srcs) if i >= T - 1 else b for i, b in enumerate(oi)]
    dst = (ae[1] == target).nonzero(as_tuple=True)[0]
    inj = set(range(split, split + len(srcs)))
    pre = torch.tensor([c for c in dst.tolist() if c not in inj], dtype=torch.long, device=orc.device)
    z0 = _enc(orc, edges, labels, oi)
    cd = (edges[1] == target).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        R0 = _ptrust(orc, z0, target, edges, cd).mean().item()
        R_avg = _ptrust(orc, z0, target, ae, dst).mean().item()       # dilution at clean emb
    z1 = _enc(orc, ae, al, ai)
    with torch.no_grad():
        Rf = _ptrust(orc, z1, target, ae, dst).mean().item()
        Rp = _ptrust(orc, z1, target, ae, pre).mean().item() if len(pre) else float('nan')
    return R0, Rf, Rp, R_avg


def _pool(orc, edges, target, rng):
    ex = set(edges[0, edges[1] == target].tolist()); ex.add(target)
    uni = [n for n in range(orc.num_nodes) if n not in ex]
    return [int(s) for s in (rng.choice(uni, C_POOL, replace=False) if len(uni) > C_POOL else uni)]


def collect(ds):
    trained = train_model(ds)
    orc = ReputationAttackOracle(trained['model'], trained['index_list'], trained['device'])
    edges, labels = trained['edges'], trained['labels']
    strata = get_targets(orc, edges, labels, per_stratum=25)
    all_reps = orc.compute_all_reputations(edges, labels)
    N = len(all_reps)
    topk = max(1, int(0.10 * N))
    rank_all = (-all_reps).argsort().argsort()  # 0 = highest reputation
    rng = np.random.RandomState(SEED)
    rows = []
    for stratum in ('high', 'moderate', 'low'):
        for t in strata.get(stratum, []):
            pool = _pool(orc, edges, t, rng)
            if len(pool) < max(BUDGETS):
                continue
            cf = orc.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
            order = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])]
            rec = {'ds': ds, 'stratum': stratum, 'target': int(t)}
            indeg = int((edges[1] == t).sum().item())
            rec['in_degree'] = indeg
            for B in BUDGETS:
                srcs = order[:B]
                R0, Rf_d, Rp_d, Ravg = _legit(orc, edges, labels, t, srcs, 'distrust')
                R0t, Rf_t, Rp_t, _ = _legit(orc, edges, labels, t, srcs, 'trust')
                rec[f'R0_B{B}'] = R0
                rec[f'dr_full_dis_B{B}'] = Rf_d - R0
                rec[f'dr_legit_dis_B{B}'] = Rp_d - R0
                rec[f'dr_full_tru_B{B}'] = Rf_t - R0t
                rec[f'mech_B{B}'] = Ravg - R0
                if B == 1:
                    # appended (silent) with same source
                    dr_app = orc.score_edge_set(edges, labels, srcs, t, 'distrust')
                    rec['dr_appended_B1'] = float(dr_app)
                    rec['margin'] = R0 - 0.5
            rows.append(rec)
    del orc, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return rows, {'N': N, 'topk': topk}


def analyze(rows):
    def col(name):
        return [r[name] for r in rows if name in r and r[name] == r[name]]
    out = {}
    # (A) paired tests, B=1
    full = col('dr_full_dis_B1'); legit = col('dr_legit_dis_B1')
    app = col('dr_appended_B1'); tru = col('dr_full_tru_B1')
    tests = {
        'appended_vs_inwindow': _wilcoxon([f - a for f, a in zip(full, app)]),
        'naive_vs_leakfree':    _wilcoxon([f - l for f, l in zip(full, legit)]),
        'distrust_vs_trust':    _wilcoxon([d - t for d, t in zip(full, tru)]),
        'leakfree_vs_zero':     _wilcoxon(legit),
        'appended_vs_zero':     _wilcoxon(app),
    }
    holm = _holm({k: v['p'] for k, v in tests.items()})
    for k in tests:
        tests[k]['p_holm'] = holm[k]
    out['paired_tests'] = tests
    # (B) 3-way decomposition (B=1 and B=5), shares of total naive drop
    for B in [1, 5]:
        tot = np.array(col(f'dr_full_dis_B{B}'))
        mech = np.array(col(f'mech_B{B}'))
        prop = np.array(col(f'dr_legit_dis_B{B}'))
        self_e = tot - mech - prop
        denom = np.where(np.abs(tot) > 1e-6, tot, np.nan)
        out[f'decomp_B{B}'] = {
            'mean_total': float(np.nanmean(tot)),
            'mean_mechanical': float(np.nanmean(mech)),
            'mean_self_edge': float(np.nanmean(self_e)),
            'mean_propagation': float(np.nanmean(prop)),
            'prop_share_pct': float(np.nanmedian(prop / denom) * 100),
            'self_share_pct': float(np.nanmedian(self_e / denom) * 100),
            'mech_share_pct': float(np.nanmedian(mech / denom) * 100),
        }
    # (C) distribution of leakage-free dR (B=1) by stratum
    dist = {}
    for stratum in ('high', 'moderate', 'low'):
        vals = [r['dr_legit_dis_B1'] for r in rows
                if r['stratum'] == stratum and r.get('dr_legit_dis_B1') == r.get('dr_legit_dis_B1')]
        if vals:
            dist[stratum] = {'n': len(vals), 'median': float(np.median(vals)),
                             'q25': float(np.percentile(vals, 25)), 'q75': float(np.percentile(vals, 75)),
                             'worst': float(np.min(vals))}
    out['distribution'] = dist
    return out


def dose_response(rows):
    X, y = [], []
    for r in rows:
        for B in BUDGETS:
            k = f'dr_legit_dis_B{B}'
            if k in r and r[k] == r[k] and 'margin' in r:
                X.append([B, r['margin'], r['in_degree']]); y.append(r[k])
    X = np.array(X, float); y = np.array(y, float)
    if len(y) < 10:
        return {}
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xd = np.column_stack([np.ones(len(y)), Xs])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    sigma2 = (resid @ resid) / (len(y) - Xd.shape[1])
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    names = ['intercept', 'budget', 'clean_margin', 'in_degree']
    return {names[i]: {'coef': float(beta[i]), 'ci': [float(beta[i] - 1.96 * se[i]), float(beta[i] + 1.96 * se[i])]}
            for i in range(4)}


def decision_cost(ds, rows_meta_rows):
    """Rank shift and top-k ejection for low+moderate targets at B=1 (recomputed here)."""
    return rows_meta_rows  # computed inline in collect_rank below


def collect_rank(ds):
    trained = train_model(ds)
    orc = ReputationAttackOracle(trained['model'], trained['index_list'], trained['device'])
    edges, labels = trained['edges'], trained['labels']
    strata = get_targets(orc, edges, labels, per_stratum=25)
    reps0 = orc.compute_all_reputations(edges, labels)
    N = len(reps0); topk = max(1, int(0.10 * N))
    order0 = (-reps0).argsort()
    rank0 = torch.empty(N, dtype=torch.long, device=reps0.device)
    rank0[order0] = torch.arange(N, device=reps0.device)
    top0 = set(order0[:topk].tolist())
    rng = np.random.RandomState(SEED)
    shifts, ejected, n = [], 0, 0
    for stratum in ('moderate', 'low'):
        for t in strata.get(stratum, [])[:15]:
            pool = _pool(orc, edges, t, rng)
            if len(pool) < 1:
                continue
            cf = orc.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
            src = [min(cf, key=cf.get)]
            # in-window injected graph, recompute all reps
            m = orc.model; T = m.args.train_time_slots
            split = m.index_list[T - 1] + 1; oi = list(m.index_list)
            ne = torch.tensor([[src[0]], [t]], dtype=torch.long, device=orc.device)
            nl = torch.tensor([[0.0, 1.0]], dtype=torch.float, device=orc.device)
            ae = torch.cat([edges[:, :split], ne, edges[:, split:]], 1)
            al = torch.cat([labels[:split], nl, labels[split:]], 0)
            ai = [b + 1 if i >= T - 1 else b for i, b in enumerate(oi)]
            try:
                m.index_list = ai
                reps1 = orc.compute_all_reputations(ae, al)
            finally:
                m.index_list = oi
            order1 = (-reps1).argsort()
            rank1 = torch.empty(N, dtype=torch.long, device=reps1.device)
            rank1[order1] = torch.arange(N, device=reps1.device)
            shifts.append(int((rank1[t] - rank0[t]).item()))
            top1 = set(order1[:topk].tolist())
            if t in top0 and t not in top1:
                ejected += 1
            if t in top0:
                n += 1
    del orc, trained, edges, labels; gc.collect(); torch.cuda.empty_cache()
    return {'mean_rank_shift': float(np.mean(shifts)) if shifts else None,
            'median_rank_shift': float(np.median(shifts)) if shifts else None,
            'topk_ejected': ejected, 'topk_eligible': n, 'topk': topk, 'N': N}


def forest_figure(multiseed):
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ys, labels_, xs, los, his = [], [], [], [], []
    y = 0
    pooled = {}
    for ds in ['otc', 'alpha']:
        per = multiseed[ds]['per_seed']
        ks, ns = [], []
        for sk, v in per.items():
            n = v['n']; pct = v['flip_legit_pct'] or 0
            k = round(pct / 100 * n); ks.append(k); ns.append(n)
            lo, hi = _cp(k, n)
            ys.append(y); labels_.append(f"{ds.upper()} {sk}")
            xs.append(pct); los.append(pct - lo * 100); his.append(hi * 100 - pct)
            y += 1
        K, Ntot = sum(ks), sum(ns)
        plo, phi = _cp(K, Ntot)
        pooled[ds] = (100 * K / Ntot, plo * 100, phi * 100, y)
        y += 0.6
    ax.errorbar(xs, ys, xerr=[los, his], fmt='o', ms=3.2, color='#4477AA',
                ecolor='#88AACC', elinewidth=0.9, capsize=1.6, lw=0)
    for ds, (p, lo, hi, yy) in pooled.items():
        ax.errorbar([p], [yy], xerr=[[p - lo], [hi - p]], fmt='D', ms=5,
                    color='#CC3311', ecolor='#CC3311', elinewidth=1.2, capsize=2, lw=0)
        labels_.append(f"{ds.upper()} pooled"); ys.append(yy)
    ax.set_yticks(ys); ax.set_yticklabels(labels_, fontsize=7)
    ax.set_xlabel('propagation-only single-edge flip (%)', fontsize=8)
    ax.axvline(50, color='0.6', ls=':', lw=0.7)
    ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FDIR, f'fig_forest.{ext}'))
    plt.close(fig)


DS_NAME = {'otc': 'OTC', 'alpha': 'Alpha'}
STRATUM_NAME = {'high': 'high', 'moderate': 'mod.', 'low': 'low'}


def violin_figure(all_rows):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    data, ticks = [], []
    pos = 1
    cols = {'otc': '#4477AA', 'alpha': '#EE7733'}
    xt, xl = [], []
    for ds in ['otc', 'alpha']:
        for stratum in ['high', 'moderate', 'low']:
            vals = [r['dr_legit_dis_B1'] for r in all_rows
                    if r['ds'] == ds and r['stratum'] == stratum
                    and r.get('dr_legit_dis_B1') == r.get('dr_legit_dis_B1')]
            if len(vals) >= 3:
                vp = ax.violinplot([vals], positions=[pos], widths=0.8, showmedians=True)
                for b in vp['bodies']:
                    b.set_facecolor(cols[ds]); b.set_alpha(0.5); b.set_edgecolor('0.3')
                for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
                    if key in vp:
                        vp[key].set_color('0.3'); vp[key].set_linewidth(0.8)
                xt.append(pos); xl.append(f"{DS_NAME[ds]}\n{STRATUM_NAME[stratum]}")
            pos += 1
        pos += 0.5
    ax.axhline(0, color='0.6', ls=':', lw=0.7)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5)
    ax.set_ylabel('propagation-only $\\Delta R$ (1 edge)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FDIR, f'fig_dist.{ext}'))
    plt.close(fig)


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {}
    all_rows = []
    for ds in dsets:
        ds = ds.strip()
        log(f"\n{'='*56}\n  STATS RIGOR — {ds}\n{'='*56}")
        rows, meta = collect(ds)
        all_rows += rows
        an = analyze(rows)
        an['dose_response'] = dose_response(rows)
        an['decision_cost'] = collect_rank(ds)
        an['meta'] = meta
        results[ds] = an
        pt = an['paired_tests']
        log(f"  paired (Holm p): appended-vs-inwindow {pt['appended_vs_inwindow']['p_holm']:.1e} "
            f"(d={pt['appended_vs_inwindow']['cohen_d']:.2f}); naive-vs-leakfree {pt['naive_vs_leakfree']['p_holm']:.1e}; "
            f"dis-vs-tru {pt['distrust_vs_trust']['p_holm']:.1e}; "
            f"leakfree-vs-0 {pt['leakfree_vs_zero']['p_holm']:.1e}; appended-vs-0 {pt['appended_vs_zero']['p_holm']:.2f}")
        log(f"  decomp B1 shares: prop {an['decomp_B1']['prop_share_pct']:.0f}% "
            f"self {an['decomp_B1']['self_share_pct']:.0f}% mech {an['decomp_B1']['mech_share_pct']:.0f}%")
        log(f"  decision: mean rank shift {an['decision_cost']['mean_rank_shift']}, "
            f"top-k ejected {an['decision_cost']['topk_ejected']}/{an['decision_cost']['topk_eligible']}")
        log(f"  dose-response beta(budget,margin,indeg): "
            f"{ {k: round(v['coef'],3) for k,v in an['dose_response'].items() if k!='intercept'} }")

    try:
        ms = json.load(open(os.path.join(RDIR, 'multiseed_flip.json')))
        forest_figure(ms)
        log("  wrote fig_forest")
    except Exception as e:
        log(f"  forest skipped: {e}")
    violin_figure(all_rows)
    log("  wrote fig_dist")

    with open(os.path.join(RDIR, 'stats_rigor.json'), 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {os.path.join(RDIR, 'stats_rigor.json')}")
