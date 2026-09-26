"""
Merged Experiment Common — Combines Ai_Attacking (paper 1290) and Grill/GRAIL (paper 590)
experiments into a single unified module.

Experiments:
  E1: Gradient validation on existing edges (both papers)
  E2: GRAIL-GRAD vs GRAIL-FWD overlap measurement (Grill)
  E3: Attack effectiveness — B=1 and B=5 comparison (Ai_Attacking)
  E4: Budget efficiency curves (Ai_Attacking)
  E5: Surrogate transfer / black-box (both papers)
  E6: Stronger model + degree-matched stealth (Grill)
  E7: Influence distribution — Gini, degree correlation (Ai_Attacking)
  E8: Attack edge detection — AUC 0.96 (Ai_Attacking)
  E9-E10: Sparsity ablation, bootstrap CI (Grill)
"""
import gc, os, json, time, warnings, copy
import torch
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from project_paths import RESULTS_DIR
from mycode.mainz_protocol import train_oracle_mainz_model
from mycode.trust_influence import (
    ReputationAttackOracle, CounterfactualSimulator,
    detect_attack_edges, analyze_gradient_failures,
)
from mycode.attacks_gpu import identify_good_nodes_from_tensors
from mycode import baselines_sota

warnings.filterwarnings('ignore')

RDIR = str(RESULTS_DIR / "unified")
FDIR = os.path.join(RDIR, "figures")
os.makedirs(FDIR, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})

COL = {
    'AI Oracle': '#E63946', 'GRAIL-FWD': '#E63946', 'GRAIL-GRAD': '#9B59B6',
    'Random': '#A8DADC', 'Degree': '#457B9D', 'PageRank': '#1D3557',
    'Betweenness': '#2A9D8F', 'Expert': '#F4A261', 'Surrogate': '#9B59B6',
    'PRBCD': '#6A4C93', 'InfluenceEdit': '#8338EC', 'NodeInjection': '#FB5607',
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wilcoxon_test(a, b):
    d = np.array(a) - np.array(b)
    d = d[d != 0]
    if len(d) < 5:
        return 1.0
    try:
        return float(sp_stats.wilcoxon(d).pvalue)
    except Exception:
        return 1.0


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    ps = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / max(ps, 1e-12))


def train_model(ds, seed=42, layers=None, epochs=50):
    log(f"Training GDTE on {ds} (seed={seed}" + (f", layers={layers}" if layers else "") + ")...")
    trained = train_oracle_mainz_model(ds, seed=seed, layers=layers, epochs=epochs)
    p = trained["perf"]
    log(f"  MCC={p['MCC']:.3f}±{p['MCC_std']:.3f}, "
        f"AUC={p['AUC']:.3f}±{p['AUC_std']:.3f}, BAcc={p['BAcc']:.3f}±{p['BAcc_std']:.3f}")
    return trained


def get_targets(oracle, edges, labels, per_stratum=10):
    strata = oracle.select_stratified_targets(edges, labels, per_stratum=per_stratum)
    return strata


def score_attack(oracle, edges, labels, target, sources, sign='distrust'):
    return oracle.score_edge_set(edges, labels, sources, target, sign)


def bootstrap_ci(data, n_boot=1000, ci=0.95):
    data = np.array(data)
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(data)), float(lo), float(hi)


# ════════════════════════════════════════════════════════════════════
# E1: Gradient Validation (both papers)
# ════════════════════════════════════════════════════════════════════
def run_E1(oracle, edges, labels, strata, ds):
    log(f"E1: Gradient validation ({ds})...")
    res = {}
    for sn, targets in strata.items():
        if not targets:
            continue
        rs, recs = [], []
        for v in targets[:5]:
            r = oracle.validate_influence_map(edges, labels, v, top_n=50)
            rs.append(r['pearson_r'])
            recs.append(r['recall_at_10'])
        res[sn] = {
            'r': float(np.mean(rs)),
            'r_std': float(np.std(rs)),
            'recall': float(np.mean(recs)),
            'n': len(rs),
        }
        log(f"  {sn}: r={np.mean(rs):.3f}±{np.std(rs):.3f}, R@10={np.mean(recs):.2f}")
    if strata.get('moderate'):
        v = strata['moderate'][0]
        r = oracle.validate_influence_map(edges, labels, v, top_n=50)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(r['gradient_magnitudes'], r['actual_shifts'], alpha=.6, s=30,
                   c='#E63946', edgecolors='w', lw=.3)
        z = np.polyfit(r['gradient_magnitudes'], r['actual_shifts'], 1)
        xl = np.linspace(0, max(r['gradient_magnitudes']), 100)
        ax.plot(xl, np.polyval(z, xl), '--', color='#1D3557', lw=1.5)
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Actual |ΔR|')
        ax.set_title(f'{ds} Moderate (r={r["pearson_r"]:.3f})')
        ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(f'{FDIR}/e1_scatter_{ds}.pdf')
        fig.savefig(f'{FDIR}/e1_scatter_{ds}.png')
        plt.close()
    return res


# ════════════════════════════════════════════════════════════════════
# E2: GRAIL-GRAD vs GRAIL-FWD Overlap (Grill)
# ════════════════════════════════════════════════════════════════════
def run_E2_grad_vs_fwd(oracle, edges, labels, ds, C_values=[50, 100]):
    log(f"E2: GRAIL-GRAD vs GRAIL-FWD ({ds})...")
    strata = get_targets(oracle, edges, labels, per_stratum=5)
    targets = strata.get('moderate', []) + strata.get('low', [])
    log(f"  {len(targets)} targets (moderate+low)")
    B = 5
    results = {}

    # Build a fixed candidate pool per target so both methods score the same C nodes.
    # This lets us compute Spearman ρ over the full pool rather than only the tiny
    # shared top-B intersection (which is near-empty at 2–8% overlap).
    def _sample_candidates(v, C):
        existing = set(
            (int(edges[0, i]), int(edges[1, i])) for i in range(edges.shape[1])
        )
        pool = [s for s in range(oracle.num_nodes) if s != v and (s, v) not in existing]
        if len(pool) > C:
            rng = np.random.RandomState(42)
            pool = list(rng.choice(pool, C, replace=False))
        return pool

    for C in C_values:
        # Pre-sample candidate pools once (used by both methods)
        candidate_pools = [_sample_candidates(v, C) for v in targets]

        t_grad_start = time.time()
        grad_all_scores = []  # list of dicts {src: grad_score} over full pool
        for v, cands in zip(targets, candidate_pools):
            gs = oracle.score_candidates_gradient(edges, labels, v, cands, sign='distrust')
            grad_all_scores.append(gs)
        t_grad = (time.time() - t_grad_start) * 1000 / max(len(targets), 1)

        t_fwd_start = time.time()
        fwd_all_scores = []  # list of dicts {src: fwd_score} over full pool
        for v, cands in zip(targets, candidate_pools):
            fr = oracle.rank_new_edges(edges, labels, v, candidate_sources=cands,
                                       sign='distrust', top_k=C)
            fwd_all_scores.append({s: sc for s, sc in fr})
        t_fwd = (time.time() - t_fwd_start) * 1000 / max(len(targets), 1)

        overlaps, rhos, grad_drs, fwd_drs = [], [], [], []
        for i, v in enumerate(targets):
            g_scores = grad_all_scores[i]   # {src: grad}
            f_scores = fwd_all_scores[i]    # {src: fwd_delta_r}

            # Top-B selections for overlap and actual ΔR
            grad_top = sorted(g_scores.items(), key=lambda x: x[1])[:B]   # most-negative first
            fwd_top  = sorted(f_scores.items(), key=lambda x: x[1])[:B]
            grad_sources = [s for s, _ in grad_top]
            fwd_sources  = [s for s, _ in fwd_top]

            overlap = len(set(grad_sources) & set(fwd_sources)) / max(B, 1)
            overlaps.append(overlap)

            gdr = oracle.score_edge_set(edges, labels, grad_sources, v, 'distrust')
            grad_drs.append(float(gdr))
            fdr = oracle.score_edge_set(edges, labels, fwd_sources, v, 'distrust')
            fwd_drs.append(float(fdr))

            # Spearman ρ over the full candidate pool (both methods scored all C nodes)
            shared = [s for s in g_scores if s in f_scores]
            if len(shared) > 2:
                gv = [g_scores[s] for s in shared]
                fv = [f_scores[s] for s in shared]
                r, _ = sp_stats.spearmanr(gv, fv)
                rhos.append(float(r))

        def _boot_ci(arr, n_boot=1000):
            arr = np.array(arr)
            if len(arr) < 2:
                return float(np.mean(arr)), float(np.mean(arr)), float(np.mean(arr))
            means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
            return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

        grad_mean, grad_lo, grad_hi = _boot_ci(grad_drs)
        fwd_mean, fwd_lo, fwd_hi = _boot_ci(fwd_drs)
        ov_mean, ov_lo, ov_hi = _boot_ci(overlaps)

        n_mod = len(strata.get('moderate', []))
        per_stratum = {}
        if n_mod > 0 and n_mod < len(targets):
            per_stratum['moderate'] = {
                'grad_dr_mean': float(np.mean(grad_drs[:n_mod])),
                'fwd_dr_mean': float(np.mean(fwd_drs[:n_mod])),
                'overlap_mean': float(np.mean(overlaps[:n_mod])),
                'n': n_mod,
            }
            per_stratum['low'] = {
                'grad_dr_mean': float(np.mean(grad_drs[n_mod:])),
                'fwd_dr_mean': float(np.mean(fwd_drs[n_mod:])),
                'overlap_mean': float(np.mean(overlaps[n_mod:])),
                'n': len(targets) - n_mod,
            }

        results[f'C{C}'] = {
            'grad_dr_mean': grad_mean,
            'grad_dr_ci': [grad_lo, grad_hi],
            'fwd_dr_mean': fwd_mean,
            'fwd_dr_ci': [fwd_lo, fwd_hi],
            'overlap_mean': ov_mean,
            'overlap_ci': [ov_lo, ov_hi],
            'rho_mean': float(np.mean(rhos)) if rhos else 0.0,
            'rho_values': rhos,
            'speedup': round(t_fwd / max(t_grad, 1), 1),
            'grad_time_ms': round(t_grad, 1),
            'fwd_time_ms': round(t_fwd, 1),
            'grad_drs': grad_drs,
            'fwd_drs': fwd_drs,
            'n_targets': len(targets),
            'grad_succ': float(np.mean([1 for d in grad_drs if d < 0]) if any(d < 0 for d in grad_drs) else 0.0),
            'fwd_succ': float(np.mean([1 for d in fwd_drs if d < 0]) if any(d < 0 for d in fwd_drs) else 0.0),
            'per_stratum': per_stratum,
        }
        log(f"  C={C}: Grad ΔR={grad_mean:+.4f} [{grad_lo:+.4f},{grad_hi:+.4f}], "
            f"Fwd ΔR={fwd_mean:+.4f} [{fwd_lo:+.4f},{fwd_hi:+.4f}], "
            f"Overlap={ov_mean:.0%} [{ov_lo:.0%},{ov_hi:.0%}], "
            f"ρ={np.mean(rhos) if rhos else 0:.3f}, "
            f"Speedup={results[f'C{C}']['speedup']}x, n={len(targets)}")
    return results


# ════════════════════════════════════════════════════════════════════
# E3: Attack Effectiveness — B=1 and B=5 (Ai_Attacking)
# ════════════════════════════════════════════════════════════════════
def run_E3_attack_effectiveness(oracle, edges, labels, strata, good_nodes, ds, budget=1):
    log(f"E3: Attack effectiveness ({ds}, B={budget})...")
    methods = ['AI Oracle', 'Random', 'Degree', 'PageRank', 'Betweenness', 'Expert',
               'PRBCD', 'InfluenceEdit', 'NodeInjection']
    rng = np.random.RandomState(42)
    ex = set((edges[0, i].item(), edges[1, i].item()) for i in range(edges.shape[1]))
    results = {}
    for sn, targets in strata.items():
        if not targets:
            continue
        shifts = {m: [] for m in methods}
        for t in targets:
            br = oracle.compute_reputation(edges, labels, t)
            if br < 0.1:
                continue
            ai = oracle.rank_new_edges(edges, labels, t, sign='distrust',
                                       top_k=budget, max_candidates=80)
            shifts['AI Oracle'].append(
                float(oracle.score_edge_set(edges, labels, [s for s, _ in ai], t, 'distrust')))
            an = list(range(oracle.num_nodes))
            rng.shuffle(an)
            rc = [n for n in an if n != t and (n, t) not in ex][:budget]
            shifts['Random'].append(score_attack(oracle, edges, labels, t, rc))
            shifts['Degree'].append(
                score_attack(oracle, edges, labels, t,
                             oracle.baseline_degree_ranking(edges, t, top_k=budget)))
            shifts['PageRank'].append(
                score_attack(oracle, edges, labels, t,
                             oracle.baseline_pagerank_ranking(edges, t, top_k=budget)))
            shifts['Betweenness'].append(
                score_attack(oracle, edges, labels, t,
                             oracle.baseline_betweenness_ranking(edges, t, top_k=budget)))
            shifts['Expert'].append(
                score_attack(oracle, edges, labels, t,
                             oracle.baseline_expert_heuristic_ranking(
                                 edges, labels, t, good_nodes, top_k=budget)))
            # ── SOTA 2024–26 baselines (utility comparison) ──
            shifts['PRBCD'].append(
                score_attack(oracle, edges, labels, t,
                             baselines_sota.rank_prbcd(
                                 oracle, edges, labels, t, budget=budget,
                                 top_k=budget, max_candidates=80)))
            shifts['InfluenceEdit'].append(
                score_attack(oracle, edges, labels, t,
                             baselines_sota.rank_influence_edge_edit(
                                 oracle, edges, labels, t,
                                 top_k=budget, max_candidates=80)))
            shifts['NodeInjection'].append(
                score_attack(oracle, edges, labels, t,
                             baselines_sota.rank_node_injection(
                                 oracle, edges, labels, t,
                                 top_k=budget, max_candidates=80)))
        sr = {}
        ai_arr = np.array(shifts['AI Oracle'])
        for m in methods:
            a = np.array(shifts[m]) if shifts[m] else np.array([0.])
            e = {'mean': float(a.mean()), 'std': float(a.std()), 'n': len(a)}
            if m != 'AI Oracle' and len(a) >= 5:
                e['wilcoxon_p'] = wilcoxon_test(ai_arr[:len(a)], a)
                e['cohens_d'] = cohens_d(ai_arr[:len(a)], a)
            sr[m] = e
        results[sn] = sr
        log(f"  {sn}: AI={ai_arr.mean():.5f}, Expert={np.mean(shifts['Expert']):.5f}")
    for sn in ['moderate', 'low']:
        if sn not in results:
            continue
        sr = results[sn]
        fig, ax = plt.subplots(figsize=(7, 4))
        ms = [abs(sr.get(m, {}).get('mean', 0)) for m in methods]
        ss = [sr.get(m, {}).get('std', 0) for m in methods]
        ax.bar(range(len(methods)), ms, yerr=ss,
               color=[COL.get(m, '#999') for m in methods],
               edgecolor='w', lw=.5, capsize=3)
        for i, m in enumerate(methods):
            p = sr.get(m, {}).get('wilcoxon_p', 1)
            if p < 0.01:
                ax.text(i, ms[i] + ss[i] + .001, '**', ha='center', fontsize=12)
            elif p < 0.05:
                ax.text(i, ms[i] + ss[i] + .001, '*', ha='center', fontsize=12)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
        ax.set_ylabel('Mean |ΔR|')
        ax.set_title(f'{ds} {sn.capitalize()} (B={budget})')
        ax.grid(True, axis='y', alpha=.3)
        fig.tight_layout()
        fig.savefig(f'{FDIR}/e3_{ds}_{sn}_B{budget}.pdf')
        fig.savefig(f'{FDIR}/e3_{ds}_{sn}_B{budget}.png')
        plt.close()
    return results


# ════════════════════════════════════════════════════════════════════
# E4: Budget Efficiency Curves (Ai_Attacking)
# ════════════════════════════════════════════════════════════════════
def run_E4_budget_efficiency(oracle, edges, labels, strata, ds):
    log(f"E4: Budget efficiency ({ds})...")
    budgets = [1, 3, 5, 7, 10]
    targets = (strata.get('moderate', []) + strata.get('low', []))[:5]
    if not targets:
        return {}
    methods = ['AI Oracle', 'Degree', 'PageRank']
    res = {m: {b: [] for b in budgets} for m in methods}
    for ti, t in enumerate(targets):
        log(f"  Target {ti + 1}/{len(targets)}")
        for b in budgets:
            ai = oracle.rank_new_edges(edges, labels, t, sign='distrust',
                                       top_k=b, max_candidates=80)
            res['AI Oracle'][b].append(
                float(oracle.score_edge_set(edges, labels, [s for s, _ in ai], t, 'distrust')))
            res['Degree'][b].append(
                score_attack(oracle, edges, labels, t,
                             oracle.baseline_degree_ranking(edges, t, top_k=b)))
            res['PageRank'][b].append(
                score_attack(oracle, edges, labels, t,
                             oracle.baseline_pagerank_ranking(edges, t, top_k=b)))
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        mn = [abs(np.mean(res[m][b])) for b in budgets]
        sd = [np.std(res[m][b]) for b in budgets]
        ax.errorbar(budgets, mn, yerr=sd, marker='o', label=m,
                    color=COL.get(m, '#333'), lw=2, ms=5, capsize=3)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Mean |ΔR|')
    ax.set_title(f'Budget Efficiency — {ds}')
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{FDIR}/e4_budget_{ds}.pdf')
    fig.savefig(f'{FDIR}/e4_budget_{ds}.png')
    plt.close()
    return {m: {str(b): {'mean': float(np.mean(res[m][b])),
                          'std': float(np.std(res[m][b]))}
                for b in budgets}
            for m in methods}


# ════════════════════════════════════════════════════════════════════
# E5: Surrogate Transfer / Black-Box (both papers)
# ════════════════════════════════════════════════════════════════════
def run_E5_transfer(defender_oracle, edges, labels, strata, ds, device):
    log(f"E5: Black-box transfer ({ds})...")
    surr = train_model(ds, seed=123)
    surr_oracle = ReputationAttackOracle(surr['model'], surr['index_list'], surr['device'])
    targets = (strata.get('moderate', []) + strata.get('low', []))[:5]
    if not targets:
        return {}
    whitebox_shifts, blackbox_shifts = [], []
    for t in targets:
        wb = defender_oracle.rank_new_edges(edges, labels, t,
                                            sign='distrust', top_k=3, max_candidates=80)
        wb_shift = defender_oracle.score_edge_set(edges, labels, [s for s, _ in wb], t, 'distrust')
        bb = surr_oracle.rank_new_edges(edges, labels, t,
                                        sign='distrust', top_k=3, max_candidates=80)
        bb_nodes = [n for n, _ in bb]
        bb_shift = score_attack(defender_oracle, edges, labels, t, bb_nodes)
        whitebox_shifts.append(wb_shift)
        blackbox_shifts.append(bb_shift)
        log(f"  Node {t}: WB={wb_shift:.5f}, BB={bb_shift:.5f}")
    wb_mag = abs(np.mean(whitebox_shifts))
    bb_mag = abs(np.mean(blackbox_shifts))
    transfer = bb_mag / max(wb_mag, 1e-12)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [wb_mag, bb_mag], color=['#E63946', '#9B59B6'], edgecolor='w')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['White-Box\n(Own Model)', 'Black-Box\n(Surrogate)'])
    ax.set_ylabel('Mean |ΔR|')
    ax.set_title(f'Attack Transferability — {ds}')
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{FDIR}/e5_transfer_{ds}.pdf')
    fig.savefig(f'{FDIR}/e5_transfer_{ds}.png')
    plt.close()
    del surr_oracle, surr
    gc.collect()
    torch.cuda.empty_cache()
    return {'wb_mean': float(np.mean(whitebox_shifts)),
            'bb_mean': float(np.mean(blackbox_shifts)),
            'transfer_ratio': float(transfer),
            'n': len(targets)}


# ════════════════════════════════════════════════════════════════════
# E6: Stronger Model + Degree-Matched Stealth (Grill)
# ════════════════════════════════════════════════════════════════════
def run_E6_stronger_stealth(oracle, edges, labels, ds, good_nodes):
    log(f"E6: Stronger model + stealth ({ds})...")

    # --- Stronger model (deeper GDTE) ---
    log(f"  E6a: Stronger model ({ds})")
    deeper = train_model(ds, layers=[64, 64, 64], epochs=100)
    deeper_oracle = ReputationAttackOracle(deeper['model'], deeper['index_list'], deeper['device'])
    strata_deep = get_targets(deeper_oracle, deeper['edges'], deeper['labels'], per_stratum=5)
    tgts_deep = (strata_deep.get('moderate', []) + strata_deep.get('low', []))[:10]
    deeper_drs = []
    for v in tgts_deep:
        sources = [s for s, _ in deeper_oracle.rank_new_edges(
            deeper['edges'], deeper['labels'], v,
            sign='distrust', top_k=3, max_candidates=50)]
        deeper_drs.append(float(deeper_oracle.score_edge_set(
            deeper['edges'], deeper['labels'], sources, v, 'distrust')))
    stronger_result = {
        'default_auc': oracle.model.AUC if hasattr(oracle.model, 'AUC') else 0.0,
        'deeper_auc': deeper['perf']['AUC'],
        'deeper_dr': float(np.mean(deeper_drs)) if deeper_drs else 0.0,
    }
    log(f"    Deeper AUC={deeper['perf']['AUC']:.3f}, Deeper ΔR={np.mean(deeper_drs):+.4f}")
    del deeper, deeper_oracle
    gc.collect()
    torch.cuda.empty_cache()

    # --- Degree-matched stealth ---
    log(f"  E6b: Degree-matched stealth ({ds})")
    strata_stealth = get_targets(oracle, edges, labels, per_stratum=5)
    src = edges[0].cpu().numpy()
    out_deg = np.bincount(src, minlength=oracle.num_nodes)
    med_deg = float(np.median(out_deg[out_deg > 0]))

    stealth_targets = []
    for sn in ['moderate', 'low']:
        stealth_targets.extend(strata_stealth.get(sn, []))

    stealth_results = {}
    for budget in [1, 3, 5]:
        unconst_drs = []
        dm_drs = []
        for v in stealth_targets:
            sources_u = [s for s, _ in oracle.rank_new_edges(
                edges, labels, v, sign='distrust', top_k=budget, max_candidates=50)]
            dr_u = oracle.score_edge_set(edges, labels, sources_u, v, 'distrust')
            unconst_drs.append(float(dr_u))

            all_cands = oracle.rank_new_edges(edges, labels, v,
                                              sign='distrust', top_k=50, max_candidates=100)
            dm_sources = []
            for s, sc in all_cands:
                if 0.5 * med_deg <= out_deg[s] <= 2.0 * med_deg:
                    dm_sources.append(s)
                    if len(dm_sources) >= budget:
                        break
            if len(dm_sources) < budget:
                dm_sources = [s for s, _ in all_cands[:budget]]
            dr_dm = oracle.score_edge_set(edges, labels, dm_sources, v, 'distrust')
            dm_drs.append(float(dr_dm))

        stealth_results[f'B{budget}'] = {
            'unconstrained': float(np.mean(unconst_drs)),
            'degree_matched': float(np.mean(dm_drs)),
        }
        log(f"    B={budget}: Unconst={np.mean(unconst_drs):+.4f}, DM={np.mean(dm_drs):+.4f}")

    return {'stronger_model': stronger_result, 'degree_matched_stealth': stealth_results}


# ════════════════════════════════════════════════════════════════════
# E7: Influence Distribution — Gini, Degree Correlation (Ai_Attacking)
# ════════════════════════════════════════════════════════════════════
def run_E7_influence_distribution(oracle, edges, labels, strata, ds):
    log(f"E7: Influence distribution ({ds})...")
    targets = (strata.get('moderate', []) + strata.get('low', []))[:3]
    if not targets:
        return {}
    all_mags = []
    sa = edges[0].cpu().numpy()
    dm = np.bincount(sa, minlength=oracle.num_nodes)
    for t in targets:
        im = oracle.compute_influence_map(edges, labels, t, top_k=20)
        all_mags.append(im.influence_magnitudes.cpu().numpy())
    combined = np.concatenate(all_mags)
    cd = np.tile(dm[sa], len(all_mags))[:len(combined)]
    sm = np.sort(combined)
    n = len(sm)
    gini = (2 * np.sum(np.arange(1, n + 1) * sm)) / (n * np.sum(sm) + 1e-12) - (n + 1) / n
    corr, _ = sp_stats.pearsonr(combined, cd[:len(combined)].astype(float))
    th = np.percentile(combined, 95)
    t5 = combined[combined >= th].sum() / (combined.sum() + 1e-12)

    fig = plt.figure(figsize=(14, 4))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, figure=fig, wspace=.35)
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(combined[combined > 0], bins=50, color='#E63946', alpha=.8, edgecolor='w')
    ax1.set_xlabel('Influence Magnitude')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Distribution')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=.3)
    ax2 = fig.add_subplot(gs[1])
    idx = np.random.choice(len(combined), min(2000, len(combined)), replace=False)
    ax2.scatter(cd[idx], combined[idx], alpha=.3, s=10, c='#457B9D')
    ax2.set_xlabel('Source Degree')
    ax2.set_ylabel('Influence')
    ax2.set_title(f'(b) vs Degree (r={corr:.3f})')
    ax2.grid(True, alpha=.3)
    ax3 = fig.add_subplot(gs[2])
    sc = np.cumsum(sm) / (sm.sum() + 1e-12)
    pct = np.linspace(0, 100, len(sc))
    ax3.plot(pct, sc * 100, color='#1D3557', lw=2)
    ax3.axvline(x=95, color='#E63946', ls='--', alpha=.5, label=f'Top5%={t5:.0%}')
    ax3.set_xlabel('Percentile')
    ax3.set_ylabel('Cumul. Influence (%)')
    ax3.set_title('(c) Concentration')
    ax3.legend()
    ax3.grid(True, alpha=.3)
    fig.savefig(f'{FDIR}/e7_influence_{ds}.pdf', bbox_inches='tight')
    fig.savefig(f'{FDIR}/e7_influence_{ds}.png', bbox_inches='tight')
    plt.close()
    return {'gini': float(gini), 'degree_corr': float(corr), 'top5_share': float(t5)}


# ════════════════════════════════════════════════════════════════════
# E8: Attack Edge Detection — AUC 0.96 (Ai_Attacking)
# ════════════════════════════════════════════════════════════════════
def run_E8_detection(oracle, edges, labels, strata, ds):
    log(f"E8: Defence — attack edge detection ({ds})...")
    targets = (strata.get('moderate', []) + strata.get('low', []))[:3]
    if not targets:
        return {}
    all_atk_edges, all_atk_labels = [], []
    for t in targets:
        ranked = oracle.rank_new_edges(edges, labels, t,
                                       sign='distrust', top_k=5, max_candidates=80)
        for src, _ in ranked:
            all_atk_edges.append([src, t])
            all_atk_labels.append([0.0, 1.0])
    if not all_atk_edges:
        return {}
    ae = torch.tensor(all_atk_edges, dtype=torch.long, device=oracle.device).T
    al = torch.tensor(all_atk_labels, device=oracle.device)
    res = detect_attack_edges(oracle, edges, labels, ae, al)
    log(f"  Detection AUC={res['auc']:.3f}, Acc={res['accuracy']:.3f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'Attack Detection ROC — {ds} (AUC={res["auc"]:.3f})')
    ax.annotate(f'AUC = {res["auc"]:.3f}', xy=(.6, .3), fontsize=14,
                color='#E63946', fontweight='bold')
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{FDIR}/e8_defence_{ds}.pdf')
    fig.savefig(f'{FDIR}/e8_defence_{ds}.png')
    plt.close()
    return res


# ════════════════════════════════════════════════════════════════════
# E9-E10: Sparsity Ablation + Bootstrap CI (Grill)
# ════════════════════════════════════════════════════════════════════
def run_E9_sparsity_ablation(oracle, edges, labels, ds):
    log(f"E9: Sparsity ablation ({ds})...")
    strata = get_targets(oracle, edges, labels, per_stratum=5)
    targets = strata.get('moderate', []) + strata.get('low', [])

    src_np = edges[0].cpu().numpy()
    in_deg = np.bincount(edges[1].cpu().numpy(), minlength=oracle.num_nodes)
    good_nodes = identify_good_nodes_from_tensors(edges, labels, num_nodes=oracle.num_nodes)

    results = []
    for i, v in enumerate(targets):
        deg = int(in_deg[v])
        ai_cands = oracle.rank_new_edges(edges, labels, v,
                                         sign='distrust', top_k=3, max_candidates=50)
        ai_sources = [s for s, _ in ai_cands]
        ai_dr = oracle.score_edge_set(edges, labels, ai_sources, v, 'distrust')
        exp_sources = oracle.baseline_expert_heuristic_ranking(
            edges, labels, v, good_nodes, top_k=3)
        exp_dr = oracle.score_edge_set(edges, labels, exp_sources, v, 'distrust')
        results.append({
            'target': int(v),
            'in_degree': deg,
            'grail_dr': float(ai_dr),
            'expert_dr': float(exp_dr),
        })
        if (i + 1) % 10 == 0:
            log(f"  Processed {i + 1}/{len(targets)} targets...")
    return results


def run_E10_bootstrap_ci(oracle, edges, labels, ds, good_nodes):
    log(f"E10: Bootstrap CI ({ds})...")
    strata = get_targets(oracle, edges, labels, per_stratum=5)
    all_t = []
    for sn in ['moderate', 'low']:
        all_t.extend(strata.get(sn, []))

    results = {}
    for budget in [1, 3, 5]:
        all_shifts = []
        for v in all_t:
            ai = oracle.rank_new_edges(edges, labels, v,
                                       sign='distrust', top_k=budget, max_candidates=80)
            all_shifts.append(
                oracle.score_edge_set(edges, labels, [s for s, _ in ai], v, 'distrust'))
        if all_shifts:
            mean, lo, hi = bootstrap_ci(all_shifts)
            results[f'B{budget}'] = {'mean': mean, 'ci_lo': lo, 'ci_hi': hi}
            log(f"  B={budget}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return results


# ════════════════════════════════════════════════════════════════════
# Dataset Runner
# ════════════════════════════════════════════════════════════════════
def run_dataset(ds):
    log(f"\n{'=' * 60}\n  DATASET: {ds}\n{'=' * 60}")
    trained = train_model(ds)
    model, device = trained['model'], trained['device']
    edges, labels = trained['edges'], trained['labels']
    oracle = ReputationAttackOracle(model, trained['index_list'], device)
    good_nodes = identify_good_nodes_from_tensors(edges, labels)
    strata = oracle.select_stratified_targets(edges, labels, per_stratum=10)
    for k, v in strata.items():
        log(f"  {k}: {len(v)} nodes")

    r = {
        'performance': trained['performance'],
        'strata_sizes': {k: len(v) for k, v in strata.items()},
    }
    r['E1'] = run_E1(oracle, edges, labels, strata, ds)
    r['E2'] = run_E2_grad_vs_fwd(oracle, edges, labels, ds)
    r['E3_B1'] = run_E3_attack_effectiveness(oracle, edges, labels, strata, good_nodes, ds, budget=1)
    r['E3_B5'] = run_E3_attack_effectiveness(oracle, edges, labels, strata, good_nodes, ds, budget=5)
    r['E4'] = run_E4_budget_efficiency(oracle, edges, labels, strata, ds)
    r['E5'] = run_E5_transfer(oracle, edges, labels, strata, ds, device)
    r['E6'] = run_E6_stronger_stealth(oracle, edges, labels, ds, good_nodes)
    r['E7'] = run_E7_influence_distribution(oracle, edges, labels, strata, ds)
    r['E8'] = run_E8_detection(oracle, edges, labels, strata, ds)
    r['E9'] = run_E9_sparsity_ablation(oracle, edges, labels, ds)
    r['E10'] = run_E10_bootstrap_ci(oracle, edges, labels, ds, good_nodes)
    del model, oracle
    gc.collect()
    torch.cuda.empty_cache()
    return r
