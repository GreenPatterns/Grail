"""
Phase 0 — LINCHPIN (go/no-go for the USENIX "false sense of safety" thesis).

The Phase-0 diagnosis revealed something stronger than expected: the original
attack inserts the candidate edge PAST the trained snapshot horizon
(index_list[-1], an unprocessed snapshot). There the structural layer never
processes the edge, so (a) its LABEL has zero gradient and (b) the exact
counterfactual ΔR collapses to a label-independent reputation-average effect.
In other words the paper's "distrust injection" never actually injected distrust.

This driver quantifies, across many targets and both datasets:

  A) APPENDED counterfactual (paper's GRAIL-Fwd placement): is it label-independent
     (distrust ΔR ≈ trust ΔR) and weak (mean ΔR ≈ 0)?
  B) PROCESSED counterfactual (edge spliced into the last TRAINED snapshot): is it
     label-dependent and strong (mean ΔR strongly negative)?  -> the true threat.
  C) Does the corrected single-edge LABEL gradient (same processed placement)
     correlate with the PROCESSED counterfactual?  -> gradient guidance is real.

GATE: B shows a large appended->processed damage gap AND C is positive
=> the fear is real and the current paper masked it via placement -> proceed.

Run on GPU (ps env):
  GRAIL_DATASETS=otc,alpha python code/experiments/run_p0_linchpin.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json
import gc
import numpy as np
import torch
from scipy import stats as sp_stats

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle


C_POOL = int(os.environ.get("GRAIL_P0_C", "50"))
B = 5
SEED = 42


def _spearman(d_a, d_b):
    keys = [k for k in d_a if k in d_b]
    if len(keys) < 3:
        return float("nan")
    a = [d_a[k] for k in keys]
    b = [d_b[k] for k in keys]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    r, _ = sp_stats.spearmanr(a, b)
    return float(r)


def _sample_pool(oracle, edges, target, n, rng):
    existing_src = set(edges[0, edges[1] == target].tolist())
    universe = [s for s in range(oracle.num_nodes) if s != target and s not in existing_src]
    if len(universe) > n:
        universe = list(rng.choice(universe, n, replace=False))
    return [int(s) for s in universe]


def _topB_dr_processed(oracle, edges, labels, scores, target):
    """Rank by score (ascending: most-negative first), realize processed top-B ΔR."""
    items = sorted(scores.items(), key=lambda x: x[1])
    srcs = [s for s, _ in items[:B]]
    # Joint processed ΔR via repeated single-edge processed scoring is not additive;
    # we report the mean single-edge processed ΔR of the selected top-B as the proxy.
    proc = oracle.score_candidates_counterfactual_processed(
        edges, labels, target, srcs, sign='distrust')
    return float(np.mean([proc[s] for s in srcs])) if srcs else 0.0


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 0 LINCHPIN — {ds}\n{'='*64}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]

    strata = get_targets(oracle, edges, labels, per_stratum=10)
    targets = strata.get("moderate", []) + strata.get("low", [])
    n_cap = int(os.environ.get("GRAIL_P0_NTARGETS", str(len(targets))))
    targets = targets[:n_cap]
    log(f"  {len(targets)} targets (moderate+low), C={C_POOL}")

    rng = np.random.RandomState(SEED)

    label_indep_appended = []   # mean |distrust - trust| ΔR, appended
    label_dep_processed = []    # mean |distrust - trust| ΔR, processed
    mean_dr_appended = []
    mean_dr_processed = []
    rho_grad_vs_processed = []
    rho_grad_vs_appended = []

    for ti, t in enumerate(targets):
        pool = _sample_pool(oracle, edges, t, C_POOL, rng)
        if len(pool) < 5:
            continue

        # A) appended counterfactual (paper's placement), distrust + trust
        cf_app_dis = dict((int(s), float(d)) for s, d in oracle.rank_new_edges(
            edges, labels, t, candidate_sources=pool, sign='distrust', top_k=len(pool)))
        cf_app_tru = dict((int(s), float(d)) for s, d in oracle.rank_new_edges(
            edges, labels, t, candidate_sources=pool, sign='trust', top_k=len(pool)))

        # B) processed counterfactual (spliced into last trained snapshot)
        cf_proc_dis = oracle.score_candidates_counterfactual_processed(
            edges, labels, t, pool, sign='distrust')
        cf_proc_tru = oracle.score_candidates_counterfactual_processed(
            edges, labels, t, pool, sign='trust')

        # C) corrected single-edge label gradient (same processed placement)
        g_proc = oracle.score_candidates_gradient_single(edges, labels, t, pool, sign='distrust')

        label_indep_appended.append(np.mean([abs(cf_app_dis[s] - cf_app_tru[s]) for s in pool]))
        label_dep_processed.append(np.mean([abs(cf_proc_dis[s] - cf_proc_tru[s]) for s in pool]))
        mean_dr_appended.append(np.mean([cf_app_dis[s] for s in pool]))
        mean_dr_processed.append(np.mean([cf_proc_dis[s] for s in pool]))
        rho_grad_vs_processed.append(_spearman(g_proc, cf_proc_dis))
        rho_grad_vs_appended.append(_spearman(g_proc, cf_app_dis))

        log(f"  [{ti+1}/{len(targets)}] t={t}  "
            f"appended ΔR={mean_dr_appended[-1]:+.4f} (lblΔ={label_indep_appended[-1]:.1e})  "
            f"processed ΔR={mean_dr_processed[-1]:+.4f} (lblΔ={label_dep_processed[-1]:.3f})  "
            f"ρ(grad,proc)={rho_grad_vs_processed[-1]:+.3f}")

    def _m(x):
        x = [v for v in x if v == v]
        return float(np.mean(x)) if x else float("nan")

    res = {
        "n_targets": len(mean_dr_processed),
        "C": C_POOL,
        "appended_mean_dr": _m(mean_dr_appended),
        "processed_mean_dr": _m(mean_dr_processed),
        "appended_label_delta": _m(label_indep_appended),
        "processed_label_delta": _m(label_dep_processed),
        "damage_amplification": (_m(mean_dr_processed) / _m(mean_dr_appended)
                                 if _m(mean_dr_appended) not in (0.0,) else float("nan")),
        "rho_grad_vs_processed": _m(rho_grad_vs_processed),
        "rho_grad_vs_appended": _m(rho_grad_vs_appended),
        "rho_grad_vs_processed_values": [v for v in rho_grad_vs_processed if v == v],
        # clean forward-chaining link prediction of the same trained model (Table 1)
        "clean_perf": {k: trained["perf"].get(k) for k in ("AUC", "AUC_std", "MCC", "MCC_std", "BAcc")},
        "train_edges": int(edges.shape[1]),
    }
    log(f"\n  SUMMARY [{ds}]")
    log(f"    appended  (paper):  mean ΔR={res['appended_mean_dr']:+.4f}   "
        f"label-Δ={res['appended_label_delta']:.2e}  (≈0 => label ignored)")
    log(f"    processed (real) :  mean ΔR={res['processed_mean_dr']:+.4f}   "
        f"label-Δ={res['processed_label_delta']:.3f}  (large => true distrust attack)")
    log(f"    ρ(grad, processed-cf) = {res['rho_grad_vs_processed']:+.3f}   "
        f"ρ(grad, appended-cf) = {res['rho_grad_vs_appended']:+.3f}")

    del trained, oracle
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}

    out = RESULTS_DIR / "unified" / os.environ.get("GRAIL_P0_OUT", "p0_linchpin.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {out}")

    log(f"\n{'='*64}\n  GATE VERDICT\n{'='*64}")
    for ds, r in results.items():
        strong_processed = r["processed_mean_dr"] < 2 * r["appended_mean_dr"] - 1e-6 \
            and r["processed_mean_dr"] < -0.02
        grad_positive = (r["rho_grad_vs_processed"] == r["rho_grad_vs_processed"]) \
            and r["rho_grad_vs_processed"] > 0.05
        passed = strong_processed and grad_positive
        log(f"  {ds}: processed ΔR={r['processed_mean_dr']:+.4f} vs appended "
            f"{r['appended_mean_dr']:+.4f}; ρ(grad,proc)={r['rho_grad_vs_processed']:+.3f} "
            f"=> {'PASS — fear is real & was masked' if passed else 'REVIEW'}")


if __name__ == "__main__":
    main()
