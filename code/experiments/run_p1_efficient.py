"""
Phase 1 — EFFICIENT guided attack at the correct (processed) placement.

Phase 0 showed: at the correct placement a single/few distrust edges are devastating,
but the per-candidate counterfactual costs O(C) forward passes. Phase 1 asks: can a
GRADIENT method recover that ranking far more cheaply?

Methods compared (all at processed placement; ΔR via score_edge_set_processed):
  Counterfactual (GOLD)  : O(C) forward, exact.
  Batch-grad (trust)     : ONE backward, all C candidates, non-saturating trust background.
  Batch-grad (neutral)   : ONE backward, neutral [0.5,0.5] background.
  IntegratedGrad         : ~16 backward (joint), trust->distrust label path.
  Greedy-grad            : B backward, re-rank after each committed edge.
  Expert (behavioral)    : no model access.
  Random                 : weak baseline.
  PRBCD (adapted)        : Geisler et al. 2021, relaxed distrust-label strengths (baselines_sota).
  Influence edit         : Heo et al. 2025 influence-function edge edits (baselines_sota).
  Node injection         : low-degree source pool ranked by processed counterfactual (baselines_sota).

Reports per method: mean realized top-B ΔR, Spearman vs counterfactual, cost (passes),
flip rate. Success = an efficient gradient method matches counterfactual ΔR at << cost.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p1_efficient.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch
from scipy import stats as sp_stats

from project_paths import RESULTS_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors
from mycode import baselines_sota

SEED = 42
C_POOL = 60
B = 5
IG_STEPS = 16


def spearman(da, db):
    ks = [k for k in da if k in db]
    if len(ks) < 3:
        return float('nan')
    a, b = [da[k] for k in ks], [db[k] for k in ks]
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(sp_stats.spearmanr(a, b)[0])


def topB(scores, b):
    return [s for s, _ in sorted(scores.items(), key=lambda x: x[1])[:b]]


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 1 EFFICIENT — {ds}\n{'='*64}")
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    good = identify_good_nodes_from_tensors(edges, labels, num_nodes=o.num_nodes)
    rng = np.random.RandomState(SEED)
    strata = o.select_stratified_targets(edges, labels, per_stratum=8)
    targets = (strata.get("moderate", []) + strata.get("low", []))[:16]

    methods = ["counterfactual", "batch_trust", "batch_neutral", "ig", "greedy",
               "expert", "random", "prbcd", "influence", "node_injection"]
    dr = {m: [] for m in methods}
    dr_app = {m: [] for m in methods}  # SAME top-B set, APPENDED placement (Q7)
    rho = {m: [] for m in ["batch_trust", "batch_neutral", "ig"]}
    flips = {m: 0 for m in methods}
    n_elig = 0

    for ti, t in enumerate(targets):
        existing = set(edges[0, edges[1] == t].tolist())
        pool = [int(s) for s in rng.permutation(o.num_nodes)
                if s != t and s not in existing][:C_POOL]
        rb = o.compute_reputation(edges, labels, t)
        elig = rb >= 0.5
        if elig:
            n_elig += 1

        cf = o.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
        bt = o.score_candidates_gradient_batch_processed(edges, labels, t, pool, baseline='trust')
        bn = o.score_candidates_gradient_batch_processed(edges, labels, t, pool, baseline='neutral')
        ig = o.score_candidates_integrated_gradient_processed(edges, labels, t, pool, steps=IG_STEPS)
        rho["batch_trust"].append(spearman(bt, cf))
        rho["batch_neutral"].append(spearman(bn, cf))
        rho["ig"].append(spearman(ig, cf))

        sets = {
            "counterfactual": topB(cf, B),
            "batch_trust": topB(bt, B),
            "batch_neutral": topB(bn, B),
            "ig": topB(ig, B),
            "greedy": o.rank_new_edges_greedy_processed(edges, labels, t, pool, B),
            "expert": [s for s in o.baseline_expert_heuristic_ranking(edges, labels, t, good, top_k=B)
                       if s in pool][:B] or o.baseline_expert_heuristic_ranking(edges, labels, t, good, top_k=B),
            "random": pool[:B],
            # published attack families, adapted to reputation; every ranker
            # sees the SAME pool, and its top-B is scored at the processed placement
            "prbcd": baselines_sota.rank_prbcd(
                o, edges, labels, t, candidate_sources=pool, budget=B,
                max_candidates=C_POOL, top_k=B)[:B],
            "influence": baselines_sota.rank_influence_edge_edit(
                o, edges, labels, t, candidate_sources=pool,
                max_candidates=C_POOL, top_k=B)[:B],
            "node_injection": baselines_sota.rank_node_injection(
                o, edges, labels, t, candidate_sources=pool,
                max_candidates=C_POOL, top_k=B)[:B],
        }
        for m, srcs in sets.items():
            d, _, ra = o.score_edge_set_processed(edges, labels, srcs, t, 'distrust', return_after=True)
            dr[m].append(d)
            # Same selected edges, appended placement: shows every method is
            # null at the wrong placement, end-to-end (not just by argument).
            dr_app[m].append(o.score_edge_set(edges, labels, srcs, t, 'distrust'))
            if elig and ra < 0.5:
                flips[m] += 1
        log(f"  [{ti+1}/{len(targets)}] t={t}  cf={dr['counterfactual'][-1]:+.3f} "
            f"batchT={dr['batch_trust'][-1]:+.3f} ig={dr['ig'][-1]:+.3f} "
            f"greedy={dr['greedy'][-1]:+.3f} expert={dr['expert'][-1]:+.3f}")

    def m(x):
        x = [v for v in x if v == v]
        return float(np.mean(x)) if x else float('nan')

    costs = {"counterfactual": f"{C_POOL} fwd", "batch_trust": "1 bwd",
             "batch_neutral": "1 bwd", "ig": f"{IG_STEPS} bwd", "greedy": f"{B} bwd",
             "expert": "0 (model-free)", "random": "0",
             "prbcd": "30 bwd", "influence": "2 bwd", "node_injection": "~15 fwd"}
    res = {"n_targets": len(targets), "n_eligible": n_elig, "C": C_POOL, "B": B,
           "mean_dr": {k: m(v) for k, v in dr.items()},
           "mean_dr_appended": {k: m(v) for k, v in dr_app.items()},
           "flip_rate": {k: flips[k] / max(n_elig, 1) for k in methods},
           "rho_vs_cf": {k: m(v) for k, v in rho.items()},
           "cost": costs}
    log(f"\n  SUMMARY [{ds}] (n={len(targets)}, C={C_POOL}, B={B})")
    for k in methods:
        extra = f"  rho_cf={res['rho_vs_cf'][k]:+.3f}" if k in rho else ""
        log(f"    {k:>14}: ΔR_proc={res['mean_dr'][k]:+.4f}  ΔR_app={res['mean_dr_appended'][k]:+.4f}  "
            f"flip={res['flip_rate'][k]:.0%}  cost={costs[k]}{extra}")

    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    outp = RESULTS_DIR / "unified" / os.environ.get("GRAIL_P1_OUT", "p1_efficient.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {outp}")


if __name__ == "__main__":
    main()
