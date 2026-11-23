"""
Source selection with both scores (E10 with Check 3; DSN revision).

Reruns E10 (run_p1_efficient.py) with its configuration: seed 42, 16 strata targets (8 moderate
+ 8 low), the same C = 60 candidates per target (RandomState(42) permutation), the same nine
strategies (plus batch_neutral, not in the paper) called in the same order, B = 5 distrust edges.
Each strategy's B edges are scored
  in-window     : R_tot with score_edge_set_processed, as in E10, and R_tot and R_prop with
                  run_loo_flip_table._variants (E4's code: R_prop averages only the target's
                  pre-existing raters, with embeddings recomputed after the attack); the two
                  R_tot values must agree (self-check);
  out-of-window : R_tot with score_edge_set (edges appended after the window), as in E10.
Shares: a strategy's mean shift in percent of the optimized strategy's mean in-window shift
under the same score (out-of-window: of the optimized in-window R_tot shift, as in Fig. S2).
Flips: eligible targets (clean R >= 0.5) whose attacked score is below 0.5, counted for the low
stratum and for all 16 targets, under each score. GPU training is not bit-deterministic, so the
model, and with it the strata, can differ slightly from the published run
(results/unified/p1_sota*.json, which this script leaves untouched).

JSON keys keep E10's names (`counterfactual` = optimized, `expert` = heuristic, `mean_dr` =
in-window total shift, `mean_dr_appended` = out-of-window total shift) and add `mean_dr_prop`,
`share_{tot,prop,appended}_pct`, `flips` and the per-target `rows`.

Outputs: results/p1_sota_prop_{otc,alpha,edgfull,epn}.json, rewritten after every target.
Run: CUDA_VISIBLE_DEVICES=0 GRAIL_DATASETS=otc,alpha python code/experiments/run_p1_prop.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch

from project_paths import RESULTS_DIR, PROJECT_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors
from mycode import baselines_sota
from run_p1_efficient import SEED, C_POOL, B, IG_STEPS, spearman, topB
from run_loo_flip_table import _variants

TAU = 0.5
OPT = "counterfactual"
METHODS = ["counterfactual", "batch_trust", "batch_neutral", "ig", "greedy",
           "expert", "random", "prbcd", "influence", "node_injection"]
RHO = ["batch_trust", "batch_neutral", "ig"]
OUT_NAME = {"otc": "otc", "alpha": "alpha", "edg60000": "edgfull", "epn30000": "epn"}


def select_sets(o, edges, labels, t, pool, good):
    """E10's source sets for one target, in run_p1_efficient's call order."""
    cf = o.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
    bt = o.score_candidates_gradient_batch_processed(edges, labels, t, pool, baseline='trust')
    bn = o.score_candidates_gradient_batch_processed(edges, labels, t, pool, baseline='neutral')
    ig = o.score_candidates_integrated_gradient_processed(edges, labels, t, pool, steps=IG_STEPS)
    rho = {"batch_trust": spearman(bt, cf), "batch_neutral": spearman(bn, cf), "ig": spearman(ig, cf)}
    greedy = o.rank_new_edges_greedy_processed(edges, labels, t, pool, B)
    # E10 calls the (deterministic) heuristic twice when none of its B best is a candidate
    heur = o.baseline_expert_heuristic_ranking(edges, labels, t, good, top_k=B)
    sets = {
        "counterfactual": topB(cf, B), "batch_trust": topB(bt, B), "batch_neutral": topB(bn, B),
        "ig": topB(ig, B), "greedy": greedy,
        "expert": [s for s in heur if s in pool][:B] or heur,
        "random": pool[:B],
        "prbcd": baselines_sota.rank_prbcd(o, edges, labels, t, candidate_sources=pool, budget=B,
                                           max_candidates=C_POOL, top_k=B)[:B],
        "influence": baselines_sota.rank_influence_edge_edit(o, edges, labels, t, candidate_sources=pool,
                                                             max_candidates=C_POOL, top_k=B)[:B],
        "node_injection": baselines_sota.rank_node_injection(o, edges, labels, t, candidate_sources=pool,
                                                             max_candidates=C_POOL, top_k=B)[:B],
    }
    return sets, rho


def score_set(o, edges, labels, t, srcs):
    d, _, r_tot = o.score_edge_set_processed(edges, labels, srcs, t, 'distrust', return_after=True)
    r0, r_tot_v, r_prop = _variants(o, edges, labels, t, srcs)
    assert abs(r_tot_v - r_tot) < 1e-4, (t, srcs, r_tot, r_tot_v)   # same in-window placement
    return {"srcs": [int(s) for s in srcs], "R_tot": r_tot, "R_prop": r_prop,
            "dR_tot": d, "dR_prop": r_prop - r0,
            "dR_app": float(o.score_edge_set(edges, labels, srcs, t, 'distrust'))}


def summarize(rows):
    mean = {k: {m: float(np.nanmean([r["s"][m][k] for r in rows])) for m in METHODS}
            for k in ("dR_tot", "dR_prop", "dR_app")}
    md, mp, ma = mean["dR_tot"], mean["dR_prop"], mean["dR_app"]
    elig = [r for r in rows if r["eligible"]]
    low = [r for r in elig if r["stratum"] == "low"]

    def k(rs, m, score):
        return int(sum(r["s"][m][score] < TAU for r in rs))

    flips = {m: {"low_tot": k(low, m, "R_tot"), "low_prop": k(low, m, "R_prop"), "n_low_eligible": len(low),
                 "all_tot": k(elig, m, "R_tot"), "all_prop": k(elig, m, "R_prop"), "n_eligible": len(elig)}
             for m in METHODS}
    return {"n_targets": len(rows), "n_eligible": len(elig),
            "mean_dr": md, "mean_dr_prop": mp, "mean_dr_appended": ma,
            "share_tot_pct": {m: 100 * md[m] / md[OPT] for m in METHODS},
            "share_prop_pct": {m: 100 * mp[m] / mp[OPT] for m in METHODS},
            "share_appended_pct": {m: 100 * ma[m] / md[OPT] for m in METHODS},
            "flips": flips,
            "flip_rate": {m: flips[m]["all_tot"] / max(len(elig), 1) for m in METHODS},
            "rho_vs_cf": {m: float(np.nanmean([r["rho"][m] for r in rows])) for m in RHO}}


def run_dataset(ds):
    log(f"\n{'=' * 64}\n  E10 WITH BOTH SCORES — {ds}\n{'=' * 64}")
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    good = identify_good_nodes_from_tensors(edges, labels, num_nodes=o.num_nodes)
    rng = np.random.RandomState(SEED)
    strata = o.select_stratified_targets(edges, labels, per_stratum=8)
    targets = (strata.get("moderate", []) + strata.get("low", []))[:16]
    low = set(strata.get("low", []))
    path = RESULTS_DIR / f"p1_sota_prop_{OUT_NAME.get(ds, ds)}.json"
    head = {"dataset": ds, "seed": SEED, "C": C_POOL, "B": B, "tau": TAU,
            "perf": {k: tr["perf"].get(k) for k in ("AUC", "MCC")},
            "targets_moderate": [int(t) for t in targets if t not in low],
            "targets_low": [int(t) for t in targets if t in low]}
    rows = []
    for ti, t in enumerate(targets):
        existing = set(edges[0, edges[1] == t].tolist())
        pool = [int(s) for s in rng.permutation(o.num_nodes) if s != t and s not in existing][:C_POOL]
        rb = o.compute_reputation(edges, labels, t)
        sets, rho = select_sets(o, edges, labels, t, pool, good)
        row = {"t": int(t), "stratum": "low" if t in low else "moderate", "R0": rb,
               "eligible": bool(rb >= TAU), "rho": rho,
               "s": {m: score_set(o, edges, labels, t, srcs) for m, srcs in sets.items()}}
        rows.append(row)
        with open(path, "w") as f:
            json.dump({**head, **summarize(rows), "rows": rows}, f, indent=1)
        s = row["s"]
        log(f"  [{ti + 1}/{len(targets)}] t={t} {row['stratum']:8s} R0={rb:.3f}  tot/prop: "
            + "  ".join(f"{n}={s[m]['dR_tot']:+.3f}/{s[m]['dR_prop']:+.3f}"
                        for m, n in (("counterfactual", "cf"), ("expert", "expert"),
                                     ("random", "random"), ("prbcd", "prbcd"))))
    S = summarize(rows)
    log(f"\n  SUMMARY [{ds}] (n={len(rows)}, eligible={S['n_eligible']}, C={C_POOL}, B={B})")
    for m in METHODS:
        fl = S["flips"][m]
        log(f"    {m:>14}: dR_tot={S['mean_dr'][m]:+.4f} ({S['share_tot_pct'][m]:5.1f}%)  "
            f"dR_prop={S['mean_dr_prop'][m]:+.4f} ({S['share_prop_pct'][m]:5.1f}%)  "
            f"dR_app={S['mean_dr_appended'][m]:+.4f}  low flips T/P={fl['low_tot']}/{fl['low_prop']} "
            f"of {fl['n_low_eligible']}  all T/P={fl['all_tot']}/{fl['all_prop']} of {fl['n_eligible']}")
    log(f"Wrote {path.relative_to(PROJECT_DIR)}")
    del tr, o; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in os.environ.get("GRAIL_DATASETS", "otc,alpha").split(","):
        run_dataset(ds.strip())
