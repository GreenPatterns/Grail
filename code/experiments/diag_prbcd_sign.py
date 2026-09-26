"""Diagnostic: is PRBCD's projected-gradient direction correct, or is the
differentiable relaxation genuinely anti-aligned with executable impact?

For a few targets, compares the TRUE counterfactual delta-R (via score_edge_set)
of edge sets chosen by:
  Grad       -- GRAIL-Grad single-pass gradient ranking
  Fwd        -- GRAIL-Fwd exact counterfactual reference
  PRBCD_min  -- our PRBCD (descends reputation; the E3 setting)
  PRBCD_max  -- PRBCD with the step direction flipped

Interpretation: if PRBCD_min ~ Grad (both weak/positive) AND PRBCD_max is NOT
strongly negative, then the relaxation is misaligned in BOTH directions, i.e.
the "counterproductive SOTA attack" result is the genuine anti-alignment
phenomenon (consistent with the paper's GRAIL-Grad +0.03), not a sign bug.
If instead PRBCD_max ~ Fwd (strongly negative), the descent sign was wrong.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode import baselines_sota


def main():
    ds = os.environ.get("GRAIL_DATASETS", "otc").split(",")[0]
    B = int(os.environ.get("GRAIL_BUDGET", "5"))
    tr = train_model(ds, seed=42)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    e, l = tr["edges"], tr["labels"]
    strata = get_targets(o, e, l, per_stratum=3)
    targets = [t for ts in strata.values() for t in ts]
    pool = list(range(o.num_nodes))
    rows = {"Grad": [], "Fwd": [], "PRBCD_min": [], "PRBCD_max": []}
    for t in targets:
        grad = [s for s, _ in o.rank_new_edges_gradient(e, l, t, candidate_sources=pool, top_k=B, max_candidates=80)]
        fwd = [s for s, _ in o.rank_new_edges(e, l, t, candidate_sources=pool, top_k=B, max_candidates=80)]
        pmin = baselines_sota.rank_prbcd(o, e, l, t, candidate_sources=pool, budget=B, top_k=B, max_candidates=80, minimize=True)
        pmax = baselines_sota.rank_prbcd(o, e, l, t, candidate_sources=pool, budget=B, top_k=B, max_candidates=80, minimize=False)
        rows["Grad"].append(o.score_edge_set(e, l, grad, t, "distrust"))
        rows["Fwd"].append(o.score_edge_set(e, l, fwd, t, "distrust"))
        rows["PRBCD_min"].append(o.score_edge_set(e, l, pmin, t, "distrust"))
        rows["PRBCD_max"].append(o.score_edge_set(e, l, pmax, t, "distrust"))
    log(f"=== PRBCD sign diagnostic ({ds}, B={B}, n={len(targets)} targets) ===")
    for k, v in rows.items():
        log(f"  {k:10s} mean true ΔR = {np.mean(v):+.5f}   (more negative = stronger attack)")
    print("DIAG DONE")


if __name__ == "__main__":
    main()
