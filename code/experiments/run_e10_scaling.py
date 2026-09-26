"""E10: attack-utility scaling curve on subsampled Epinions.

Full-graph Epinions (131k nodes) OOMs at ~45 GiB on a 48 GiB A6000, so we
subsample induced top-degree subgraphs at N in {1k,5k,10k,30k}, train GDTE on
each, and measure best-method delta-R vs graph size. This directly answers the
"does the (null) result scale?" critique.

Subsamples are regenerated from epinions.csv (the single source of truth) into
data/cyberdata/epn{N}.csv + epn{N}-rating.txt, kept row-aligned by construction.

Runs on GPU: CUDA_VISIBLE_DEVICES=0 python experiments/run_e10_scaling.py
Env knobs: GRAIL_SCALE_N (default "1000,5000,10000,30000").
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import numpy as np
import pandas as pd

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors
from mycode import baselines_sota

DATA_ROOT = os.environ.get(
    "GRAIL_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cyberdata"),
)


def make_epn_subsample(N):
    """Induced top-N-degree subgraph of Epinions, renumbered to 0..N'-1.

    Regenerates BOTH the CSV (source,target,rating,time) and the space-separated
    rating.txt (src dst trust distrust) from the same time-sorted edge list, so
    snapshot boundaries (from the CSV) align with edges (from the rating.txt).
    """
    csv_out = os.path.join(DATA_ROOT, f"epn{N}.csv")
    rt_out = os.path.join(DATA_ROOT, f"epn{N}-rating.txt")
    if os.path.exists(csv_out) and os.path.exists(rt_out):
        return
    df = pd.read_csv(os.path.join(DATA_ROOT, "epinions.csv"))
    df.columns = ["source", "target", "rating", "time"]
    deg = pd.concat([df.source, df.target]).value_counts()
    keep = set(deg.index[:N])
    sub = df[df.source.isin(keep) & df.target.isin(keep)].copy()
    sub = sub.sort_values("time", kind="mergesort").reset_index(drop=True)
    nodes = pd.unique(pd.concat([sub.source, sub.target]))
    remap = {int(n): i for i, n in enumerate(nodes)}
    sub.source = sub.source.map(remap)
    sub.target = sub.target.map(remap)
    sub.time = range(len(sub))
    sub.to_csv(csv_out, index=False)
    with open(rt_out, "w") as f:
        for s, t, r in zip(sub.source.tolist(), sub.target.tolist(), sub.rating.tolist()):
            trust, distrust = (1, 0) if r > 0 else (0, 1)
            f.write(f"{s} {t} {trust} {distrust}\n")
    log(f"  subsample epn{N}: nodes={len(nodes)} edges={len(sub)}")


def measure(ds, budget=5):
    tr = train_model(ds, seed=42)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    e, l = tr["edges"], tr["labels"]
    strata = get_targets(o, e, l, per_stratum=5)
    gn = identify_good_nodes_from_tensors(e, l, num_nodes=o.num_nodes)
    targets = [t for ts in strata.values() for t in ts]
    res = {m: [] for m in ["Fwd", "Expert", "Grad", "PRBCD"]}
    for t in targets:
        if o.compute_reputation(e, l, t) < 0.1:
            continue
        fwd = [s for s, _ in o.rank_new_edges(e, l, t, top_k=budget, max_candidates=80)]
        grad = [s for s, _ in o.rank_new_edges_gradient(e, l, t, top_k=budget, max_candidates=80)]
        prb = baselines_sota.rank_prbcd(o, e, l, t, budget=budget, top_k=budget, max_candidates=80)
        exp = o.baseline_expert_heuristic_ranking(e, l, t, gn, top_k=budget)
        res["Fwd"].append(o.score_edge_set(e, l, fwd, t, "distrust"))
        res["Grad"].append(o.score_edge_set(e, l, grad, t, "distrust"))
        res["PRBCD"].append(o.score_edge_set(e, l, prb, t, "distrust"))
        res["Expert"].append(o.score_edge_set(e, l, exp, t, "distrust"))
    return {
        m: {"mean": float(np.mean(v)) if v else 0.0,
            "std": float(np.std(v)) if v else 0.0, "n": len(v)}
        for m, v in res.items()
    }


def main():
    Ns = [int(x) for x in os.environ.get("GRAIL_SCALE_N", "1000,5000,10000,30000").split(",")]
    p = RESULTS_DIR / "unified" / os.environ.get("GRAIL_E10_OUT", "e10_scaling.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for N in Ns:
        log(f"=== E10 scaling N={N} ===")
        make_epn_subsample(N)
        out[str(N)] = measure(f"epn{N}")
        log(f"  N={N}: " + ", ".join(f"{m}={out[str(N)][m]['mean']:+.4f}" for m in out[str(N)]))
        # write incrementally so a later OOM (large N) can't lose earlier results
        with open(p, "w") as f:
            json.dump(out, f, indent=2, default=str)
    log(f"Saved {p}")


if __name__ == "__main__":
    main()
