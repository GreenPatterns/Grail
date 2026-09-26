"""E4 (cost-normalized utility): attacker compute cost vs. attack utility.

Shows the gradient/SOTA attacks buy no cost-effective advantage: the only
effective methods are the model-free Expert (negligible cost) and the exact
counterfactual Fwd (expensive); Grad is cheap-but-counterproductive and PRBCD
is expensive-but-counterproductive. Reports, per method, mean delta-R and mean
wall-clock to produce the attack edge set over stratified targets.

Runs on GPU: CUDA_VISIBLE_DEVICES=0 python experiments/run_e4_cost.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import numpy as np
import torch

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors
from mycode import baselines_sota


def _timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0


def main():
    B = int(os.environ.get("GRAIL_BUDGET", "5"))
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    methods = ["Fwd", "Grad", "Expert", "Degree", "PRBCD", "InfluenceEdit", "NodeInjection"]
    out = {}
    for ds in datasets:
        tr = train_model(ds, seed=42)
        o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
        e, l = tr["edges"], tr["labels"]
        strata = get_targets(o, e, l, per_stratum=10)
        gn = identify_good_nodes_from_tensors(e, l, num_nodes=o.num_nodes)
        targets = [t for ts in strata.values() for t in ts]
        rec = {m: {"dr": [], "sec": []} for m in methods}
        for t in targets:
            if o.compute_reputation(e, l, t) < 0.1:
                continue
            runners = {
                "Fwd": lambda: [s for s, _ in o.rank_new_edges(e, l, t, top_k=B, max_candidates=80)],
                "Grad": lambda: [s for s, _ in o.rank_new_edges_gradient(e, l, t, top_k=B, max_candidates=80)],
                "Expert": lambda: o.baseline_expert_heuristic_ranking(e, l, t, gn, top_k=B),
                "Degree": lambda: o.baseline_degree_ranking(e, t, top_k=B),
                "PRBCD": lambda: baselines_sota.rank_prbcd(o, e, l, t, budget=B, top_k=B, max_candidates=80),
                "InfluenceEdit": lambda: baselines_sota.rank_influence_edge_edit(o, e, l, t, top_k=B, max_candidates=80),
                "NodeInjection": lambda: baselines_sota.rank_node_injection(o, e, l, t, top_k=B, max_candidates=80),
            }
            for m in methods:
                sources, sec = _timed(runners[m])
                rec[m]["dr"].append(o.score_edge_set(e, l, sources, t, "distrust"))
                rec[m]["sec"].append(sec)
        out[ds] = {
            m: {
                "mean_dr": float(np.mean(v["dr"])) if v["dr"] else 0.0,
                "mean_sec": float(np.mean(v["sec"])) if v["sec"] else 0.0,
                "n": len(v["dr"]),
            }
            for m, v in rec.items()
        }
        log(f"{ds}: " + ", ".join(
            f"{m}(ΔR={out[ds][m]['mean_dr']:+.3f}, {out[ds][m]['mean_sec']*1000:.0f}ms)" for m in methods))
    p = RESULTS_DIR / "unified" / "e4_cost.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Saved {p}")


if __name__ == "__main__":
    main()
