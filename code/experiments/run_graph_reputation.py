"""
Non-GNN reputation baseline: is the single-edge exposure specific to GNN reputation?

Same targets (low and moderate GDTE strata, <= 25 each), same C=50 candidate pools and
the same in-window distrust injections, scored by
  GDTE      : mean predicted trust, total effect and propagation-only (Eq. 3)
  FG        : Fairness-Goodness goodness (Kumar et al. 2016), mycode/reputation_systems.py,
              on the SAME +1/-1 labels GDTE sees; reputation (g + 1) / 2, gate 0.5
  fraction, beta, wilson : the label-space aggregators of the paper's Table 7
for three attackers:
  random   : B random candidates
  gnn_opt  : top-B by GDTE's processed single-edge counterfactual (transferred to FG)
  fg_opt   : top-B by FG's own single-edge counterfactual (adaptive against FG)
Per (stratum, attacker, B, function): eligible targets (clean reputation >= 0.5 under that
function), flip rate with Clopper-Pearson interval, mean shift, and, for FG and GDTE-free
comparison, the mean rank shift of the target among all users (gate-free).

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_graph_reputation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.reputation_systems import fairness_goodness
from run_loo_flip_table import _pool, _variants, _cp
from run_repbaselines_calib import _label_reps

SEED = 42
BUDGETS = [1, 3, 5]
PER_STRATUM = 25
C_POOL = 50
ATTACKERS = ["random", "gnn_opt", "fg_opt"]
FUNCS = ["gdte_total", "gdte_prop", "fg", "fraction", "beta", "wilson"]


def _rank_of(scores, node):
    """1 = highest score."""
    return int((scores > scores[node]).sum()) + 1


def run_dataset(ds):
    log(f"\n{'='*60}\n  GRAPH REPUTATION BASELINE — {ds}\n{'='*60}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]
    n = oracle.num_nodes
    src = edges[0].cpu().numpy(); dst = edges[1].cpu().numpy()
    w = np.where(labels[:, 0].cpu().numpy() >= 0.5, 1.0, -1.0)
    trust = labels[:, 0].cpu().numpy() >= 0.5
    _, g0 = fairness_goodness(src, dst, w, n)
    strata = get_targets(oracle, edges, labels, per_stratum=PER_STRATUM)
    rng = np.random.RandomState(SEED)

    def fg_with(t, srcs):
        s2 = np.concatenate([src, np.asarray(srcs, dtype=np.int64)])
        d2 = np.concatenate([dst, np.full(len(srcs), t, dtype=np.int64)])
        w2 = np.concatenate([w, -np.ones(len(srcs))])
        return fairness_goodness(s2, d2, w2, n)[1]

    out = {}
    for stratum in os.environ.get("GRAIL_STRATA", "low,moderate").split(","):
        cells = {(a, f, B): {"flip": [], "dr": [], "rank": []}
                 for a in ATTACKERS for f in FUNCS for B in BUDGETS}
        for ti, t in enumerate(strata.get(stratum, [])):
            pool = _pool(oracle, edges, t, C_POOL, rng)
            if len(pool) < max(BUDGETS):
                continue
            cf = oracle.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
            fg_single = {s: fg_with(t, [s])[t] for s in pool}
            order = {"random": list(pool),
                     "gnn_opt": [s for s, _ in sorted(cf.items(), key=lambda x: x[1])],
                     "fg_opt": [s for s, _ in sorted(fg_single.items(), key=lambda x: x[1])]}
            k_in = int(((dst == t) & trust).sum()); n_in = int((dst == t).sum())
            lab0 = _label_reps(k_in, n_in)
            r0_fg = (g0[t] + 1) / 2
            rank0_fg = _rank_of(g0, t)
            for a in ATTACKERS:
                for B in BUDGETS:
                    srcs = order[a][:B]
                    R0, Rf, Rp = _variants(oracle, edges, labels, t, srcs)
                    g1 = fg_with(t, srcs)
                    r1_fg = (g1[t] + 1) / 2
                    lab1 = _label_reps(k_in, n_in + B)
                    pairs = {"gdte_total": (R0, Rf), "gdte_prop": (R0, Rp), "fg": (r0_fg, r1_fg),
                             "fraction": (lab0["signed_mean"], lab1["signed_mean"]),
                             "beta": (lab0["beta_bayes"], lab1["beta_bayes"]),
                             "wilson": (lab0["wilson"], lab1["wilson"])}
                    for f, (r0, r1) in pairs.items():
                        if r0 != r0 or r1 != r1 or r0 < 0.5:
                            continue
                        c = cells[(a, f, B)]
                        c["flip"].append(int(r1 < 0.5)); c["dr"].append(r1 - r0)
                        if f == "fg":
                            c["rank"].append(_rank_of(g1, t) - rank0_fg)
            if (ti + 1) % 5 == 0:
                log(f"  [{stratum} {ti+1}/{len(strata.get(stratum, []))}] t={t}")
        for (a, f, B), c in cells.items():
            k, m = sum(c["flip"]), len(c["flip"])
            out[f"{stratum}|{a}|{f}|B{B}"] = {
                "n_eligible": m, "flip_pct": round(100 * k / m, 1) if m else None, "flip_ci": _cp(k, m),
                "dR": round(float(np.mean(c["dr"])), 4) if c["dr"] else None,
                "rank_shift": round(float(np.mean(c["rank"])), 1) if c["rank"] else None,
            }
        for a in ATTACKERS:
            row = {f: out[f"{stratum}|{a}|{f}|B1"]["flip_pct"] for f in FUNCS}
            log(f"  {stratum} {a:>8} B1 flip %: {row}")
    del trained, oracle
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_OUT", "graph_reputation.json"))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved {path}")
