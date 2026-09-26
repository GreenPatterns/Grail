"""
Reputation-function baselines, paired: every function is scored on the SAME targets,
the SAME candidate pools and the SAME in-window distrust injections in ONE run, so the
comparison carries no run-to-run training noise. It merges run_robust_functional.py
(robust GNN aggregation) and run_graph_reputation.py (non-GNN reputation):

  GDTE mean / median / trimmed (10%) : aggregate of the incoming raters' predicted trust
      (mycode/reputation.py), total effect (all raters) and propagation-only
      (pre-existing raters, attacked embeddings; Eq. 3 of the paper)
  FG      : Fairness-Goodness goodness (Kumar et al. 2016) on the same +1/-1 labels,
            reputation (g + 1) / 2 (mycode/reputation_systems.py)
  fraction, beta, wilson : label-space aggregators (run_repbaselines_calib._label_reps)

Attackers (top-B of one C=50 pool per target):
  random     : B random candidates
  mean_opt   : processed single-edge counterfactual under the GDTE mean
  median_opt : the same counterfactual under the GDTE median (adaptive against the median)
  fg_opt     : FG's own single-edge counterfactual (adaptive against FG)

Per (stratum, attacker, function, B): eligible targets (clean reputation >= 0.5 under that
function), flip rate with a Clopper-Pearson interval, and mean shift. Per-target values are
kept for paired analyses.

Run: GRAIL_DATASETS=otc,alpha GRAIL_OUT=reputation_baselines.json \
     python code/experiments/run_reputation_baselines.py
     GRAIL_FG_WARM=1 warm-starts every attacked FG solve from the clean fixed point (same
     fixed point, far fewer iterations; used for the 689k-edge dated Epinions).
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
from mycode.reputation import aggregate_reputation
from mycode.reputation_systems import fairness_goodness
from run_loo_flip_table import _p_trust, _encode, _pool, _cp
from run_repbaselines_calib import _label_reps
from run_robust_functional import _vectors, _cf_ranking

SEED = 42
BUDGETS = [1, 3, 5]
PER_STRATUM = 25
C_POOL = 50
AGGS = ["mean", "median", "trimmed"]
ATTACKERS = ["random", "mean_opt", "median_opt", "fg_opt"]
FUNCS = ([f"gdte_{a}_{e}" for a in AGGS for e in ("total", "prop")]
         + ["fg", "fraction", "beta", "wilson"])


def _fg_goodness_after(src, dst, w, n, target, srcs, init=None):
    """FG goodness of every node after adding distrust edges srcs -> target."""
    s2 = np.concatenate([src, np.asarray(srcs, dtype=np.int64)])
    d2 = np.concatenate([dst, np.full(len(srcs), target, dtype=np.int64)])
    w2 = np.concatenate([w, -np.ones(len(srcs))])
    return fairness_goodness(s2, d2, w2, n, init=init)[1]


def _clean_reps(p0, g0, t, k_in, n_in):
    lab = _label_reps(k_in, n_in)
    reps = {f"gdte_{a}_{e}": aggregate_reputation(p0, a).item() for a in AGGS for e in ("total", "prop")}
    reps.update(fg=(g0[t] + 1) / 2, fraction=lab["signed_mean"], beta=lab["beta_bayes"], wilson=lab["wilson"])
    return reps


def _attacked_reps(p_full, p_pre, g1, t, k_in, n_in, B):
    lab = _label_reps(k_in, n_in + B)
    reps = {}
    for a in AGGS:
        reps[f"gdte_{a}_total"] = aggregate_reputation(p_full, a).item()
        reps[f"gdte_{a}_prop"] = aggregate_reputation(p_pre, a).item() if p_pre is not None else float('nan')
    reps.update(fg=(g1[t] + 1) / 2, fraction=lab["signed_mean"], beta=lab["beta_bayes"], wilson=lab["wilson"])
    return reps


def run_dataset(ds):
    log(f"\n{'='*60}\n  REPUTATION BASELINES (paired) — {ds}\n{'='*60}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]
    n = oracle.num_nodes
    src = edges[0].cpu().numpy(); dst = edges[1].cpu().numpy()
    trust = labels[:, 0].cpu().numpy() >= 0.5
    w = np.where(trust, 1.0, -1.0)
    f0, g0 = fairness_goodness(src, dst, w, n)
    warm = (f0, g0) if os.environ.get("GRAIL_FG_WARM") == "1" else None
    strata = get_targets(oracle, edges, labels, per_stratum=PER_STRATUM)
    rng = np.random.RandomState(SEED)
    z0 = _encode(oracle, edges, labels, list(oracle.model.index_list))
    out, per_target = {}, []
    for stratum in os.environ.get("GRAIL_STRATA", "low,moderate").split(","):
        cells = {(a, f, B): {"flip": [], "dr": []} for a in ATTACKERS for f in FUNCS for B in BUDGETS}
        for ti, t in enumerate(strata.get(stratum, [])):
            pool = _pool(oracle, edges, t, C_POOL, rng)
            if len(pool) < max(BUDGETS):
                continue
            fg_single = {s: _fg_goodness_after(src, dst, w, n, t, [s], warm)[t] for s in pool}
            order = {"random": list(pool),
                     "mean_opt": _cf_ranking(oracle, edges, labels, t, pool, "mean"),
                     "median_opt": _cf_ranking(oracle, edges, labels, t, pool, "median"),
                     "fg_opt": [s for s, _ in sorted(fg_single.items(), key=lambda x: x[1])]}
            k_in, n_in = int(((dst == t) & trust).sum()), int((dst == t).sum())
            cdst = (edges[1] == t).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                p0 = _p_trust(oracle, z0, t, edges, cdst)
            r0 = _clean_reps(p0, g0, t, k_in, n_in)
            row = {"stratum": stratum, "target": int(t), "clean": r0, "attacked": {}}
            for a in ATTACKERS:
                for B in BUDGETS:
                    srcs = order[a][:B]
                    p_full, p_pre = _vectors(oracle, edges, labels, t, srcs, z0, p0)
                    g1 = _fg_goodness_after(src, dst, w, n, t, srcs, warm)
                    r1 = _attacked_reps(p_full, p_pre, g1, t, k_in, n_in, B)
                    row["attacked"][f"{a}|B{B}"] = r1
                    for f in FUNCS:
                        if r0[f] < 0.5 or r1[f] != r1[f]:
                            continue
                        c = cells[(a, f, B)]
                        c["flip"].append(int(r1[f] < 0.5)); c["dr"].append(r1[f] - r0[f])
            per_target.append(row)
            if (ti + 1) % 5 == 0:
                log(f"  [{stratum} {ti+1}/{len(strata.get(stratum, []))}] t={t}")
        for (a, f, B), c in cells.items():
            k, m = sum(c["flip"]), len(c["flip"])
            out[f"{stratum}|{a}|{f}|B{B}"] = {
                "n_eligible": m, "flip_pct": round(100 * k / m, 1) if m else None,
                "flip_ci": _cp(k, m), "dR": round(float(np.mean(c["dr"])), 4) if c["dr"] else None}
        for a in ATTACKERS:
            log(f"  {stratum} {a:>10} B1 flip %: "
                + str({f: out[f'{stratum}|{a}|{f}|B1']['flip_pct'] for f in FUNCS}))
    out["per_target"] = per_target
    del trained, oracle
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_OUT", "reputation_baselines.json"))
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    log(f"\nSaved {path}")
