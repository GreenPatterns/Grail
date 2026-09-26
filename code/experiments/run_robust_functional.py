"""
Robust-aggregation baseline: does a robust reputation functional blunt the attack?

The paper's reputation is the MEAN of the incoming raters' predicted trust (Eq. 1). A
classic defense against a few bad raters is a robust aggregate. This runner scores the
same in-window distrust injections under three functionals (mycode/reputation.py):
  mean     (Eq. 1)
  median
  trimmed  (10% trimmed mean)
for three attackers, following Mujkanovic et al. (2022) for the adaptive one:
  mean_opt   : top-B sources by the processed single-edge counterfactual under the MEAN
  median_opt : top-B sources by the same counterfactual under the MEDIAN (adaptive)
  random     : B random non-neighbor sources
Each (attacker, functional) pair reports the total-effect and propagation-only flip rate
(eligible: clean reputation >= 0.5 under that functional) and mean shift.

Pools (C=50) are drawn once per target, and both counterfactual rankings are computed
once per target and reused for every budget B in {1,3,5}.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_robust_functional.py
     (GRAIL_STRATA=low,moderate, GRAIL_OUT=robust_functional.json by default)
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
from run_loo_flip_table import _p_trust, _encode, _pool, _cp

SEED = 42
BUDGETS = [1, 3, 5]
PER_STRATUM = 25
C_POOL = 50
MODES = ["mean", "median", "trimmed"]
ATTACKERS = ["mean_opt", "median_opt", "random"]


def _vectors(oracle, edges, labels, target, srcs, z0, p0):
    """Predicted-trust vectors after injecting distrust edges from srcs at the processed
    placement: (all raters incl. injected, pre-existing raters only)."""
    B = len(srcs)
    T = oracle.model.args.train_time_slots
    split = oracle.model.index_list[T - 1] + 1
    orig_idx = list(oracle.model.index_list)
    ne = torch.tensor([list(srcs), [target] * B], dtype=torch.long, device=oracle.device)
    nl = torch.tensor([[0.0, 1.0]] * B, dtype=torch.float, device=oracle.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], dim=1)
    al = torch.cat([labels[:split], nl, labels[split:]], dim=0)
    aidx = [b + B if i >= T - 1 else b for i, b in enumerate(orig_idx)]
    dst = (ae[1] == target).nonzero(as_tuple=True)[0]
    inj = set(range(split, split + B))
    pre = torch.tensor([c for c in dst.tolist() if c not in inj], dtype=torch.long, device=oracle.device)
    z1 = _encode(oracle, ae, al, aidx)
    with torch.no_grad():
        p_full = _p_trust(oracle, z1, target, ae, dst)
        p_pre = _p_trust(oracle, z1, target, ae, pre) if len(pre) else None
    return p_full, p_pre


def _cf_ranking(oracle, edges, labels, target, pool, mode):
    prev = oracle.rep_mode
    oracle.rep_mode = mode
    try:
        cf = oracle.score_candidates_counterfactual_processed(edges, labels, target, pool, sign='distrust')
    finally:
        oracle.rep_mode = prev
    return [s for s, _ in sorted(cf.items(), key=lambda x: x[1])]


def run_dataset(ds):
    log(f"\n{'='*60}\n  ROBUST FUNCTIONAL — {ds}\n{'='*60}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]
    strata = get_targets(oracle, edges, labels, per_stratum=PER_STRATUM)
    rng = np.random.RandomState(SEED)
    z0 = _encode(oracle, edges, labels, list(oracle.model.index_list))
    out = {}
    for stratum in os.environ.get("GRAIL_STRATA", "low,moderate").split(","):
        cells = {(a, m, B): {"ft": [], "fp": [], "dt": [], "dp": []}
                 for a in ATTACKERS for m in MODES for B in BUDGETS}
        for ti, t in enumerate(strata.get(stratum, [])):
            pool = _pool(oracle, edges, t, C_POOL, rng)
            if len(pool) < max(BUDGETS):
                continue
            order = {"mean_opt": _cf_ranking(oracle, edges, labels, t, pool, "mean"),
                     "median_opt": _cf_ranking(oracle, edges, labels, t, pool, "median"),
                     "random": list(pool)}
            cdst = (edges[1] == t).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                p0 = _p_trust(oracle, z0, t, edges, cdst)
            r0 = {m: aggregate_reputation(p0, m).item() for m in MODES}
            for a in ATTACKERS:
                for B in BUDGETS:
                    p_full, p_pre = _vectors(oracle, edges, labels, t, order[a][:B], z0, p0)
                    for m in MODES:
                        if r0[m] < 0.5:
                            continue
                        rf = aggregate_reputation(p_full, m).item()
                        rp = aggregate_reputation(p_pre, m).item() if p_pre is not None else float('nan')
                        c = cells[(a, m, B)]
                        c["ft"].append(int(rf < 0.5)); c["dt"].append(rf - r0[m])
                        if rp == rp:
                            c["fp"].append(int(rp < 0.5)); c["dp"].append(rp - r0[m])
            if (ti + 1) % 5 == 0:
                log(f"  [{stratum} {ti+1}/{len(strata.get(stratum, []))}] t={t}")
        for (a, m, B), c in cells.items():
            n, k_t, k_p = len(c["ft"]), sum(c["ft"]), sum(c["fp"])
            out[f"{stratum}|{a}|{m}|B{B}"] = {
                "n_eligible": n,
                "flip_total_pct": round(100 * k_t / n, 1) if n else None,
                "flip_total_ci": _cp(k_t, n),
                "flip_prop_pct": round(100 * k_p / len(c["fp"]), 1) if c["fp"] else None,
                "flip_prop_ci": _cp(k_p, len(c["fp"])),
                "dR_total": round(float(np.mean(c["dt"])), 4) if c["dt"] else None,
                "dR_prop": round(float(np.mean(c["dp"])), 4) if c["dp"] else None,
            }
        for m in MODES:
            row = [out[f"{stratum}|{a}|{m}|B1"]["flip_prop_pct"] for a in ATTACKERS]
            log(f"  {stratum} {m:>8} B1 prop-only flip (mean_opt, median_opt, random): {row}")
    del trained, oracle
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_OUT", "robust_functional.json"))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved {path}")
