"""
Leakage-free flip table (reviewer blocker #2, headline configuration).

Reproduces the main flip table's setup -- OPTIMIZED (per-candidate processed
counterfactual) sources, per stratum, budgets {1,3,5} -- but reports BOTH:

  flip_total     : naive processed reputation over ALL raters (injected + legit)
                   == the paper's current number
  flip_preexist  : leakage-free reputation over PRE-EXISTING (legit) raters only,
                   embeddings from enc(G+A). Removes self-edge conditioning and
                   mechanical aggregation; keeps genuine propagation to v*.

If flip_preexist stays high, the headline is a real vulnerability. If it falls,
the naive number was inflated by self-conditioning.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_loo_flip_table.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch
import torch.nn.functional as F

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
BUDGETS = [1, 3, 5]
PER_STRATUM = 25
C_POOL = 50


def _cp(k, n, alpha=0.05):
    from scipy import stats as sp
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else sp.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sp.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (round(float(lo), 3), round(float(hi), 3))


def _p_trust(oracle, z, target, aug_edges, cols):
    src = aug_edges[0, cols]
    feats = torch.cat((z[src], z[target].unsqueeze(0).expand(len(cols), -1)), dim=1)
    return F.softmax(feats @ oracle.model.regression_weights, dim=1)[:, 0]


def _encode(oracle, edges, labels, idx):
    orig = oracle.model.index_list
    try:
        oracle.model.index_list = idx
        with torch.no_grad():
            return oracle._compute_node_embeddings(edges, labels)
    finally:
        oracle.model.index_list = orig


def _pool(oracle, edges, target, C, rng):
    existing = set(edges[0, edges[1] == target].tolist()); existing.add(target)
    uni = [n for n in range(oracle.num_nodes) if n not in existing]
    if len(uni) > C:
        uni = list(rng.choice(uni, C, replace=False))
    return [int(s) for s in uni]


def _variants(oracle, edges, labels, target, srcs):
    """Return (R0, R_full, R_preexist) at processed placement, distrust."""
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

    z0 = _encode(oracle, edges, labels, orig_idx)
    cdst = (edges[1] == target).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        R0 = _p_trust(oracle, z0, target, edges, cdst).mean().item()
    z1 = _encode(oracle, ae, al, aidx)
    with torch.no_grad():
        R_full = _p_trust(oracle, z1, target, ae, dst).mean().item()
        R_pre = _p_trust(oracle, z1, target, ae, pre).mean().item() if len(pre) else float('nan')
    return R0, R_full, R_pre


def run_dataset(ds):
    log(f"\n{'='*60}\n  LEAKAGE-FREE FLIP TABLE — {ds}\n{'='*60}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]
    strata = get_targets(oracle, edges, labels, per_stratum=PER_STRATUM)
    rng = np.random.RandomState(SEED)
    out = {}
    for stratum in os.environ.get("GRAIL_STRATA", "high,moderate,low").split(","):
        ts = strata.get(stratum, [])
        for B in BUDGETS:
            ft, fp, drt, drp = [], [], [], []
            for t in ts:
                pool = _pool(oracle, edges, t, C_POOL, rng)
                if len(pool) < B:
                    continue
                # optimized: rank by processed single-edge counterfactual, take top-B
                cf = oracle.score_candidates_counterfactual_processed(
                    edges, labels, t, pool, sign='distrust')
                srcs = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:B]]
                R0, Rf, Rp = _variants(oracle, edges, labels, t, srcs)
                if R0 != R0 or R0 < 0.5:   # only targets initially above the gate are eligible
                    continue
                drt.append(Rf - R0); drp.append(Rp - R0)
                ft.append(1 if Rf < 0.5 else 0)
                fp.append(1 if (Rp == Rp and Rp < 0.5) else 0)
            n = len(ft)
            out[f"{stratum}_B{B}"] = {
                "n_eligible": n,
                "dR_total": round(float(np.mean(drt)), 4) if drt else None,
                "dR_preexist": round(float(np.mean(drp)), 4) if drp else None,
                "flip_total_pct": round(100 * np.mean(ft), 1) if n else None,
                "flip_preexist_pct": round(100 * np.mean(fp), 1) if n else None,
                "flip_total_ci": _cp(sum(ft), n),
                "flip_preexist_ci": _cp(sum(fp), n),
            }
            r = out[f"{stratum}_B{B}"]
            log(f"  {stratum:8s} B={B}: n={n:2d}  "
                f"flip_total={r['flip_total_pct']}%  flip_preexist={r['flip_preexist_pct']}%  "
                f"(dR {r['dR_total']} / {r['dR_preexist']})")
    del oracle, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_OUT", "loo_flip_table.json"))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
