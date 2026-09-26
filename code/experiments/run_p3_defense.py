"""
Phase 3 — rater-history weighting DEFENSE (supports the S7 recommendation).

The paper recommends down-weighting low-history raters rather than recency. Here
we measure that defense directly. A naive attacker (no model access) injects B
distrust edges at the correct processed placement, choosing sources two ways:
  * random non-neighbors  (established accounts),
  * low-history non-neighbors (the fresh / Sybil-account threat: lowest
    out-degree nodes, a proxy for newly created accounts), and
  * high-history non-neighbors (the ADAPTIVE attacker, in the sense of
    Mujkanovic et al. 2022: it knows the rule and rates only from accounts the
    rule weights fully, out-degree >= h0).
We then score the SAME attack under two reputation functionals:
  * 'mean'           (Eq. 1, undefended), and
  * 'rater_weighted' (weight u by min(1, hist(u)/h0), the defense).
Flip rate = fraction of eligible (R>=0.5) victims pushed below 0.5.

Defense works if rater-weighting sharply cuts the flip rate of the low-history
attacker; honestly, it should help less against established-account sources.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_defense.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch
from scipy.stats import beta

from project_paths import RESULTS_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
PER_STRATUM = 16
H0 = 5.0  # rater-history saturation: accounts with >=5 ratings get full weight


def pick_sources(source_kind, cands, outdeg, v, B, seed=SEED, h0=H0):
    """Attacker source choice for victim v. 'high_history' is the adaptive
    attacker: it only uses accounts the rater-history rule weights fully."""
    if source_kind == 'random':
        perm = np.random.RandomState(seed + int(v)).permutation(cands)
        return [int(s) for s in perm[:B]]
    if source_kind == 'low_history':  # lowest out-degree (fresh-account proxy)
        return sorted(cands, key=lambda s: outdeg[s])[:B]
    if source_kind == 'high_history':
        full = [s for s in cands if outdeg[s] >= h0]
        perm = np.random.RandomState(seed + int(v)).permutation(full)
        return [int(s) for s in perm[:B]]
    raise ValueError(f"unknown source_kind {source_kind!r}")


def cp(k, n):
    if n == 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 3 DEFENSE (rater-history weighting) — {ds}\n{'='*64}")
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"],
                               rep_mode='mean', rater_h0=H0)
    edges, labels = tr["edges"], tr["labels"]
    rng = np.random.RandomState(SEED)
    outdeg = torch.bincount(edges[0], minlength=o.num_nodes).cpu().numpy()
    strata = o.select_stratified_targets(edges, labels, per_stratum=PER_STRATUM)

    def flip_cell(victims, source_kind, B, mode):
        o.rep_mode = mode
        flips, elig, drs = 0, 0, []
        for v in victims:
            existing = set(edges[0, edges[1] == v].tolist())
            cands = [s for s in range(o.num_nodes) if s != v and s not in existing]
            srcs = pick_sources(source_kind, cands, outdeg, v, B)
            rb = o.compute_reputation(edges, labels, v)
            _, _, ra = o.score_edge_set_processed(edges, labels, srcs, v, 'distrust', return_after=True)
            drs.append(ra - rb)
            if rb >= 0.5:
                elig += 1
                flips += int(ra < 0.5)
        return {"dr": float(np.mean(drs)), "flip_rate": flips / max(elig, 1),
                "k_n": [flips, elig], "flip_ci": cp(flips, elig)}

    out = {}
    for stratum in ["low", "moderate"]:
        victims = strata.get(stratum, [])
        if not victims:
            continue
        for source_kind in ["random", "low_history", "high_history"]:
            for B in (1, 5):
                key = f"{stratum}_{source_kind}_B{B}"
                cell = {m: flip_cell(victims, source_kind, B, m)
                        for m in ("mean", "rater_weighted")}
                out[key] = cell
                log(f"  {key:>26}: mean flip={cell['mean']['flip_rate']:.0%} "
                    f"{cell['mean']['flip_ci']}  ->  rater-weighted flip="
                    f"{cell['rater_weighted']['flip_rate']:.0%} {cell['rater_weighted']['flip_ci']}")
    o.rep_mode = 'mean'
    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    out = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p3_defense.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {out}")
    log(f"\n{'='*64}\n  DEFENSE HEADLINE (low stratum, B=1)\n{'='*64}")
    for ds, r in results.items():
        for sk in ["random", "low_history", "high_history"]:
            k = f"low_{sk}_B1"
            if k in r:
                log(f"  {ds} {sk:>11}: undefended {r[k]['mean']['flip_rate']:.0%} "
                    f"-> rater-weighted {r[k]['rater_weighted']['flip_rate']:.0%}")


if __name__ == "__main__":
    main()
