"""
Phase 3 — margin / threshold robustness (addresses "are you just flipping users
already at 0.5?").

Reports, per dataset and stratum:
  * clean reputation distribution (mean, sd, quartiles);
  * distance-to-threshold before attack (R_clean - 0.5);
  * single-edge (B=1) and B=5 flip rate as a function of clean margin band;
  * flip rate under alternative gate thresholds {0.4, 0.5, 0.6};
  * Spearman(clean in-degree, |dR|) to separate margin from degree effects.

All at the correct processed placement, optimized counterfactual sources.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_margin.py
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

SEED = 42
PER_STRATUM = 25
CAND = 60
THRESHOLDS = [0.4, 0.5, 0.6]


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 3 MARGIN/THRESHOLD — {ds}\n{'='*64}")
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    rng = np.random.RandomState(SEED)
    in_deg = np.bincount(edges[1].cpu().numpy(), minlength=o.num_nodes)
    strata = o.select_stratified_targets(edges, labels, per_stratum=PER_STRATUM)

    out = {"clean_dist": {}, "by_threshold": {}, "by_margin": {}, "indeg_rho": {}}
    # pooled records across strata for margin-banded analysis
    rec = []  # (stratum, r_clean, dr_B1, dr_B5, in_deg)
    for stratum in ["high", "moderate", "low"]:
        victims = strata.get(stratum, [])
        if not victims:
            continue
        cleans = []
        for v in victims:
            rb = o.compute_reputation(edges, labels, v)
            cleans.append(rb)
            existing = set(edges[0, edges[1] == v].tolist())
            pool = [int(s) for s in rng.permutation(o.num_nodes)
                    if s != v and s not in existing][:CAND]
            sc = o.score_candidates_counterfactual_processed(edges, labels, v, pool, sign='distrust')
            opt = [s for s, _ in sorted(sc.items(), key=lambda x: x[1])]
            _, _, ra1 = o.score_edge_set_processed(edges, labels, opt[:1], v, 'distrust', return_after=True)
            _, _, ra5 = o.score_edge_set_processed(edges, labels, opt[:5], v, 'distrust', return_after=True)
            rec.append((stratum, rb, ra1, ra5, int(in_deg[v])))
        c = np.array(cleans)
        out["clean_dist"][stratum] = {
            "mean": float(c.mean()), "sd": float(c.std()),
            "q25": float(np.percentile(c, 25)), "median": float(np.median(c)),
            "q75": float(np.percentile(c, 75)),
            "mean_margin_to_0.5": float((c - 0.5).mean()),
            "min": float(c.min()), "max": float(c.max()),
        }

    arr_clean = np.array([r[1] for r in rec])
    arr_ra1 = np.array([r[2] for r in rec])
    arr_ra5 = np.array([r[3] for r in rec])
    arr_indeg = np.array([r[4] for r in rec])

    # flip rate at alternative thresholds (eligible = clean above threshold)
    for thr in THRESHOLDS:
        elig = arr_clean >= thr
        n = int(elig.sum())
        f1 = float(np.mean(arr_ra1[elig] < thr)) if n else float("nan")
        f5 = float(np.mean(arr_ra5[elig] < thr)) if n else float("nan")
        out["by_threshold"][str(thr)] = {"n_eligible": n, "flip_B1": f1, "flip_B5": f5}
        log(f"  threshold {thr}: eligible n={n}  flip B1={f1:.0%}  flip B5={f5:.0%}")

    # flip rate (gate 0.5) by clean-margin band, to test margin dependence
    elig = arr_clean >= 0.5
    margin = arr_clean[elig] - 0.5
    ra1e, ra5e = arr_ra1[elig], arr_ra5[elig]
    bands = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.0)]
    for lo, hi in bands:
        m = (margin >= lo) & (margin < hi)
        n = int(m.sum())
        f1 = float(np.mean(ra1e[m] < 0.5)) if n else float("nan")
        f5 = float(np.mean(ra5e[m] < 0.5)) if n else float("nan")
        out["by_margin"][f"{lo:.2f}-{hi:.2f}"] = {"n": n, "flip_B1": f1, "flip_B5": f5}
        log(f"  margin [{lo:.2f},{hi:.2f}): n={n}  flip B1={f1:.0%}  flip B5={f5:.0%}")

    dr1 = arr_ra1 - arr_clean
    rho = float(sp_stats.spearmanr(arr_indeg, np.abs(dr1))[0]) if len(rec) > 2 else float("nan")
    out["indeg_rho"] = {"spearman_indeg_absdrB1": rho,
                        "indeg_min": int(arr_indeg.min()), "indeg_max": int(arr_indeg.max())}
    log(f"  Spearman(in-degree, |dR_B1|) = {rho:+.3f}  (in-deg {arr_indeg.min()}-{arr_indeg.max()})")

    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    outp = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p3_margin.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {outp}")


if __name__ == "__main__":
    main()
