"""
Hierarchical / seed-clustered re-analysis of the headline contrasts (reviewer #5).

The node-level Wilcoxon test treats targets as independent, but targets share a
graph and a trained model, so the effective sample size is closer to the number
of seeds. This script recomputes each headline contrast across five seeds and
reports, per contrast:
  - median and Cliff's delta (rank dominance effect size on the paired diffs),
  - a seed-CLUSTERED bootstrap 95% CI on the median (the seed is the cluster),
  - per-seed medians (direction consistency) and a seed-level sign-test p
    (unit = seed, n=5), which is deliberately conservative.

Contrasts (B=1, pooled strata):
  appended_vs_inwindow = dr_full_dis - dr_appended
  naive_vs_prop        = dr_full_dis - dr_legit_dis
  distrust_vs_trust    = dr_full_dis - dr_full_tru
  prop_vs_zero         = dr_legit_dis
  appended_vs_zero     = dr_appended

Run: GRAIL_DATASETS=otc,alpha CUDA_VISIBLE_DEVICES=0 \
     python code/experiments/run_stats_hier.py
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
from run_stats_rigor import _legit, _pool

SEEDS = [int(x) for x in os.environ.get("GRAIL_SEEDS", "42,1,2,3,4").split(",")]  # genuine seeds since the seed-bug fix
DATASETS = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")


def cliffs_delta(d):
    d = np.asarray([x for x in d if x == x], float)
    n = len(d)
    if n == 0:
        return float('nan')
    return float((np.sum(d > 0) - np.sum(d < 0)) / n)   # dominance vs 0 (paired)


def seed_cluster_ci(per_seed_vals, n_boot=3000):
    """Bootstrap over seeds (clusters): resample the seeds, pool their diffs, median."""
    S = len(per_seed_vals)
    if S == 0:
        return (float('nan'), float('nan'))
    rng = np.random.RandomState(0)
    meds = []
    for _ in range(n_boot):
        idx = rng.randint(0, S, S)
        pooled = np.concatenate([per_seed_vals[i] for i in idx]) if S else np.array([])
        if len(pooled):
            meds.append(float(np.median(pooled)))
    return (round(float(np.percentile(meds, 2.5)), 4),
            round(float(np.percentile(meds, 97.5)), 4))


def sign_test_p(seed_medians):
    """Two-sided sign test on seed-level medians vs 0 (unit = seed, n=5)."""
    from scipy import stats as sp
    m = np.asarray([x for x in seed_medians if x == x], float)
    n = len(m)
    if n == 0:
        return float('nan')
    k = int(np.sum(m > 0))
    n_nz = int(np.sum(m != 0))
    if n_nz == 0:
        return 1.0
    p = 2 * sp.binom.cdf(min(k, n_nz - k), n_nz, 0.5)
    return float(min(1.0, p))


def collect_seed(ds, seed):
    trained = train_model(ds, seed=seed)
    orc = ReputationAttackOracle(trained['model'], trained['index_list'], trained['device'])
    edges, labels = trained['edges'], trained['labels']
    strata = get_targets(orc, edges, labels, per_stratum=25)
    rng = np.random.RandomState(1234 + seed)
    rows = []
    for stratum in ('high', 'moderate', 'low'):
        for t in strata.get(stratum, []):
            pool = _pool(orc, edges, t, rng)
            if len(pool) < 1:
                continue
            cf = orc.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
            srcs = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:1]]
            R0, Rf_d, Rp_d, _ = _legit(orc, edges, labels, t, srcs, 'distrust')
            R0t, Rf_t, _, _ = _legit(orc, edges, labels, t, srcs, 'trust')
            if R0 != R0:
                continue
            dr_app = float(orc.score_edge_set(edges, labels, srcs, t, 'distrust'))
            rows.append({
                'full': Rf_d - R0, 'legit': Rp_d - R0,
                'trust': Rf_t - R0t, 'appended': dr_app,
            })
    del orc, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return rows


CONTRASTS = {
    'appended_vs_inwindow': lambda r: r['full'] - r['appended'],
    'naive_vs_prop':        lambda r: r['full'] - r['legit'],
    'distrust_vs_trust':    lambda r: r['full'] - r['trust'],
    'prop_vs_zero':         lambda r: r['legit'],
    'appended_vs_zero':     lambda r: r['appended'],
}


def run_dataset(ds):
    log(f"\n{'#'*60}\n#  HIERARCHICAL STATS — {ds}\n{'#'*60}")
    per_seed_rows = {s: collect_seed(ds, s) for s in SEEDS}
    out = {}
    for name, fn in CONTRASTS.items():
        per_seed_vals = [np.asarray([fn(r) for r in per_seed_rows[s]], float) for s in SEEDS]
        pooled = np.concatenate(per_seed_vals)
        seed_meds = [float(np.median(v)) if len(v) else float('nan') for v in per_seed_vals]
        out[name] = {
            'n_pooled': int(len(pooled)),
            'median': round(float(np.median(pooled)), 4),
            'cliffs_delta': round(cliffs_delta(pooled), 3),
            'seedcluster_ci_median': seed_cluster_ci(per_seed_vals),
            'per_seed_median': [round(x, 4) for x in seed_meds],
            'seed_sign_test_p': round(sign_test_p(seed_meds), 4),
        }
        r = out[name]
        log(f"  {name:22s}: n={r['n_pooled']:3d} med={r['median']:+.4f} "
            f"cliff={r['cliffs_delta']:+.3f} seedCI{r['seedcluster_ci_median']} "
            f"perseed={r['per_seed_median']} signp={r['seed_sign_test_p']}")
    return out


def main():
    allout = {}
    for ds in DATASETS:
        allout[ds.strip()] = run_dataset(ds.strip())
        path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_OUT", "stats_hier.json"))
        with open(path, "w") as f:
            json.dump(allout, f, indent=2)
        log(f"saved {path}")
    log("DONE")


if __name__ == "__main__":
    main()
