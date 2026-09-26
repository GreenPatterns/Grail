"""
Reputation functions compared on a COMMON target set (Reviewer 2): every target with in-degree
>= 2 in the processed window, the same candidate pools as run_cohort_seeds.py, the same
injections for every function, one model (seed 42). Per target and attacker it records the
clean and attacked reputation under
  GDTE mean / median, total effect and propagation-only (mycode/fast_s6.S6Scorer),
  Fairness-Goodness (Kumar et al. 2016; warm-started), trust fraction, Beta mean, Wilson bound,
so any subset (targets eligible under every function, a margin band) can be analyzed
afterwards (code/analyze_common_set.py).

Attackers, each choosing from the target's 50-candidate pool:
  random   : pool[:B]
  gdte_opt : the B candidates whose single distrust edge lowers GDTE's total-effect R most
  fg_opt   : the B candidates whose single distrust edge lowers Fairness-Goodness most
Budgets B in {1, 5}.

Run: GRAIL_DATASETS=otc python code/experiments/run_common_set.py
     writes results/unified/common_set_{ds}.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc, time
import numpy as np
import torch

from project_paths import RESULTS_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.fast_s6 import S6Scorer
from mycode.reputation_systems import fairness_goodness
from run_cohort_seeds import candidate_pool
from run_repbaselines_calib import _label_reps
from run_reputation_baselines import _fg_goodness_after

SEED = 42
BUDGETS = (1, 5)
ATTACKERS = ("random", "gdte_opt", "fg_opt")


def _gdte(p_pre, p_inj=None):
    allp = p_pre if p_inj is None else torch.cat([p_pre, p_inj])
    med = lambda x: float(torch.quantile(x, 0.5)) if len(x) else float('nan')
    return {"gdte_mean_total": float(allp.mean()), "gdte_mean_prop": float(p_pre.mean()),
            "gdte_median_total": med(allp), "gdte_median_prop": med(p_pre)}


def _labels(k, n):
    lab = _label_reps(k, n)
    return {"fraction": lab["signed_mean"], "beta": lab["beta_bayes"], "wilson": lab["wilson"]}


def run(ds):
    t0 = time.time()
    tr = train_model(ds, seed=SEED)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    f = S6Scorer(o, edges, labels)
    src = edges[0].cpu().numpy(); dst = edges[1].cpu().numpy()
    trust = labels[:, 0].cpu().numpy() >= 0.5
    w = np.where(trust, 1.0, -1.0)
    n = o.num_nodes
    f0, g0 = fairness_goodness(src, dst, w, n)
    warm = (f0, g0)
    indeg = np.bincount(dst, minlength=n)
    rows = []
    targets = [int(t) for t in np.nonzero(indeg >= 2)[0]]
    for i, t in enumerate(targets):
        raters = src[dst == t].tolist()
        k_in, n_in = int(trust[dst == t].sum()), int((dst == t).sum())
        pool = candidate_pool(t, raters, n)
        gd = {s: f.attacked(t, [s])[0] for s in pool}
        fg = {s: _fg_goodness_after(src, dst, w, n, t, [s], warm)[t] for s in pool}
        order = {"random": pool, "gdte_opt": sorted(pool, key=gd.get), "fg_opt": sorted(pool, key=fg.get)}
        clean = {**_gdte(f.clean_vector(t)), "fg": (g0[t] + 1) / 2, **_labels(k_in, n_in)}
        att = {}
        for a in ATTACKERS:
            for B in BUDGETS:
                s_b = [int(s) for s in order[a][:B]]
                p_pre, p_inj = f.attacked_vectors(t, s_b)
                g1 = _fg_goodness_after(src, dst, w, n, t, s_b, warm)
                att[f"{a}|B{B}"] = {**_gdte(p_pre, p_inj), "fg": (g1[t] + 1) / 2, **_labels(k_in, n_in + B)}
        rows.append({"t": t, "indeg": int(indeg[t]), "clean": clean, "attacked": att})
        if (i + 1) % 250 == 0:
            log(f"  [{ds}] {i+1}/{len(targets)} targets, {(time.time()-t0)/60:.1f} min")
    out = {"dataset": ds, "seed": SEED, "n_targets": len(rows), "perf": {k: tr["perf"].get(k) for k in ("AUC", "MCC")},
           "minutes": round((time.time() - t0) / 60, 1), "rows": rows}
    with open(RESULTS_DIR / "unified" / f"common_set_{ds}.json", "w") as fh:
        json.dump(out, fh)
    log(f"  saved common_set_{ds}.json ({len(rows)} targets, {out['minutes']} min)")
    del tr, o, f
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in os.environ.get("GRAIL_DATASETS", "otc").split(","):
        run(ds.strip())
