"""
Complete-cohort, multi-seed single-edge attack (Reviewer 2's rebuttal request): propagation-
only and total-effect flip rates for random and optimized sources, on EVERY eligible target
(in-degree >= 2, clean reputation >= 0.5) of each genuinely seeded model, reported by the
target's distance to the 0.5 gate.

Per target t, one candidate pool of C = 50 non-neighbor sources is drawn with a target-
specific RNG (RandomState(POOL_SEED + t)), so the pool, and therefore the random source
(pool[0]), is the same for every training seed. The optimized source is the candidate whose
single distrust edge lowers the total-effect reputation most, by the exact processed-placement
counterfactual (the paper's optimized source). Every edge is spliced at the end of the last
processed snapshot and scored by mycode/fast_s6.S6Scorer, which reproduces the full forward
pass (tests/test_fast_s6.py).

Seeds are genuine since the seed-bug fix (tests/test_seed_propagation.py): each seed retrains
GDTE from its own initialization and node features.

Run: GRAIL_DATASETS=otc GRAIL_SEEDS=42,1,2 python code/experiments/run_cohort_seeds.py
     writes results/unified/cohort/cohort_{ds}_s{seed}.json per (dataset, seed)
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

C_POOL = 50
POOL_SEED = 20260925
OUT_DIR = RESULTS_DIR / "unified" / "cohort"


def candidate_pool(target, raters, num_nodes, C=C_POOL):
    """C distinct non-neighbor sources, fixed per target across training seeds."""
    rng = np.random.RandomState(POOL_SEED + int(target))
    banned = set(raters) | {int(target)}
    pool = []
    for s in rng.permutation(num_nodes):
        if int(s) not in banned:
            pool.append(int(s))
            if len(pool) == C:
                break
    return pool


def run(ds, seed):
    t0 = time.time()
    tr = train_model(ds, seed=seed)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    f = S6Scorer(o, edges, labels)
    dst = edges[1].cpu().numpy(); src = edges[0].cpu().numpy()
    indeg = np.bincount(dst, minlength=o.num_nodes)
    rows = []
    cand = np.nonzero(indeg >= 2)[0]
    for i, t in enumerate(cand):
        t = int(t)
        r0 = f.clean(t)
        if r0 < 0.5:
            continue
        raters = src[dst == t].tolist()
        pool = candidate_pool(t, raters, o.num_nodes)
        rf_r, rp_r = f.attacked(t, [pool[0]])
        s_opt, rf_o, rp_o = f.best_single(t, pool)
        rows.append({"t": t, "r0": round(r0, 6), "indeg": int(indeg[t]),
                     "random": {"src": pool[0], "total": round(rf_r, 6), "prop": round(rp_r, 6)},
                     "opt": {"src": int(s_opt), "total": round(rf_o, 6), "prop": round(rp_o, 6)}})
        if (i + 1) % 500 == 0:
            log(f"  [{ds} s{seed}] {i+1}/{len(cand)} candidates scanned, {len(rows)} eligible")
    perf = {k: tr["perf"].get(k) for k in ("AUC", "MCC", "BAcc")}
    out = {"dataset": ds, "seed": seed, "C": C_POOL, "pool_seed": POOL_SEED, "perf": perf,
           "n_eligible": len(rows), "minutes": round((time.time() - t0) / 60, 1), "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"cohort_{ds}_s{seed}.json", "w") as fh:
        json.dump(out, fh)
    fl = lambda k, m: np.mean([r[k][m] < 0.5 for r in rows]) * 100
    log(f"  {ds} seed {seed}: n={len(rows)} AUC={perf['AUC']:.3f} | flip% random total/prop "
        f"{fl('random','total'):.1f}/{fl('random','prop'):.1f} | optimized {fl('opt','total'):.1f}/"
        f"{fl('opt','prop'):.1f} | {out['minutes']} min")
    del tr, o, f
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    for ds in os.environ.get("GRAIL_DATASETS", "otc").split(","):
        for seed in [int(s) for s in os.environ.get("GRAIL_SEEDS", "42").split(",")]:
            run(ds.strip(), seed)
