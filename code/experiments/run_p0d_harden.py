"""
Time-decayed mean (E12; Section VII) and its plain-mean reference, which also gives Section V's
single-run example (one optimized edge on the OTC low stratum, 68% under the plain mean).

Each stratum (high / moderate / low, up to GRAIL_P0D_PER_STRATUM = 25 targets per dataset, one
run) receives B in {1, 3, 5, 10} in-window distrust edges from RANDOM sources or OPTIMIZED ones
(the B best single candidates by exact in-window scoring); the paper reports B = 1 and 5.
GRAIL_REP_MODE=decayed scores the reputation with the time-decayed mean of Eq. 7 (weight
GRAIL_DECAY_LAMBDA ** age, default 0.8) instead of the mean of Eq. 1.

Metrics: mean shift, flip rate (R falls below the threshold 0.5) with a bootstrap 95% interval,
and the out-of-window shift of the same sources for contrast (key `dr_appended_paper`, where
"paper" marks the appended placement of the first, flawed evaluation). Edges are spliced into the
processed window and re-embedded with frozen weights (score_edge_set_processed); nothing is
retrained.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p0d_harden.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch

from project_paths import RESULTS_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
BUDGETS = [1, 3, 5, 10]
PER_STRATUM = int(os.environ.get("GRAIL_P0D_PER_STRATUM", "25"))
CAND_POOL = 60  # pool for optimized source selection


def boot_ci(x, n=1000):
    x = np.array(x)
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else 0.0, 0.0, 0.0
    m = [np.mean(np.random.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 0d HARDEN — {ds}\n{'='*64}")
    tr = train_model(ds)
    rep_mode = os.environ.get("GRAIL_REP_MODE", "mean")
    decay_lambda = float(os.environ.get("GRAIL_DECAY_LAMBDA", "0.8"))
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"],
                               rep_mode=rep_mode, decay_lambda=decay_lambda)
    if rep_mode != "mean":
        log(f"  [reputation functional = {rep_mode}, decay_lambda={decay_lambda}]")
    edges, labels = tr["edges"], tr["labels"]
    rng = np.random.RandomState(SEED)

    strata = o.select_stratified_targets(edges, labels, per_stratum=PER_STRATUM)
    out = {}
    for stratum in ["high", "moderate", "low"]:
        victims = strata.get(stratum, [])
        if not victims:
            continue
        log(f"  stratum={stratum}  n={len(victims)}")
        # Precompute candidate pools + optimized ranking per target once.
        pools, opt_rank = {}, {}
        for v in victims:
            existing = set(edges[0, edges[1] == v].tolist())
            pool = [int(s) for s in rng.permutation(o.num_nodes)
                    if s != v and s not in existing][:CAND_POOL]
            pools[v] = pool
            sc = o.score_candidates_counterfactual_processed(edges, labels, v, pool, sign='distrust')
            opt_rank[v] = [s for s, _ in sorted(sc.items(), key=lambda x: x[1])]  # most damaging first

        for B in BUDGETS:
            dr_rand, dr_opt, dr_app, flips_rand, flips_opt = [], [], [], 0, 0
            n_eligible = 0
            for v in victims:
                pool = pools[v]
                rand_src = pool[:B]
                opt_src = opt_rank[v][:B]
                dr_r, rb, ra_r = o.score_edge_set_processed(edges, labels, rand_src, v, 'distrust', return_after=True)
                dr_o, _, ra_o = o.score_edge_set_processed(edges, labels, opt_src, v, 'distrust', return_after=True)
                dr_a = o.score_edge_set(edges, labels, opt_src, v, 'distrust')  # out-of-window (appended)
                dr_rand.append(dr_r); dr_opt.append(dr_o); dr_app.append(dr_a)
                if rb >= 0.5:
                    n_eligible += 1
                    flips_rand += int(ra_r < 0.5)
                    flips_opt += int(ra_o < 0.5)
            m_r, lo_r, hi_r = boot_ci(dr_rand)
            m_o, lo_o, hi_o = boot_ci(dr_opt)
            m_a, _, _ = boot_ci(dr_app)
            out[f"{stratum}_B{B}"] = {
                "n": len(victims), "n_eligible_flip": n_eligible,
                "dr_random": m_r, "dr_random_ci": [lo_r, hi_r],
                "dr_optimized": m_o, "dr_optimized_ci": [lo_o, hi_o],
                "dr_appended_paper": m_a,
                "flip_rate_random": flips_rand / max(n_eligible, 1),
                "flip_rate_optimized": flips_opt / max(n_eligible, 1),
            }
            log(f"    B={B:2d}: ΔR rand={m_r:+.4f}[{lo_r:+.3f},{hi_r:+.3f}] "
                f"opt={m_o:+.4f}[{lo_o:+.3f},{hi_o:+.3f}] appended={m_a:+.4f} | "
                f"flip% rand={flips_rand/max(n_eligible,1):.0%} opt={flips_opt/max(n_eligible,1):.0%} "
                f"(elig {n_eligible})")

    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    outp = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p0d_harden.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {outp}")

    log(f"\n{'='*64}\n  HARDENING HEADLINES (B=5, optimized)\n{'='*64}")
    for ds, r in results.items():
        for stratum in ["high", "moderate", "low"]:
            k = f"{stratum}_B5"
            if k in r:
                e = r[k]
                log(f"  {ds} {stratum:>8}: ΔR_opt={e['dr_optimized']:+.4f}  "
                    f"flip%={e['flip_rate_optimized']:.0%}  "
                    f"(appended/paper={e['dr_appended_paper']:+.4f})")


if __name__ == "__main__":
    main()
