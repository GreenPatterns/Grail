"""
Reviewer-rebuttal runs for the AAAI revision.

Block A (#4/#5): FIXED target cohort reused across 5 seeds, larger n, exact
  Clopper-Pearson intervals + seed-clustered bootstrap CI. Separates training
  variance from target-sampling variance (same node IDs every seed).
Block B (#2): STRICT query-edge-masked encoding. When scoring P(trust|u,v*)
  for a pre-existing rater u, the edge (u,v*) is removed from the graph used to
  compute embeddings. This kills same-edge conditioning for legit edges too, the
  strongest form of the leakage control. Reports flip_masked vs flip_preexist.
Block C (#8): random-source distribution. 50 random source-sets per target;
  optimizer regret = percentile of the optimized dR within the random dR dist.

Run: GRAIL_DATASETS=otc,alpha CUDA_VISIBLE_DEVICES=0 \
     python code/experiments/run_reviewer_rebuttal.py
     (GRAIL_SEEDS=42,1,... sets the seeds, GRAIL_BLOCKS=B runs only the masked control,
      GRAIL_OUT names the output file)
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

SEEDS = [int(x) for x in os.environ.get("GRAIL_SEEDS", "42,1,2,3,4").split(",")]  # genuine seeds since the seed-bug fix
DATASETS = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
C_POOL = 50
COHORT_LOW = 25          # fixed low-stratum cohort size (used for masked + random)
COHORT_MOD = 40          # moderate cohort for the larger-n cheap block
N_RAND = 50              # random source-sets per target (Block C)
MASK_CAP = 30            # cap raters re-encoded per target in Block B


# ── shared helpers ───────────────────────────────────────────────────
def _cp(k, n, alpha=0.05):
    from scipy import stats as sp
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else sp.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sp.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (round(float(lo), 3), round(float(hi), 3))


def _seed_cluster_ci(per_seed_hits, per_seed_n, n_boot=2000):
    """Bootstrap over seeds (clusters): resample the 5 seeds with replacement,
    pool their hit/n, take the flip rate. Returns (lo, hi) of the 95% interval."""
    per_seed_hits = np.asarray(per_seed_hits, float)
    per_seed_n = np.asarray(per_seed_n, float)
    S = len(per_seed_hits)
    if S == 0 or per_seed_n.sum() == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(0)
    rates = []
    for _ in range(n_boot):
        idx = rng.randint(0, S, S)
        n = per_seed_n[idx].sum()
        if n == 0:
            continue
        rates.append(100.0 * per_seed_hits[idx].sum() / n)
    return (round(float(np.percentile(rates, 2.5)), 1),
            round(float(np.percentile(rates, 97.5)), 1))


def _p_trust(oracle, z, target, edges, cols):
    src = edges[0, cols]
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


def _build_attacked(oracle, edges, labels, target, srcs):
    """Splice B distrust edges into the last trained snapshot (processed placement)."""
    B = len(srcs)
    T = oracle.model.args.train_time_slots
    split = oracle.model.index_list[T - 1] + 1
    orig_idx = list(oracle.model.index_list)
    ne = torch.tensor([list(srcs), [target] * B], dtype=torch.long, device=oracle.device)
    nl = torch.tensor([[0.0, 1.0]] * B, dtype=torch.float, device=oracle.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], dim=1)
    al = torch.cat([labels[:split], nl, labels[split:]], dim=0)
    aidx = [b + B if i >= T - 1 else b for i, b in enumerate(orig_idx)]
    inj = set(range(split, split + B))
    return ae, al, aidx, split, inj


def _variants(oracle, edges, labels, target, srcs):
    """(R0, R_full, R_preexist) at processed placement, distrust — cheap (2 encodes)."""
    ae, al, aidx, split, inj = _build_attacked(oracle, edges, labels, target, srcs)
    dst = (ae[1] == target).nonzero(as_tuple=True)[0]
    pre = torch.tensor([c for c in dst.tolist() if c not in inj], dtype=torch.long, device=oracle.device)
    z0 = _encode(oracle, edges, labels, list(oracle.model.index_list))
    cdst = (edges[1] == target).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        R0 = _p_trust(oracle, z0, target, edges, cdst).mean().item()
    z1 = _encode(oracle, ae, al, aidx)
    with torch.no_grad():
        R_full = _p_trust(oracle, z1, target, ae, dst).mean().item()
        R_pre = _p_trust(oracle, z1, target, ae, pre).mean().item() if len(pre) else float('nan')
    return R0, R_full, R_pre


def _masked_rep(oracle, edges, labels, idx, target, rater_cols, rng):
    """Query-edge-masked reputation: mean over rater edges of P(trust|u,v*)
    where THAT edge is removed from the encoder graph. One encode per rater."""
    cols = rater_cols.tolist()
    if len(cols) > MASK_CAP:
        cols = list(rng.choice(cols, MASK_CAP, replace=False))
    preds = []
    for c in cols:
        keep = torch.ones(edges.shape[1], dtype=torch.bool, device=edges.device)
        keep[c] = False
        em = edges[:, keep]; lm = labels[keep]
        idx_m = [b - 1 if b >= c else b for b in idx]   # boundary shift for removed col
        u = int(edges[0, c])
        with torch.no_grad():
            zm = _encode(oracle, em, lm, idx_m)
            feat = torch.cat((zm[u], zm[target]), dim=0).unsqueeze(0)
            p = F.softmax(feat @ oracle.model.regression_weights, dim=1)[0, 0].item()
        preds.append(p)
    return (float(np.mean(preds)) if preds else float('nan')), len(cols)


# ── Block A: fixed cohort, larger n, CIs (#4/#5) ─────────────────────
def block_A(ds, cohort, models):
    log(f"\n[A] fixed-cohort flip table — {ds}  (low {len(cohort['low'])}, mod {len(cohort['moderate'])})")
    res = {}
    for stratum in ("low", "moderate"):
        for B in (1, 5):
            per_seed_t, per_seed_p = [], []   # hits
            per_seed_nt = []                  # eligible n
            dR_t_all, dR_p_all = [], []
            for seed in SEEDS:
                oracle, edges, labels = models[seed]
                rng = np.random.RandomState(1000 + seed)
                ht = hp = n = 0
                for t in cohort[stratum]:
                    pool = _pool(oracle, edges, t, C_POOL, rng)
                    if len(pool) < B:
                        continue
                    cf = oracle.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
                    srcs = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:B]]
                    R0, Rf, Rp = _variants(oracle, edges, labels, t, srcs)
                    if R0 != R0 or R0 < 0.5:
                        continue
                    n += 1
                    ht += 1 if Rf < 0.5 else 0
                    hp += 1 if (Rp == Rp and Rp < 0.5) else 0
                    dR_t_all.append(Rf - R0); dR_p_all.append(Rp - R0)
                per_seed_t.append(ht); per_seed_p.append(hp); per_seed_nt.append(n)
            N = sum(per_seed_nt)
            res[f"{stratum}_B{B}"] = {
                "per_seed_n": per_seed_nt,
                "n_pooled": N,
                "flip_total_pct": round(100 * sum(per_seed_t) / N, 1) if N else None,
                "flip_preexist_pct": round(100 * sum(per_seed_p) / N, 1) if N else None,
                "flip_total_cp_ci": _cp(sum(per_seed_t), N),
                "flip_preexist_cp_ci": _cp(sum(per_seed_p), N),
                "flip_total_seedboot_ci": _seed_cluster_ci(per_seed_t, per_seed_nt),
                "flip_preexist_seedboot_ci": _seed_cluster_ci(per_seed_p, per_seed_nt),
                "flip_total_per_seed_pct": [round(100 * h / max(nn, 1), 1) for h, nn in zip(per_seed_t, per_seed_nt)],
                "flip_preexist_per_seed_pct": [round(100 * h / max(nn, 1), 1) for h, nn in zip(per_seed_p, per_seed_nt)],
                "dR_preexist_mean": round(float(np.mean(dR_p_all)), 4) if dR_p_all else None,
                "dR_preexist_median": round(float(np.median(dR_p_all)), 4) if dR_p_all else None,
            }
            r = res[f"{stratum}_B{B}"]
            log(f"  {stratum:8s} B={B}: N={N:3d}  total={r['flip_total_pct']}% "
                f"CP{r['flip_total_cp_ci']} seed{r['flip_total_seedboot_ci']} | "
                f"pre={r['flip_preexist_pct']}% CP{r['flip_preexist_cp_ci']} seed{r['flip_preexist_seedboot_ci']}")
    return res


# ── Block B: strict query-edge-masked encoding (#2) ──────────────────
def block_B(ds, cohort, models):
    log(f"\n[B] query-edge-masked encoding (strict) — {ds}  low B=1")
    per_seed_p, per_seed_m, per_seed_n = [], [], []
    dR_m_all, dR_p_all = [], []
    for seed in SEEDS:
        oracle, edges, labels = models[seed]
        rng = np.random.RandomState(2000 + seed)
        orig_idx = list(oracle.model.index_list)
        hp = hm = n = 0
        for t in cohort["low"]:
            pool = _pool(oracle, edges, t, C_POOL, rng)
            if not pool:
                continue
            cf = oracle.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
            srcs = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:1]]
            ae, al, aidx, split, inj = _build_attacked(oracle, edges, labels, t, srcs)
            dst = (ae[1] == t).nonzero(as_tuple=True)[0]
            pre = torch.tensor([c for c in dst.tolist() if c not in inj], dtype=torch.long, device=oracle.device)
            # clean full R0 for eligibility
            z0 = _encode(oracle, edges, labels, orig_idx)
            cdst = (edges[1] == t).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                R0 = _p_trust(oracle, z0, t, edges, cdst).mean().item()
            if R0 != R0 or R0 < 0.5 or len(pre) == 0:
                continue
            # leakage-free (preexist, unmasked encoder) for reference
            z1 = _encode(oracle, ae, al, aidx)
            with torch.no_grad():
                Rp = _p_trust(oracle, z1, t, ae, pre).mean().item()
            # STRICT masked: attacked-graph masked, and clean-graph masked baseline
            Rm_att, _ = _masked_rep(oracle, ae, al, aidx, t, pre, rng)
            Rm_cln, _ = _masked_rep(oracle, edges, labels, orig_idx, t, cdst, rng)
            if Rm_att != Rm_att or Rm_cln != Rm_cln:
                continue
            n += 1
            hp += 1 if Rp < 0.5 else 0
            hm += 1 if Rm_att < 0.5 else 0
            dR_p_all.append(Rp - R0)
            dR_m_all.append(Rm_att - Rm_cln)      # masked propagation-only shift
            gc.collect()
        per_seed_p.append(hp); per_seed_m.append(hm); per_seed_n.append(n)
        log(f"  seed {seed}: n={n:2d}  preexist_flip={100*hp/max(n,1):.0f}%  masked_flip={100*hm/max(n,1):.0f}%")
    N = sum(per_seed_n)
    out = {
        "config": "low stratum, B=1, optimized source, distrust; each scored edge removed from encoder",
        "per_seed_n": per_seed_n,
        "n_pooled": N,
        "flip_preexist_pct": round(100 * sum(per_seed_p) / N, 1) if N else None,
        "flip_masked_pct": round(100 * sum(per_seed_m) / N, 1) if N else None,
        "flip_masked_cp_ci": _cp(sum(per_seed_m), N),
        "flip_masked_seedboot_ci": _seed_cluster_ci(per_seed_m, per_seed_n),
        "flip_masked_per_seed_pct": [round(100 * h / max(nn, 1), 1) for h, nn in zip(per_seed_m, per_seed_n)],
        "dR_masked_mean": round(float(np.mean(dR_m_all)), 4) if dR_m_all else None,
        "dR_masked_median": round(float(np.median(dR_m_all)), 4) if dR_m_all else None,
        "dR_preexist_mean": round(float(np.mean(dR_p_all)), 4) if dR_p_all else None,
    }
    log(f"  POOLED N={N}: preexist={out['flip_preexist_pct']}%  MASKED={out['flip_masked_pct']}% "
        f"CP{out['flip_masked_cp_ci']} seed{out['flip_masked_seedboot_ci']}  dR_masked={out['dR_masked_median']}")
    return out


# ── Block C: random-source distribution (#8) ─────────────────────────
def block_C(ds, cohort, models):
    log(f"\n[C] random-source distribution — {ds}  low  (seed 42)")
    oracle, edges, labels = models[42]
    rng = np.random.RandomState(7)
    res = {}
    for B in (1, 5):
        regrets, opt_drs, rand_meds = [], [], []
        upper_q = 0; cnt = 0
        for t in cohort["low"]:
            pool = _pool(oracle, edges, t, C_POOL, rng)
            if len(pool) < B:
                continue
            cf = oracle.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
            opt = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:B]]
            R0o, Rfo, _ = _variants(oracle, edges, labels, t, opt)
            if R0o != R0o or R0o < 0.5:
                continue
            opt_dr = Rfo - R0o
            rand = []
            for _ in range(N_RAND):
                srcs = list(rng.choice(pool, B, replace=False))
                _, Rf, _ = _variants(oracle, edges, labels, t, [int(s) for s in srcs])
                rand.append(Rf - R0o)
            rand = np.array(rand)
            # regret = percentile of optimized dR within random dist (lower dR = more damage)
            pct = 100.0 * float((rand <= opt_dr).mean())   # % of random at least as damaging
            regrets.append(pct); opt_drs.append(opt_dr); rand_meds.append(float(np.median(rand)))
            if opt_dr <= np.percentile(rand, 25):   # optimized in the most-damaging quartile
                upper_q += 1
            cnt += 1
        res[f"B{B}"] = {
            "n": cnt,
            "opt_dR_mean": round(float(np.mean(opt_drs)), 4) if opt_drs else None,
            "rand_dR_median_mean": round(float(np.mean(rand_meds)), 4) if rand_meds else None,
            "rand_frac_ge_optimized_mean_pct": round(float(np.mean(regrets)), 1) if regrets else None,
            "optimized_in_random_top_quartile_pct": round(100.0 * upper_q / cnt, 1) if cnt else None,
        }
        r = res[f"B{B}"]
        log(f"  B={B}: n={cnt}  opt dR={r['opt_dR_mean']}  rand median dR={r['rand_dR_median_mean']}  "
            f"random>=opt in {r['rand_frac_ge_optimized_mean_pct']}% of draws")
    return res


def run_dataset(ds):
    log(f"\n{'#'*64}\n#  REVIEWER REBUTTAL — {ds}\n{'#'*64}")
    # train all seeds, build oracles
    models = {}
    ref = None
    for seed in SEEDS:
        tr = train_model(ds, seed=seed)
        orc = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
        models[seed] = (orc, tr["edges"], tr["labels"])
        if seed == 42:
            ref = (orc, tr["edges"], tr["labels"])
    # fixed cohort from seed-42 clean model
    orc, e0, l0 = ref
    strata = orc.select_stratified_targets(e0, l0, per_stratum=max(COHORT_LOW, COHORT_MOD))
    cohort = {"low": strata["low"][:COHORT_LOW], "moderate": strata["moderate"][:COHORT_MOD]}
    log(f"fixed cohort: low={len(cohort['low'])} moderate={len(cohort['moderate'])} (node IDs reused across seeds)")
    blocks = os.environ.get("GRAIL_BLOCKS", "A,B,C").split(",")   # e.g. GRAIL_BLOCKS=B
    out = {"cohort_low_ids": cohort["low"], "cohort_moderate_ids": cohort["moderate"], "seeds": SEEDS}
    if "A" in blocks:
        out["A_fixed_cohort"] = block_A(ds, cohort, models)
    if "B" in blocks:
        out["B_query_edge_masked"] = block_B(ds, cohort, models)
    if "C" in blocks:
        out["C_random_distribution"] = block_C(ds, cohort, models)
    for seed in SEEDS:
        del models[seed]
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    allout = {}
    for ds in DATASETS:
        allout[ds] = run_dataset(ds)
        path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_OUT", "reviewer_rebuttal.json"))
        with open(path, "w") as f:
            json.dump(allout, f, indent=2)
        log(f"\nsaved {path}")
    log("\nDONE")


if __name__ == "__main__":
    main()
