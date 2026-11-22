"""
Propagation-only score under retraining (E8 retraining and E9 poisoning controls with Check 3;
DSN revision WP4).

Recomputes, with the procedure of run_poison_matrix.py (whose retrain() it calls), both the
total score R_tot (all raters, attack sources included) and the propagation-only score R_prop
(pre-existing raters only, embeddings of the retrained poisoned model on the poisoned graph).
Each poisoned retraining is compared with the clean retraining of the same seed; a target-run
pair is eligible if its clean R >= 0.5, and it flips if the poisoned score is < 0.5.

  part 'causal' (E8 retraining): 24 low-stratum targets of a clean seed-42 W = 8 model; per
      target 5 random sources (RandomState(42), as in run_poison_matrix.causal_retrain); GDTE is
      retrained on s0..s7 with the distrust edges in the new snapshot s7, at B = 5 and at B = 1
      (the first of the same five sources), seeds 42,1,2,3,4.
  part 'matrix' (E9): the five controls of run_poison_matrix.poison_matrix (none, dis_victim,
      tru_victim, dis_unrelated, dis_randsnap) for its 24 moderate-stratum targets, W = 7,
      edges in s6 (dis_randsnap: a random processed snapshot), B = 5, same seeds.

Outputs: results/retrain_prop.json (per-seed per-target reputations and summaries).
Run: CUDA_VISIBLE_DEVICES=1 GRAIL_DATASETS=otc,alpha python code/experiments/run_retrain_prop.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch
import torch.nn.functional as F

from project_paths import RESULTS_DIR
from experiment_common import log
from mycode.mainz_protocol import build_mainz_args
from mycode.dataset import get_snapshot_index
from run_poison_matrix import retrain, DIS, TRU

SEEDS = [int(x) for x in os.environ.get("GRAIL_SEEDS", "42,1,2,3,4").split(",")]
N_VICTIMS = 24
B = 5
EPOCHS = int(os.environ.get("GRAIL_EPOCHS", "50"))
TAU = 0.5
BOOT_SEED = 12345
N_BOOT = 10000


def base_index(ds):
    a = build_mainz_args(ds)
    return get_snapshot_index(a.time_slots, a.data_path, homogeneous_edges=a.homogeneous_edges)


def reps(orc, e, l, victims, inj_lo, inj_hi):
    """{v: (R_tot, R_prop)}; columns inj_lo..inj_hi-1 of e are the injected attack edges."""
    with torch.no_grad():
        z = orc._compute_node_embeddings(e, l)
        out = {}
        for v in victims:
            cols = (e[1] == v).nonzero(as_tuple=True)[0]
            p = F.softmax(torch.cat((z[e[0, cols]], z[v].unsqueeze(0).expand(len(cols), -1)), 1)
                          @ orc.model.regression_weights, 1)[:, 0]
            pre = (cols < inj_lo) | (cols >= inj_hi)
            out[int(v)] = (round(p.mean().item(), 6), round(p[pre].mean().item(), 6))
    return out


def run_retrain(ds, inj, seed, T_train, snap, victims, bidx):
    """Retrain (run_poison_matrix.retrain) and score every victim under both scores."""
    s = (T_train - 1) if snap is None else snap
    lo = bidx[s] + 1
    orc, e, l = retrain(ds, inj, seed, T_train=T_train, snap_for_inject=snap, epochs=EPOCHS)
    if inj:   # self-check: the injected columns hold exactly the planned edges
        got = e[:, lo:lo + len(inj)].cpu().numpy().T.tolist()
        assert got == [[s_, d_] for s_, d_, _ in inj], "injected edges not where expected"
    r = reps(orc, e, l, victims, lo, lo + len(inj))
    del orc, e, l
    gc.collect(); torch.cuda.empty_cache()
    return r


def seed_boot_ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return [None, None]
    m = v[np.random.RandomState(BOOT_SEED).randint(0, len(v), (N_BOOT, len(v)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def summarize(clean, att, victims):
    """clean[seed][v] = (R, R); att[seed][v] = (R_tot, R_prop). Rates over eligible pairs."""
    per_seed, pool = {}, {'k_tot': 0, 'k_prop': 0, 'n': 0}
    for sd in clean:
        el = [v for v in victims if clean[sd][v][0] >= TAU]
        dt = [att[sd][v][0] - clean[sd][v][0] for v in el]
        dp = [att[sd][v][1] - clean[sd][v][0] for v in el]
        kt = sum(att[sd][v][0] < TAU for v in el); kp = sum(att[sd][v][1] < TAU for v in el)
        per_seed[str(sd)] = {'n_eligible': len(el), 'k_flip_tot': int(kt), 'k_flip_prop': int(kp),
                             'flip_tot_pct': round(100 * kt / len(el), 1) if el else None,
                             'flip_prop_pct': round(100 * kp / len(el), 1) if el else None,
                             'mean_dR_tot_eligible': round(float(np.mean(dt)), 4) if el else None,
                             'mean_dR_prop_eligible': round(float(np.mean(dp)), 4) if el else None,
                             'mean_dR_tot_all': round(float(np.mean([att[sd][v][0] - clean[sd][v][0] for v in victims])), 4),
                             'mean_dR_prop_all': round(float(np.mean([att[sd][v][1] - clean[sd][v][0] for v in victims])), 4)}
        pool['k_tot'] += int(kt); pool['k_prop'] += int(kp); pool['n'] += len(el)
    out = {'per_seed': per_seed, 'pooled_n_eligible': pool['n'],
           'pooled_flip_tot_pct': round(100 * pool['k_tot'] / pool['n'], 1) if pool['n'] else None,
           'pooled_flip_prop_pct': round(100 * pool['k_prop'] / pool['n'], 1) if pool['n'] else None}
    for k in ('flip_tot_pct', 'flip_prop_pct', 'mean_dR_tot_eligible', 'mean_dR_prop_eligible',
              'mean_dR_tot_all', 'mean_dR_prop_all'):
        v = [p[k] for p in per_seed.values() if p[k] is not None]
        out[k] = {'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
                  'min': min(v), 'max': max(v), 'ci95_seed_boot': seed_boot_ci(v)}
    return out


def causal(ds, bidx):
    log(f"\n  E8 RETRAINING with R_prop (W = 8, edges in s7) — {ds}")
    o0, e0, l0 = retrain(ds, [], 42, T_train=8, epochs=EPOCHS)
    strata = o0.select_stratified_targets(e0, l0, per_stratum=max(8, N_VICTIMS))
    victims = [int(v) for v in (strata.get('low', []) + strata.get('moderate', []))[:N_VICTIMS]]
    n_low = sum(v in strata.get('low', []) for v in victims)
    rng = np.random.RandomState(42)
    plan = {}
    for v in victims:
        ex = set(e0[0, e0[1] == v].tolist())
        plan[v] = [int(s) for s in rng.permutation(o0.num_nodes) if s != v and s not in ex][:B]
    del o0, e0, l0; gc.collect(); torch.cuda.empty_cache()
    inj5 = [(s, v, DIS) for v in victims for s in plan[v]]
    inj1 = [(plan[v][0], v, DIS) for v in victims]
    clean, a5, a1 = {}, {}, {}
    for sd in SEEDS:
        clean[sd] = run_retrain(ds, [], sd, 8, None, victims, bidx)
        a5[sd] = run_retrain(ds, inj5, sd, 8, 7, victims, bidx)
        a1[sd] = run_retrain(ds, inj1, sd, 8, 7, victims, bidx)
        for name, a in (('B5', a5), ('B1', a1)):
            el = [v for v in victims if clean[sd][v][0] >= TAU]
            log(f"    seed {sd} {name}: eligible {len(el)} flip tot {sum(a[sd][v][0] < TAU for v in el)} "
                f"prop {sum(a[sd][v][1] < TAU for v in el)}")
    res = {'seeds': SEEDS, 'victims': victims, 'n_low_stratum': n_low, 'plan': {str(v): plan[v] for v in victims},
           'raw': {str(sd): {'clean': clean[sd], 'B5': a5[sd], 'B1': a1[sd]} for sd in SEEDS},
           'B5': summarize(clean, a5, victims), 'B1': summarize(clean, a1, victims)}
    for k in ('B5', 'B1'):
        r = res[k]
        log(f"    {k}: pooled flip tot {r['pooled_flip_tot_pct']}% prop {r['pooled_flip_prop_pct']}% (n={r['pooled_n_eligible']}); "
            f"mean dR tot {r['mean_dR_tot_eligible']['mean']} prop {r['mean_dR_prop_eligible']['mean']}")
    return res


def matrix(ds, bidx):
    log(f"\n  E9 POISONING CONTROLS with R_prop (W = 7) — {ds}")
    o0, e0, l0 = retrain(ds, [], 42, epochs=EPOCHS)
    strata = o0.select_stratified_targets(e0, l0, per_stratum=max(8, N_VICTIMS))
    victims = [int(v) for v in (strata.get('moderate', []) + strata.get('low', []))[:N_VICTIMS]]
    rng = np.random.RandomState(42)
    plan = {}
    for v in victims:
        ex = set(e0[0, e0[1] == v].tolist())
        plan[v] = [int(s) for s in rng.permutation(o0.num_nodes) if s != v and s not in ex][:B]
    del o0, e0, l0; gc.collect(); torch.cuda.empty_cache()
    controls = ['none', 'dis_victim', 'tru_victim', 'dis_unrelated', 'dis_randsnap']
    data = {c: {} for c in controls}
    for sd in SEEDS:
        for c in controls:
            snap = None
            if c == 'none':
                inj = []
            elif c == 'dis_victim':
                inj = [(s, v, DIS) for v in victims for s in plan[v]]
            elif c == 'tru_victim':
                inj = [(s, v, TRU) for v in victims for s in plan[v]]
            elif c == 'dis_unrelated':
                inj = [(s, victims[0], DIS) for s in plan[victims[0]]]
            else:  # dis_randsnap: same draw as run_poison_matrix
                snap = int(np.random.RandomState(sd).randint(0, 6))
                inj = [(s, v, DIS) for v in victims for s in plan[v]]
            data[c][sd] = run_retrain(ds, inj, sd, 7, snap, victims, bidx)
        log(f"    seed {sd} done")
    res = {'seeds': SEEDS, 'victims': victims, 'raw': {c: {str(sd): data[c][sd] for sd in SEEDS} for c in controls},
           'controls': {}}
    for c in controls[1:]:
        res['controls'][c] = summarize(data['none'], data[c], victims)
        r = res['controls'][c]
        log(f"    {c:14s}: pooled flip tot {r['pooled_flip_tot_pct']}% prop {r['pooled_flip_prop_pct']}%; "
            f"mean dR (all 24) tot {r['mean_dR_tot_all']['mean']} prop {r['mean_dR_prop_all']['mean']}")
    return res


if __name__ == "__main__":
    parts = os.environ.get("GRAIL_PARTS", "causal,matrix").split(",")
    path = os.path.join(os.environ.get("GRAIL_OUT_DIR", str(RESULTS_DIR)), "retrain_prop.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for ds in os.environ.get("GRAIL_DATASETS", "otc,alpha").split(","):
        ds = ds.strip()
        bidx = base_index(ds)
        results.setdefault(ds, {})
        if "causal" in parts:
            results[ds]['causal_retrain'] = causal(ds, bidx)
        if "matrix" in parts:
            results[ds]['poison_matrix'] = matrix(ds, bidx)
        with open(path, "w") as f:
            json.dump(results, f, indent=1)
        log(f"Wrote {path}")
