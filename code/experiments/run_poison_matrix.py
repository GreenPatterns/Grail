"""
Poisoning control matrix + periodic-retraining causal regime
(reviewer blockers #11 and #1-regimeB).

#11: the single distrust-minus-trust control is extended to a full placebo
matrix, retrained across seeds {42,1,2} so the attack effect is read against the
distribution of ordinary retraining variation. Controls per seed:
  none          : clean retrain (no injected edge)      -> retraining-variance ref
  dis_victim    : B distrust edges (src->victim) in s6   -> the attack
  tru_victim    : same B edges labelled TRUST            -> label control
  dis_unrelated : B distrust edges to an UNRELATED node  -> placebo (cross-talk)
  dis_randsnap  : B distrust edges to victim, placed in a RANDOM processed snapshot
                  -> tests timestamp/snapshot sensitivity

#1-regimeB: PERIODIC RETRAINING through the next snapshot. A fresh model is
trained with train_time_slots=8 (the backbone rebuilt to length 8) on s0..s7,
clean vs with the attack edge baked into the NEW snapshot s7. This is the
deployment-faithful "retrain through s7" regime (no retroactive rewrite).

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_poison_matrix.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch

from project_paths import RESULTS_DIR
from experiment_common import log
import mycode.gcn3 as gcn3_mod
from mycode.gcn3 import GCNTrainer
from mycode.mainz_protocol import build_mainz_args
from mycode.utils import read_graph
from mycode.dataset import get_snapshot_index
from mycode.trust_influence import ReputationAttackOracle

SEEDS = [int(x) for x in os.environ.get("GRAIL_SEEDS", "42,1,2").split(",")]  # genuine seeds since the seed-bug fix
B = 5
N_VICTIMS = int(os.environ.get("GRAIL_NVICTIMS", "6"))
EPOCHS = 50
DIS, TRU = [0.0, 1.0], [1.0, 0.0]


def _cp_ci(k, n, alpha=0.05):
    from scipy import stats as sp
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else float(sp.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sp.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (round(lo, 3), round(hi, 3))


def retrain(ds, injections, seed, T_train=7, snap_for_inject=None, epochs=EPOCHS):
    """injections: list of (src,dst,label_vec). snap_for_inject: processed snapshot
    index to splice into (default = last processed T_train-1). T_train sets the
    backbone length (8 => causal retrain through s7)."""
    args = build_mainz_args(ds, epochs=epochs)
    args.train_time_slots = T_train
    graph = read_graph(args)
    base_index = get_snapshot_index(args.time_slots, args.data_path,
                                    homogeneous_edges=args.homogeneous_edges)
    snap = (T_train - 1) if snap_for_inject is None else snap_for_inject
    split = base_index[snap] + 1
    k = len(injections)
    if k:
        pe = np.array([[s, d] for s, d, _ in injections], dtype=float)
        pl = np.array([lab for _, _, lab in injections], dtype=float)
        edges = np.vstack([graph['edges'][:split], pe, graph['edges'][split:]])
        labels = np.vstack([graph['labels'][:split], pl, graph['labels'][split:]])
        pindex = [b + k if i >= snap else b for i, b in enumerate(base_index)]
    else:
        edges, labels, pindex = graph['edges'], graph['labels'], list(base_index)
    pgraph = {'edges': edges, 'labels': labels, 'ecount': len(edges), 'ncount': graph['ncount']}
    orig = gcn3_mod.get_snapshot_index
    gcn3_mod.get_snapshot_index = lambda *a, **kw: list(pindex)
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        args.seed = seed   # DGTEN, SL and setup_features reseed from args.seed (seed-bug fix)
        trainer = GCNTrainer(args, pgraph, use_GPU=True)
        trainer.setup_dataset()
        trainer.create_and_train_model(startmsg=f"POISON-{ds}-s{seed}-T{T_train}-k{k}")
    finally:
        gcn3_mod.get_snapshot_index = orig
    orc = ReputationAttackOracle(trainer.model, trainer.index_list, trainer.device)
    return orc, trainer.train_edges_final, trainer.train_labels_final


def poison_matrix(ds):
    log(f"\n{'='*62}\n  POISON MATRIX — {ds}\n{'='*62}")
    # fix victims + sources from a clean seed-42 model
    o0, e0, l0 = retrain(ds, [], 42)
    strata = o0.select_stratified_targets(e0, l0, per_stratum=max(8, N_VICTIMS))
    victims = (strata.get('moderate', []) + strata.get('low', []))[:N_VICTIMS]
    rng = np.random.RandomState(42)
    plan, unrelated = {}, {}
    for v in victims:
        ex = set(e0[0, e0[1] == v].tolist())
        srcs = [int(s) for s in rng.permutation(o0.num_nodes) if s != v and s not in ex][:B]
        plan[v] = srcs
    unrelated_target = int(victims[0])
    del o0; gc.collect(); torch.cuda.empty_cache()

    controls = ['none', 'dis_victim', 'tru_victim', 'dis_unrelated', 'dis_randsnap']
    # per control: {seed: reputation} per victim, so a failed retrain cannot shift later seeds
    data = {c: {int(v): {} for v in victims} for c in controls}
    for seed in SEEDS:
        for c in controls:
            if c == 'none':
                inj = []
            elif c == 'dis_victim':
                inj = [(s, int(v), DIS) for v in victims for s in plan[v]]
            elif c == 'tru_victim':
                inj = [(s, int(v), TRU) for v in victims for s in plan[v]]
            elif c == 'dis_unrelated':
                inj = [(s, unrelated_target, DIS) for s in plan[victims[0]]]
            elif c == 'dis_randsnap':
                # place victim distrust edges in a random processed snapshot (0..5)
                rs = np.random.RandomState(seed).randint(0, 6)
                inj = [(s, int(v), DIS) for v in victims for s in plan[v]]
            snap = rs if c == 'dis_randsnap' else None
            try:
                orc, e, l = retrain(ds, inj, seed, snap_for_inject=snap)
                for v in victims:
                    data[c][int(v)][seed] = orc.compute_reputation(e, l, int(v))
                del orc, e, l; gc.collect(); torch.cuda.empty_cache()
            except Exception as ex:
                log(f"  [warn] {c} seed{seed} failed: {ex}")
    # aggregate: mean victim R per control (over seeds+victims); dR vs 'none'
    def meanR(c):
        vals = [x for v in victims for x in data[c][int(v)].values()]
        return float(np.mean(vals)) if vals else float('nan')
    def flip_and_dr(c):
        drs, flips = [], []
        for v in victims:
            base = data['none'][int(v)]
            att = data[c][int(v)]
            for sd in sorted(set(base) & set(att)):      # pair each seed with its own clean retrain
                drs.append(att[sd] - base[sd])
                if base[sd] >= 0.5 and att[sd] < 0.5:
                    flips.append(1)
                elif base[sd] >= 0.5:
                    flips.append(0)
        return (round(float(np.mean(drs)), 4) if drs else None,
                round(100 * np.mean(flips), 1) if flips else None,
                _cp_ci(sum(flips), len(flips)))
    # retraining variance: std of clean victim R across seeds
    clean_var = float(np.mean([np.std(list(data['none'][int(v)].values())) for v in victims
                               if len(data['none'][int(v)]) > 1] or [0.0]))
    res = {'B': B, 'seeds': SEEDS, 'n_victims': len(victims),
           'retrain_std_clean': round(clean_var, 4), 'controls': {}}
    for c in controls:
        dr, fl, ci = flip_and_dr(c)
        res['controls'][c] = {'mean_R': round(meanR(c), 4), 'dR_vs_none': dr, 'flip_pct': fl, 'flip_ci': ci}
        log(f"  {c:14s}: meanR={res['controls'][c]['mean_R']}  dR_vs_none={dr}  flip={fl}% CI{ci}")
    log(f"  retraining std (clean, across seeds) = {clean_var:.4f}  "
        f"(attack dR must exceed this to be real)")
    return res


def causal_retrain(ds):
    """#1 regime B: retrain through the next snapshot s7 (T=8), clean vs attack in s7.
    Victims and sources are fixed from a clean seed-42 model; every seed in SEEDS then
    retrains clean and attacked models, and each victim's flip is read against its own
    seed's clean retrain (GPU training is not bit-deterministic, so one seed is one draw)."""
    log(f"\n  CAUSAL RETRAIN (T=8, attack in new snapshot s7) — {ds}")
    o0, e0, l0 = retrain(ds, [], 42, T_train=8)
    strata = o0.select_stratified_targets(e0, l0, per_stratum=max(8, N_VICTIMS))
    victims = (strata.get('low', []) + strata.get('moderate', []))[:N_VICTIMS]
    rng = np.random.RandomState(42)
    plan = {}
    for v in victims:
        ex = set(e0[0, e0[1] == v].tolist())
        plan[int(v)] = [int(s) for s in rng.permutation(o0.num_nodes) if s != v and s not in ex][:B]
    del o0; gc.collect(); torch.cuda.empty_cache()
    inj = [(s, int(v), DIS) for v in victims for s in plan[int(v)]]
    per_seed = {}
    for seed in SEEDS:
        oc, ec, lc = retrain(ds, [], seed, T_train=8)
        r_clean = {int(v): oc.compute_reputation(ec, lc, int(v)) for v in victims}
        del oc, ec, lc; gc.collect(); torch.cuda.empty_cache()
        od, ed, ld = retrain(ds, inj, seed, T_train=8, snap_for_inject=7)
        r_att = {int(v): od.compute_reputation(ed, ld, int(v)) for v in victims}
        del od, ed, ld; gc.collect(); torch.cuda.empty_cache()
        elig = [int(v) for v in victims if r_clean[int(v)] >= 0.5]
        per_seed[seed] = {'n_eligible': len(elig), 'k_flip': sum(r_att[v] < 0.5 for v in elig),
                          'mean_dR': float(np.mean([r_att[int(v)] - r_clean[int(v)] for v in victims]))}
        log(f"    seed {seed}: eligible {len(elig)}, flipped {per_seed[seed]['k_flip']}, mean dR {per_seed[seed]['mean_dR']:+.3f}")
    k = sum(c['k_flip'] for c in per_seed.values()); n = sum(c['n_eligible'] for c in per_seed.values())
    rates = [100 * c['k_flip'] / c['n_eligible'] for c in per_seed.values() if c['n_eligible']]
    res = {'seeds': SEEDS, 'n_victims': len(victims), 'per_seed': per_seed,
           'mean_dR': round(float(np.mean([c['mean_dR'] for c in per_seed.values()])), 4),
           'flip_pct': round(100 * k / n, 1) if n else None, 'flip_ci': _cp_ci(k, n),
           'flip_pct_per_seed_mean': round(float(np.mean(rates)), 1) if rates else None,
           'flip_pct_per_seed_std': round(float(np.std(rates, ddof=1)), 1) if len(rates) > 1 else None,
           'n_eligible': n, 'k_flip': k}
    log(f"    causal-retrain T=8: pooled flip={res['flip_pct']}% CI{res['flip_ci']} over {n} victim-model pairs; "
        f"per-seed {res['flip_pct_per_seed_mean']}±{res['flip_pct_per_seed_std']}; mean dR={res['mean_dR']}")
    return res


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    parts = os.environ.get("GRAIL_POISON_PARTS", "matrix,causal").split(",")
    results = {}
    for ds in dsets:
        ds = ds.strip()
        results[ds] = {}
        if "matrix" in parts:
            results[ds]['matrix'] = poison_matrix(ds)
        if "causal" in parts:
            results[ds]['causal_retrain'] = causal_retrain(ds)
    path = os.path.join(str(RESULTS_DIR / "unified"), os.environ.get("GRAIL_POISON_OUT","poison_matrix.json"))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
