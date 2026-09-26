"""
Phase 0c — the UNIMPEACHABLE realism test: genuine retraining with poison edges.

Phase 0/0b measured the realistic deployment (fixed weights, forward-pass the current
graph, attacker's recent rating inside the processed window). The strongest reviewer
objection is "frozen weights fed a perturbed input may overstate the effect vs an
honestly retrained model." This script removes that objection: it BAKES B poison
distrust edges into the TRAINING data, retrains GDTE from scratch, and measures the
victim's reputation under the genuinely retrained model.

Design (cancels retraining noise):
  R_clean(v)    : victim reputation, clean model.
  R_distrust(v) : retrain with B (src->v) edges labelled DISTRUST, measure v.
  R_trust(v)    : retrain with the SAME B edges labelled TRUST, same seed -> isolates
                  the LABEL-poison effect (everything else identical).
  Placebo       : retrain with B distrust edges to an UNRELATED node w; victims' R
                  should be ~unchanged (bounds cross-talk / retraining noise).

If R_distrust < R_trust and R_distrust < R_clean across victims, genuine training-time
poisoning works -> the threat model is real beyond any frozen-weight artifact.

Run: GRAIL_DATASETS=otc python code/experiments/run_p0c_poison_retrain.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

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

SEED = 42
B = int(os.environ.get("GRAIL_P0C_B", "5"))
N_VICTIMS = int(os.environ.get("GRAIL_P0C_NVICTIMS", "6"))
EPOCHS = 50


def train_model_with_injection(ds, injections, seed=SEED, epochs=EPOCHS):
    """injections: list of (src, dst, label_vec). Returns (oracle, edges, labels)."""
    args = build_mainz_args(ds, epochs=epochs)
    args.train_time_slots = 7
    graph = read_graph(args)
    base_index = get_snapshot_index(
        args.time_slots, args.data_path, homogeneous_edges=args.homogeneous_edges)
    T = args.train_time_slots
    split = base_index[T - 1] + 1
    k = len(injections)

    if k:
        pe = np.array([[s, d] for s, d, _ in injections], dtype=float)
        pl = np.array([lab for _, _, lab in injections], dtype=float)
        edges = np.vstack([graph['edges'][:split], pe, graph['edges'][split:]])
        labels = np.vstack([graph['labels'][:split], pl, graph['labels'][split:]])
        pindex = [b + k if i >= T - 1 else b for i, b in enumerate(base_index)]
    else:
        edges, labels, pindex = graph['edges'], graph['labels'], list(base_index)

    pgraph = {'edges': edges, 'labels': labels,
              'ecount': len(edges), 'ncount': graph['ncount']}

    orig = gcn3_mod.get_snapshot_index
    gcn3_mod.get_snapshot_index = lambda *a, **kw: list(pindex)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.seed = seed   # DGTEN, SL and setup_features reseed from args.seed (seed-bug fix)
        trainer = GCNTrainer(args, pgraph, use_GPU=True)
        trainer.setup_dataset()
        trainer.create_and_train_model(startmsg=f"POISON-{ds}-k{k}")
    finally:
        gcn3_mod.get_snapshot_index = orig

    oracle = ReputationAttackOracle(trainer.model, trainer.index_list, trainer.device)
    return oracle, trainer.train_edges_final, trainer.train_labels_final


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 0c POISON-RETRAIN — {ds}\n{'='*64}")
    DIS, TRU = [0.0, 1.0], [1.0, 0.0]

    # Clean model: pick victims, sources, baseline reputations.
    log("  Training CLEAN model...")
    o0, e0, l0 = train_model_with_injection(ds, [])
    strata = o0.select_stratified_targets(e0, l0, per_stratum=8)
    victims = (strata.get("moderate", []) + strata.get("low", []))[:N_VICTIMS]
    rng = np.random.RandomState(SEED)

    plan = {}
    r_clean = {}
    for v in victims:
        existing = set(e0[0, e0[1] == v].tolist())
        srcs = [int(s) for s in rng.permutation(o0.num_nodes)
                if s != v and s not in existing][:B]
        plan[v] = srcs
        r_clean[v] = o0.compute_reputation(e0, l0, v)
    placebo_node = victims[0]
    placebo_victim = victims[-1]
    del o0; gc.collect(); torch.cuda.empty_cache()

    rows = []
    for v in victims:
        inj_dis = [(s, v, DIS) for s in plan[v]]
        inj_tru = [(s, v, TRU) for s in plan[v]]
        od, ed, ld = train_model_with_injection(ds, inj_dis)
        r_dis = od.compute_reputation(ed, ld, v)
        del od; gc.collect(); torch.cuda.empty_cache()
        ot, et, lt = train_model_with_injection(ds, inj_tru)
        r_tru = ot.compute_reputation(et, lt, v)
        del ot; gc.collect(); torch.cuda.empty_cache()
        rows.append({"victim": int(v), "r_clean": r_clean[v],
                     "r_distrust": r_dis, "r_trust": r_tru,
                     "dr_vs_clean": r_dis - r_clean[v],
                     "dr_distrust_minus_trust": r_dis - r_tru})
        log(f"  v={v}: R_clean={r_clean[v]:.4f}  R_distrust={r_dis:.4f}  "
            f"R_trust={r_tru:.4f}  ΔR(vs clean)={r_dis-r_clean[v]:+.4f}  "
            f"ΔR(dis-tru)={r_dis-r_tru:+.4f}")

    # Placebo: poison an UNRELATED node, check another victim barely moves.
    inj_plac = [(s, placebo_node, DIS) for s in plan[placebo_node]]
    op, ep, lp = train_model_with_injection(ds, inj_plac)
    r_plac = op.compute_reputation(ep, lp, placebo_victim)
    placebo_shift = r_plac - r_clean[placebo_victim]
    del op; gc.collect(); torch.cuda.empty_cache()
    log(f"  placebo: poison node {placebo_node}, measure victim {placebo_victim}: "
        f"ΔR={placebo_shift:+.4f} (should be ~0)")

    res = {
        "rows": rows,
        "mean_dr_vs_clean": float(np.mean([r["dr_vs_clean"] for r in rows])),
        "mean_dr_distrust_minus_trust": float(np.mean([r["dr_distrust_minus_trust"] for r in rows])),
        "placebo_shift": float(placebo_shift),
        "B": B, "n_victims": len(rows),
    }
    log(f"\n  SUMMARY [{ds}] genuine retrain, B={B}:")
    log(f"    mean ΔR vs clean       = {res['mean_dr_vs_clean']:+.4f}")
    log(f"    mean ΔR (distrust-trust) = {res['mean_dr_distrust_minus_trust']:+.4f}")
    log(f"    placebo shift            = {res['placebo_shift']:+.4f}")
    return res


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    out = RESULTS_DIR / "unified" / "p0c_poison_retrain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {out}")

    log(f"\n{'='*64}\n  POISON-RETRAIN VERDICT\n{'='*64}")
    for ds, r in results.items():
        genuine = r["mean_dr_vs_clean"] < -0.01 and r["mean_dr_distrust_minus_trust"] < -0.01 \
            and abs(r["placebo_shift"]) < abs(r["mean_dr_vs_clean"])
        log(f"  {ds}: ΔR_vs_clean={r['mean_dr_vs_clean']:+.4f}  "
            f"ΔR_dis-tru={r['mean_dr_distrust_minus_trust']:+.4f}  "
            f"placebo={r['placebo_shift']:+.4f} => "
            f"{'PASS — survives genuine retrain' if genuine else 'WASHES OUT under retrain'}")


if __name__ == "__main__":
    main()
