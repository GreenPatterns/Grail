#!/usr/bin/env python
"""
Silent-No-Op Diagnostic — minimal, self-contained demo.
=================================================================
Demonstrates Artifact 1 of the paper (Proposition 1): an attack edge appended after the
processed window (out-of-window) is a *silent no-op*: its label never reaches the encoder, so
trust and distrust give identical reputation shifts. The same edge inside the processed window
(in-window, at the end of the last processed snapshot s6) is label-sensitive and damaging.

It trains GDTE (seed 42) and prints, for eight strata targets (the moderate stratum first):
  * out-of-window: mean total shift ΔR and label sensitivity Λ = mean |ΔR_distrust − ΔR_trust| (≈ 0)
  * in-window    : mean total shift ΔR and Λ (clearly > 0)
The code keeps its older names for the two placements: 'appended' (out-of-window) and
'processed' (in-window).

A reader needs only a trained model and a few forward passes — no attack
optimization. On Bitcoin-OTC it runs in about a minute on one GPU and in about
nine minutes on CPU (CUDA_VISIBLE_DEVICES="", eight threads), almost all of it
training.

Usage:
    GRAIL_DATASETS=otc python code/silent_noop_demo.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "common"))

import numpy as np
import torch

from experiment_common import train_model, get_targets
from mycode.trust_influence import ReputationAttackOracle

DIS, TRU = [0.0, 1.0], [1.0, 0.0]
PLACEMENT = {"appended": "out-of-window", "processed": "in-window"}   # code name -> paper term


def shift(oracle, edges, labels, target, src, where, label_vec):
    """Total shift ΔR of one attack edge, placed in-window ('processed': end of the last
    processed snapshot) or out-of-window ('appended': after the processed window)."""
    idx = list(oracle.model.index_list)
    T = oracle.model.args.train_time_slots
    if where == "appended":
        split, bump_from = edges.shape[1], len(idx) - 1
    else:                                   # in-window: end of the last processed snapshot
        split, bump_from = idx[T - 1] + 1, T - 1
    ne = torch.tensor([[src], [target]], dtype=torch.long, device=oracle.device)
    nl = torch.tensor([label_vec], dtype=torch.float, device=oracle.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], dim=1)
    al = torch.cat([labels[:split], nl, labels[split:]], dim=0)
    aug = [b + 1 if i >= bump_from else b for i, b in enumerate(idx)]
    base = oracle.compute_reputation(edges, labels, target)
    orig = oracle.model.index_list
    try:
        oracle.model.index_list = aug
        with torch.no_grad():
            emb = oracle._compute_node_embeddings(ae, al)
            new = oracle._reputation_score(emb, target, ae).item()
    finally:
        oracle.model.index_list = orig
    return new - base


def main():
    ds = os.environ.get("GRAIL_DATASETS", "otc").split(",")[0]
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    strata = get_targets(o, edges, labels, per_stratum=8)
    targets = (strata.get("moderate", []) + strata.get("low", []))[:8]
    rng = np.random.RandomState(42)

    res = {"appended": {"dr": [], "ld": []}, "processed": {"dr": [], "ld": []}}
    for t in targets:
        existing = set(edges[0, edges[1] == t].tolist())
        src = next(s for s in rng.permutation(o.num_nodes)
                   if s != t and s not in existing)
        for where in ("appended", "processed"):
            d_dis = shift(o, edges, labels, t, src, where, DIS)
            d_tru = shift(o, edges, labels, t, src, where, TRU)
            res[where]["dr"].append(d_dis)
            res[where]["ld"].append(abs(d_dis - d_tru))

    print(f"\n=== Silent-No-Op Diagnostic on {ds} (n={len(targets)} targets) ===")
    for where in ("appended", "processed"):
        dr = float(np.mean(res[where]["dr"]))
        ld = float(np.mean(res[where]["ld"]))
        print(f"  {PLACEMENT[where]:>13}: mean ΔR = {dr:+.4f}   Λ = {ld:.6f}")
    appended_silent = abs(np.mean(res["appended"]["dr"])) < 0.03 \
        and np.mean(res["appended"]["ld"]) < 1e-3
    processed_live = np.mean(res["processed"]["dr"]) < -0.03 \
        and np.mean(res["processed"]["ld"]) > 0.02
    print(f"\n  out-of-window edge is a silent, label-independent no-op: {appended_silent}")
    print(f"  in-window edge is label-sensitive and damaging:          {processed_live}")
    print("\n  => A 'robust' verdict from an out-of-window test is a silent no-op (Artifact 1).")


if __name__ == "__main__":
    main()
