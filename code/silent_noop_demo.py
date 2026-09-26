#!/usr/bin/env python
"""
Silent-No-Op Diagnostic — minimal, self-contained reviewer demo.
=================================================================
Demonstrates the paper's central, directly-checkable claim: an attack edge
APPENDED past the processed snapshot window is a *silent no-op* (its label never
reaches the encoder, so trust and distrust give identical reputation shifts),
whereas the SAME edge placed inside the processed window is label-sensitive and
damaging.

It prints, for a handful of low-reputation targets:
  * appended placement : mean ΔR  and  label-Δ = |ΔR_distrust − ΔR_trust|  (≈ 0)
  * processed placement: mean ΔR  and  label-Δ  (clearly > 0)

A reviewer needs only a trained model and a few forward passes — no attack
optimization. Runs in a couple of minutes on a single GPU (≈ a minute on CPU
for BTC-OTC).

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


def shift(oracle, edges, labels, target, src, where, label_vec):
    """ΔR of one injected edge, placed either 'processed' (last trained snapshot)
    or 'appended' (past the processed window)."""
    idx = list(oracle.model.index_list)
    T = oracle.model.args.train_time_slots
    if where == "appended":
        split, bump_from = edges.shape[1], len(idx) - 1
    else:                                   # processed: into last trained snapshot
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
        print(f"  {where:>9}: mean ΔR = {dr:+.4f}   label-Δ = {ld:.6f}")
    appended_silent = abs(np.mean(res["appended"]["dr"])) < 0.03 \
        and np.mean(res["appended"]["ld"]) < 1e-3
    processed_live = np.mean(res["processed"]["dr"]) < -0.03 \
        and np.mean(res["processed"]["ld"]) > 0.02
    print(f"\n  appended placement is a SILENT, LABEL-INDEPENDENT no-op: {appended_silent}")
    print(f"  processed placement is LABEL-SENSITIVE and damaging:       {processed_live}")
    print("\n  => The 'robust' verdict in the appended evaluation is a placement artifact.")


if __name__ == "__main__":
    main()
