"""
Per-snapshot sweep and sign control (E1; Fig. 2 and Section VI-A): one distrust edge from a
random source, inserted into each processed snapshot s0..s6 in turn or out-of-window (appended
after the processed window), for 12 strata targets (8 moderate, 4 low). JSON keys keep the code's
names: `appended` is out-of-window, and `sweep_label_delta` holds the label sensitivity Lambda.

The two placements correspond to two deployment regimes (Section II):

  frozen embeddings (TM-A in the code): train_edges_final = snapshots 0..W-1; the model embeds
       once and scores a new link with the head on the endpoint embeddings (gcn3.test_model).
       A new edge is never re-embedded, so its label is irrelevant.
  re-embedding (TM-B in the code): the attack edge is spliced INTO a processed snapshot and the
       model re-embeds, so its label matters.

The script checks:

  V1 Structural facts: |train_edges_final|, len(index_list), where an out-of-window edge lands.
  V2 Placement sweep: insert the SAME distrust edge into snapshot s = 0..W-1 and out-of-window;
     record the shift and Lambda. If every processed snapshot responds and only the
     out-of-window position is ~0, the effect is processed vs not, not a chosen snapshot.
  V3 Sign control at the last processed snapshot (Check 4): a trust edge should RAISE the score,
     a distrust edge LOWER it, and a neutral edge fall in between.
  V4 Rater-count confound: correlate |shift| with the target's number of raters; report Lambda
     (independent of averaging) separately.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p0b_verify.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch
from scipy import stats as sp_stats

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42


def insert_at_snapshot(oracle, edges, labels, target, src, snapshot, label_vec):
    """Splice one edge into a given snapshot (shift boundaries >= snapshot)."""
    idx = list(oracle.model.index_list)
    if snapshot == "appended":
        split = edges.shape[1]
        bump_from = len(idx) - 1
    else:
        split = idx[snapshot] + 1
        bump_from = snapshot
    ne = torch.tensor([[src], [target]], dtype=torch.long, device=oracle.device)
    nl = torch.tensor([label_vec], dtype=torch.float, device=oracle.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], dim=1)
    al = torch.cat([labels[:split], nl, labels[split:]], dim=0)
    aug_idx = [b + 1 if i >= bump_from else b for i, b in enumerate(idx)]
    orig = oracle.model.index_list
    base = oracle.compute_reputation(edges, labels, target)
    try:
        oracle.model.index_list = aug_idx
        with torch.no_grad():
            emb = oracle._compute_node_embeddings(ae, al)
            new = oracle._reputation_score(emb, target, ae).item()
    finally:
        oracle.model.index_list = orig
    return new - base


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 0b VERIFY — {ds}\n{'='*64}")
    layers_env = os.environ.get("GRAIL_LAYERS", "")
    layers = [int(x) for x in layers_env.split(",")] if layers_env else None
    if layers:
        log(f"  architecture variant: layers={layers}")
    tr = train_model(ds, layers=layers)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    T = o.model.args.train_time_slots
    idx = list(o.model.index_list)

    # V1 structural facts
    n_edges = edges.shape[1]
    last_trained_boundary = idx[T - 1]
    log(f"  V1: |train_edges_final|={n_edges}  len(index_list)={len(idx)}  "
        f"T={T}  last_trained_boundary=idx[{T-1}]={last_trained_boundary}")
    log(f"      idx[T-1..end]={idx[T-1:]}  -> appended edge index {n_edges} is "
        f"{'PAST' if n_edges > last_trained_boundary else 'within'} the processed window")

    strata = get_targets(o, edges, labels, per_stratum=8)
    targets = (strata.get("moderate", []) + strata.get("low", []))[:12]
    in_deg = np.bincount(edges[1].cpu().numpy(), minlength=o.num_nodes)
    rng = np.random.RandomState(SEED)

    DIS, TRU, NEU = [0., 1.], [1., 0.], [0.5, 0.5]
    sweep_dr = {f"s{s}": [] for s in range(T)}
    sweep_dr["appended"] = []
    sweep_lbldelta = {f"s{s}": [] for s in range(T)}
    sweep_lbldelta["appended"] = []
    sign_trust, sign_neutral, sign_distrust = [], [], []
    indeg_list, absdr_processed = [], []

    for t in targets:
        existing_src = set(edges[0, edges[1] == t].tolist())
        src = next(s for s in rng.permutation(o.num_nodes)
                   if s != t and s not in existing_src)

        for s in list(range(T)) + ["appended"]:
            key = f"s{s}" if s != "appended" else "appended"
            d_dis = insert_at_snapshot(o, edges, labels, t, src, s, DIS)
            d_tru = insert_at_snapshot(o, edges, labels, t, src, s, TRU)
            sweep_dr[key].append(d_dis)
            sweep_lbldelta[key].append(abs(d_dis - d_tru))

        # V3 sign test at last trained snapshot
        sign_distrust.append(insert_at_snapshot(o, edges, labels, t, src, T - 1, DIS))
        sign_neutral.append(insert_at_snapshot(o, edges, labels, t, src, T - 1, NEU))
        sign_trust.append(insert_at_snapshot(o, edges, labels, t, src, T - 1, TRU))

        indeg_list.append(int(in_deg[t]))
        absdr_processed.append(abs(sweep_dr[f"s{T-1}"][-1]))

    def m(x): return float(np.mean(x)) if x else float("nan")

    log(f"  V2 placement sweep — mean ΔR (distrust) by snapshot:")
    for s in list(range(T)) + ["appended"]:
        key = f"s{s}" if s != "appended" else "appended"
        log(f"      {key:>9}: ΔR={m(sweep_dr[key]):+.4f}   label-Δ={m(sweep_lbldelta[key]):.4f}")
    log(f"  V3 sign test @ snap {T-1}: trust={m(sign_trust):+.4f}  "
        f"neutral={m(sign_neutral):+.4f}  distrust={m(sign_distrust):+.4f}")
    rho_indeg = float(sp_stats.spearmanr(indeg_list, absdr_processed)[0]) if len(indeg_list) > 2 else float("nan")
    log(f"  V4 in-degree vs |ΔR_processed|: ρ={rho_indeg:+.3f}  "
        f"(in-deg range {min(indeg_list)}-{max(indeg_list)})")

    res = {
        "n_edges": int(n_edges), "len_index_list": len(idx), "T": int(T),
        "last_trained_boundary": int(last_trained_boundary),
        "appended_is_past_window": bool(n_edges > last_trained_boundary),
        "sweep_dr": {k: m(v) for k, v in sweep_dr.items()},
        "sweep_label_delta": {k: m(v) for k, v in sweep_lbldelta.items()},
        "sign_trust": m(sign_trust), "sign_neutral": m(sign_neutral),
        "sign_distrust": m(sign_distrust),
        "indeg_vs_absdr_rho": rho_indeg,
        "indeg": indeg_list, "absdr_processed": absdr_processed,
    }
    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    out = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p0b_verify.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {out}")

    log(f"\n{'='*64}\n  VERIFICATION VERDICT\n{'='*64}")
    for ds, r in results.items():
        processed_strong = all(
            r["sweep_dr"][f"s{s}"] < -0.01 for s in range(r["T"])
        )
        appended_null = abs(r["sweep_dr"]["appended"]) < 0.02
        label_alive = all(r["sweep_label_delta"][f"s{s}"] > 0.02 for s in range(r["T"]))
        label_dead_appended = r["sweep_label_delta"]["appended"] < 1e-3
        sign_ok = r["sign_trust"] > r["sign_neutral"] > r["sign_distrust"]
        log(f"  {ds}: processed_strong={processed_strong} appended_null={appended_null} "
            f"label_alive_processed={label_alive} label_dead_appended={label_dead_appended} "
            f"sign_monotonic={sign_ok}")


if __name__ == "__main__":
    main()
