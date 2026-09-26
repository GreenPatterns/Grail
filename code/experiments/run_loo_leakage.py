"""
Leave-one-edge-out / self-conditioning decomposition (reviewer blocker #2).

The reported processed-placement drop was decomposed (p3_decompose) into an
"averaging" term (label-independent dilution at frozen embeddings) and an
"embedding" term (re-embedding the labelled edges). The reviewer's concern is
that the EMBEDDING term itself may be self-label leakage: an injected edge
(s, v*, distrust) is encoded, then P(trust | s, v*) -- computed from embeddings
that just ingested that edge's own distrust label -- is averaged back into R(v*).
That would be nearly tautological rather than a security vulnerability.

This script splits the embedding term into two pieces, at the PROCESSED
placement (edge spliced into the last trained snapshot, TM-B):

  R0          clean R(v*): mean P(trust|u,v*) over PRE-EXISTING raters, enc(G)
  R_full      mean over ALL raters (pre-existing + injected), enc(G+A)  [reported]
  R_preexist  mean over PRE-EXISTING raters ONLY, enc(G+A)              [isolation]
  R_looself   mean over ALL raters, but each injected edge (s_i,v*) scored with
              enc(G+A \ {that edge}) -- removes self-conditioning from self-edges

Decomposition of the reported drop:
  dR_total     = R_full     - R0
  dR_preexist  = R_preexist - R0   <-- pure propagation to LEGIT raters
                                       (no mechanical dilution, no self-edge,
                                        no self-conditioning)
  self+mech    = R_full - R_preexist

Decisive reading:
  * dR_preexist strongly negative, label-sensitive, and flips targets
      => the injected distrust genuinely changes how the model embeds v* and
         scores its legitimate raters. NOT leakage. Strong claim holds.
  * dR_preexist ~ 0 / label-dead
      => the reported drop was self-edge/mechanical (leakage). Strong claim falls.

Sources are RANDOM non-neighbours (the naive-attacker headline). Reports the
label-Delta diagnostic (|distrust - trust|) under every scoring variant.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_loo_leakage.py
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

SEED = 42
BUDGETS = [1, 5]
PER_STRATUM = 10          # moderate + low => up to 20 targets/dataset
CLOPPER = None


def _clopper_pearson(k, n, alpha=0.05):
    from scipy import stats as sp
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else sp.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sp.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (round(float(lo), 3), round(float(hi), 3))


def _p_trust_for_cols(oracle, node_emb, target, aug_edges, cols):
    """P(trust | u, target) for the incoming edges at the given column indices,
    using the supplied node embeddings. Mirrors _reputation_score's head."""
    src = aug_edges[0, cols]
    src_emb = node_emb[src]
    dst_emb = node_emb[target].unsqueeze(0).expand_as(src_emb)
    feats = torch.cat((src_emb, dst_emb), dim=1)
    logits = feats @ oracle.model.regression_weights
    return F.softmax(logits, dim=1)[:, 0]


def _encode(oracle, edges, labels, index_list):
    orig = oracle.model.index_list
    try:
        oracle.model.index_list = index_list
        with torch.no_grad():
            return oracle._compute_node_embeddings(edges, labels)
    finally:
        oracle.model.index_list = orig


def _random_nonneighbors(oracle, edges, target, B, rng):
    existing = set()
    dst_mask = (edges[1] == target)
    for s in edges[0, dst_mask].tolist():
        existing.add(s)
    existing.add(target)
    universe = [n for n in range(oracle.num_nodes) if n not in existing]
    if len(universe) < B:
        return None
    return [int(s) for s in rng.choice(universe, B, replace=False)]


def _score_variants(oracle, edges, labels, target, srcs, sign):
    """Return (R0, R_full, R_preexist, R_looself) at processed placement."""
    B = len(srcs)
    lab_vec = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
    T = oracle.model.args.train_time_slots
    split = oracle.model.index_list[T - 1] + 1
    orig_idx = list(oracle.model.index_list)

    new_edges = torch.tensor([list(srcs), [target] * B], dtype=torch.long, device=oracle.device)
    new_labels = torch.tensor([lab_vec] * B, dtype=torch.float, device=oracle.device)
    aug_edges = torch.cat([edges[:, :split], new_edges, edges[:, split:]], dim=1)
    aug_labels = torch.cat([labels[:split], new_labels, labels[split:]], dim=0)
    aug_index = [b + B if i >= T - 1 else b for i, b in enumerate(orig_idx)]

    # incoming columns of target in aug graph; injected edges are at [split, split+B)
    dst_cols = (aug_edges[1] == target).nonzero(as_tuple=True)[0]
    injected_set = set(range(split, split + B))
    inj_cols = torch.tensor([c for c in dst_cols.tolist() if c in injected_set],
                            dtype=torch.long, device=oracle.device)
    pre_cols = torch.tensor([c for c in dst_cols.tolist() if c not in injected_set],
                            dtype=torch.long, device=oracle.device)

    # clean R0 over pre-existing raters (same rater set, clean embeddings)
    z0 = _encode(oracle, edges, labels, orig_idx)
    clean_dst = (edges[1] == target).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        R0 = _p_trust_for_cols(oracle, z0, target, edges, clean_dst).mean().item()

    # re-embedded graph
    z1 = _encode(oracle, aug_edges, aug_labels, aug_index)
    with torch.no_grad():
        p_all = _p_trust_for_cols(oracle, z1, target, aug_edges, dst_cols)
        R_full = p_all.mean().item()
        R_pre = (_p_trust_for_cols(oracle, z1, target, aug_edges, pre_cols).mean().item()
                 if len(pre_cols) > 0 else float('nan'))

    # leave-one-out on the SELF edges: re-encode without each injected edge,
    # score that injected edge's own P(trust) with the edge absent from encoding
    with torch.no_grad():
        p_pre = _p_trust_for_cols(oracle, z1, target, aug_edges, pre_cols) if len(pre_cols) > 0 \
            else torch.tensor([], device=oracle.device)
        loo_self_p = []
        for j in range(B):
            keep = [c for c in range(aug_edges.shape[1]) if c != (split + j)]
            keep = torch.tensor(keep, dtype=torch.long, device=oracle.device)
            e_j = aug_edges[:, keep]
            l_j = aug_labels[keep]
            idx_j = [b - 1 if i >= T - 1 else b for i, b in enumerate(aug_index)]
            z_j = _encode(oracle, e_j, l_j, idx_j)
            s_j = srcs[j]
            src_emb = z_j[s_j].unsqueeze(0)
            dst_emb = z_j[target].unsqueeze(0)
            logit = torch.cat((src_emb, dst_emb), dim=1) @ oracle.model.regression_weights
            loo_self_p.append(F.softmax(logit, dim=1)[0, 0].item())
        loo_self_p = torch.tensor(loo_self_p, device=oracle.device) if loo_self_p \
            else torch.tensor([], device=oracle.device)
        allp = torch.cat([p_pre, loo_self_p]) if len(p_pre) else loo_self_p
        R_looself = allp.mean().item() if len(allp) else float('nan')

    return R0, R_full, R_pre, R_looself


def run_dataset(ds):
    log(f"\n{'='*64}\n  LOO / SELF-CONDITIONING LEAKAGE TEST — {ds}\n{'='*64}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]

    strata = get_targets(oracle, edges, labels, per_stratum=PER_STRATUM)
    targets = strata.get("moderate", []) + strata.get("low", [])
    log(f"  {len(targets)} targets (moderate+low)")

    rng = np.random.RandomState(SEED)
    out = {}
    for B in BUDGETS:
        rows = {"dR_total": [], "dR_preexist": [], "dR_looself": [],
                "lblD_total": [], "lblD_preexist": [],
                "flip_total": [], "flip_preexist": [], "flip_looself": [],
                "prop_share": []}
        for t in targets:
            srcs = _random_nonneighbors(oracle, edges, t, B, rng)
            if srcs is None:
                continue
            R0d, Rf_d, Rp_d, Rl_d = _score_variants(oracle, edges, labels, t, srcs, 'distrust')
            R0t, Rf_t, Rp_t, _ = _score_variants(oracle, edges, labels, t, srcs, 'trust')
            if R0d != R0d:  # nan guard
                continue
            dR_total = Rf_d - R0d
            dR_pre = Rp_d - R0d
            dR_loo = Rl_d - R0d
            rows["dR_total"].append(dR_total)
            rows["dR_preexist"].append(dR_pre)
            rows["dR_looself"].append(dR_loo)
            rows["lblD_total"].append(abs((Rf_d - R0d) - (Rf_t - R0t)))
            rows["lblD_preexist"].append(abs((Rp_d - R0d) - (Rp_t - R0t)))
            rows["flip_total"].append(1 if (R0d >= 0.5 and Rf_d < 0.5) else 0)
            rows["flip_preexist"].append(1 if (R0d >= 0.5 and Rp_d < 0.5) else 0)
            rows["flip_looself"].append(1 if (R0d >= 0.5 and Rl_d < 0.5) else 0)
            if abs(dR_total) > 1e-9:
                rows["prop_share"].append(dR_pre / dR_total)

        def _m(x):
            x = [v for v in x if v == v]
            return float(np.mean(x)) if x else float('nan')
        n = len(rows["dR_total"])
        elig = sum(1 for i in range(n) if True)  # all counted; flips over n
        res = {
            "n": n,
            "dR_total": round(_m(rows["dR_total"]), 4),
            "dR_preexist": round(_m(rows["dR_preexist"]), 4),
            "dR_looself": round(_m(rows["dR_looself"]), 4),
            "label_delta_total": float(f"{_m(rows['lblD_total']):.2e}"),
            "label_delta_preexist": float(f"{_m(rows['lblD_preexist']):.2e}"),
            "flip_total_pct": round(100 * _m(rows["flip_total"]), 1),
            "flip_preexist_pct": round(100 * _m(rows["flip_preexist"]), 1),
            "flip_looself_pct": round(100 * _m(rows["flip_looself"]), 1),
            "flip_total_ci": _clopper_pearson(sum(rows["flip_total"]), n),
            "flip_preexist_ci": _clopper_pearson(sum(rows["flip_preexist"]), n),
            "propagation_share_mean": round(_m(rows["prop_share"]), 3),
        }
        out[f"B{B}"] = res
        log(f"  B={B}: dR_total={res['dR_total']} dR_preexist={res['dR_preexist']} "
            f"dR_looself={res['dR_looself']} | lblD_pre={res['label_delta_preexist']} | "
            f"flip_total={res['flip_total_pct']}% flip_preexist={res['flip_preexist_pct']}% "
            f"| prop_share={res['propagation_share_mean']}")
    del oracle, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {}
    for ds in dsets:
        results[ds] = run_dataset(ds.strip())
    path = os.path.join(str(RESULTS_DIR / "unified"), "loo_leakage.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
    log("\n=== SUMMARY (decisive: is dR_preexist still large & label-sensitive?) ===")
    for ds, r in results.items():
        for b, v in r.items():
            log(f"{ds} {b}: dR_total={v['dR_total']:+.3f}  dR_preexist={v['dR_preexist']:+.3f}  "
                f"prop_share={v['propagation_share_mean']}  "
                f"lblD_preexist={v['label_delta_preexist']:.1e}  "
                f"flip_pre={v['flip_preexist_pct']}%")
