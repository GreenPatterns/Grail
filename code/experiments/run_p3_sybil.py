"""
Phase 3 — genuine new-account (Sybil) attack. Instead of choosing existing
non-neighbor sources, we inject B \emph{brand-new} zero-history nodes (generic
features = column-mean of X, i.e. an average-looking fresh account) that each
rate the target with distrust, spliced into the processed window and re-embedded.
This tests deployment realism: can an attacker who only creates new accounts
(rather than controlling established nodes) still flip low-reputation targets?

We compare three source regimes at the correct placement:
  existing  : random existing non-neighbor accounts (current setup),
  dormant   : lowest-out-degree existing accounts (minimal history),
  new-Sybil : freshly injected nodes with no prior edges.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_sybil.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import json, gc
import numpy as np
import torch
from scipy.stats import beta
from project_paths import RESULTS_DIR
from experiment_common import train_model, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
PER_STRATUM = 16


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def score_new_nodes(o, edges, labels, X, target, B, sign='distrust'):
    """Inject B new zero-history nodes (mean features) rating `target`, processed
    placement, re-embed. Returns (dr, r_before, r_after)."""
    N = X.shape[0]
    newfeat = X.mean(0, keepdim=True).repeat(B, 1)
    X_ext = torch.cat([X, newfeat], 0)
    new_ids = list(range(N, N + B))
    T = o.model.args.train_time_slots
    split = o.model.index_list[T - 1] + 1
    lab = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
    ne = torch.tensor([new_ids, [target] * B], dtype=torch.long, device=o.device)
    nl = torch.tensor([lab] * B, dtype=torch.float, device=o.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], dim=1)
    al = torch.cat([labels[:split], nl, labels[split:]], dim=0)
    aug_index = [b + B if i >= T - 1 else b for i, b in enumerate(o.model.index_list)]
    rb = o.compute_reputation(edges, labels, target)
    sl = o.model.structural_layer
    ox, on, oi = sl.X_raw, o.model.nodes, o.model.index_list
    try:
        sl.X_raw = X_ext
        o.model.nodes = N + B
        o.model.index_list = aug_index
        with torch.no_grad():
            emb = o._compute_node_embeddings(ae, al)
            ra = o._reputation_score(emb, target, ae).item()
    finally:
        sl.X_raw, o.model.nodes, o.model.index_list = ox, on, oi
    return ra - rb, rb, ra


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 3 SYBIL (new-account injection) — {ds}\n{'='*64}")
    tr = train_model(ds)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    X = o.model.X if torch.is_tensor(o.model.X) else torch.as_tensor(o.model.X, dtype=torch.float)
    X = X.float().to(o.device)
    outdeg = torch.bincount(edges[0], minlength=o.num_nodes).cpu().numpy()
    rng = np.random.RandomState(SEED)
    strata = o.select_stratified_targets(edges, labels, per_stratum=PER_STRATUM)
    vics = strata.get("low", [])
    out = {}
    for B in (1, 5):
        cells = {k: {"flips": 0, "elig": 0, "dr": []} for k in ("existing", "dormant", "new_sybil")}
        for v in vics:
            existing = set(edges[0, edges[1] == v].tolist())
            cands = [s for s in range(o.num_nodes) if s != v and s not in existing]
            srcs = {
                "existing": [int(s) for s in rng.permutation(cands)[:B]],
                "dormant": sorted(cands, key=lambda s: outdeg[s])[:B],
            }
            for k, ss in srcs.items():
                dr, rb, ra = o.score_edge_set_processed(edges, labels, ss, v, 'distrust', return_after=True)
                cells[k]["dr"].append(dr)
                if rb >= 0.5:
                    cells[k]["elig"] += 1; cells[k]["flips"] += int(ra < 0.5)
            dr, rb, ra = score_new_nodes(o, edges, labels, X, v, B, 'distrust')
            cells["new_sybil"]["dr"].append(dr)
            if rb >= 0.5:
                cells["new_sybil"]["elig"] += 1; cells["new_sybil"]["flips"] += int(ra < 0.5)
        for k, c in cells.items():
            fr = c["flips"] / max(c["elig"], 1)
            out[f"{k}_B{B}"] = {"flip_rate": fr, "dr": float(np.mean(c["dr"])),
                                "k_n": [c["flips"], c["elig"]], "flip_ci": cp(c["flips"], c["elig"])}
            log(f"  B={B} {k:>10}: flip={fr:.0%} {cp(c['flips'], c['elig'])} ΔR={np.mean(c['dr']):+.3f} "
                f"(k/n={c['flips']}/{c['elig']})")
    del tr, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    outp = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p3_sybil.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {outp}")


if __name__ == "__main__":
    main()
