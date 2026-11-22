"""
Phase 2 — silent-no-op SIGNATURE + vulnerability on INDEPENDENT architectures.

Shows the placement hazard is a property of the prefix-snapshot pattern, not of
GDTE, by reproducing it on two further, deliberately different families:
  * SignedSnapshotGNN  — SGCN/SDGNN-style signed mean pools + GRU temporal head;
  * EvolveGCNLite       — EvolveGCN-style GRU weight-evolution + signed graph conv.
GDTE itself uses spectral-gated RFF layers + S3R attention, so the three span
distinct structural and temporal mechanisms.

For each architecture we (a) train to a held-out link-prediction AUC (transductive
10% edge hold-out, so the number is generalization, not in-sample fit), then run
the SAME diagnostic used for GDTE (run_p0b_verify):
  * placement sweep: distrust edge into each processed snapshot s0..s_{T-1} vs
    "appended"; record ΔR and label-Δ (Proposition 1 predicts label-Δ=0 appended);
  * sign test @ last processed snapshot: trust raises / neutral ~0 / distrust lowers;
  * flip grid: optimized single/five-edge threshold-flip rate, low+moderate strata.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_p2_arch_signature.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.signed_snapshot_gnn import SignedSnapshotGNN
from mycode.evolvegcn_lite import EvolveGCNLite
from run_p0b_verify import insert_at_snapshot

SEED = 42
EPOCHS = int(os.environ.get("GRAIL_P2_EPOCHS", "200"))
PER_STRATUM = 16
ARCHS = [("SignedSnapshotGNN", SignedSnapshotGNN), ("EvolveGCN", EvolveGCNLite)]


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(.975, k + 1, n - k)
    return [round(float(lo), 3), round(float(hi), 3)]


def train_arch(model_cls, name, edges, labels, index_list, args, X, device):
    """Train an independent architecture with a transductive 10% held-out split,
    returning the model and its held-out link-prediction AUC."""
    torch.manual_seed(SEED)
    model = model_cls(device, args, X, num_labels=2, index_list=index_list)
    y = (labels[:, 1] > labels[:, 0]).long().to(device)
    T = args.train_time_slots
    n_proc = index_list[T - 1] + 1
    g = torch.Generator().manual_seed(SEED)
    held = torch.zeros(edges.shape[1], dtype=torch.bool)
    sel = torch.arange(n_proc)[torch.randperm(n_proc, generator=g)[: max(1, n_proc // 10)]]
    held[sel] = True
    train_mask = ~held
    cw = torch.bincount(y[train_mask], minlength=2).float()
    cw = (cw.sum() / (cw + 1.0)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        temp = model.s3r(model._process_structural_layer(edges, labels))[:, T - 1, :]
        feats = torch.cat((temp[edges[0, train_mask]], temp[edges[1, train_mask]]), dim=1)
        loss = F.cross_entropy(feats @ model.regression_weights, y[train_mask], weight=cw)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        temp = model.s3r(model._process_structural_layer(edges, labels))[:, T - 1, :]
        fh = torch.cat((temp[edges[0, held]], temp[edges[1, held]]), dim=1)
        ph = torch.softmax(fh @ model.regression_weights, dim=1)[:, 0]
        yh = (1 - y[held]).cpu().numpy()
        try:
            auc = roc_auc(yh, ph.cpu().numpy())
        except Exception:
            auc = float("nan")
    log(f"  [{name}] train loss={loss.item():.3f}  held-out link AUC={auc:.3f}")
    return model, auc


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")


def diagnose(o, edges, labels, strata, T, auc, name):
    rng = np.random.RandomState(SEED)
    targets = (strata.get("moderate", []) + strata.get("low", []))[:12]
    DIS, TRU, NEU = [0., 1.], [1., 0.], [0.5, 0.5]
    sweep_dr = {f"s{s}": [] for s in range(T)}; sweep_dr["appended"] = []
    sweep_ld = {f"s{s}": [] for s in range(T)}; sweep_ld["appended"] = []
    s_tru, s_neu, s_dis = [], [], []
    for t in targets:
        existing = set(edges[0, edges[1] == t].tolist())
        src = next(s for s in rng.permutation(o.num_nodes) if s != t and s not in existing)
        for s in list(range(T)) + ["appended"]:
            key = f"s{s}" if s != "appended" else "appended"
            d_dis = insert_at_snapshot(o, edges, labels, t, src, s, DIS)
            d_tru = insert_at_snapshot(o, edges, labels, t, src, s, TRU)
            sweep_dr[key].append(d_dis); sweep_ld[key].append(abs(d_dis - d_tru))
        s_dis.append(insert_at_snapshot(o, edges, labels, t, src, T - 1, DIS))
        s_neu.append(insert_at_snapshot(o, edges, labels, t, src, T - 1, NEU))
        s_tru.append(insert_at_snapshot(o, edges, labels, t, src, T - 1, TRU))

    def m(x): return float(np.mean(x)) if x else float("nan")

    flip = {}
    frng = np.random.RandomState(SEED)
    for stratum in ["low", "moderate"]:
        vics = strata.get(stratum, [])[:PER_STRATUM]
        if not vics:
            continue
        for B in (1, 3, 5):
            drs, flips, elig = [], 0, 0
            for v in vics:
                existing = set(edges[0, edges[1] == v].tolist())
                pool = [int(s) for s in frng.permutation(o.num_nodes)
                        if s != v and s not in existing][:50]
                sc = o.score_candidates_counterfactual_processed(edges, labels, v, pool, sign='distrust')
                opt = [s for s, _ in sorted(sc.items(), key=lambda x: x[1])][:B]
                _, rb, ra = o.score_edge_set_processed(edges, labels, opt, v, 'distrust', return_after=True)
                drs.append(ra - rb)
                if rb >= 0.5:
                    elig += 1; flips += int(ra < 0.5)
            flip[f"{stratum}_B{B}"] = {"dr": m(drs), "flip_rate": flips / max(elig, 1),
                                       "k_n": [flips, elig], "flip_ci": cp(flips, elig)}
            log(f"    flip {stratum:>8} B={B}: ΔR={m(drs):+.4f} flip={flips/max(elig,1):.0%} "
                f"{cp(flips, elig)} (k/n={flips}/{elig})")
    log(f"    appended ΔR={m(sweep_dr['appended']):+.4f} label-Δ={m(sweep_ld['appended']):.5f}  "
        f"| sign test trust={m(s_tru):+.3f}/neu={m(s_neu):+.3f}/dis={m(s_dis):+.3f}")
    return {
        "heldout_link_auc": float(auc), "T": int(T),
        "sweep_dr": {k: m(v) for k, v in sweep_dr.items()},
        "sweep_label_delta": {k: m(v) for k, v in sweep_ld.items()},
        "sign_trust": m(s_tru), "sign_neutral": m(s_neu), "sign_distrust": m(s_dis),
        "flip": flip,
    }


def run_dataset(ds):
    log(f"\n{'='*64}\n  PHASE 2 ARCH SIGNATURE — {ds}\n{'='*64}")
    tr = train_model(ds)
    device, edges, labels = tr["device"], tr["edges"], tr["labels"]
    args, index_list, X = tr["model"].args, tr["index_list"], tr["model"].X
    del tr; gc.collect(); torch.cuda.empty_cache()
    out = {}
    for name, cls in ARCHS:
        log(f"  --- {name} ---")
        model, auc = train_arch(cls, name, edges, labels, index_list, args, X, device)
        o = ReputationAttackOracle(model, index_list, device)
        T = o.model.args.train_time_slots
        strata = get_targets(o, edges, labels, per_stratum=PER_STRATUM)
        out[name] = diagnose(o, edges, labels, strata, T, auc, name)
        del model, o; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    datasets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds: run_dataset(ds) for ds in datasets}
    out = RESULTS_DIR / "unified" / os.environ.get("GRAIL_OUT", "p2_arch_signature.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nSaved {out}")

    log(f"\n{'='*64}\n  SIGNATURE VERDICT (independent architectures)\n{'='*64}")
    for ds, archs in results.items():
        for name, r in archs.items():
            T = r["T"]
            appended_null = abs(r["sweep_dr"]["appended"]) < 0.02
            label_dead = r["sweep_label_delta"]["appended"] < 1e-3
            label_alive = all(r["sweep_label_delta"][f"s{s}"] > 0.01 for s in range(T))
            sign_ok = r["sign_trust"] > r["sign_neutral"] > r["sign_distrust"]
            log(f"  {ds}/{name}: appended_null={appended_null} label_dead={label_dead} "
                f"label_alive={label_alive} sign_monotonic={sign_ok} (AUC={r['heldout_link_auc']:.3f})")


if __name__ == "__main__":
    main()
