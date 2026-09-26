"""
Non-GNN reputation baselines under the same attack (blocker #3) + calibration
of the GNN reputation head (blocker #10).

BASELINES. The reviewer asks: how much MORE vulnerable is the GNN than a simple
reputation system fed the SAME ratings? A label-space reputation (fraction of
trust among incoming raters) mechanically drops when distrust edges are added --
that is shared by any reputation system, not a GNN-specific flaw. We compare, on
targets initially above the 0.5 gate, the single-/five-edge distrust-injection
flip rate of:
  signed_mean : #trust / #incoming            (empirical trust fraction)
  beta_bayes  : (1+#trust)/(2+#incoming)      (Beta(1,1) posterior mean)
  wilson      : Wilson 95% lower bound of the trust proportion
  gnn_full    : GNN reputation over ALL raters (naive; == paper's number)
The GNN's EXTRA, non-mechanical exposure is the leakage-free propagation term
measured separately in loo_flip_table.json.

CALIBRATION. Over the incoming rater edges that actually feed reputation, we
compare the head's predicted P(trust|u,v) to the true edge sign and report
Brier score, expected calibration error (ECE, 10 bins), a reliability curve, and
AUC on the same set. This is in-sample over the processed window (the edges the
reputation average is taken over); we label it as such.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_repbaselines_calib.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc, math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
BUDGETS = [1, 5]
PER_STRATUM = 25


def _wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def _label_reps(n_trust, n_tot):
    """Return dict of label-space reputations given (#trust, #incoming)."""
    if n_tot == 0:
        return {"signed_mean": 0.0, "beta_bayes": 0.5, "wilson": 0.0}
    return {
        "signed_mean": n_trust / n_tot,
        "beta_bayes": (1 + n_trust) / (2 + n_tot),
        "wilson": _wilson_lower(n_trust, n_tot),
    }


def baselines(oracle, edges, labels, strata):
    """Flip rates of label-space reputations under B distrust injections."""
    trust_ind = (labels[:, 0] >= 0.5)   # class 0 = trust
    dst = edges[1]
    methods = ["signed_mean", "beta_bayes", "wilson"]
    out = {}
    for stratum in ("high", "moderate", "low"):
        ts = strata.get(stratum, [])
        for B in BUDGETS:
            elig = {m: 0 for m in methods}
            flip = {m: 0 for m in methods}
            for t in ts:
                mask = (dst == t)
                n_tot = int(mask.sum().item())
                if n_tot == 0:
                    continue
                n_trust = int(trust_ind[mask].sum().item())
                clean = _label_reps(n_trust, n_tot)
                after = _label_reps(n_trust, n_tot + B)   # add B distrust raters
                for m in methods:
                    if clean[m] >= 0.5:            # eligible: above gate when clean
                        elig[m] += 1
                        if after[m] < 0.5:
                            flip[m] += 1
            for m in methods:
                out[f"{m}_{stratum}_B{B}"] = {
                    "n_eligible": elig[m],
                    "flip_pct": round(100 * flip[m] / elig[m], 1) if elig[m] else None,
                }
    return out


def calibration(oracle, edges, labels, max_edges=40000):
    """Brier / ECE / reliability / AUC of P(trust|u,v) over the rater edges."""
    with torch.no_grad():
        z = oracle._compute_node_embeddings(edges, labels)
        E = edges.shape[1]
        idx = np.arange(E)
        if E > max_edges:
            idx = np.random.RandomState(SEED).choice(E, max_edges, replace=False)
        idx = torch.as_tensor(idx, dtype=torch.long, device=edges.device)
        u = edges[0, idx]; v = edges[1, idx]
        feats = torch.cat((z[u], z[v]), dim=1)
        p = F.softmax(feats @ oracle.model.regression_weights, dim=1)[:, 0]
        y = (labels[idx, 0] >= 0.5).float()   # 1 = trust
    p = p.cpu().numpy(); y = y.cpu().numpy()
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, 11)
    ece = 0.0; rel = []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < 9 else p <= bins[i + 1])
        if m.sum() == 0:
            rel.append(None); continue
        conf = float(p[m].mean()); acc = float(y[m].mean()); w = m.sum() / len(p)
        ece += w * abs(conf - acc)
        rel.append({"bin": [round(bins[i], 2), round(bins[i + 1], 2)],
                    "conf": round(conf, 3), "emp_trust": round(acc, 3), "n": int(m.sum())})
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = None
    return {"brier": round(brier, 4), "ece": round(float(ece), 4),
            "auc_insample": round(auc, 4) if auc else None,
            "base_trust_rate": round(float(y.mean()), 4),
            "n_edges": int(len(p)), "reliability": rel}


def run_dataset(ds):
    log(f"\n{'='*60}\n  REP-BASELINES + CALIBRATION — {ds}\n{'='*60}")
    trained = train_model(ds)
    oracle = ReputationAttackOracle(trained["model"], trained["index_list"], trained["device"])
    edges, labels = trained["edges"], trained["labels"]
    strata = get_targets(oracle, edges, labels, per_stratum=PER_STRATUM)
    res = {"baselines": baselines(oracle, edges, labels, strata),
           "calibration": calibration(oracle, edges, labels)}
    c = res["calibration"]
    log(f"  calibration: Brier={c['brier']} ECE={c['ece']} AUC(in-sample)={c['auc_insample']} "
        f"base_trust={c['base_trust_rate']}")
    for stratum in ("low", "moderate"):
        for B in BUDGETS:
            sm = res["baselines"].get(f"signed_mean_{stratum}_B{B}", {})
            bb = res["baselines"].get(f"beta_bayes_{stratum}_B{B}", {})
            log(f"  {stratum} B={B}: signed_mean flip={sm.get('flip_pct')}% "
                f"beta flip={bb.get('flip_pct')}% (n_elig={sm.get('n_eligible')})")
    del oracle, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), "repbaselines_calib.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
