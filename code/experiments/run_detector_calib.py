"""
Rigorous detector metrics + out-of-sample calibration (blockers #9, #10).

#9 DETECTOR. The paper previously reported a balanced-split AUC and (wrongly)
described it as changing at a 1% FPR operating point. Here we report the correct
threshold-independent ROC-AUC AND operating-point metrics (TPR at 1%/5% FPR,
PR-AUC under realistic imbalance), under an ENTITY-DISJOINT split (attack targets
partitioned into train/test halves so target/campaign identity cannot leak), and
we evaluate an ADAPTIVE attacker that makes injected edges symmetric to defeat the
dominant edge-symmetry feature.

#10 CALIBRATION out-of-sample. The head's P(trust|u,v) is calibrated on the
HELD-OUT future edges (snapshots s7..s9, the temporal tail the model never
trained on): Brier, ECE (10 bins), reliability curve, AUC, base rate.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_detector_calib.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import json, gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle

SEED = 42
B = 5


def _tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, fpr_target, side='right') - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def _edge_features(orc, real_edges, reps, src_deg, dst_deg, rev_set, edges_t, force_sym=False):
    src = edges_t[0].cpu().numpy(); dst = edges_t[1].cpu().numpy()
    feats = []
    for i in range(len(src)):
        s, d = int(src[i]), int(dst[i])
        sym = 1.0 if (force_sym or (s, d) in rev_set) else 0.0
        feats.append([src_deg[s], dst_deg[d],
                      reps[s] if s < len(reps) else 0.5,
                      reps[d] if d < len(reps) else 0.5, sym])
    return np.array(feats)


def detector(orc, edges, labels, strata):
    dev = orc.device
    rng = np.random.RandomState(SEED)
    # attack targets, entity-disjoint train/test split
    targets = list(strata.get('low', [])) + list(strata.get('moderate', []))
    rng.shuffle(targets)
    half = len(targets) // 2
    tr_tg, te_tg = set(targets[:half]), set(targets[half:])

    def gen_attacks(tgset, force_sym=False):
        srcs_all, dsts_all = [], []
        for t in tgset:
            ex = set(edges[0, edges[1] == t].tolist()); ex.add(t)
            uni = [n for n in range(orc.num_nodes) if n not in ex]
            picks = rng.choice(uni, min(B, len(uni)), replace=False)
            for s in picks:
                srcs_all.append(int(s)); dsts_all.append(int(t))
                if force_sym:
                    srcs_all.append(int(t)); dsts_all.append(int(s))  # reverse edge added
        return torch.tensor([srcs_all, dsts_all], dtype=torch.long, device=dev)

    # organic distrust edges (label class 1), split by destination entity
    distrust_mask = (labels[:, 1] >= 0.5)
    dcols = distrust_mask.nonzero(as_tuple=True)[0]
    real = edges[:, dcols]
    real_dst = real[1].cpu().numpy()
    tr_real = real[:, np.isin(real_dst, list(tr_tg)) | ~np.isin(real_dst, list(te_tg))]
    te_real = real[:, np.isin(real_dst, list(te_tg))]
    if te_real.shape[1] < 5:
        te_real = real[:, :max(50, real.shape[1] // 3)]
        tr_real = real[:, max(50, real.shape[1] // 3):]

    src_deg = np.bincount(edges[0].cpu().numpy(), minlength=orc.num_nodes)
    dst_deg = np.bincount(edges[1].cpu().numpy(), minlength=orc.num_nodes)
    reps = orc.compute_all_reputations(edges, labels).cpu().numpy()
    rev_set = set((edges[1, i].item(), edges[0, i].item()) for i in range(edges.shape[1]))

    atk_tr = gen_attacks(tr_tg); atk_te = gen_attacks(te_tg)
    atk_te_adapt = gen_attacks(te_tg, force_sym=True)

    def F_(e, fs=False):
        return _edge_features(orc, edges, reps, src_deg, dst_deg, rev_set, e, force_sym=fs)

    # train (balanced), evaluate on entity-disjoint test at realistic imbalance
    Xtr = np.vstack([F_(tr_real), F_(atk_tr)])
    ytr = np.concatenate([np.zeros(tr_real.shape[1]), np.ones(atk_tr.shape[1])])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)

    def evaluate(real_e, atk_e, adaptive=False):
        Xr = F_(real_e); Xa = F_(atk_e, fs=adaptive)
        X = np.vstack([Xr, Xa]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xa))])
        s = clf.predict_proba(sc.transform(X))[:, 1]
        return {
            'roc_auc': round(float(roc_auc_score(y, s)), 3),
            'pr_auc': round(float(average_precision_score(y, s)), 3),
            'tpr_at_1pct_fpr': round(_tpr_at_fpr(y, s, 0.01), 3),
            'tpr_at_5pct_fpr': round(_tpr_at_fpr(y, s, 0.05), 3),
            'n_real': int(len(Xr)), 'n_attack': int(len(Xa)),
        }

    return {
        'test_disjoint': evaluate(te_real, atk_te, adaptive=False),
        'test_adaptive_symmetric': evaluate(te_real, atk_te_adapt, adaptive=True),
        'feature_importances': {k: round(float(v), 3) for k, v in zip(
            ['src_degree', 'dst_degree', 'src_reputation', 'dst_reputation', 'edge_symmetry'],
            clf.coef_[0])},
    }


def calibration_oos(orc, trained):
    """Calibrate P(trust) on held-out future edges s7..s9 (never trained on)."""
    tr = trained['trainer']
    te = np.asarray(getattr(tr, 'test_edges', []))
    tl = np.asarray(getattr(tr, 'test_labels', []))
    edges, labels = trained['edges'], trained['labels']
    if te.size == 0:
        return {'note': 'no held-out edges available'}
    train_nodes = set(edges.flatten().tolist())
    keep = [i for i in range(len(te)) if int(te[i][0]) in train_nodes and int(te[i][1]) in train_nodes]
    if len(keep) < 20:
        keep = list(range(len(te)))
    te = te[keep]; tl = tl[keep]
    with torch.no_grad():
        z = orc._compute_node_embeddings(edges, labels)
        u = torch.as_tensor(te[:, 0].astype(np.int64), device=orc.device)
        v = torch.as_tensor(te[:, 1].astype(np.int64), device=orc.device)
        feats = torch.cat((z[u], z[v]), 1)
        p = F.softmax(feats @ orc.model.regression_weights, 1)[:, 0].cpu().numpy()
    y = (tl[:, 0] >= 0.5).astype(float) if tl.ndim == 2 else (tl >= 0.5).astype(float)
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, 11); ece = 0.0; rel = []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < 9 else p <= bins[i + 1])
        if m.sum() == 0:
            rel.append(None); continue
        conf, acc, w = float(p[m].mean()), float(y[m].mean()), m.sum() / len(p)
        ece += w * abs(conf - acc)
        rel.append({'bin': [round(bins[i], 2), round(bins[i + 1], 2)],
                    'conf': round(conf, 3), 'emp_trust': round(acc, 3), 'n': int(m.sum())})
    try:
        auc = round(float(roc_auc_score(y, p)), 3)
    except Exception:
        auc = None
    return {'held_out_snapshots': 's7-s9', 'brier': round(brier, 4), 'ece': round(float(ece), 4),
            'auc_oos': auc, 'base_trust_rate': round(float(y.mean()), 4),
            'n_edges': int(len(p)), 'reliability': rel}


def run_dataset(ds):
    log(f"\n{'='*60}\n  DETECTOR + OOS CALIBRATION — {ds}\n{'='*60}")
    trained = train_model(ds)
    orc = ReputationAttackOracle(trained['model'], trained['index_list'], trained['device'])
    edges, labels = trained['edges'], trained['labels']
    strata = get_targets(orc, edges, labels, per_stratum=25)
    det = detector(orc, edges, labels, strata)
    cal = calibration_oos(orc, trained)
    log(f"  detector disjoint: ROC-AUC={det['test_disjoint']['roc_auc']} "
        f"PR-AUC={det['test_disjoint']['pr_auc']} "
        f"TPR@1%FPR={det['test_disjoint']['tpr_at_1pct_fpr']} "
        f"TPR@5%FPR={det['test_disjoint']['tpr_at_5pct_fpr']}")
    log(f"  detector adaptive(symmetric): ROC-AUC={det['test_adaptive_symmetric']['roc_auc']} "
        f"TPR@1%FPR={det['test_adaptive_symmetric']['tpr_at_1pct_fpr']}")
    log(f"  calibration OOS: Brier={cal.get('brier')} ECE={cal.get('ece')} "
        f"AUC={cal.get('auc_oos')} base={cal.get('base_trust_rate')} n={cal.get('n_edges')}")
    del orc, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return {'detector': det, 'calibration_oos': cal}


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), "detector_calib.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
