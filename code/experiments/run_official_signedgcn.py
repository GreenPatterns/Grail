"""
Official-architecture reproduction on PyTorch-Geometric SignedGCN (reviewer #3).

SignedGCN is a STATIC signed GNN, so it has no temporal window; we therefore
reproduce the two artifacts through the general data-flow control that
Proposition 1 rests on, namely whether the injected edge enters the encoder:

  in-encoder   (analog of in-window): the injected edge is added to the
               message-passing graph, so it re-embeds the target -> label-sensitive,
               single-edge propagation flip.
  not-in-encoder (analog of out-of-window / appended): the injected edge enters
               only the reputation average, scored from clean embeddings the edge
               never touched -> label-independent no-op (Prop 1).

Reputation(v) = mean over incoming edges (u,v) of P(positive | u,v), where P is
SignedGCN's own discriminator softmax. We report the four-cell placement/label
diagnostic and the single-edge low-stratum flip (total vs propagation-only).

This covers the propagation and self-conditioning claims on an official public
implementation; the temporal out-of-window artifact itself is shown on the
temporal models (Section Generality). Single seed (42) unless GRAIL_SEEDS set.

Run: GRAIL_DATASETS=otc,alpha CUDA_VISIBLE_DEVICES=0 \
     python code/experiments/run_official_signedgcn.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch

from project_paths import RESULTS_DIR
from experiment_common import log
from mycode.mainz_protocol import build_mainz_args
from mycode.utils import read_graph
from mycode.dataset import get_snapshot_index
from torch_geometric.nn import SignedGCN

DATASETS = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
SEEDS = [int(s) for s in os.environ.get("GRAIL_SEEDS", "42").split(",")]
T_TRAIN = 7
EPOCHS = 120
C_POOL = 30
PER_STRATUM = 25
DEV = 'cuda'


def _cp(k, n, alpha=0.05):
    from scipy import stats as sp
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else round(float(sp.beta.ppf(alpha / 2, k, n - k + 1)), 3)
    hi = 1.0 if k == n else round(float(sp.beta.ppf(1 - alpha / 2, k + 1, n - k)), 3)
    return (lo, hi)


def load_split(ds):
    """Processed-window (s0..s6) pos/neg edges + held-out tail (s7..s9) for AUC."""
    args = build_mainz_args(ds, epochs=1)
    args.train_time_slots = T_TRAIN
    g = read_graph(args)
    E = np.asarray(g['edges'], int)
    L = np.asarray(g['labels'], float)
    idx = get_snapshot_index(args.time_slots, args.data_path,
                             homogeneous_edges=args.homogeneous_edges)
    split = idx[T_TRAIN - 1] + 1                 # end of processed window
    tr, te = slice(0, split), slice(split, len(E))
    pos_tr = E[tr][L[tr][:, 0] >= 0.5]
    neg_tr = E[tr][L[tr][:, 1] >= 0.5]
    pos_te = E[te][L[te][:, 0] >= 0.5]
    neg_te = E[te][L[te][:, 1] >= 0.5]
    return g['ncount'], pos_tr, neg_tr, pos_te, neg_te


def to_ei(a):
    return torch.tensor(a.T, dtype=torch.long, device=DEV) if len(a) else torch.zeros((2, 0), dtype=torch.long, device=DEV)


def p_pos(model, z, ei):
    """P(positive | edges) from the discriminator softmax (class 0)."""
    if ei.shape[1] == 0:
        return torch.zeros(0, device=DEV)
    logits = model.discriminate(z, ei)          # log-softmax over [pos, neg, none]
    return logits.exp()[:, 0]


def reputation(model, z, in_src, target):
    """mean P(pos | u,target) over incoming source nodes in_src."""
    if len(in_src) == 0:
        return float('nan')
    ei = torch.tensor([list(in_src), [target] * len(in_src)], dtype=torch.long, device=DEV)
    return float(p_pos(model, z, ei).mean())


def run_seed(ds, seed):
    n, pos_tr, neg_tr, pos_te, neg_te = load_split(ds)
    torch.manual_seed(seed); np.random.seed(seed)
    pos, neg = to_ei(pos_tr), to_ei(neg_tr)
    model = SignedGCN(48, 48, num_layers=2, lamb=5).to(DEV)
    x = model.create_spectral_features(pos, neg, num_nodes=n)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        z = model(x, pos, neg)
        loss = model.loss(z, pos, neg)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z = model(x, pos, neg)
        auc, f1 = model.test(z, to_ei(pos_te), to_ei(neg_te))
    log(f"  [{ds} s{seed}] SignedGCN held-out AUC={auc:.3f} F1={f1:.3f}")

    # incoming source map on the processed window
    dst = pos_tr.T[1].tolist() + neg_tr.T[1].tolist() if len(pos_tr) or len(neg_tr) else []
    from collections import defaultdict
    incoming = defaultdict(list)
    for arr in (pos_tr, neg_tr):
        for u, v in arr:
            incoming[int(v)].append(int(u))
    with torch.no_grad():
        z_clean = model(x, pos, neg)
    clean_R = {v: reputation(model, z_clean, incoming[v], v) for v in incoming if len(incoming[v]) >= 2}
    # low stratum: clean R in [0.5, 0.6)
    low = [v for v, r in clean_R.items() if r == r and 0.5 <= r < 0.6]
    low = sorted(low, key=lambda v: abs(len(incoming[v]) - 30))[:PER_STRATUM]
    rng = np.random.RandomState(seed)

    def inject(target, s, sign, in_encoder):
        """Return propagation-only and total reputation after one edge."""
        if in_encoder:
            if sign == 'distrust':
                neg2 = torch.cat([neg, torch.tensor([[s], [target]], device=DEV)], 1); pos2 = pos
            else:
                pos2 = torch.cat([pos, torch.tensor([[s], [target]], device=DEV)], 1); neg2 = neg
            with torch.no_grad():
                z2 = model(x, pos2, neg2)
        else:
            z2 = z_clean
        prop = reputation(model, z2, incoming[target], target)                  # pre-existing only
        total = reputation(model, z2, incoming[target] + [s], target)           # incl injected
        return prop, total

    # 4-cell diagnostic (optimized source = worst-case over pool), pooled low
    cells = {k: [] for k in ('inenc_dis', 'inenc_tru', 'notenc_dis', 'notenc_tru')}
    flip_total, flip_prop = [], []
    for v in low:
        pool = [u for u in rng.permutation(n)[:C_POOL * 2] if u != v and u not in set(incoming[v])][:C_POOL]
        if not pool:
            continue
        # optimized: source giving lowest in-encoder distrust propagation R
        best_s, best_r = None, 1.0
        for s in pool:
            pr, _ = inject(v, int(s), 'distrust', True)
            if pr == pr and pr < best_r:
                best_r, best_s = pr, int(s)
        if best_s is None:
            continue
        base = clean_R[v]
        pr_d, tot_d = inject(v, best_s, 'distrust', True)
        pr_t, _ = inject(v, best_s, 'trust', True)
        npr_d, ntot_d = inject(v, best_s, 'distrust', False)
        npr_t, _ = inject(v, best_s, 'trust', False)
        cells['inenc_dis'].append(pr_d - base)
        cells['inenc_tru'].append(pr_t - base)
        cells['notenc_dis'].append(ntot_d - base)
        cells['notenc_tru'].append((reputation(model, z_clean, incoming[v] + [best_s], v)) - base)  # trust not-enc == distrust not-enc (label-blind)
        if base >= 0.5:
            flip_total.append(1 if tot_d < 0.5 else 0)
            flip_prop.append(1 if (pr_d == pr_d and pr_d < 0.5) else 0)
    def agg(a):
        a = [x for x in a if x == x]
        return round(float(np.mean(a)), 4) if a else None
    n_el = len(flip_total)
    res = {
        'auc': round(float(auc), 3), 'f1': round(float(f1), 3),
        'n_low': len(low), 'n_eligible': n_el,
        'dR_inenc_distrust': agg(cells['inenc_dis']),
        'dR_inenc_trust': agg(cells['inenc_tru']),
        'dR_notenc_distrust': agg(cells['notenc_dis']),
        'dR_notenc_trust': agg(cells['notenc_tru']),
        'label_delta_inenc': round(abs((agg(cells['inenc_dis']) or 0) - (agg(cells['inenc_tru']) or 0)), 4),
        'label_delta_notenc': round(abs((agg(cells['notenc_dis']) or 0) - (agg(cells['notenc_tru']) or 0)), 4),
        'flip_total_pct': round(100 * np.mean(flip_total), 1) if n_el else None,
        'flip_prop_pct': round(100 * np.mean(flip_prop), 1) if n_el else None,
        'flip_total_ci': _cp(sum(flip_total), n_el),
        'flip_prop_ci': _cp(sum(flip_prop), n_el),
    }
    log(f"  [{ds} s{seed}] in-enc dR dis/tru={res['dR_inenc_distrust']}/{res['dR_inenc_trust']} "
        f"lblΔ={res['label_delta_inenc']} | not-enc dR dis/tru={res['dR_notenc_distrust']}/{res['dR_notenc_trust']} "
        f"lblΔ={res['label_delta_notenc']} | flip total/prop={res['flip_total_pct']}/{res['flip_prop_pct']}% (n={n_el})")
    del model, z, z_clean; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    out = {}
    for ds in DATASETS:
        ds = ds.strip()
        out[ds] = {f"seed{s}": run_seed(ds, s) for s in SEEDS}
        path = os.path.join(str(RESULTS_DIR / "unified"), "official_signedgcn.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        log(f"saved {path}")
    log("DONE")


if __name__ == "__main__":
    main()
