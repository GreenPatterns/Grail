"""
Multi-seed stability of the leakage-free single-edge flip (reviewer blocker #7).

The main flip result used one trained model (seed 42). This retrains GDTE from
scratch under seeds {42,1,2,3,4} and, for each, recomputes the headline low-
stratum single-edge (B=1) optimized-source flip rate under BOTH the naive and the
leakage-free score. Reports mean +/- std across seeds, so the directional claim is
read against training variance rather than a single draw.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_multiseed_flip.py
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

SEEDS = [42, 1, 2, 3, 4]
PER_STRATUM = 25
C_POOL = 50
B = 1


def _ptrust(orc, z, target, edges, cols):
    s = edges[0, cols]
    feats = torch.cat((z[s], z[target].unsqueeze(0).expand(len(cols), -1)), 1)
    return F.softmax(feats @ orc.model.regression_weights, 1)[:, 0]


def _enc(orc, edges, labels, idx):
    sI = orc.model.index_list
    try:
        orc.model.index_list = idx
        with torch.no_grad():
            return orc._compute_node_embeddings(edges, labels)
    finally:
        orc.model.index_list = sI


def _variants(orc, edges, labels, target, srcs):
    m = orc.model
    T = m.args.train_time_slots
    split = m.index_list[T - 1] + 1
    oi = list(m.index_list)
    ne = torch.tensor([list(srcs), [target] * len(srcs)], dtype=torch.long, device=orc.device)
    nl = torch.tensor([[0.0, 1.0]] * len(srcs), dtype=torch.float, device=orc.device)
    ae = torch.cat([edges[:, :split], ne, edges[:, split:]], 1)
    al = torch.cat([labels[:split], nl, labels[split:]], 0)
    ai = [b + len(srcs) if i >= T - 1 else b for i, b in enumerate(oi)]
    dst = (ae[1] == target).nonzero(as_tuple=True)[0]
    inj = set(range(split, split + len(srcs)))
    pre = torch.tensor([c for c in dst.tolist() if c not in inj], dtype=torch.long, device=orc.device)
    z0 = _enc(orc, edges, labels, oi)
    cd = (edges[1] == target).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        R0 = _ptrust(orc, z0, target, edges, cd).mean().item()
    z1 = _enc(orc, ae, al, ai)
    with torch.no_grad():
        Rf = _ptrust(orc, z1, target, ae, dst).mean().item()
        Rp = _ptrust(orc, z1, target, ae, pre).mean().item() if len(pre) else float('nan')
    return R0, Rf, Rp


def _pool(orc, edges, target, C, rng):
    ex = set(edges[0, edges[1] == target].tolist()); ex.add(target)
    uni = [n for n in range(orc.num_nodes) if n not in ex]
    return [int(s) for s in (rng.choice(uni, C, replace=False) if len(uni) > C else uni)]


def one_seed(ds, seed):
    trained = train_model(ds, seed=seed)
    orc = ReputationAttackOracle(trained['model'], trained['index_list'], trained['device'])
    edges, labels = trained['edges'], trained['labels']
    strata = get_targets(orc, edges, labels, per_stratum=PER_STRATUM)
    rng = np.random.RandomState(42)
    ts = strata.get('low', [])
    fn, fl = [], []
    auc = trained['perf']['AUC']
    for t in ts:
        pool = _pool(orc, edges, t, C_POOL, rng)
        if len(pool) < B:
            continue
        cf = orc.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
        srcs = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:B]]
        R0, Rf, Rp = _variants(orc, edges, labels, t, srcs)
        if R0 != R0 or R0 < 0.5:
            continue
        fn.append(1 if Rf < 0.5 else 0)
        if Rp == Rp:
            fl.append(1 if Rp < 0.5 else 0)
    del orc, trained, edges, labels
    gc.collect(); torch.cuda.empty_cache()
    return {'auc': round(float(auc), 3),
            'n': len(fn),
            'flip_naive_pct': round(100 * np.mean(fn), 1) if fn else None,
            'flip_legit_pct': round(100 * np.mean(fl), 1) if fl else None}


def run_dataset(ds):
    log(f"\n{'='*60}\n  MULTI-SEED FLIP — {ds}\n{'='*60}")
    per = {}
    for s in SEEDS:
        r = one_seed(ds, s)
        per[f"seed{s}"] = r
        log(f"  seed {s}: AUC={r['auc']} n={r['n']} flip_naive={r['flip_naive_pct']}% flip_legit={r['flip_legit_pct']}%")
    naive = [per[k]['flip_naive_pct'] for k in per if per[k]['flip_naive_pct'] is not None]
    legit = [per[k]['flip_legit_pct'] for k in per if per[k]['flip_legit_pct'] is not None]
    summ = {
        'per_seed': per,
        'flip_naive_mean': round(float(np.mean(naive)), 1), 'flip_naive_std': round(float(np.std(naive)), 1),
        'flip_legit_mean': round(float(np.mean(legit)), 1), 'flip_legit_std': round(float(np.std(legit)), 1),
    }
    log(f"  SUMMARY {ds}: flip_naive {summ['flip_naive_mean']}+/-{summ['flip_naive_std']}%  "
        f"flip_legit {summ['flip_legit_mean']}+/-{summ['flip_legit_std']}%")
    return summ


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), "multiseed_flip.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
