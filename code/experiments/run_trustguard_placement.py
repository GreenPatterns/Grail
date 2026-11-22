"""
Placement signature on an official trust-GNN release: TrustGuard (Wang et al., IEEE TDSC
2024; https://github.com/Jieerbobo/TrustGuard). The released code is used unchanged, in its
default configuration (robust aggregation off), trained with its own trainer on its own
Bitcoin-OTC / Bitcoin-Alpha files at T_train = 7 of K = 10 snapshots.

TrustGuard's forward pass reads snapshot i as edges[index_list[i-1]:index_list[i]] for
i < T only, the prefix-snapshot pattern of Proposition 1. We repeat the paper's placement
sweep (run_p0b_verify.py): one random non-neighbor source per target, a distrust and a
trust edge spliced into each processed snapshot s0..s6 or appended past the window, and
report the mean reputation shift and label-Delta = |dR_distrust - dR_trust|. Reputation is
Eq. 1 of the paper: mean softmax(W [z_u || z_v])_0 over the target's incoming raters.
A single-edge flip test follows on low-reputation targets (clean R in [0.5, 0.6)): the most
damaging of C = 50 non-neighbor candidates by exact counterfactual at s6, naive and
propagation-only.

Run (GPU, ps env):
  TRUSTGUARD_DIR=<clone>/code GRAIL_DATASETS=otc,alpha \
      python code/experiments/run_trustguard_placement.py
"""
import sys, os, json, gc

import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
T_TRAIN = 7
N_SWEEP = 12
N_FLIP = 25
C_POOL = 50
DISTRUST, TRUST = [0., 1.], [1., 0.]
FILES = {'otc': ('../data/bitcoinotc-rating.txt', '../data/bitcoinotc.csv', 8),
         'alpha': ('../data/bitcoinalpha-rating.txt', '../data/bitcoinalpha.csv', 16)}


def splice_edge(edges, labels, index_list, T, pos, src, dst, label):
    """Insert edge (src, dst, label) at the end of snapshot `pos` (0 <= pos < T), shifting
    later boundaries, or past the processed window when pos == 'appended' (only the
    boundaries of unprocessed snapshots move). Returns new (edges, labels, index_list)."""
    if pos == 'appended':
        at, bump_from = index_list[T - 1], T
    else:
        at, bump_from = index_list[pos], pos
    ne = torch.tensor([[src], [dst]], dtype=edges.dtype, device=edges.device)
    nl = torch.tensor([label], dtype=labels.dtype, device=labels.device)
    e = torch.cat([edges[:, :at], ne, edges[:, at:]], dim=1)
    l = torch.cat([labels[:at], nl, labels[at:]], dim=0)
    idx = [b + 1 if i >= bump_from else b for i, b in enumerate(index_list)]
    return e, l, idx


def _embed(model, edges, labels, index_list):
    y = labels[:, 1].long()
    with torch.no_grad():
        _, z = model(edges, y, labels, index_list)
    return z


def _p_trust(model, z, src, dst):
    feats = torch.cat((z[src], z[dst]), 1)
    return F.softmax(torch.mm(feats, model.regression_weights), dim=1)[:, 0]


def _reputation(model, z, edges, t, exclude=None):
    cols = (edges[1] == t).nonzero(as_tuple=True)[0]
    if exclude is not None:
        cols = cols[cols != exclude]
    if len(cols) == 0:
        return float('nan')
    return _p_trust(model, z, edges[0, cols], edges[1, cols]).mean().item()


def _train(ds):
    edge_path, data_path, heads = FILES[ds]
    sys.argv = [sys.argv[0]]
    from arg_parser import parameter_parser
    from utils import read_graph
    from gcn import GCNTrainer
    args = parameter_parser()
    args.edge_path, args.data_path, args.attention_head = edge_path, data_path, heads
    args.train_time_slots, args.seed = T_TRAIN, SEED
    trainer = GCNTrainer(args, read_graph(args))
    trainer.setup_dataset()
    trainer.create_and_train_model()
    trainer.model.eval()
    perf = trainer.logs['performance'][-1]
    return trainer, {'MCC': float(perf[1]), 'AUC': float(perf[2])}


def run_dataset(ds):
    trainer, perf = _train(ds)
    model = trainer.model
    edges, labels = trainer.train_edges, trainer.y_train
    idx = [int(b) for b in trainer.index_list]
    n = int(edges.max().item()) + 1
    z0 = _embed(model, edges, labels, idx)
    indeg = np.bincount(edges[1].cpu().numpy(), minlength=n)
    rep0 = {t: _reputation(model, z0, edges, t) for t in np.nonzero(indeg >= 2)[0]}
    rng = np.random.RandomState(SEED)

    def non_neighbors(t):
        raters = set(edges[0, edges[1] == t].tolist())
        return [s for s in rng.permutation(n) if s != t and s not in raters]

    # placement sweep over moderate + low targets
    pool = [t for t, r in rep0.items() if 0.5 <= r < 0.9]
    targets = [int(t) for t in rng.permutation(pool)[:N_SWEEP]]
    positions = list(range(T_TRAIN)) + ['appended']
    dr = {str(p): [] for p in positions}
    ld = {str(p): [] for p in positions}
    for t in targets:
        s = int(non_neighbors(t)[0])
        for p in positions:
            shift = {}
            for name, lab in (('dis', DISTRUST), ('tru', TRUST)):
                e, l, i2 = splice_edge(edges, labels, idx, T_TRAIN, p, s, t, lab)
                shift[name] = _reputation(model, _embed(model, e, l, i2), e, t) - rep0[t]
            dr[str(p)].append(shift['dis']); ld[str(p)].append(abs(shift['dis'] - shift['tru']))
    sweep = {'sweep_dr': {k: float(np.mean(v)) for k, v in dr.items()},
             'sweep_label_delta': {k: float(np.mean(v)) for k, v in ld.items()},
             'targets': targets}

    # single-edge flip test on low-reputation targets (exact counterfactual at s6)
    low = [int(t) for t in rng.permutation([t for t, r in rep0.items() if 0.5 <= r < 0.6])[:N_FLIP]]
    flips_total, flips_prop, drs = [], [], []
    for t in low:
        cands = non_neighbors(t)[:C_POOL]
        best = None
        for s in cands:
            e, l, i2 = splice_edge(edges, labels, idx, T_TRAIN, T_TRAIN - 1, int(s), t, DISTRUST)
            z1 = _embed(model, e, l, i2)
            r_tot = _reputation(model, z1, e, t)
            if best is None or r_tot < best[0]:
                at = idx[T_TRAIN - 1]
                best = (r_tot, _reputation(model, z1, e, t, exclude=at))
        flips_total.append(int(best[0] < 0.5)); flips_prop.append(int(best[1] < 0.5))
        drs.append(best[0] - rep0[t])
    flip = {'n_eligible': len(low),
            'flip_total_pct': round(100 * np.mean(flips_total), 1) if low else None,
            'flip_prop_pct': round(100 * np.mean(flips_prop), 1) if low else None,
            'k_total': int(sum(flips_total)), 'k_prop': int(sum(flips_prop)),
            'mean_dR_total': float(np.mean(drs)) if drs else None}
    out = {'clean_perf_last_epoch': perf, 'index_list': idx, 'T': T_TRAIN, **sweep, 'flip_low_B1': flip}
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    tg = os.environ.get('TRUSTGUARD_DIR')
    if not tg:
        raise SystemExit('set TRUSTGUARD_DIR to the TrustGuard/code directory of the official clone')
    tg = os.path.abspath(tg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results',
                            'unified', os.environ.get('GRAIL_OUT', 'p0b_trustguard.json'))
    sys.path.insert(0, tg)
    os.chdir(tg)                      # TrustGuard resolves its data paths relative to code/
    torch.manual_seed(SEED)
    res = {ds: run_dataset(ds) for ds in os.environ.get('GRAIL_DATASETS', 'otc,alpha').split(',')}
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    for ds, r in res.items():
        print(ds, 'label-delta', {k: f"{v:.2e}" for k, v in r['sweep_label_delta'].items()})
        print(ds, 'flip', r['flip_low_B1'])
    print('saved', os.path.abspath(out_path))
