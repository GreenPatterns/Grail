"""
TrustGuard case study, robust aggregation OFF (release default) and ON. Writes
results/trustguard_casestudy_<dataset>.json (per-seed records and a summary) for the seeds 40-44 of
TrustGuard's arg_parser, with T = 7 processed snapshots of 10.

Part A, reproduction of TrustGuard's published robustness result (TDSC 2024, Sec. V-C, Fig. 3a/3d,
single-timeslot task): train on snapshots 1-7, MCC on the observed edges of snapshot 8, collaborative
bad-mouthing. The attack code is not released; the protocol follows the paper's text: targets are 10% of
the "good" test-set nodes (more trust than distrust among their training-window edges); each target gets
as many distrust edges as its training-window degree, from distinct training-window nodes not linked to
it; each edge goes to a uniformly drawn processed snapshot; the model is retrained (poisoning). MCC is
taken at the epoch main.py selects (max MCC+AUC+BA+F1-macro) and at the last epoch. Conditions: clean,
clean2 (identical retraining, GPU noise), inwin_dis (the published attack), inwin_tru (label swap),
app_dis / app_tru (same edges after the last processed edge, inside the training tensors: the loss reads
them, the encoder does not), graph_dis / graph_tru (in-window, left out of the loss: the encoder reads
them, the loss does not). Per condition also the targets' R_tot / R_prop against the clean model of the
same seed, the injected edges' aggregation coefficients (TrustGuard's Fig. 4), and the change of each
target's embedding when the injected edges are removed under fixed weights (reachability).

Part B, single-edge evasion test on the clean model: low-reputation targets (clean R in [0.5, 0.6),
>= 2 raters; at most 100 per run), C = 50 candidates that do not rate the target (same per seed and
target for OFF and ON), one distrust edge at the end of s6; random source = first candidate, optimized =
lowest R_tot (as run_trustguard_placement.py). For the random source also the label swap and the
embedding change, in-window and appended after the window.

Deviations (also in the JSON): robust aggregation is switched on through the constructor default of
convolution.Convolution; its CPU coefficient is replaced by trustguard_robust_gpu.ro_coefficient_gpu,
checked against the released function on one forward pass ('gpu_equivalence'); per-call prints silenced.
--features node2vec replaces the release's random node features by 64-d node2vec features of the processed
window (the paper's stated input; walk settings in N2V), written to ..._node2vec.json.

--attack-degree test is the alternative reading of the attack size (observed test edges of the target);
--max-low 0 skips part B. `--self-check` tests the edge-placement helpers without TrustGuard.

Run (GPU 0):
  TRUSTGUARD_DIR=<clone>/code CUDA_VISIBLE_DEVICES=0 python code/experiments/run_trustguard_casestudy.py --dataset otc [--features node2vec]
"""
import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_trustguard_placement import FILES, _embed, _p_trust, splice_edge  # noqa: E402
from trustguard_robust_gpu import ro_coefficient_gpu, ro_coefficient_reference  # noqa: E402

T = 7
SEEDS = (40, 41, 42, 43, 44)
C_POOL = 50
TAU = 0.5
LOW = (0.5, 0.6)
TARGET_FRAC = 0.10
DISTRUST, TRUST = [0., 1.], [1., 0.]
AGGS = (('mean', False), ('robust', True))
CONDITIONS = ('inwin_dis', 'inwin_tru', 'app_dis', 'app_tru', 'graph_dis', 'graph_tru')
N2V = dict(embedding_dim=64, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, p=1.0, q=1.0)
N2V_EPOCHS = 100
# TrustGuard bars of Fig. 3(a) OTC and 3(d) Alpha, single-timeslot, read from the rendered figure (+-0.002)
PUBLISHED = {'otc': {'attack_without_defense': 0.516, 'attack_with_defense': 0.543, 'no_attack': 0.558},
             'alpha': {'attack_without_defense': 0.420, 'attack_with_defense': 0.466, 'no_attack': 0.527}}
DEVIATIONS = [
    'robust aggregation ON = constructor default robust_aggr=True (release: False, convolution.py:64, no flag)',
    'robust coefficient computed by ro_coefficient_gpu (same formula, on GPU); the released CPU function '
    'returns a CPU tensor that fails in the GPU forward pass and uses the sparse .A removed in SciPy 1.14',
    "per-call print of the robust layers silenced",
    'node features: the release\'s random N(0,1) 64-d features; the paper states node2vec features',
    'attack code not released: bad-mouthing protocol reconstructed from the paper text (see docstring)',
]
TG = {}


@contextlib.contextmanager
def quiet():
    with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        yield


def load_trustguard(tg_dir):
    sys.path.insert(0, tg_dir)
    os.chdir(tg_dir)                      # TrustGuard resolves its data paths relative to code/
    import arg_parser, convolution, gcn, utils  # noqa: E401
    convolution.ro_coefficient = ro_coefficient_gpu
    convolution.print = lambda *a, **k: None
    TG.update(arg_parser=arg_parser, convolution=convolution, gcn=gcn, utils=utils)


def set_robust(on):
    TG['convolution'].Convolution.__init__.__defaults__ = (bool(on), False, True)


def make_trainer(ds, seed):
    edge_path, data_path, heads = FILES[ds]
    sys.argv = sys.argv[:1]
    args = TG['arg_parser'].parameter_parser()
    args.edge_path, args.data_path, args.attention_head = edge_path, data_path, heads
    args.train_time_slots, args.seed = T, seed
    with quiet():
        tr = TG['gcn'].GCNTrainer(args, TG['utils'].read_graph(args))
        tr.setup_dataset()
    tr.index_list = [int(b) for b in tr.index_list]
    return tr


@contextlib.contextmanager
def exclude_from_loss(mask):
    """Keep the edges in `mask` in the encoder's input but out of TrustGuard's loss (graph-only)."""
    if mask is None:
        yield
        return
    cls, keep = TG['gcn'].TrustGuard, ~mask
    orig = cls.calculate_loss_function

    def loss(self, z, edges, target):
        if edges.size(1) == keep.numel():
            edges, target = edges[:, keep], target[keep]
        return orig(self, z, edges, target)
    cls.calculate_loss_function = loss
    try:
        yield
    finally:
        cls.calculate_loss_function = orig


def node2vec_features(e, idx, n, seed):
    """node2vec features (64-d, the paper's stated input) of the undirected processed window."""
    from torch_geometric.nn import Node2Vec
    torch.manual_seed(seed)
    w = e[:, :idx[T - 1]]
    m = Node2Vec(torch.cat([w, w.flip(0)], 1), num_nodes=n, sparse=True, **N2V).to(e.device)
    opt = torch.optim.SparseAdam(list(m.parameters()), lr=0.01)
    for _ in range(N2V_EPOCHS):
        for pos, neg in m.loader(batch_size=128, shuffle=True, num_workers=0):
            opt.zero_grad()
            m.loss(pos.to(e.device), neg.to(e.device)).backward()
            opt.step()
    return m.embedding.weight.detach().clone()


def train(ds, seed, robust, graph=None, features=None, loss_mask=None):
    """Release trainer on the clean graph or on graph = (edges, labels, index_list); optional node
    features replacing the release's random ones, and edges left out of the loss."""
    set_robust(robust)
    tr = make_trainer(ds, seed)
    if graph is not None:
        tr.train_edges, tr.y_train, tr.index_list = graph[0], graph[1], list(graph[2])
        tr.y = graph[1][:, 1].long()
    if features is not None:
        tr.X = features
    with quiet(), exclude_from_loss(loss_mask):
        tr.create_and_train_model()
    tr.model.eval()
    perf = tr.logs['performance'][1:]
    best = max(perf, key=lambda r: r[1] + r[2] + r[3] + r[6])          # main.py:26-29
    keys = ('epoch', 'MCC', 'AUC', 'BA', 'AP', 'F1_micro', 'F1_macro')
    return tr, {'best': dict(zip(keys, map(float, best))), 'last': dict(zip(keys, map(float, perf[-1])))}


def clean_graph(tr):
    return tr.train_edges, tr.y_train, list(tr.index_list)


def insert_in_window(e, l, idx, ne, nl, snap):
    """Append each new edge at the end of its processed snapshot snap[i]; later boundaries move.
    Returns edges, labels, index_list and the mask of the new edges."""
    pe, pl, pm, new_idx, start, added = [], [], [], list(idx), 0, 0
    for s in range(T):
        m = torch.as_tensor(snap == s, device=e.device)
        pe += [e[:, start:idx[s]], ne[:, m]]
        pl += [l[start:idx[s]], nl[m]]
        pm += [torch.zeros(idx[s] - start, dtype=torch.bool), torch.ones(int(m.sum()), dtype=torch.bool)]
        added += int(m.sum())
        new_idx[s] = idx[s] + added
        start = idx[s]
    pe.append(e[:, start:]); pl.append(l[start:]); pm.append(torch.zeros(e.size(1) - start, dtype=torch.bool))
    new_idx[T:] = [b + added for b in idx[T:]]
    return torch.cat(pe, 1), torch.cat(pl, 0), new_idx, torch.cat(pm).to(e.device)


def append_after_window(e, l, idx, ne, nl):
    """New edges after the last processed edge, inside the training tensors; index_list[:T] unchanged."""
    at, k = idx[T - 1], ne.size(1)
    mask = torch.zeros(e.size(1) + k, dtype=torch.bool, device=e.device)
    mask[at:at + k] = True
    return (torch.cat([e[:, :at], ne, e[:, at:]], 1), torch.cat([l[:at], nl, l[at:]], 0),
            [b if i < T else b + k for i, b in enumerate(idx)], mask)


def bad_mouthing(tr, seed, degree='train'):
    """Targets and attack edges of the collaborative bad-mouthing attack (paper text, Sec. V-A1).
    degree: 'train' = the target's number of edges in the processed window (primary reading),
    'test' = its number of observed test edges (alternative reading, fixed before its results)."""
    rng = np.random.RandomState(seed)
    e = tr.train_edges.cpu().numpy()
    trust = tr.y_train[:, 0].cpu().numpy() == 1
    n = tr.X.shape[0]
    pos = np.bincount(e[:, trust].ravel(), minlength=n)
    neg = np.bincount(e[:, ~trust].ravel(), minlength=n)
    obs = np.asarray(tr.obs, dtype=np.int64)
    deg = pos + neg if degree == 'train' else np.bincount(obs.ravel(), minlength=n)
    test_nodes = np.unique(obs.ravel())
    good = test_nodes[pos[test_nodes] > neg[test_nodes]]
    targets = np.sort(rng.choice(good, int(round(TARGET_FRAC * len(good))), replace=False))
    present = np.unique(e)
    src, dst = [], []
    for t in targets:
        linked = np.union1d(e[0, e[1] == t], e[1, e[0] == t])
        pool = np.setdiff1d(present, np.append(linked, t))
        d = int(deg[t])
        src.append(rng.choice(pool, d, replace=False))
        dst.append(np.full(d, t))
    attack = np.stack([np.concatenate(src), np.concatenate(dst)])
    snap = rng.randint(0, T, attack.shape[1])
    info = {'n_test_nodes': int(len(test_nodes)), 'n_good_test_nodes': int(len(good)),
            'n_targets': int(len(targets)), 'n_attack_edges': int(attack.shape[1]),
            'attack_edges_per_snapshot': np.bincount(snap, minlength=T).tolist(),
            'n_clean_train_edges': int(e.shape[1]), 'targets': targets.tolist()}
    return targets, attack, snap, info


@torch.no_grad()
def all_reputations(model, z, e):
    p = _p_trust(model, z, e[0], e[1])
    n = z.size(0)
    s = torch.zeros(n, device=z.device).index_add_(0, e[1], p)
    c = torch.zeros(n, device=z.device).index_add_(0, e[1], torch.ones_like(p))
    return (s / c.clamp_min(1)).cpu().numpy(), c.cpu().numpy().astype(int)


def _pt(model, z, src, t):
    src = torch.as_tensor(np.asarray(src), device=z.device, dtype=torch.long)
    return _p_trust(model, z, src, torch.full_like(src, int(t)))


@torch.no_grad()
def target_view(model0, g0, model1, g1, targets, attack):
    """Clean R (clean model, clean graph) and R_prop / R_tot (poisoned model, poisoned graph)."""
    z0, z1 = _embed(model0, *g0), _embed(model1, *g1)
    e0 = g0[0].cpu().numpy()
    rows = []
    for t in targets:
        pre, srcs = e0[0, e0[1] == t], attack[0, attack[1] == t]
        p1 = _pt(model1, z1, pre, t)
        rows.append({'t': int(t), 'n_raters': int(len(pre)), 'm': int(len(srcs)),
                     'R_clean': _pt(model0, z0, pre, t).mean().item() if len(pre) else None,
                     'R_prop': p1.mean().item() if len(pre) else None,
                     'R_tot': (p1.sum() + _pt(model1, z1, srcs, t).sum()).item() / (len(pre) + len(srcs))})
    return rows


@torch.no_grad()
def injected_coefficients(model, g, mask, robust):
    """Mean aggregation weight of the injected edges per layer: 'in' = weight of the source in the
    target's in-aggregation, 'out' = weight of the target in the source's out-aggregation. Robust: the
    robust coefficient; mean aggregator: 1/in-degree, 1/out-degree within the edge's snapshot."""
    e, l, idx = g
    conv, calls = TG['convolution'], []
    if robust:
        def record(x, prune, row, col):
            out = ro_coefficient_gpu(x, prune, row, col)
            calls.append(out.squeeze(1))
            return out
        conv.ro_coefficient = record
        try:
            _embed(model, e, l, idx)
        finally:
            conv.ro_coefficient = ro_coefficient_gpu
    vals, start, n = {}, 0, int(e.max()) + 1
    for s in range(T):
        m, es = mask[start:idx[s]], e[:, start:idx[s]]
        if m.any():
            for layer in range(3):
                for j, d in enumerate(('out', 'in')):
                    if robust:                      # call order per snapshot: (L1 out, in), (L2 ...), (L3 ...)
                        v = calls[6 * s + 2 * layer + j]
                    else:                           # in: 1/in-degree of the target; out: 1/out-degree of the source
                        v = 1.0 / torch.bincount(es[j], minlength=n).float()[es[j]]
                    vals.setdefault(f'L{layer + 1}_{d}', []).append((v[m], v[~m]))
        start = idx[s]
    out = {}
    for k, v in vals.items():
        inj, org = torch.cat([x[0] for x in v]), torch.cat([x[1] for x in v])
        out[k] = {'mean': inj.mean().item(), 'share_zero': (inj == 0).float().mean().item(), 'n': int(inj.numel()),
                  'organic_mean': org.mean().item(), 'organic_share_zero': (org == 0).float().mean().item()}
    return out


@torch.no_grad()
def reachability(model, g_with, g_without, targets):
    z1, z0 = _embed(model, *g_with), _embed(model, *g_without)
    t = torch.as_tensor(np.asarray(targets), device=z1.device)
    return (z1[t] - z0[t]).norm(dim=1).cpu().numpy().tolist()


@torch.no_grad()
def evasion(model, g, seed, max_low):
    """Single-edge test on low-reputation targets with fixed weights (checks 3a and 3b)."""
    e, l, idx = g
    z0 = _embed(model, e, l, idx)
    rep0, cnt = all_reputations(model, z0, e)
    low = np.where((cnt >= 2) & (rep0 >= LOW[0]) & (rep0 < LOW[1]))[0]
    n_low_all = int(len(low))
    if len(low) > max_low:
        low = np.sort(np.random.RandomState(seed).choice(low, max_low, replace=False))
    e_np, n = e.cpu().numpy(), z0.size(0)
    recs = []
    for t in low:
        t = int(t)
        pre = e_np[0, e_np[1] == t]
        excl = set(pre.tolist()) | {t}
        cands = [int(s) for s in np.random.RandomState([seed, t]).permutation(n) if s not in excl][:C_POOL]

        def attacked(s, pos, label):
            e2, l2, i2 = splice_edge(e, l, idx, T, pos, s, t, label)
            z1 = _embed(model, e2, l2, i2)
            p_pre = _pt(model, z1, pre, t)
            r_tot = (p_pre.sum() + _pt(model, z1, [s], t).sum()).item() / (len(pre) + 1)
            return r_tot, p_pre.mean().item(), (z1[t] - z0[t]).norm().item()

        cand = [attacked(s, T - 1, DISTRUST) for s in cands]
        k = int(np.argmin([c[0] for c in cand]))
        swap = {'tru_in': attacked(cands[0], T - 1, TRUST), 'dis_out': attacked(cands[0], 'appended', DISTRUST),
                'tru_out': attacked(cands[0], 'appended', TRUST)}
        r0 = float(rep0[t])
        recs.append({
            't': t, 'n_raters': int(len(pre)), 'R_clean': r0,
            'random': {'src': cands[0], 'R_tot': cand[0][0], 'R_prop': cand[0][1], 'dz': cand[0][2]},
            'optimized': {'src': cands[k], 'R_tot': cand[k][0], 'R_prop': cand[k][1], 'dz': cand[k][2]},
            'Lambda_in': abs(cand[0][0] - swap['tru_in'][0]), 'Lambda_out': abs(swap['dis_out'][0] - swap['tru_out'][0]),
            'dR_tot_dis_out': swap['dis_out'][0] - r0, 'dR_prop_dis_out': swap['dis_out'][1] - r0,
            'dR_tot_tru_in': swap['tru_in'][0] - r0, 'dR_prop_tru_in': swap['tru_in'][1] - r0,
            'dz_in': cand[0][2], 'dz_out': swap['dis_out'][2],
            'cands_R_tot_R_prop': [[round(c[0], 5), round(c[1], 5)] for c in cand]})
    return {'n_low_all': n_low_all, 'n_low_evaluated': len(recs), 'targets': recs}


def gpu_equivalence(model, g):
    """Compare ro_coefficient_gpu with the released function on every robust call of one forward pass."""
    conv, diffs = TG['convolution'], []

    def both(x, prune, row, col):
        row, col = np.array(row), np.array(col)
        t0 = time.time()
        ref = ro_coefficient_reference(x.cpu(), prune, row, col).numpy().ravel()
        secs = time.time() - t0
        out = ro_coefficient_gpu(x, prune, row, col)
        d = np.abs(ref - out.cpu().numpy().ravel())
        diffs.append((len(row), float(d.max()), int((d > 1e-5).sum()), secs))
        return out
    conv.ro_coefficient = both
    try:
        _embed(model, *g)
    finally:
        conv.ro_coefficient = ro_coefficient_gpu
    return {'calls': len(diffs), 'edge_values': sum(d[0] for d in diffs), 'max_abs_diff': max(d[1] for d in diffs),
            'values_differing_gt_1e-5': sum(d[2] for d in diffs),
            'reference_cpu_seconds_per_forward': round(sum(d[3] for d in diffs), 1)}


# ---------------------------------------------------------------- summaries
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def mstd(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return None
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'per_seed': v}


def rate_block(kn):
    """kn: per-seed (k, n). Per-seed rate, mean +- SD over seeds, pooled rate with exact 95% interval."""
    kk, nn = sum(k for k, _ in kn), sum(n for _, n in kn)
    return {'per_seed_k_n': [list(x) for x in kn],
            'rate_pct': mstd([100 * k / n if n else None for k, n in kn]),
            'pooled_k': kk, 'pooled_n': nn, 'pooled_pct': 100 * kk / nn if nn else None,
            'pooled_cp95_pct': [100 * x for x in cp(kk, nn)] if nn else None}


def summarize(runs):
    s = {}
    for agg, _ in AGGS:
        rs = [runs[k][agg] for k in sorted(runs) if agg in runs[k]]
        if not rs:
            continue
        a = {'seeds_done': [r['seed'] for r in rs]}
        for cond in ('clean', 'clean2') + CONDITIONS:
            a[f'MCC_best_{cond}'] = mstd([r['repro'][cond]['metrics']['best']['MCC'] for r in rs if cond in r['repro']])
            a[f'MCC_last_{cond}'] = mstd([r['repro'][cond]['metrics']['last']['MCC'] for r in rs if cond in r['repro']])
        for cond in ('clean2',) + CONDITIONS:
            for which in ('best', 'last'):
                a[f'dMCC_{which}_{cond}'] = mstd([r['repro'][cond]['metrics'][which]['MCC'] - r['repro']['clean']['metrics'][which]['MCC']
                                                  for r in rs if cond in r['repro']])
        for cond in CONDITIONS:
            rows =[(r, [x for x in r['repro'][cond]['targets'] if x['R_clean'] is not None and x['R_clean'] >= TAU
                         and x['n_raters'] >= 2]) for r in rs if cond in r['repro']]
            a[f'targets_{cond}'] = {
                'flip_prop': rate_block([(sum(x['R_prop'] < TAU for x in el), len(el)) for _, el in rows]),
                'flip_tot': rate_block([(sum(x['R_tot'] < TAU for x in el), len(el)) for _, el in rows]),
                'shift_prop': mstd([np.mean([x['R_prop'] - x['R_clean'] for x in el]) if el else None for _, el in rows]),
                'shift_tot': mstd([np.mean([x['R_tot'] - x['R_clean'] for x in el]) if el else None for _, el in rows]),
                'reach_dz_mean': mstd([np.mean(r['repro'][cond]['reach_dz']) for r, _ in rows]),
                'coef': {k: mstd([r['repro'][cond]['coef'][k]['mean'] for r, _ in rows]) for k in rows[0][0]['repro'][cond]['coef']},
                'coef_share_zero': {k: mstd([r['repro'][cond]['coef'][k]['share_zero'] for r, _ in rows]) for k in rows[0][0]['repro'][cond]['coef']},
                'coef_organic_share_zero': {k: mstd([r['repro'][cond]['coef'][k]['organic_share_zero'] for r, _ in rows]) for k in rows[0][0]['repro'][cond]['coef']}}
        for which in ('best', 'last'):
            for mode in ('inwin', 'app', 'graph'):
                a[f'Lambda_MCC_{which}_{mode}'] = mstd([abs(r['repro'][f'{mode}_dis']['metrics'][which]['MCC'] - r['repro'][f'{mode}_tru']['metrics'][which]['MCC'])
                                                        for r in rs if f'{mode}_tru' in r['repro']])
        for mode in ('inwin', 'app', 'graph'):
            a[f'Lambda_R_targets_{mode}'] = mstd([np.mean([abs(x['R_tot'] - y['R_tot']) for x, y in zip(r['repro'][f'{mode}_dis']['targets'], r['repro'][f'{mode}_tru']['targets'])])
                                                 for r in rs if f'{mode}_tru' in r['repro']])
        ev = [r['evasion'] for r in rs if 'evasion' in r]
        b = {'n_low_all': [x['n_low_all'] for x in ev], 'n_low_evaluated': [x['n_low_evaluated'] for x in ev]}
        for src in ('random', 'optimized'):
            for sc in ('tot', 'prop'):
                key = f'R_{sc}'
                b[f'{src}_{sc}_flip'] = rate_block([(sum(x[src][key] < TAU for x in v['targets']), len(v['targets'])) for v in ev])
                b[f'{src}_{sc}_shift'] = mstd([np.mean([x[src][key] - x['R_clean'] for x in v['targets']]) if v['targets'] else None for v in ev])
        for k in ('Lambda_in', 'Lambda_out', 'dz_in', 'dz_out', 'dR_tot_dis_out', 'dR_prop_dis_out', 'dR_tot_tru_in'):
            b[k] = mstd([np.mean([x[k] for x in v['targets']]) if v['targets'] else None for v in ev])
        b['Lambda_out_max'] = max((x['Lambda_out'] for v in ev for x in v['targets']), default=None)
        b['dz_out_max'] = max((x['dz_out'] for v in ev for x in v['targets']), default=None)
        b['Lambda_in_min'] = min((x['Lambda_in'] for v in ev for x in v['targets']), default=None)
        b['all_candidates_flip_tot'] = rate_block([(sum(c[0] < TAU for x in v['targets'] for c in x['cands_R_tot_R_prop']),
                                                    sum(len(x['cands_R_tot_R_prop']) for x in v['targets'])) for v in ev])
        b['all_candidates_flip_prop'] = rate_block([(sum(c[1] < TAU for x in v['targets'] for c in x['cands_R_tot_R_prop']),
                                                     sum(len(x['cands_R_tot_R_prop']) for x in v['targets'])) for v in ev])
        s[agg] = {'repro': a, 'evasion': b}
    return s


def commit_of(tg_dir):
    try:
        return subprocess.check_output(['git', '-C', tg_dir, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


def run_seed_agg(ds, seed, robust, attack_pack, max_low, check_equiv, feats):
    """feats: None (release features) or {'clean': X, 'inwin': X} node2vec features of the clean and
    of the in-window poisoned processed window (appended edges lie outside it: clean features)."""
    targets, attack, snap, _ = attack_pack
    fx = (lambda k: None) if feats is None else feats.get
    out = {'seed': seed, 'robust_aggregation': robust, 'repro': {}}
    t0 = time.time()
    clean_tr, m = train(ds, seed, robust, features=fx('clean'))
    g0 = clean_graph(clean_tr)
    out['repro']['clean'] = {'metrics': m}
    if check_equiv and robust:
        out['gpu_equivalence'] = gpu_equivalence(clean_tr.model, g0)
    if max_low > 0:
        out['evasion'] = evasion(clean_tr.model, g0, seed, max_low)
    ev = out.get('evasion', {'n_low_evaluated': 0, 'n_low_all': 0})
    print(f'  [{ds} s{seed} {"robust" if robust else "mean"}] clean MCC best {m["best"]["MCC"]:.4f} last {m["last"]["MCC"]:.4f}; '
          f'evasion {ev["n_low_evaluated"]}/{ev["n_low_all"]} low targets ({time.time() - t0:.0f}s)', flush=True)
    _, m2 = train(ds, seed, robust, features=fx('clean'))
    out['repro']['clean2'] = {'metrics': m2}
    e, l, idx = g0
    ne = torch.as_tensor(attack, device=e.device, dtype=torch.long)
    for cond in CONDITIONS:
        lab = DISTRUST if cond.endswith('dis') else TRUST
        nl = torch.tensor([lab] * ne.size(1), device=e.device, dtype=l.dtype)
        if cond.startswith('app'):
            e1, l1, i1, mask = append_after_window(e, l, idx, ne, nl)
        else:
            e1, l1, i1, mask = insert_in_window(e, l, idx, ne, nl, snap)
        tr1, m1 = train(ds, seed, robust, (e1, l1, i1), features=fx('clean' if cond.startswith('app') else 'inwin'),
                        loss_mask=mask if cond.startswith('graph') else None)
        g1 = (e1, l1, i1)
        out['repro'][cond] = {
            'metrics': m1,
            'targets': target_view(clean_tr.model, g0, tr1.model, g1, targets, attack),
            'coef': injected_coefficients(tr1.model, g1, mask, robust),
            'reach_dz': reachability(tr1.model, g1, g0, targets)}
        print(f'    {cond}: MCC best {m1["best"]["MCC"]:.4f} last {m1["last"]["MCC"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
        del tr1
    del clean_tr
    gc.collect(); torch.cuda.empty_cache()
    return out


def self_check():
    """In-window edges land inside their snapshot's slice; appended edges lie outside every processed slice."""
    idx = [3, 5, 9, 12, 14, 15, 18, 20, 22, 25]
    e = torch.arange(2 * 18).view(2, 18)
    l = torch.zeros(18, 2)
    ne, nl, snap = torch.tensor([[100, 101, 102], [200, 201, 202]]), torch.ones(3, 2), np.array([0, 6, 3])
    e1, l1, i1, m1 = insert_in_window(e, l, idx, ne, nl, snap)
    starts = [0] + i1[:T - 1]
    for k, s in enumerate(snap):
        pos = int((e1[0] == ne[0, k]).nonzero())
        assert starts[s] <= pos < i1[s] and bool(m1[pos]), (k, pos)
    assert i1[T:] == [b + 3 for b in idx[T:]] and int(m1.sum()) == 3 and torch.equal(e1[:, ~m1], e)
    e2, l2, i2, m2 = append_after_window(e, l, idx, ne, nl)
    assert i2[:T] == idx[:T] and bool(m2[idx[T - 1]:idx[T - 1] + 3].all()) and torch.equal(e2[:, ~m2], e)
    print('self-check passed')


def main():
    if '--self-check' in sys.argv:
        return self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(FILES))
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--max-low', type=int, default=100, help='0 skips the evasion test (part B)')
    ap.add_argument('--features', default='release', choices=('release', 'node2vec'))
    ap.add_argument('--attack-degree', default='train', choices=('train', 'test'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tg = os.environ.get('TRUSTGUARD_DIR')
    if not tg:
        raise SystemExit('set TRUSTGUARD_DIR to the code/ directory of the official TrustGuard clone')
    tg = os.path.abspath(tg)
    suffix = ('' if a.features == 'release' else '_node2vec') + ('' if a.attack_degree == 'train' else '_testdeg')
    out_path = os.path.abspath(a.out or os.path.join(HERE, '..', '..', 'results', f'trustguard_casestudy_{a.dataset}{suffix}.json'))
    load_trustguard(tg)
    seeds = [int(x) for x in a.seeds.split(',')]
    res = {'dataset': a.dataset, 'trustguard_commit': commit_of(tg),
           'config': {'T': T, 'seeds': seeds, 'C_pool': C_POOL, 'tau': TAU, 'low': LOW, 'max_low': a.max_low,
                      'target_frac': TARGET_FRAC, 'epochs': 50, 'attention_heads': FILES[a.dataset][2],
                      'features': a.features, 'node2vec': {**N2V, 'epochs': N2V_EPOCHS} if a.features == 'node2vec' else None,
                      'attack_degree': a.attack_degree},
           'published_fig3_single_timeslot_MCC': PUBLISHED[a.dataset], 'deviations': DEVIATIONS,
           'attack': {}, 'runs': {}}
    for seed in seeds:
        base = make_trainer(a.dataset, seed)
        pack = bad_mouthing(base, seed, a.attack_degree)
        res['attack'][str(seed)] = pack[3]
        feats = None
        if a.features == 'node2vec':
            e, l, idx = clean_graph(base)
            e1, _, i1, _ = insert_in_window(e, l, idx, torch.as_tensor(pack[1], device=e.device),
                                            l[:1].repeat(pack[1].shape[1], 1), pack[2])
            feats = {'clean': node2vec_features(e, idx, base.X.shape[0], seed),
                     'inwin': node2vec_features(e1, i1, base.X.shape[0], seed)}
        del base
        for agg, robust in AGGS:
            res['runs'].setdefault(str(seed), {})[agg] = run_seed_agg(a.dataset, seed, robust, pack, a.max_low,
                                                                        check_equiv=(seed == seeds[0]), feats=feats)
            res['summary'] = summarize(res['runs'])
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=1)
    print('saved', os.path.relpath(out_path, os.path.join(HERE, '..', '..')))


if __name__ == '__main__':
    main()
