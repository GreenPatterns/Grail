"""
Placement test on the official DySAT release (Sankar et al., WSDM 2020;
https://github.com/aravindsankar28/DySAT, commit 777f290). Runs in a Python 3.7 / TensorFlow
1.15 environment; the released training script is executed unchanged through runpy, with one
compatibility shim: the Python-2 code indexes map() results, so its modules get a list-
returning map.

Bitcoin-OTC / Alpha are split into ten equal-time-span snapshots as in the paper (unsigned,
undirected graphs: DySAT never reads ratings). DySAT is trained with its default flags on
time_steps = 8; its script replaces the last input graph by the previous snapshot's edges and
evaluates the embeddings at step 6, so snapshots s0..s6 are processed and s7 is not.
Reputation of v is the mean predicted trust over v's directed raters in s0..s6, from a
class-balanced logistic-regression trust head on Hadamard features (DySAT's own evaluation
classifier type) fitted to the window's rating signs. A distrust edge is injected into s0..s6 or into s7, the
embeddings are recomputed by feeding the trained model the modified adjacency, and we report
the target's embedding change and the reputation shift (total effect and propagation-only).

Run (dysat env, CPU):
  DYSAT_DIR=<clone> GRAIL_DATASETS=otc,alpha <dysat-env>/bin/python \
      code/experiments/run_dysat_placement.py
"""
import builtins, json, os, runpy, sys

import numpy as np

K, TIME_STEPS, SEED, N_SWEEP, N_FLIP, C_POOL = 10, 8, 42, 12, 25, 50
DATA_ROOT = os.environ.get('GRAIL_DATA_ROOT', 'data/cyberdata')
CSV = {'otc': 'bitcoinotc.csv', 'alpha': 'bitcoinalpha.csv'}


def list_map(f, *xs):
    return list(builtins.map(f, *xs))


def list_range(*a):
    return list(builtins.range(*a))


def released_defaults(repo):
    """The argparse defaults of the released run_script.py, which writes them to the config
    that train.py reads (flags.py alone defaults to one epoch)."""
    import ast
    tree = ast.parse(open(os.path.join(repo, 'run_script.py')).read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument':
            name = node.args[0].s.lstrip('-')
            kw = {k.arg: k.value for k in node.keywords}
            if 'default' in kw:
                val = ast.literal_eval(kw['default'])
                if isinstance(val, str) and val in ('True', 'False'):
                    val = val == 'True'
                out[name] = val
    return out


def snapshots(ds):
    """Directed (src, dst, sign, snapshot) with contiguous ids, ten equal-span snapshots."""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_ROOT, CSV[ds]))
    df.columns = ['source', 'target', 'rating', 'time']   # the Alpha header has 'target ' (trailing space)
    df = df.sort_values('time', kind='mergesort').reset_index(drop=True)
    ids = pd.unique(pd.concat([df.source, df.target]))
    remap = {v: i for i, v in enumerate(ids)}
    src = df.source.map(remap).to_numpy(); dst = df.target.map(remap).to_numpy()
    t = df.time.to_numpy().astype(float)
    bounds = t.min() + (t.max() - t.min()) * np.arange(1, K) / K
    snap = np.searchsorted(bounds, t, side='left')
    return src, dst, (df.rating.to_numpy() > 0).astype(int), snap, len(ids)


def write_graphs(repo, name, src, dst, snap, n):
    import networkx as nx
    graphs = []
    for k in range(K):
        g = nx.MultiGraph(); g.add_nodes_from(range(n))
        g.add_edges_from(zip(src[snap == k].tolist(), dst[snap == k].tolist()))
        graphs.append(g)
    os.makedirs(os.path.join(repo, 'data', name), exist_ok=True)
    arr = np.empty(len(graphs), dtype=object)   # 1-D: np.array() would iterate the graphs' nodes
    for i, gr in enumerate(graphs):
        arr[i] = gr
    np.savez(os.path.join(repo, 'data', name, 'graphs.npz'), graph=arr)
    return graphs


def run(ds, repo):
    name = f'grail_{ds}'
    src, dst, sign, snap, n = snapshots(ds)
    write_graphs(repo, name, src, dst, snap, n)
    os.chdir(repo); sys.path.insert(0, repo)
    # the Python-2 code uses implicit relative imports inside utils/ and models/DySAT/
    for sub in ('utils', os.path.join('models', 'DySAT')):
        sys.path.insert(1, os.path.join(repo, sub))
    import importlib
    import utils.preprocess as pp
    # Python-2 semantics for map/range in every released module (lists, not iterators)
    for mod in ('utils.preprocess', 'utils.utilities', 'utils.random_walk', 'utils.minibatch',
                'models.DySAT.models', 'models.DySAT.layers', 'utilities', 'random_walk'):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        m.map, m.range = list_map, list_range
    sys.argv = ['train.py', f'--dataset={name}', f'--time_steps={TIME_STEPS}', '--featureless=True',
                f'--base_model=DySAT', f'--model=grail_{ds}']
    if os.environ.get('DYSAT_EPOCHS'):
        sys.argv.append(f"--epochs={int(os.environ['DYSAT_EPOCHS'])}")
    if os.environ.get('DYSAT_N_FLIP'):
        globals()['N_FLIP'] = int(os.environ['DYSAT_N_FLIP'])
    # run_script.py normally writes this config and then launches train.py; write it ourselves
    cfg_dir = os.path.join(repo, 'logs', f'DySAT_grail_{ds}')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg = released_defaults(repo)            # every default of the released run_script.py
    cfg.update({'dataset': name, 'time_steps': TIME_STEPS, 'base_model': 'DySAT', 'model': f'grail_{ds}'})
    cfg.pop('min_time', None); cfg.pop('max_time', None); cfg.pop('run_parallel', None)
    if os.environ.get('DYSAT_EPOCHS'):
        cfg['epochs'] = int(os.environ['DYSAT_EPOCHS'])
    with open(os.path.join(cfg_dir, f'flags_{name}.json'), 'w') as fh:
        json.dump(cfg, fh)
    import tensorflow as tf                  # TF 1.15 dropped the tf.contrib.linalg alias (TF <= 1.14)
    tf.contrib.linalg = tf.linalg
    from absl import flags as absl_flags     # newer absl defines log_dir, which DySAT redefines
    if 'log_dir' in absl_flags.FLAGS:
        delattr(absl_flags.FLAGS, 'log_dir')
    # The released script selects the epoch with the best validation AUC but keeps only that
    # epoch's embeddings; wrap its evaluate_classifier to also keep that epoch's weights.
    import eval.link_prediction as lp
    orig_eval, best = lp.evaluate_classifier, {'val': -1.0, 'weights': None}

    def evaluate_classifier(*a, **kw):
        out = orig_eval(*a, **kw)
        val_auc = out[0]['HAD'][1]
        caller = sys._getframe(1).f_globals
        if val_auc > best['val'] and 'sess' in caller:      # strict: the release keeps the first maximum
            best['val'] = val_auc
            best['weights'] = caller['sess'].run(tf.global_variables())
        return out
    lp.evaluate_classifier = evaluate_classifier
    g = runpy.run_path(os.path.join(repo, 'train.py'), init_globals={'map': list_map, 'range': list_range}, run_name='__main__')
    sess, model, ph = g['sess'], g['model'], g['placeholders']
    feats_train, adjs = g['feats_train'], g['adjs']
    idx_emb = model.final_output_embeddings.get_shape()[1] - 2
    for var, val in zip(tf.global_variables(), best['weights']):
        var.load(val, sess)
    held_out = {'selected_epoch': int(g['best_epoch']), 'val_auc_had': float(g['val_results']['HAD'][1]),
                'test_auc_had': float(g['test_results']['HAD'][1])}

    import networkx as nx
    import scipy.sparse as sp

    def embed(extra):
        """Final embeddings (step TIME_STEPS - 2) with extra undirected edges {step: [(u, v)]}."""
        feed = {ph['spatial_drop']: 0.0, ph['temporal_drop']: 0.0}
        for i in range(TIME_STEPS):
            a = adjs[i].tolil(copy=True)
            for u, v in extra.get(i, []):
                a[u, v] += 1; a[v, u] += 1
            feed[ph['adjs'][i]] = pp.normalize_graph_gcn(a.tocsr())
            feed[ph['features'][i]] = feats_train[i]
        return sess.run(model.final_output_embeddings, feed_dict=feed)[:, idx_emb, :]

    # The released script feeds graph TIME_STEPS-1 with the edges of snapshot TIME_STEPS-2, so a
    # rating placed in snapshot 7 never reaches the input: model it by leaving the feed unchanged.
    win = snap <= TIME_STEPS - 2
    z0 = embed({})
    # the restored weights must reproduce the embeddings the release reported for that epoch
    ref = np.asarray(g['epochs_embeds'][g['best_epoch']])
    held_out['max_abs_diff_vs_selected_epoch_embeddings'] = float(np.abs(z0 - ref).max())
    assert held_out['max_abs_diff_vs_selected_epoch_embeddings'] < 1e-4, held_out
    from sklearn.linear_model import LogisticRegression
    had = lambda z, u, v: z[u] * z[v]
    # class-balanced, like the class-weighted losses of the trust GNNs (91% of ratings are trust)
    head = LogisticRegression(max_iter=2000, class_weight='balanced').fit(had(z0, src[win], dst[win]), sign[win])
    p_trust = lambda z, u, v: head.predict_proba(had(z, np.asarray(u), np.asarray(v)))[:, 1]
    raters = {}
    for u, v in zip(src[win], dst[win]):
        raters.setdefault(int(v), []).append(int(u))
    rep0 = {v: float(p_trust(z0, r, [v] * len(r)).mean()) for v, r in raters.items() if len(r) >= 2}
    rng = np.random.RandomState(SEED)

    def pool(v):
        banned = set(raters[v]) | {v}
        return [int(s) for s in rng.permutation(n) if int(s) not in banned][:C_POOL]

    def attacked(v, s_new, step):
        # an edge in s_{T-2} also enters the copied last input graph, as the released script
        # would copy it; an edge in s_{T-1} never reaches the input
        extra = {step: [(s_new, v)]}
        if step == TIME_STEPS - 2:
            extra[TIME_STEPS - 1] = [(s_new, v)]
        z = embed(extra) if step <= TIME_STEPS - 2 else z0
        pre = p_trust(z, raters[v], [v] * len(raters[v]))
        own = p_trust(z, [s_new], [v])
        return z, float(np.concatenate([pre, own]).mean()), float(pre.mean())

    positions = list(range(TIME_STEPS - 1)) + [TIME_STEPS - 1]
    targets = [v for v in rng.permutation(sorted(rep0)) if 0.5 <= rep0[v] < 0.9][:N_SWEEP]
    sweep = {str(p): {'dR_total': [], 'dR_prop': [], 'emb_change': []} for p in positions}
    for v in targets:
        s_new = pool(v)[0]
        for p in positions:
            z, rt, rp = attacked(v, s_new, p)
            sweep[str(p)]['dR_total'].append(rt - rep0[v]); sweep[str(p)]['dR_prop'].append(rp - rep0[v])
            sweep[str(p)]['emb_change'].append(float(np.linalg.norm(z[v] - z0[v])))
    sweep = {p: {k: float(np.mean(x)) for k, x in c.items()} for p, c in sweep.items()}
    low = [v for v in rng.permutation(sorted(rep0)) if 0.5 <= rep0[v] < 0.6][:N_FLIP]
    ft, fp = [], []
    rt_rand, rp_rand = [], []
    for v in low:
        top = None
        for j, s_new in enumerate(pool(v)):
            _, rt, rp = attacked(v, s_new, TIME_STEPS - 2)
            if j == 0:
                rt_rand.append(int(rt < 0.5)); rp_rand.append(int(rp < 0.5))
            if top is None or rt < top[0]:
                top = (rt, rp)
        ft.append(int(top[0] < 0.5)); fp.append(int(top[1] < 0.5))
    sess.close()
    return {'time_steps': TIME_STEPS, 'processed_snapshots': [0, TIME_STEPS - 2], 'n_rep_targets': len(rep0),
            'held_out': held_out,
            'sweep': sweep, 'flip_low_B1': {'n': len(low), 'k_total': int(sum(ft)), 'k_prop': int(sum(fp)),
                            'random_k_total': int(sum(rt_rand)), 'random_k_prop': int(sum(rp_rand))}}


if __name__ == '__main__':
    repo = os.path.abspath(os.environ['DYSAT_DIR'])
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for ds in os.environ.get('GRAIL_DATASETS', 'otc').split(','):
        out[ds] = run(ds, repo)
        print(ds, json.dumps(out[ds], indent=1), flush=True)
    path = os.path.join(here, '..', '..', 'results', 'unified', os.environ.get('GRAIL_OUT', 'dysat_placement.json'))
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', os.path.abspath(path))
