"""
Placement test on the official EvolveGCN release (Pareja et al., AAAI 2020;
https://github.com/IBM/EvolveGCN, commit 90869062). The released code, its Bitcoin-OTC /
Bitcoin-Alpha edge-classification configuration (EGCN-O by default) and its own trainer are
used unchanged; the only shim is PyYAML's Loader argument, which the 2019 code omits.

EvolveGCN reads a window of num_hist_steps + 1 = 11 three-week adjacency matrices (edge
counts; the rating sign is only the prediction target) and scores an edge (u, v) from the
node embeddings at the last step. Reputation (Eq. 1 of the paper) of a target v at the first
test step T is the mean predicted trust over v's directed raters inside that window. A
distrust edge is injected at a window step (s_{T-10}..s_T) or past the window (T+1); we report
the reputation shift, its propagation-only part (pre-existing raters only), the change of the
target's embedding, and the label sensitivity. EvolveGCN never reads edge labels, so the
label swap is zero by construction at every placement: the diagnostic that exposes a no-op
here is the embedding change, which is exactly zero past the window.

Run (GPU, ps env):
  EVOLVEGCN_DIR=<clone> GRAIL_DATASETS=bitcoinotc,bitcoinalpha \
      python code/experiments/run_evolvegcn_placement.py
  (EGCN_MODEL=egcn_o|egcn_h, EGCN_EPOCHS overrides the configured 1000)
"""
import copy, functools, json, os, random, sys

import numpy as np
import torch
import yaml

SEED = 42
N_SWEEP = 12
N_FLIP = 25
C_POOL = 50
TRUST, DISTRUST = 1, 0            # EvolveGCN's labels: 1 = positive-majority rating


def _setup(repo, data, model, epochs):
    yaml.load = functools.partial(yaml.load, Loader=yaml.FullLoader)
    os.chdir(repo); sys.path.insert(0, repo)
    import utils as u
    import run_exp as rx
    import splitter as sp
    import Cross_Entropy as ce
    import trainer as tr
    tag = 'otc' if data == 'bitcoinotc' else 'alpha'
    cfg = os.path.join(repo, 'experiments', f'parameters_bitcoin_{tag}_edgecls_{model}.yaml')
    sys.argv = ['run_exp.py', '--config_file', cfg]
    args = u.parse_args(u.create_parser())
    args.data, args.use_logfile = data, False     # the released OTC config names bitcoinalpha
    if epochs:
        args.num_epochs = int(epochs)
    args.use_cuda = torch.cuda.is_available() and args.use_cuda
    args.device = 'cuda' if args.use_cuda else 'cpu'
    for f in (np.random.seed, random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
        f(args.seed)
    args.rank, args.wsize = 0, 1
    args = rx.build_random_hyper_params(args)
    dataset = rx.build_dataset(args)
    tasker = rx.build_tasker(args, dataset)
    splitter = sp.splitter(args, tasker)
    gcn = rx.build_gcn(args, tasker)
    classifier = rx.build_classifier(args, tasker)
    loss = ce.Cross_Entropy(args, dataset).to(args.device)
    trainer = tr.Trainer(args, splitter=splitter, gcn=gcn, classifier=classifier,
                         comp_loss=loss, dataset=dataset, num_classes=tasker.num_classes)
    selected = _track_best_valid(trainer)
    trainer.train()
    # The released trainer selects its reported model by the validation measure (F1 of class
    # 0, the distrust class, in these configs) but leaves the last epoch's weights in place;
    # restore the selected epoch's weights so the attacked model is the one the release reports.
    assert selected['gcn'] is not None, 'validation measure never rose above 0'
    with torch.no_grad():
        for p_, v_ in zip(trainer.gcn.parameters(), selected['gcn']):
            p_.copy_(v_.to(p_.device))
    trainer.classifier.load_state_dict({k: v.to(args.device) for k, v in selected['cls'].items()})
    # Score on CPU: EvolveGCN's sparse GPU products are non-deterministic (repeat noise about
    # 4e-3 in the embeddings), so exact zeros past the window are only visible on CPU. The
    # trained weights are loaded into CPU copies built by the same released constructors.
    # (EGCN overwrites nn.Module._parameters with a ParameterList, so state_dict() and .to()
    # fail on it; its parameters() order is fixed, so weights are copied pairwise.)
    gpu_params = [p.detach().cpu() for p in trainer.gcn.parameters()]
    sd_c = {k: v.detach().cpu() for k, v in trainer.classifier.state_dict().items()}
    args.device, args.use_cuda = 'cpu', False
    gcn_cpu = rx.build_gcn(args, tasker)
    cpu_params = list(gcn_cpu.parameters())
    assert len(cpu_params) == len(gpu_params) and all(a.shape == b.shape for a, b in zip(cpu_params, gpu_params))
    with torch.no_grad():
        for dst_p, src_p in zip(cpu_params, gpu_params):
            dst_p.copy_(src_p)
    trainer.gcn = gcn_cpu
    trainer.classifier = rx.build_classifier(args, tasker); trainer.classifier.load_state_dict(sd_c)
    trainer.args.device = 'cpu'
    held_out = {'selected_epoch': selected['epoch'], 'valid_f1_class0': selected['valid'],
                **selected['test'].get(selected['epoch'], {})}
    return u, args, dataset, tasker, trainer, held_out


def _track_best_valid(trainer):
    """Wrap the released trainer's run_epoch: keep the weights of every new best validation
    epoch (the release's own model selection) and the test measures it logs at that epoch."""
    sel = {'valid': 0.0, 'epoch': None, 'gcn': None, 'cls': None, 'test': {}}   # as the release: best starts at 0
    orig = trainer.run_epoch

    def run_epoch(split, epoch, set_name, grad):
        out = orig(split, epoch, set_name, grad)
        lg = trainer.logger
        if set_name == 'VALID' and float(out[0]) > sel['valid']:
            sel.update(valid=float(out[0]), epoch=int(epoch),
                       gcn=[q.detach().cpu().clone() for q in trainer.gcn.parameters()],
                       cls={k: v.detach().cpu().clone() for k, v in trainer.classifier.state_dict().items()})
        if set_name == 'TEST':
            _, _, f1 = lg.calc_microavg_eval_measures(lg.conf_mat_tp, lg.conf_mat_fn, lg.conf_mat_fp)
            per = [lg.calc_eval_measures_per_class(lg.conf_mat_tp, lg.conf_mat_fn, lg.conf_mat_fp, c)[2]
                   for c in range(lg.num_classes)]
            sel['test'][int(epoch)] = {'test_microavg_f1': float(f1), 'test_f1_class0': float(per[0]),
                                       'test_f1_class1': float(per[1])}
        return out
    trainer.run_epoch = run_epoch
    return sel


def _directed_edges(dataset, args, u):
    """The released loader's directed (source, target, step) triples, before it adds the
    reversed links, with its node renumbering and three-week aggregation."""
    e = dataset.load_edges(args.bitcoin_args)
    e = dataset.make_contigous_node_ids(e)
    steps = u.aggregate_by_time(e[:, 3], args.bitcoin_args.aggr_time)
    return e[:, 0].numpy(), e[:, 1].numpy(), steps.numpy()


class EGCNScorer:
    def __init__(self, u, args, dataset, tasker, trainer, T):
        self.u, self.args, self.data, self.tasker, self.tr, self.T = u, args, dataset, tasker, trainer, T
        self.clean_edges = {k: v.clone() for k, v in dataset.edges.items()}
        trainer.gcn.eval(); trainer.classifier.eval()
        # EGCN keeps its GRCU layers in a plain list, so .eval() does not reach them and their
        # RReLU activations would keep sampling random slopes; switch them explicitly.
        for layer in getattr(trainer.gcn, 'GRCU_layers', []):
            layer.eval()

    def _embed(self, extra=()):
        """Node embeddings at step T after adding directed edges (s, v, step, label)."""
        edges = {k: v.clone() for k, v in self.clean_edges.items()}
        if extra:
            rows = []
            for s, v, step, lab in extra:       # the released loader stores both directions
                rows += [[s, v, step, lab], [v, s, step, lab]]
            edges['idx'] = torch.cat([edges['idx'], torch.tensor(rows, dtype=edges['idx'].dtype)])
            edges['vals'] = torch.cat([edges['vals'], torch.ones(len(rows), dtype=edges['vals'].dtype)])
        self.data.edges = edges
        try:
            sample = torch.utils.data.default_collate([self.tasker.get_sample(self.T, test=True)])
            s = self.tr.prepare_sample(sample)
            with torch.no_grad():
                z = self.tr.gcn(s.hist_adj_list, s.hist_ndFeats_list, s.node_mask_list)
        finally:
            self.data.edges = self.clean_edges
        return z

    def p_trust(self, z, src, v):
        src = torch.as_tensor(src, device=z.device)
        cls_in = torch.cat([z[src], z[v].unsqueeze(0).expand(len(src), -1)], dim=1)
        with torch.no_grad():
            return torch.softmax(self.tr.classifier(cls_in), dim=1)[:, TRUST]


def run(ds, repo, model, epochs):
    u, args, dataset, tasker, trainer, held_out = _setup(repo, ds, model, epochs)
    T = int(np.floor(float(tasker.data.max_time) * (args.train_proportion + args.dev_proportion)))
    src, dst, step = _directed_edges(dataset, args, u)
    win = (step >= T - args.num_hist_steps) & (step <= T)
    raters = {}
    for s, v in zip(src[win], dst[win]):
        raters.setdefault(int(v), []).append(int(s))
    sc = EGCNScorer(u, args, dataset, tasker, trainer, T)
    z0 = sc._embed()
    rep0 = {v: sc.p_trust(z0, r, v).mean().item() for v, r in raters.items() if len(r) >= 2}
    rng = np.random.RandomState(SEED)
    n = dataset.num_nodes

    def attacked(v, s_new, when, label=DISTRUST):
        z = sc._embed([(s_new, v, when, label)])
        r_pre = sc.p_trust(z, raters[v], v)
        r_all = torch.cat([r_pre, sc.p_trust(z, [s_new], v)])
        return z, r_all.mean().item(), r_pre.mean().item()

    def pool(v):
        banned = set(raters[v]) | {v}
        return [int(s) for s in rng.permutation(n) if int(s) not in banned][:C_POOL]

    # placement sweep: every window step and one step past the window
    positions = list(range(T - args.num_hist_steps, T + 1)) + [T + 1]
    targets = [v for v in rng.permutation(sorted(rep0)) if 0.5 <= rep0[v] < 0.9][:N_SWEEP]
    sweep = {str(p): {"dR_total": [], "dR_prop": [], "emb_change": [], "label_delta": []} for p in positions}
    for v in targets:
        s_new = pool(v)[0]
        for p in positions:
            z_d, rt_d, rp_d = attacked(v, s_new, p, DISTRUST)
            _, rt_t, _ = attacked(v, s_new, p, TRUST)
            cell = sweep[str(p)]
            cell["dR_total"].append(rt_d - rep0[v]); cell["dR_prop"].append(rp_d - rep0[v])
            cell["emb_change"].append((z_d[v] - z0[v]).norm().item()); cell["label_delta"].append(abs(rt_d - rt_t))
    sweep = {p: {k: float(np.mean(x)) for k, x in c.items()} for p, c in sweep.items()}

    # single-edge flip test on low-reputation targets at step T: optimized source (best of the
    # pool by total effect) and random source (the pool's first candidate)
    low = [v for v in rng.permutation(sorted(rep0)) if 0.5 <= rep0[v] < 0.6][:N_FLIP]
    flips_t, flips_p, rand_t, rand_p = [], [], [], []
    for v in low:
        best = None
        cands = pool(v)
        for j, s_new in enumerate(cands):
            _, rt, rp = attacked(v, s_new, T)
            if j == 0:
                rand_t.append(int(rt < 0.5)); rand_p.append(int(rp < 0.5))
            if best is None or rt < best[0]:
                best = (rt, rp)
        flips_t.append(int(best[0] < 0.5)); flips_p.append(int(best[1] < 0.5))
    return {"model": model, "T": T, "window": [T - args.num_hist_steps, T], "epochs_cap": args.num_epochs,
            "n_targets_rep": len(rep0), "sweep": sweep, "past_window_step": T + 1,
            "held_out": held_out,
            "flip_low_B1": {"n": len(low), "k_total": int(sum(flips_t)), "k_prop": int(sum(flips_p)),
                            "total_pct": round(100 * np.mean(flips_t), 1) if low else None,
                            "prop_pct": round(100 * np.mean(flips_p), 1) if low else None,
                            "random_k_total": int(sum(rand_t)), "random_k_prop": int(sum(rand_p))}}


if __name__ == '__main__':
    repo = os.path.abspath(os.environ['EVOLVEGCN_DIR'])
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, '..', '..', 'results', 'unified')
    model = os.environ.get('EGCN_MODEL', 'egcn_o')
    res = {}
    for ds in os.environ.get('GRAIL_DATASETS', 'bitcoinotc,bitcoinalpha').split(','):
        res[ds] = run(ds, repo, model, os.environ.get('EGCN_EPOCHS'))
        print(ds, json.dumps(res[ds]["sweep"], indent=1), res[ds]["flip_low_B1"], res[ds]["held_out"], flush=True)
    path = os.path.join(out_dir, os.environ.get('GRAIL_OUT', f'evolvegcn_placement_{model}.json'))
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print('saved', os.path.abspath(path))
