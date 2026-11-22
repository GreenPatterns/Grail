"""
Budget ablation of the propagation-only fall of Table V (E4 low stratum; DSN revision WP5).

In Table V the propagation-only flip rate of the low stratum falls as the budget B grows. This
script tests three candidate explanations on the seed-42 low stratum (25 targets) of OTC and
Alpha, with B optimized in-window distrust edges spliced at the end of s6:
  (i)   selection: E4 takes the top-B candidates by single-edge TOTAL shift, not by R_prop;
  (ii)  saturation: the target's own embedding stops moving after the first edge;
  (iii) dilution: GDTE's structural layer averages a node's in-neighbor features and incoming
        label embeddings (scatter_mean in mycode/SL.py), so B sources act through their mean
        and weaker extra sources dilute the strongest one.

Part 'e4' reproduces E4's procedure exactly (run_loo_flip_table: shared RandomState(42), one
fresh pool of 50 candidates per target AND budget, top-B by the exact single-edge total shift).
Part 'ablation' fixes ONE pool of 50 candidates per target (run_cohort_seeds.candidate_pool,
independent of the model) and grows nested source sets S_1 c ... c S_5 under five rules:
  random      : pool order (no model access)
  topB_tot    : top-B by single-edge R_tot (E4's rule)       topB_prop  : top-B by single-edge R_prop
  greedy_tot  : add the source that minimizes R_tot(S+a)     greedy_prop: same for R_prop
For every single candidate and every (rule, B) it logs R_tot, R_prop and the mechanism terms.
GDTE's head is linear before the softmax, so a rater's logit change (trust minus distrust)
splits exactly into a target term w_dst.(z_v' - z_v), common to all raters, and a rater term
w_src.(z_u' - z_u). Logged: target term, |dz_v|, mean rater term, share of raters whose
embedding moves, mean prediction change of the raters (= propagation shift), the sources' own
predictions, the target's in-edges in s6 and whether each source has other s6 edges.
Scoring uses mycode/fast_s6.S6Scorer internals; a self-check compares it with the full forward
pass (run_loo_flip_table._variants).

Outputs: results/budget_ablation_{ds}.json (per-target logs and summary).
Run: CUDA_VISIBLE_DEVICES=1 GRAIL_DATASETS=otc,alpha python code/experiments/run_budget_ablation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc
import numpy as np
import torch
import torch.nn.functional as F

from project_paths import RESULTS_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.fast_s6 import S6Scorer
from run_loo_flip_table import _pool, _variants
from run_cohort_seeds import candidate_pool

SEED = 42
PER_STRATUM = int(os.environ.get("GRAIL_PER_STRATUM", "25"))   # 25 = E4; smaller only for smoke tests
C_POOL = 50
BMAX = 5
TAU = 0.5
RULES = ('random', 'topB_tot', 'topB_prop', 'greedy_tot', 'greedy_prop')
EPOCHS = int(os.environ.get("GRAIL_EPOCHS", "50"))


class Probe:
    """Exact in-window (end of s6) attacked state of one target, with embeddings."""

    def __init__(self, oracle, edges, labels):
        self.sc = S6Scorer(oracle, edges, labels)
        W = oracle.model.regression_weights
        d = W.shape[0] // 2
        self.w_src, self.w_dst = (W[:d, 0] - W[:d, 1]), (W[d:, 0] - W[d:, 1])
        sc = self.sc
        s6 = sc.edges[:, sc.lo:sc.hi]
        self.s6_deg = torch.bincount(torch.cat([s6[0], s6[1]]), minlength=oracle.num_nodes).cpu().numpy()
        self.s6_in = torch.bincount(s6[1], minlength=oracle.num_nodes).cpu().numpy()
        self.s6_in_dis = torch.bincount(s6[1], weights=sc.labels[sc.lo:sc.hi, 1],
                                        minlength=oracle.num_nodes).cpu().numpy()

    def clean(self, t):
        sc = self.sc
        pre = (sc.dst == t).nonzero(as_tuple=True)[0]
        with torch.no_grad():
            return {'z_t': sc.z0[t], 'z_pre': sc.z0[sc.edges[0, pre]], 'p_pre': sc.clean_vector(t)}

    def attacked(self, t, srcs):
        sc, B = self.sc, len(srcs)
        ne = torch.tensor([list(srcs), [t] * B], dtype=sc.edges.dtype, device=sc.device)
        nl = torch.tensor([[0.0, 1.0]] * B, dtype=sc.labels.dtype, device=sc.device)
        with torch.no_grad():
            mu6 = sc.model.structural_layer(torch.cat([sc.edges[:, sc.lo:sc.hi], ne], 1),
                                            torch.cat([sc.labels[sc.lo:sc.hi], nl], 0))
            src_pre = sc.edges[0, (sc.dst == t).nonzero(as_tuple=True)[0]]
            rows = torch.cat([torch.tensor([t], device=sc.device), src_pre, ne[0]])
            mu_rows = sc.mu[rows].clone()
            mu_rows[:, sc.T - 1, :] = mu6[rows]
            z = sc.model.s3r(mu_rows)[:, sc.T - 1, :]
            n = len(src_pre)
            z_t, z_pre, z_inj = z[0], z[1:1 + n], z[1 + n:]
            zs = torch.cat([z_pre, z_inj])
            p = F.softmax(torch.cat((zs, z_t.unsqueeze(0).expand(n + B, -1)), 1)
                          @ sc.model.regression_weights, 1)[:, 0]
        return {'z_t': z_t, 'z_pre': z_pre, 'p_pre': p[:n], 'p_inj': p[n:]}

    def diag(self, t, cl, srcs):
        a = self.attacked(t, srcs)
        dz_t = a['z_t'] - cl['z_t']
        dz_pre = a['z_pre'] - cl['z_pre']
        r0 = cl['p_pre'].mean().item()
        return {
            'R_tot': round(torch.cat([a['p_pre'], a['p_inj']]).mean().item(), 6),
            'R_prop': round(a['p_pre'].mean().item(), 6),
            'dR_prop': round(a['p_pre'].mean().item() - r0, 6),
            'tgt_logit': round(float(dz_t @ self.w_dst), 6),
            'dz_t_norm': round(float(dz_t.norm()), 6),
            'rater_logit_mean': round(float((dz_pre @ self.w_src).mean()), 6),
            # 1e-4: clean (full batch) and attacked (row subset) embeddings differ by ~1e-6 float noise
            'rater_moved_frac': round(float((dz_pre.norm(dim=1) > 1e-4).float().mean()), 4),
            'src_P_mean': round(a['p_inj'].mean().item(), 6),
            'dP_raters': [round(x, 5) for x in (a['p_pre'] - cl['p_pre']).tolist()],
        }


def nested_sets(rule, pool, singles, probe, t, cl):
    """S_1..S_BMAX for one rule; greedy rules evaluate every remaining candidate per step."""
    if rule == 'random':
        return [pool[:b] for b in range(1, BMAX + 1)]
    key = 'R_tot' if rule.endswith('_tot') else 'R_prop'
    order = sorted(pool, key=lambda s: singles[s][key])
    if rule.startswith('topB'):
        return [order[:b] for b in range(1, BMAX + 1)]
    S, out = [order[0]], [[order[0]]]
    while len(S) < BMAX:
        best = min((s for s in pool if s not in S), key=lambda s: probe.diag(t, cl, S + [s])[key])
        S = S + [best]
        out.append(list(S))
    return out


def ablation_target(probe, o, t, raters):
    pool = candidate_pool(t, raters, o.num_nodes, C=C_POOL)
    cl = probe.clean(t)
    singles = {s: probe.diag(t, cl, [s]) for s in pool}
    rec = {'t': int(t), 'R0': round(cl['p_pre'].mean().item(), 6), 'n_raters': len(raters),
           's6_in': int(probe.s6_in[t]), 's6_in_distrust': int(round(probe.s6_in_dis[t])),
           'pool': pool, 'rules': {},
           'singles': {str(s): {k: v for k, v in singles[s].items() if k != 'dP_raters'} for s in pool}}
    for rule in RULES:
        sets = nested_sets(rule, pool, singles, probe, t, cl)
        rec['rules'][rule] = [dict(probe.diag(t, cl, S), B=len(S), srcs=S,
                                   srcs_fresh_s6=[bool(probe.s6_deg[s] == 0) for s in S]) for S in sets]
    return rec


def e4_reproduction(o, edges, labels, strata):
    """E4 (Table V) low stratum, exact procedure of run_loo_flip_table (pools redrawn per B)."""
    rng = np.random.RandomState(SEED)
    out = {}
    for stratum in ('high', 'moderate', 'low'):
        for B in (1, 3, 5):
            rows = []
            for t in strata.get(stratum, []):
                pool = _pool(o, edges, t, C_POOL, rng)       # advances the shared RNG as in E4
                if stratum != 'low' or len(pool) < B:
                    continue
                cf = o.score_candidates_counterfactual_processed(edges, labels, t, pool, sign='distrust')
                srcs = [s for s, _ in sorted(cf.items(), key=lambda x: x[1])[:B]]
                R0, Rf, Rp = _variants(o, edges, labels, t, srcs)
                if R0 == R0 and R0 >= TAU:
                    rows.append({'t': int(t), 'srcs': [int(s) for s in srcs], 'R0': R0, 'R_tot': Rf, 'R_prop': Rp})
            if stratum == 'low':
                out[f"B{B}"] = {'n': len(rows),
                                'flip_tot_pct': round(100 * np.mean([r['R_tot'] < TAU for r in rows]), 1),
                                'flip_prop_pct': round(100 * np.mean([r['R_prop'] < TAU for r in rows]), 1),
                                'dR_tot': round(float(np.mean([r['R_tot'] - r['R0'] for r in rows])), 4),
                                'dR_prop': round(float(np.mean([r['R_prop'] - r['R0'] for r in rows])), 4),
                                'rows': rows}
                log(f"  E4 repro low B={B}: n={len(rows)} flip tot {out[f'B{B}']['flip_tot_pct']}% "
                    f"prop {out[f'B{B}']['flip_prop_pct']}% dR_prop {out[f'B{B}']['dR_prop']}")
    return out


# ── summary ─────────────────────────────────────────────────────────────────────────────────
def _boot_ci(v, n_boot=10000, seed=12345):
    v = np.asarray(v, float)
    m = v[np.random.RandomState(seed).randint(0, len(v), (n_boot, len(v)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def _fit(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    return {'mae': round(float(np.mean(np.abs(obs - pred))), 5),
            'r2': round(float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)), 3)}


def summarize(recs):
    S = {'n_targets': len(recs),
         'share_targets_without_s6_in_edges': round(float(np.mean([r['s6_in'] == 0 for r in recs])), 3),
         'by_rule': {}, 'combination_of_single_edges': {}, 'selection_test': {}}
    for rule in RULES:
        rb = {}
        for b in range(1, BMAX + 1):
            x = [r['rules'][rule][b - 1] for r in recs]
            rb[f"B{b}"] = {'flip_tot_pct': round(100 * np.mean([d['R_tot'] < TAU for d in x]), 1),
                           'flip_prop_pct': round(100 * np.mean([d['R_prop'] < TAU for d in x]), 1),
                           'dR_tot': round(float(np.mean([d['R_tot'] - r['R0'] for d, r in zip(x, recs)])), 4),
                           'dR_prop': round(float(np.mean([d['dR_prop'] for d in x])), 4),
                           'tgt_logit': round(float(np.mean([d['tgt_logit'] for d in x])), 4),
                           'dz_t_norm': round(float(np.mean([d['dz_t_norm'] for d in x])), 4),
                           'rater_logit_mean': round(float(np.mean([d['rater_logit_mean'] for d in x])), 4),
                           'rater_moved_frac': round(float(np.mean([d['rater_moved_frac'] for d in x])), 3),
                           'src_P_mean': round(float(np.mean([d['src_P_mean'] for d in x])), 4),
                           'share_fresh_sources': round(float(np.mean([f for d in x for f in d['srcs_fresh_s6']])), 3)}
        S['by_rule'][rule] = rb
        d51 = [r['rules'][rule][4]['dR_prop'] - r['rules'][rule][0]['dR_prop'] for r in recs]
        f51 = [int(r['rules'][rule][4]['R_prop'] < TAU) - int(r['rules'][rule][0]['R_prop'] < TAU) for r in recs]
        S['selection_test'][rule] = {
            'mean_dR_prop_B5_minus_B1': round(float(np.mean(d51)), 4), 'ci95_target_boot': _boot_ci(d51),
            'targets_weaker_at_B5': int(sum(x > 0 for x in d51)), 'targets_stronger_at_B5': int(sum(x < 0 for x in d51)),
            'flip_prop_B5_minus_B1_pp': round(100 * float(np.mean(f51)), 1)}
    # how multi-edge effects combine the single-edge effects of the same sources
    groups = (('all_targets', lambda r: True), ('no_s6_in_edges', lambda r: r['s6_in'] == 0),
              ('with_s6_in_edges', lambda r: r['s6_in'] > 0))
    for q in ('tgt_logit', 'dR_prop'):
        res = {}
        for (scope, rules), (gname, gf) in [(sr, g) for sr in (('all_rules', RULES), ('random', ('random',)),
                                                                ('topB_tot', ('topB_tot',))) for g in groups]:
            if gname != 'all_targets':
                scope = f"{scope}|{gname}"
            obs, pf, pm, ps = [], [], [], []
            for r in (x for x in recs if gf(x)):
                for rule in rules:
                    for d in r['rules'][rule][1:]:
                        sv = [r['singles'][str(s)][q] for s in d['srcs']]
                        obs.append(d[q]); pf.append(sv[0]); pm.append(float(np.mean(sv))); ps.append(float(np.sum(sv)))
            res[scope] = {'n_points': len(obs), 'first_edge_only (saturation)': _fit(obs, pf),
                          'mean_of_singles (dilution)': _fit(obs, pm), 'sum_of_singles (additive)': _fit(obs, ps),
                          'median_obs_over_first': round(float(np.median(np.asarray(obs) / np.asarray(pf))), 3)}
        S['combination_of_single_edges'][q] = res
    # B-trend split by whether the target has in-edges of its own in s6
    S['by_s6_in_edges'] = {}
    for name, f in (('no_s6_in_edges', lambda r: r['s6_in'] == 0), ('with_s6_in_edges', lambda r: r['s6_in'] > 0)):
        sub = [r for r in recs if f(r)]
        S['by_s6_in_edges'][name] = {'n': len(sub), **{rule: {
            f"B{b}": {'flip_prop_pct': round(100 * np.mean([r['rules'][rule][b - 1]['R_prop'] < TAU for r in sub]), 1),
                      'dR_prop': round(float(np.mean([r['rules'][rule][b - 1]['dR_prop'] for r in sub])), 4),
                      'tgt_logit': round(float(np.mean([r['rules'][rule][b - 1]['tgt_logit'] for r in sub])), 4)}
            for b in (1, 5)} for rule in ('random', 'topB_tot', 'greedy_prop')}} if sub else {'n': 0}
    return S


def run_dataset(ds, parts):
    log(f"\n{'=' * 60}\n  BUDGET ABLATION — {ds}\n{'=' * 60}")
    tr = train_model(ds, seed=SEED, epochs=EPOCHS)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    edges, labels = tr["edges"], tr["labels"]
    strata = get_targets(o, edges, labels, per_stratum=PER_STRATUM)
    low = [int(t) for t in strata.get('low', [])]
    out = {'dataset': ds, 'seed': SEED, 'epochs': EPOCHS, 'low_stratum': low,
           'perf': {k: tr['perf'].get(k) for k in ('AUC', 'MCC')}}
    probe = Probe(o, edges, labels)
    # self-check against the full forward pass
    rng = np.random.RandomState(0)
    for t in low[:3]:
        srcs = [int(s) for s in rng.choice(o.num_nodes, 3, replace=False) if s != t][:3]
        R0, Rf, Rp = _variants(o, edges, labels, t, srcs)
        d = probe.diag(t, probe.clean(t), srcs)
        assert abs(d['R_tot'] - Rf) < 1e-4 and abs(d['R_prop'] - Rp) < 1e-4, (t, d, Rf, Rp)
    if 'e4' in parts:
        out['e4_reproduction'] = e4_reproduction(o, edges, labels, strata)
    if 'ablation' in parts:
        src, dst = edges[0].cpu().numpy(), edges[1].cpu().numpy()
        recs = []
        for i, t in enumerate(low):
            recs.append(ablation_target(probe, o, t, src[dst == t].tolist()))
            log(f"  [{ds}] target {i + 1}/{len(low)} done")
        out['ablation'] = {'targets': recs, 'summary': summarize(recs)}
        for rule in RULES:
            rb = out['ablation']['summary']['by_rule'][rule]
            log(f"  {rule:12s} prop flip % B1..5: {[rb[f'B{b}']['flip_prop_pct'] for b in range(1, 6)]} "
                f"dR_prop: {[rb[f'B{b}']['dR_prop'] for b in range(1, 6)]}")
    path = os.path.join(os.environ.get("GRAIL_OUT_DIR", str(RESULTS_DIR)), f"budget_ablation_{ds}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    log(f"Wrote {path}")
    del probe, o, tr
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    parts = os.environ.get("GRAIL_PARTS", "e4,ablation").split(",")
    for ds in os.environ.get("GRAIL_DATASETS", "otc,alpha").split(","):
        run_dataset(ds.strip(), parts)
