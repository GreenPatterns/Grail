"""
Paired prospective replication of the sliding-window regime (E8 extension; DSN revision WP3).

Question: is in-window insertion into s6 of the frozen model ("historical insertion") a useful
proxy for a rating that arrives in the next snapshot s7 and is read by sliding-window inference
(window s1..s7, same weights)? Both placements use the code of Table S13
(run_causal_regimes.Regimes: keys inwindow_s6 and sliding_s7).

Per dataset and seed (ten seeds, 0..9; one run = one training of GDTE's W = 7 model):
  targets : low-reputation targets of the frozen clean model (>= 2 raters in s0..s6, clean R in
            [0.5, 0.6)) that are also eligible after the clean slide (>= 2 raters in s1..s7,
            clean post-slide R >= 0.5). One target set serves both placements.
  sources : one draw per target of 5 distinct accounts that are not the target and do not rate
            it in s0..s7, RandomState([SOURCE_SEED, seed, target]); B = 1 uses the first, B = 5
            all five. The same sources serve both placements.
  scores  : R_tot (all raters, sources included) and R_prop (pre-existing raters of the
            placement's window, recomputed embeddings); each placement's shift is measured
            against its own clean reputation (sliding: the clean reputation after the same
            slide). Label sensitivity (Lambda, Check 2) from the same edges labeled trust.
The pre-declared proxy criterion (PREREG) is written into the output file before any outcome
is computed, and the verdict is evaluated against it at the end.

The W = 7 model is trained with train_fixed_mainz_model, which train_model also calls; the
forward-chaining accuracy sweep of train_model is skipped (it trains and discards other models),
and perf holds the W = 7 model's held-out s7 metrics.

Outputs: results/prospective_paired_{ds}.json (per-target rows, per-seed and across-seed
summaries, paired differences, verdict), rewritten after every seed.
Run: CUDA_VISIBLE_DEVICES=1 GRAIL_DATASETS=otc,alpha python code/experiments/run_prospective_paired.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json, gc, time
import numpy as np
import torch
import torch.nn.functional as F

from project_paths import RESULTS_DIR
from experiment_common import log
from mycode.mainz_protocol import train_fixed_mainz_model
from run_causal_regimes import Regimes

SEEDS = [int(s) for s in os.environ.get("GRAIL_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
BUDGETS = (1, 5)
EPOCHS = int(os.environ.get("GRAIL_EPOCHS", "50"))
SOURCE_SEED = 20261122
BOOT_SEED = 12345
N_BOOT = 10000
TAU = 0.5
REGS = ('inwindow_s6', 'sliding_s7')

PREREG = {
    "declared": "2026-11-22, written into this file before any outcome of this experiment was computed",
    "question": ("Is in-window insertion into s6 of the frozen model (historical insertion) a useful proxy "
                 "for a rating that arrives in the next snapshot s7 under sliding-window inference "
                 "(window s1..s7, same weights)?"),
    "targets": ("per seed: low-reputation targets of the frozen clean model (>= 2 raters in s0..s6, clean R in "
                "[0.5, 0.6)) that are also eligible after the clean slide (>= 2 raters in s1..s7, clean "
                "post-slide R >= 0.5); one target set for both placements"),
    "sources": ("per seed and target: one uniform draw of 5 distinct accounts that are not the target and do not "
                "rate it in s0..s7, RandomState([SOURCE_SEED, seed, target]); B=1 uses the first, B=5 all five; "
                "the same sources in both placements"),
    "baselines": "in-window: clean frozen reputation; sliding: clean reputation after the same slide",
    "estimand": ("per seed: paired difference sliding minus in-window of (a) the propagation-only flip rate, in "
                 "percentage points, and (b) the mean propagation-only shift; across seeds: the mean of the ten "
                 "per-seed differences with a seed-clustered percentile bootstrap 95% CI (10,000 resamples of "
                 "seeds, RandomState(12345))"),
    "criterion": ("historical insertion is a useful proxy on a dataset at budget B if the 95% CI of (a) lies "
                  "within [-10, +10] percentage points AND the 95% CI of (b) lies within [-0.02, +0.02]"),
    "verdict_rule": ("evaluated per dataset and budget (B = 1 and B = 5); the dataset-level verdict is "
                     "'useful proxy' only if both budgets meet the criterion"),
    "margins": {"flip_prop_pp": 10.0, "mean_shift_prop": 0.02},
    "seeds": SEEDS, "budgets": list(BUDGETS),
}


def clean_reps(R, regime):
    """Clean reputation (Eq. 1) and rater count of every node under one placement's window."""
    edges, labels, idx, Tp, _ = R._clean_window(regime)
    z = R._enc(edges, labels, idx, Tp)
    with torch.no_grad():
        p = F.softmax(torch.cat((z[edges[0]], z[edges[1]]), 1) @ R.m.regression_weights, 1)[:, 0]
    n = R.orc.num_nodes
    cnt = torch.bincount(edges[1], minlength=n)
    s = torch.zeros(n, dtype=torch.float64, device=p.device).index_add_(0, edges[1], p.double())
    rep = (s / cnt.clamp(min=1).double()).cpu().numpy()
    return rep, cnt.cpu().numpy()


def draw_sources(seed, t, banned, num_nodes, k=max(BUDGETS)):
    rng = np.random.RandomState([SOURCE_SEED, seed, int(t)])
    out = []
    for s in rng.permutation(num_nodes):
        if int(s) not in banned:
            out.append(int(s))
            if len(out) == k:
                break
    return out


def run_seed(ds, seed):
    t0 = time.time()
    log(f"[{ds} s{seed}] training W=7 model")
    trainer, perf = train_fixed_mainz_model(ds, seed=seed, train_time_slots=7, epochs=EPOCHS, start_prefix="WP3")
    R = Regimes({'model': trainer.model, 'device': trainer.device, 'edges': trainer.train_edges_final,
                 'labels': trainer.train_labels_final, 'trainer': trainer})
    rep_in, n_in = clean_reps(R, 'inwindow_s6')
    rep_sl, n_sl = clean_reps(R, 'sliding_s7')
    low = [v for v in range(R.orc.num_nodes) if n_in[v] >= 2 and TAU <= rep_in[v] < 0.6]
    targets = [v for v in low if n_sl[v] >= 2 and rep_sl[v] >= TAU]
    E8 = R.E[:, :R.idx[7] + 1]                       # s0..s7: sources must not rate the target here
    rows = []
    for t in targets:
        banned = set(E8[0, E8[1] == t].tolist()) | {t}
        srcs = draw_sources(seed, t, banned, R.orc.num_nodes)
        row = {'t': int(t), 'n_raters_in': int(n_in[t]), 'n_raters_slide': int(n_sl[t]),
               'R0_in': round(float(rep_in[t]), 6), 'R0_slide': round(float(rep_sl[t]), 6), 'srcs': srcs}
        for reg in REGS:
            r0_vec = rep_in[t] if reg == 'inwindow_s6' else rep_sl[t]
            for B in BUDGETS:
                d = R.score(reg, t, srcs[:B], 'distrust')
                tr = R.score(reg, t, srcs[:B], 'trust')
                # self-check: the vectorized clean reputation that set eligibility equals Regimes' R0
                assert abs(d['R0'] - r0_vec) < 1e-4 and abs(tr['R0'] - r0_vec) < 1e-4, (t, reg, d['R0'], r0_vec)
                row[f"{reg}_B{B}"] = {'R0': round(d['R0'], 6), 'R_tot': round(d['R_full'], 6),
                                      'R_prop': round(d['R_legit'], 6), 'R_tot_trust': round(tr['R_full'], 6)}
        rows.append(row)
    out = {'seed': seed, 'perf_W7_heldout_s7': {k: perf.get(k) for k in ('AUC', 'MCC', 'BAcc', 'epoch')},
           'n_low_frozen': len(low), 'n_targets': len(rows),
           'n_low_frozen_not_eligible_after_slide': len(low) - len(rows),
           'mean_slide_shift_of_clean_R_on_targets': round(float(np.mean([rep_sl[t] - rep_in[t] for t in targets])), 4)
           if targets else None,
           'minutes': round((time.time() - t0) / 60, 2), 'rows': rows}
    log(f"[{ds} s{seed}] AUC={perf.get('AUC'):.3f} low={len(low)} targets={len(rows)} ({out['minutes']} min)")
    del R, trainer
    gc.collect(); torch.cuda.empty_cache()
    return out


# ── statistics ──────────────────────────────────────────────────────────────────────────────
def seed_boot_ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return [None, None]
    m = v[np.random.RandomState(BOOT_SEED).randint(0, len(v), (N_BOOT, len(v)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def across(vals):
    v = [x for x in vals if x is not None]
    return {'mean': round(float(np.mean(v)), 4) if v else None,
            'sd': round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
            'ci95_seed_boot': seed_boot_ci(v), 'n_seeds': len(v)}


def seed_metrics(rows, B):
    """Per-seed rates (%) and mean shifts for both placements, and paired differences."""
    if not rows:
        return None
    m = {'n': len(rows)}
    per = {}
    for reg in REGS:
        c = [r[f"{reg}_B{B}"] for r in rows]
        per[reg] = {
            'flip_tot': 100 * np.mean([x['R_tot'] < TAU for x in c]),
            'flip_prop': 100 * np.mean([x['R_prop'] < TAU for x in c]),
            'dR_tot': np.mean([x['R_tot'] - x['R0'] for x in c]),
            'dR_prop': np.mean([x['R_prop'] - x['R0'] for x in c]),
            'Lambda': np.mean([abs(x['R_tot'] - x['R_tot_trust']) for x in c]),
        }
    for reg, short in (('inwindow_s6', 'in'), ('sliding_s7', 'slide')):
        for k, v in per[reg].items():
            m[f"{short}_{k}"] = round(float(v), 4)
    for k in ('flip_tot', 'flip_prop', 'dR_tot', 'dR_prop'):
        m[f"diff_{k}"] = round(float(per['sliding_s7'][k] - per['inwindow_s6'][k]), 4)
    fi = [r[f"inwindow_s6_B{B}"]['R_prop'] < TAU for r in rows]
    fs = [r[f"sliding_s7_B{B}"]['R_prop'] < TAU for r in rows]
    m['agree_flip_prop_pct'] = round(100 * float(np.mean([a == b for a, b in zip(fi, fs)])), 1)
    di = [r[f"inwindow_s6_B{B}"]['R_prop'] - r[f"inwindow_s6_B{B}"]['R0'] for r in rows]
    dsl = [r[f"sliding_s7_B{B}"]['R_prop'] - r[f"sliding_s7_B{B}"]['R0'] for r in rows]
    m['pearson_dR_prop'] = round(float(np.corrcoef(di, dsl)[0, 1]), 3) if len(rows) > 2 else None
    return m


def summarize(per_seed, row_filter=None):
    res = {}
    for B in BUDGETS:
        ms = []
        for s in per_seed:
            rows = [r for r in s['rows'] if row_filter is None or row_filter(r)]
            ms.append(seed_metrics(rows, B))
        ok = [m for m in ms if m is not None]
        keys = [k for k in ok[0] if k != 'n'] if ok else []
        res[f"B{B}"] = {'per_seed': {str(s['seed']): m for s, m in zip(per_seed, ms)},
                        'n_target_seed_pairs': int(sum(m['n'] for m in ok)),
                        'across_seeds': {k: across([m[k] for m in ok]) for k in keys}}
    return res


def verdict(summary):
    mg = PREREG['margins']
    out = {}
    for B in BUDGETS:
        a = summary[f"B{B}"]['across_seeds']
        cf, cs = a['diff_flip_prop']['ci95_seed_boot'], a['diff_dR_prop']['ci95_seed_boot']
        ok_f = None not in cf and -mg['flip_prop_pp'] <= cf[0] and cf[1] <= mg['flip_prop_pp']
        ok_s = None not in cs and -mg['mean_shift_prop'] <= cs[0] and cs[1] <= mg['mean_shift_prop']
        out[f"B{B}"] = {'ci_diff_flip_prop_pp': cf, 'ci_diff_dR_prop': cs,
                        'flip_within_margin': bool(ok_f), 'shift_within_margin': bool(ok_s),
                        'useful_proxy': bool(ok_f and ok_s)}
    out['dataset_verdict'] = ('useful proxy' if all(out[f"B{B}"]['useful_proxy'] for B in BUDGETS)
                              else 'not shown to be a useful proxy (criterion not met)')
    return out


def write(path, doc):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def run_dataset(ds):
    path = os.path.join(os.environ.get("GRAIL_OUT_DIR", str(RESULTS_DIR)), f"prospective_paired_{ds}.json")
    doc = {'experiment': 'WP3: paired prospective (sliding-window) replication of E8', 'dataset': ds,
           'prereg': PREREG, 'source_seed': SOURCE_SEED, 'epochs': EPOCHS, 'status': 'running', 'per_seed': []}
    write(path, doc)                                  # the pre-declaration is on disk before any outcome
    log(f"{ds}: pre-declaration written to {path}")
    for seed in SEEDS:
        doc['per_seed'].append(run_seed(ds, seed))
        write(path, doc)
    doc['summary'] = summarize(doc['per_seed'])
    doc['verdict'] = verdict(doc['summary'])
    doc['secondary_not_predeclared'] = {
        'note': 'targets low-reputation under BOTH clean placements (post-slide R also < 0.6); not part of the verdict',
        'summary': summarize(doc['per_seed'], row_filter=lambda r: r['R0_slide'] < 0.6)}
    doc['status'] = 'done'
    write(path, doc)
    for B in BUDGETS:
        a = doc['summary'][f"B{B}"]['across_seeds']
        log(f"{ds} B={B}: in flip tot/prop {a['in_flip_tot']['mean']}/{a['in_flip_prop']['mean']} | slide "
            f"{a['slide_flip_tot']['mean']}/{a['slide_flip_prop']['mean']} | diff prop flip "
            f"{a['diff_flip_prop']['mean']} CI{a['diff_flip_prop']['ci95_seed_boot']} | diff prop dR "
            f"{a['diff_dR_prop']['mean']} CI{a['diff_dR_prop']['ci95_seed_boot']}")
    log(f"{ds}: verdict {doc['verdict']}")


if __name__ == "__main__":
    for ds in os.environ.get("GRAIL_DATASETS", "otc,alpha").split(","):
        run_dataset(ds.strip())
