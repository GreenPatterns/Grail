"""
Causal next-snapshot evaluation (reviewer blocker #1).

The reviewer objects that both the appended placement (past the horizon) and the
in-window placement (spliced retroactively into a historical snapshot s6) are
artificial. The deployment-faithful test submits the attack edge in the
chronologically NEXT snapshot and lets the model process it under a realistic
window policy, with clean and attacked runs ending at the same chronological
time.

The S3R temporal backbone is built with max_seq_len = T_train = 7, so a frozen
model cannot EXPAND its window past the training horizon (an honest architectural
fact). The two frozen-weight causal regimes it CAN realize are:

  appended     : edge past the processed horizon (s9) -- the silent-no-op bug.
  inwindow_s6  : edge spliced into the last processed snapshot s6 (paper).
  sliding_s7   : SLIDING WINDOW of length 7 ending at the NEW snapshot s7
                 (process s1..s7, drop s0). The attack edge lives in a genuinely
                 new snapshot -- no retroactive rewrite -- and the frozen model
                 slides its window forward to embed it. This is the deployment-
                 faithful frozen-inference regime.

(The periodic-retraining regime, which rebuilds the backbone at T=8, is measured
separately in run_poison_matrix.py.)

For every regime we report, at that regime's own evaluation horizon: naive dR
(all raters), leakage-free dR_legit (pre-existing raters only, Eq. R_legit),
label-Delta, and flip rate vs the regime's own clean reputation.

Run: GRAIL_DATASETS=otc,alpha python code/experiments/run_causal_regimes.py
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

SEED = 42
BUDGETS = [1, 5]
PER_STRATUM = 25


def _cp(k, n, a=0.05):
    from scipy import stats as sp
    if n == 0:
        return (0.0, 0.0)
    lo = 0.0 if k == 0 else sp.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else sp.beta.ppf(1 - a / 2, k + 1, n - k)
    return (round(float(lo), 3), round(float(hi), 3))


class Regimes:
    def __init__(self, trained):
        self.m = trained['model']; self.dev = trained['device']
        self.orc = ReputationAttackOracle(self.m, list(self.m.index_list), self.dev)
        self.pe, self.pl = trained['edges'], trained['labels']          # processed window s0..s6
        self.idx = list(self.m.index_list)                              # 10 boundaries
        fe = np.asarray(trained['trainer'].graph['edges'])
        fl = np.asarray(trained['trainer'].graph['labels'])
        self.E = torch.as_tensor(fe.T.astype(np.int64), device=self.dev)
        self.L = torch.as_tensor(fl.astype(np.float32), device=self.dev)
        self.T = 7

    def _enc(self, edges, labels, idxl, Tp):
        sT, sI = self.m.args.train_time_slots, self.m.index_list
        try:
            self.m.args.train_time_slots = Tp
            self.m.index_list = idxl
            with torch.no_grad():
                return self.orc._compute_node_embeddings(edges, labels)
        finally:
            self.m.args.train_time_slots = sT
            self.m.index_list = sI

    def _ptrust(self, z, target, edges, cols):
        s = edges[0, cols]
        feats = torch.cat((z[s], z[target].unsqueeze(0).expand(len(cols), -1)), 1)
        return F.softmax(feats @ self.m.regression_weights, 1)[:, 0]

    def _clean_window(self, regime):
        """Return (edges, labels, idx, Tp, split_for_new_edge)."""
        if regime in ('appended', 'inwindow_s6'):
            return self.pe, self.pl, list(self.idx), 7, None
        if regime == 'sliding_s7':
            lo, hi = self.idx[0] + 1, self.idx[7] + 1
            Ew, Lw = self.E[:, lo:hi].contiguous(), self.L[lo:hi].contiguous()
            nidx = [self.idx[j + 1] - lo for j in range(7)]      # s1..s7, readout s7
            return Ew, Lw, nidx, 7, None
        raise ValueError(regime)

    def _attack(self, regime, edges, labels, idx, target, srcs, sign):
        """Return (aug_edges, aug_labels, aug_idx, injected_col_set)."""
        B = len(srcs)
        lab = [1.0, 0.0] if sign == 'trust' else [0.0, 1.0]
        ne = torch.tensor([list(srcs), [target] * B], dtype=torch.long, device=self.dev)
        nl = torch.tensor([lab] * B, dtype=torch.float, device=self.dev)
        if regime == 'appended':
            ae = torch.cat([edges, ne], 1); al = torch.cat([labels, nl], 0)
            ai = list(idx); ai[-1] += B
            inj = set(range(edges.shape[1], edges.shape[1] + B))
        else:  # inwindow_s6 or sliding_s7: splice into last processed snapshot (readout step)
            split = idx[self.T - 1] + 1
            ae = torch.cat([edges[:, :split], ne, edges[:, split:]], 1)
            al = torch.cat([labels[:split], nl, labels[split:]], 0)
            ai = [b + B if i >= self.T - 1 else b for i, b in enumerate(idx)]
            inj = set(range(split, split + B))
        return ae, al, ai, inj

    def score(self, regime, target, srcs, sign):
        edges, labels, idx, Tp, _ = self._clean_window(regime)
        z0 = self._enc(edges, labels, idx, Tp)
        cdst = (edges[1] == target).nonzero(as_tuple=True)[0]
        if len(cdst) == 0:
            return None
        with torch.no_grad():
            R0 = self._ptrust(z0, target, edges, cdst).mean().item()
        ae, al, ai, inj = self._attack(regime, edges, labels, idx, target, srcs, sign)
        z1 = self._enc(ae, al, ai, Tp)
        dst = (ae[1] == target).nonzero(as_tuple=True)[0]
        pre = torch.tensor([c for c in dst.tolist() if c not in inj], dtype=torch.long, device=self.dev)
        with torch.no_grad():
            R_full = self._ptrust(z1, target, ae, dst).mean().item()
            R_leg = self._ptrust(z1, target, ae, pre).mean().item() if len(pre) else float('nan')
        return {'R0': R0, 'R_full': R_full, 'R_legit': R_leg}


def _pool(orc, edges, target, C, rng):
    ex = set(edges[0, edges[1] == target].tolist()); ex.add(target)
    uni = [n for n in range(orc.num_nodes) if n not in ex]
    return [int(s) for s in (rng.choice(uni, C, replace=False) if len(uni) > C else uni)]


def run_dataset(ds):
    log(f"\n{'='*62}\n  CAUSAL REGIMES — {ds}\n{'='*62}")
    trained = train_model(ds)
    R = Regimes(trained)
    strata = get_targets(R.orc, R.pe, R.pl, per_stratum=PER_STRATUM)
    targets = strata.get('low', []) + strata.get('moderate', [])
    rng = np.random.RandomState(SEED)
    regimes = ['appended', 'inwindow_s6', 'sliding_s7']
    out = {}
    for B in BUDGETS:
        for reg in regimes:
            drt, drl, lblt, ft, fl = [], [], [], [], []
            for t in targets:
                pool = _pool(R.orc, R.pe, t, 50, rng)
                if len(pool) < B:
                    continue
                srcs = list(rng.choice(pool, B, replace=False))  # random naive sources
                d = R.score(reg, t, srcs, 'distrust')
                tr = R.score(reg, t, srcs, 'trust')
                if d is None or tr is None or d['R0'] != d['R0']:
                    continue
                if d['R0'] < 0.5:            # eligible: above gate under this regime's clean R
                    continue
                drt.append(d['R_full'] - d['R0'])
                if d['R_legit'] == d['R_legit']:
                    drl.append(d['R_legit'] - d['R0'])
                    fl.append(1 if d['R_legit'] < 0.5 else 0)
                lblt.append(abs((d['R_full'] - d['R0']) - (tr['R_full'] - tr['R0'])))
                ft.append(1 if d['R_full'] < 0.5 else 0)
            n = len(ft)
            out[f"{reg}_B{B}"] = {
                'n_eligible': n,
                'dR_naive': round(float(np.mean(drt)), 4) if drt else None,
                'dR_legit': round(float(np.mean(drl)), 4) if drl else None,
                'label_delta': float(f"{np.mean(lblt):.2e}") if lblt else None,
                'flip_naive_pct': round(100 * np.mean(ft), 1) if n else None,
                'flip_legit_pct': round(100 * np.mean(fl), 1) if fl else None,
                'flip_naive_ci': _cp(sum(ft), n),
                'flip_legit_ci': _cp(sum(fl), len(fl)),
            }
            r = out[f"{reg}_B{B}"]
            log(f"  {reg:12s} B={B}: n={n:2d} dR_naive={r['dR_naive']} dR_legit={r['dR_legit']} "
                f"lblD={r['label_delta']} flip_naive={r['flip_naive_pct']}% flip_legit={r['flip_legit_pct']}%")
    del R, trained
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    dsets = os.environ.get("GRAIL_DATASETS", "otc,alpha").split(",")
    results = {ds.strip(): run_dataset(ds.strip()) for ds in dsets}
    path = os.path.join(str(RESULTS_DIR / "unified"), "causal_regimes.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nWrote {path}")
