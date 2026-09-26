"""Generate/refresh paper figures into figures/.

- scaling_curve.pdf  (NEW): ΔR vs subsample size N (Epinions), from e10 JSONs.
- boxplot_b5.pdf     (regenerate; was missing): per-target ΔR distribution per
                      method at B=5 (OTC+Alpha), collected on GPU.
- ablation_sparsity.pdf (regenerate; was missing): target in-degree vs ΔR
                      (GRAIL-Fwd, Alpha), collected on GPU.

Run on GPU: CUDA_VISIBLE_DEVICES=0 python experiments/make_paper_figures.py
matplotlib labels use mathtext only (no external LaTeX needed).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# embed TrueType (Type42), never Type3 bitmap fonts (which render as
# letter-spaced/dropped glyphs in some viewers and are disallowed by USENIX)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'dejavuserif'})

from project_paths import RESULTS_DIR, PROJECT_DIR
from experiment_common import train_model, get_targets, log
from mycode.trust_influence import ReputationAttackOracle
from mycode.attacks_gpu import identify_good_nodes_from_tensors
from mycode import baselines_sota

FIGS = PROJECT_DIR / "figures"
RD = RESULTS_DIR / "unified"


def fig_scaling():
    d = {}
    for fn in ["e10_scaling.json", "e10_scaling_large.json"]:
        p = RD / fn
        if p.exists():
            d.update(json.load(open(p)))
    if not d:
        log("  (no e10 scaling json found; skipping scaling_curve)")
        return
    Ns = sorted(int(k) for k in d)
    plt.figure(figsize=(5, 3.2))
    for m in ["Expert", "Fwd", "PRBCD", "Grad"]:
        ys = [d[str(N)][m]["mean"] for N in Ns]
        plt.plot(Ns, ys, marker="o", label=m)
    plt.axhline(0, color="k", lw=0.6, ls="--")
    plt.xscale("log")
    plt.xlabel("Subsample size $N$ (nodes)")
    plt.ylabel(r"mean $\Delta R$ ($B{=}5$)")
    plt.title("Attack-utility scaling (Epinions)")
    plt.legend(fontsize=8)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIGS / "scaling_curve.pdf")
    plt.close()
    log(f"  wrote scaling_curve.pdf (N={Ns})")


def _collect(ds, B=5):
    tr = train_model(ds, seed=42)
    o = ReputationAttackOracle(tr["model"], tr["index_list"], tr["device"])
    e, l = tr["edges"], tr["labels"]
    strata = get_targets(o, e, l, per_stratum=10)
    gn = identify_good_nodes_from_tensors(e, l, num_nodes=o.num_nodes)
    rows = []
    for t in [x for ts in strata.values() for x in ts]:
        if o.compute_reputation(e, l, t) < 0.1:
            continue
        fwd = [s for s, _ in o.rank_new_edges(e, l, t, top_k=B, max_candidates=80)]
        grad = [s for s, _ in o.rank_new_edges_gradient(e, l, t, top_k=B, max_candidates=80)]
        prb = baselines_sota.rank_prbcd(o, e, l, t, budget=B, top_k=B, max_candidates=80)
        exp = o.baseline_expert_heuristic_ranking(e, l, t, gn, top_k=B)
        rows.append({
            "indeg": int((e[1] == t).sum().item()),
            "Expert": o.score_edge_set(e, l, exp, t, "distrust"),
            "Fwd": o.score_edge_set(e, l, fwd, t, "distrust"),
            "Grad": o.score_edge_set(e, l, grad, t, "distrust"),
            "PRBCD": o.score_edge_set(e, l, prb, t, "distrust"),
        })
    return rows


def fig_boxplot_and_sparsity():
    rows = {ds: _collect(ds) for ds in ["otc", "alpha"]}
    methods = ["Expert", "Fwd", "PRBCD", "Grad"]
    data = [[r[m] for ds in rows for r in rows[ds]] for m in methods]
    plt.figure(figsize=(5, 3.2))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(methods) + 1), methods)
    plt.axhline(0, color="k", lw=0.6, ls="--")
    plt.ylabel(r"$\Delta R$ ($B{=}5$)")
    plt.title("Reputation shift by method (OTC+Alpha)")
    plt.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGS / "boxplot_b5.pdf")
    plt.close()
    log("  wrote boxplot_b5.pdf")

    a = rows["alpha"]
    plt.figure(figsize=(5, 3.2))
    plt.scatter([r["indeg"] for r in a], [r["Fwd"] for r in a], s=20, alpha=.7)
    plt.axhline(0, color="k", lw=0.6, ls="--")
    plt.xlabel("Target in-degree")
    plt.ylabel(r"$\Delta R$ (Grail-Fwd, $B{=}5$)")
    plt.title("Target in-degree vs. reputation shift (Alpha)")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIGS / "ablation_sparsity.pdf")
    plt.close()
    log("  wrote ablation_sparsity.pdf")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_scaling()
    fig_boxplot_and_sparsity()
    log("FIGS DONE")
