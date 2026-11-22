"""Write the main paper's replication/effect-size paragraph and the supplement's paired-contrast
table from results/unified/stats_hier_s10.json (code/experiments/run_stats_hier.py with ten
independently trained models), so that every number in them can be regenerated.

Run: python3 code/make_stats_paragraph.py      (prints the paragraph and the table rows)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'results', 'unified', 'stats_hier_s10.json')
ROWS = [('appended_vs_inwindow', 'In-win vs.\\ append'), ('distrust_vs_trust', 'Distrust vs.\\ trust'),
        ('naive_vs_prop', 'Naive vs.\\ prop.-only'), ('prop_vs_zero', 'Prop.-only vs.\\ $0$'),
        ('appended_vs_zero', 'Append vs.\\ $0$ (null)')]


def main():
    S = json.load(open(PATH))
    o, a = S['otc'], S['alpha']
    prim = ['appended_vs_inwindow', 'distrust_vs_trust', 'naive_vs_prop']
    deltas = [S[ds][c]['cliffs_delta'] for ds in ('otc', 'alpha') for c in prim]
    signp = max(S[ds][c]['seed_sign_test_p'] for ds in ('otc', 'alpha') for c in prim + ['prop_vs_zero'])
    n = o['prop_vs_zero']['n_pooled']
    para = (
        "We treat the trained model, not the target, as the unit of replication~\\citep{bouthillier2021accounting}: "
        "node-level observations share a graph and a model, so we report no node-level $p$-values. Effect size is "
        "Cliff's $\\delta$~\\citep{cliff1993dominance} of the paired differences $d_i$ against zero, "
        "$\\delta=(\\#\\{d_i>0\\}-\\#\\{d_i<0\\})/n\\in[-1,1]$, where $d_i$ is the first-named condition minus the "
        "second for target-model pair $i$. "
        f"Over ten models ($n={n}$ pairs per dataset at $B\\!=\\!1$, the strata of each model), the three primary "
        "contrasts, in-window vs.\\ appended, distrust vs.\\ trust, and naive vs.\\ propagation-only, show large rank "
        f"dominance ($\\delta$ from ${min(deltas):.2f}$ to ${max(deltas):.2f}$), and the per-model medians agree in "
        f"sign in every model (sign test $p\\le{signp:.3f}$). The propagation-only shift is itself negative with a "
        f"large effect ($\\delta={o['prop_vs_zero']['cliffs_delta']:.2f}$ OTC, ${a['prop_vs_zero']['cliffs_delta']:.2f}$ "
        "Alpha; seed-clustered $95\\%$ interval of the median "
        f"$[{o['prop_vs_zero']['seedcluster_ci_median'][0]:.3f},{o['prop_vs_zero']['seedcluster_ci_median'][1]:.3f}]$ and "
        f"$[{a['prop_vs_zero']['seedcluster_ci_median'][0]:.3f},{a['prop_vs_zero']['seedcluster_ci_median'][1]:.3f}]$), "
        "while the appended shift is a null: its median is "
        f"${o['appended_vs_zero']['median']:+.4f}$ (OTC) and ${a['appended_vs_zero']['median']:+.4f}$ (Alpha), "
        f"with $\\delta={o['appended_vs_zero']['cliffs_delta']:.2f}$ and ${a['appended_vs_zero']['cliffs_delta']:.2f}$ "
        f"and sign-test $p={o['appended_vs_zero']['seed_sign_test_p']:.2f}$ and "
        f"${a['appended_vs_zero']['seed_sign_test_p']:.2f}$ (supplement).")
    print(para)
    print()
    print('% --- supplement table rows: contrast & OTC med & OTC delta & OTC seed CI & Alpha med & Alpha delta & Alpha seed CI')
    for key, lab in ROWS:
        cells = []
        for ds in ('otc', 'alpha'):
            c = S[ds][key]
            lo, hi = c['seedcluster_ci_median']
            cells += [f"${c['median']:+.3f}$", f"${c['cliffs_delta']:+.2f}$", f"$[{lo:+.3f},{hi:+.3f}]$",
                      f"${c['seed_sign_test_p']:.3f}$"]
        print(f"{lab} & " + ' & '.join(cells) + '\\\\')


if __name__ == '__main__':
    main()
