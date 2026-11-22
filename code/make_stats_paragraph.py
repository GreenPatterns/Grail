"""Print the paired statistics of experiment E2 (Section VI-A) and the rows of Table S6 from
results/unified/stats_hier_s10.json (code/experiments/run_stats_hier.py over ten runs), so that
every number in them can be regenerated. The paragraph is an earlier wording of Section VI-A; its
numbers are those the paper cites. JSON keys keep the code's names: `appended` is
out-of-window and `naive` is the total score.

Run: python3 code/make_stats_paragraph.py      (prints the paragraph and the table rows)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'results', 'unified', 'stats_hier_s10.json')
ROWS = [('appended_vs_inwindow', 'In-window vs.\\ out-of-window'), ('distrust_vs_trust', 'Distrust vs.\\ trust'),
        ('naive_vs_prop', 'Total vs.\\ prop.-only'), ('prop_vs_zero', 'Prop.-only vs.\\ $0$'),
        ('appended_vs_zero', 'Out-of-window vs.\\ $0$')]


def main():
    S = json.load(open(PATH))
    o, a = S['otc'], S['alpha']
    prim = ['appended_vs_inwindow', 'distrust_vs_trust', 'naive_vs_prop']
    deltas = [S[ds][c]['cliffs_delta'] for ds in ('otc', 'alpha') for c in prim]
    signp = max(S[ds][c]['seed_sign_test_p'] for ds in ('otc', 'alpha') for c in prim + ['prop_vs_zero'])
    n = o['prop_vs_zero']['n_pooled']
    para = (
        "We treat the run, not the target, as the unit of replication~\\citep{bouthillier2021accounting}: "
        "the targets of one run share one network and one model, so we report no target-level $p$-values. Effect size is "
        "Cliff's $\\delta$~\\citep{cliff1993dominance} of the paired differences $d_i$ against zero, "
        "$\\delta=(\\#\\{d_i>0\\}-\\#\\{d_i<0\\})/n\\in[-1,1]$, where $d_i$ is the first-named condition minus the "
        "second for target-run pair $i$. "
        f"Over ten runs ($n={n}$ pairs per dataset at $B\\!=\\!1$, the strata of each run), the three primary "
        "contrasts, in-window vs.\\ out-of-window, distrust vs.\\ trust, and total vs.\\ propagation-only, show large rank "
        f"dominance ($\\delta$ from ${min(deltas):.2f}$ to ${max(deltas):.2f}$), and the per-run medians agree in "
        f"sign in every run (sign test $p\\le{signp:.3f}$). The propagation-only shift is itself negative with a "
        f"large effect ($\\delta={o['prop_vs_zero']['cliffs_delta']:.2f}$ OTC, ${a['prop_vs_zero']['cliffs_delta']:.2f}$ "
        "Alpha; seed-clustered $95\\%$ interval of the median "
        f"$[{o['prop_vs_zero']['seedcluster_ci_median'][0]:.3f},{o['prop_vs_zero']['seedcluster_ci_median'][1]:.3f}]$ and "
        f"$[{a['prop_vs_zero']['seedcluster_ci_median'][0]:.3f},{a['prop_vs_zero']['seedcluster_ci_median'][1]:.3f}]$), "
        "while the out-of-window shift is a null: its median is "
        f"${o['appended_vs_zero']['median']:+.4f}$ (OTC) and ${a['appended_vs_zero']['median']:+.4f}$ (Alpha), "
        f"with $\\delta={o['appended_vs_zero']['cliffs_delta']:.2f}$ and ${a['appended_vs_zero']['cliffs_delta']:.2f}$ "
        f"and sign-test $p={o['appended_vs_zero']['seed_sign_test_p']:.2f}$ and "
        f"${a['appended_vs_zero']['seed_sign_test_p']:.2f}$ (Table~\\ref{{S-tab:s-stats}}).")
    print(para)
    print()
    print('% --- Table S6 rows: contrast & OTC med & delta & run-clustered CI & sign-test p & Alpha med & delta & CI & p')
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
