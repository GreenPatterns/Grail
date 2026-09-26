"""Check the Reviewer-2 numbers in the paper's LaTeX source against the result JSONs.

Each check recomputes a value from its JSON, formats it the way the paper prints it, and asserts
that the formatted sentence fragment occurs in the paper. A failure means the prose and the data
disagree (or the prose was reworded, in which case update the fragment here, never delete the
check). The paper source is not part of this artifact; pass its path in GRAIL_PAPER_TEX.
Run: GRAIL_PAPER_TEX=<paper>.tex python3 code/check_r2_numbers.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results', 'unified')
TEX_PATH = os.environ.get('GRAIL_PAPER_TEX', '')
if not os.path.isfile(TEX_PATH):
    sys.exit('set GRAIL_PAPER_TEX to the LaTeX source of the paper (not part of this artifact)')
TEX = open(TEX_PATH).read()


def J(name):
    return json.load(open(os.path.join(R, name)))


def pct(x, nd=0):
    return f"{x:.{nd}f}"


def main():
    C, CS = J('cohort_summary.json'), J('common_set_summary.json')
    ST, MK = J('stats_hier_s10.json'), J('masked_control_s10.json')
    PM, CR = J('poison_matrix_s5.json'), J('causal_retrain_s5.json')
    o, a = C['otc'], C['alpha']
    checks = [
        # complete cohort
        (f"One optimized edge flips ${pct(o['near_gate']['opt_prop']['mean'])}\\%$ of them on OTC and ${pct(a['near_gate']['opt_prop']['mean'])}\\%$ on Alpha propagation-only, and a random edge, which needs no model access, flips ${pct(o['near_gate']['random_prop']['mean'])}\\%$ and ${pct(a['near_gate']['random_prop']['mean'])}\\%$.", 'cohort low-reputation rates'),
        (f"one optimized distrust edge pushes ${pct(o['near_gate']['opt_prop']['mean'])}\\%$ (Bitcoin-OTC) and ${pct(a['near_gate']['opt_prop']['mean'])}\\%$ (Bitcoin-Alpha) of low-reputation targets below the trust gate, propagation-only", 'abstract rates'),
        (f"The naive score, at ${pct(o['near_gate']['opt_total']['mean'])}\\%$ and ${pct(a['near_gate']['opt_total']['mean'])}\\%$, is higher in every model", 'intro naive rates'),
        (f"one optimized edge still flips ${pct(o['cells']['opt_prop']['mean'])}\\%$ (OTC) and ${pct(a['cells']['opt_prop']['mean'])}\\%$ (Alpha) of targets propagation-only.", 'whole cohort'),
        (f"propagation-only reputation of ${100 * o['shift']['all']['opt']['frac_negative']:.1f}\\%$ (OTC) and ${100 * a['shift']['all']['opt']['frac_negative']:.1f}\\%$ (Alpha) of eligible targets, by a median of ${-o['shift']['all']['opt']['median']:.2f}$ and ${-a['shift']['all']['opt']['median']:.2f}$", 'shift share and median'),
        (f"Targets within $0.05$ of the gate flip at ${pct(o['cells']['opt_prop']['by_margin']['<.05']['mean'])}\\%$ (OTC) and ${pct(a['cells']['opt_prop']['by_margin']['<.05']['mean'])}\\%$ (Alpha), and even targets with clean $R\\ge0.8$ flip at ${pct(o['cells']['opt_prop']['by_margin']['>=.3']['mean'])}\\%$ and ${pct(a['cells']['opt_prop']['by_margin']['>=.3']['mean'])}\\%$.", 'margin extremes'),
        (f"One optimized edge flips ${pct(o['by_indegree']['2-3']['opt_prop']['mean'])}\\%$ of OTC targets with two or three raters, which make up ${pct(100 * o['share_indegree_le3'])}\\%$ of the cohort, against ${pct(o['by_indegree']['>=31']['opt_prop']['mean'])}\\%$", 'raters'),
        (f"Target-model pairs (OTC / Alpha / Epn-d): ${o['near_gate']['n_pooled']}$ / ${a['near_gate']['n_pooled']}$ / ${C['edg60000']['near_gate']['n_pooled']:,}$ low-reputation".replace(',', '{,}'), 'cohort table n'),
        # statistics
        (f"sign-test $p={ST['otc']['appended_vs_zero']['seed_sign_test_p']:.2f}$ and ${ST['alpha']['appended_vs_zero']['seed_sign_test_p']:.2f}$", 'appended null p'),
        (f"$\\delta={ST['otc']['prop_vs_zero']['cliffs_delta']:.2f}$ OTC, ${ST['alpha']['prop_vs_zero']['cliffs_delta']:.2f}$ Alpha", 'prop delta'),
        # masked control
        (f"from ${MK['otc']['B_query_edge_masked']['flip_preexist_pct']:.1f}\\%$ to ${MK['otc']['B_query_edge_masked']['flip_masked_pct']:.1f}\\%$ on OTC", 'masked OTC'),
        (f"(${MK['otc']['B_query_edge_masked']['n_pooled']}$ eligible target-model pairs on OTC, ${MK['alpha']['B_query_edge_masked']['n_pooled']}$ on Alpha)", 'masked n'),
        # common set
        (f"($1{{,}}{CS['otc']['n_common'] - 1000:03d}$ on OTC, $1{{,}}{CS['alpha']['n_common'] - 1000:03d}$ on Alpha)", 'common set n'),
        (f"whereas GDTE flips ${pct(CS['otc']['per_function']['gdte_mean_total']['B1']['worst_pct'])}\\%$ (OTC) and ${pct(CS['alpha']['per_function']['gdte_mean_total']['B1']['worst_pct'])}\\%$ (Alpha) naively and ${pct(CS['otc']['per_function']['gdte_mean_prop']['B1']['worst_pct'])}\\%$ and ${pct(CS['alpha']['per_function']['gdte_mean_prop']['B1']['worst_pct'])}\\%$ propagation-only.", 'common set GDTE B1'),
        (f"The Wilson bound flips more, ${pct(CS['otc']['per_function']['wilson']['B1']['worst_pct'])}\\%$ and ${pct(CS['alpha']['per_function']['wilson']['B1']['worst_pct'])}\\%$", 'common set Wilson B1'),
        (f"(Wilson ${CS['otc']['common_by_raters']['>10']['wilson|B1']['pct']:.1f}\\%$ and ${CS['alpha']['common_by_raters']['>10']['wilson|B1']['pct']:.1f}\\%$)", 'Wilson >10'),
        (f"while GDTE still flips ${pct(CS['otc']['common_by_raters']['>10']['gdte_mean_total|B1']['pct'])}\\%$ and ${pct(CS['alpha']['common_by_raters']['>10']['gdte_mean_total|B1']['pct'])}\\%$ naively and ${pct(CS['otc']['common_by_raters']['>10']['gdte_mean_prop|B1']['pct'])}\\%$ and ${pct(CS['alpha']['common_by_raters']['>10']['gdte_mean_prop|B1']['pct'])}\\%$ propagation-only", 'GDTE >10'),
        (f"the Wilson bound flips ${pct(CS['otc']['per_function']['wilson']['B5']['worst_pct'])}\\%$ of both common sets, against GDTE's naive ${pct(CS['otc']['per_function']['gdte_mean_total']['B5']['worst_pct'])}\\%$ and ${pct(CS['alpha']['per_function']['gdte_mean_total']['B5']['worst_pct'])}\\%$", 'B5'),
        # retraining
        (f"lower victim reputation by ${-PM['otc']['matrix']['controls']['dis_victim']['dR_vs_none']:.2f}$ (OTC) and ${-PM['alpha']['matrix']['controls']['dis_victim']['dR_vs_none']:.2f}$ (Alpha)", 'poison dR'),
        (f"(standard deviation ${PM['otc']['matrix']['retrain_std_clean']:.3f}$ OTC, ${PM['alpha']['matrix']['retrain_std_clean']:.3f}$ Alpha)", 'poison std'),
        (f"flip ${CR['otc']['causal_retrain']['flip_pct']:.0f}\\%$ of the eligible OTC target-model pairs (${CR['otc']['causal_retrain']['k_flip']}/{CR['otc']['causal_retrain']['n_eligible']}$", 'causal OTC'),
        (f"and ${CR['alpha']['causal_retrain']['flip_pct'] // 1:.0f}\\%$ on Alpha (${CR['alpha']['causal_retrain']['k_flip']}/{CR['alpha']['causal_retrain']['n_eligible']}$", 'causal Alpha'),
    ]
    # the complete dated Epinions (edg60000)
    LF, LI = J('loo_flip_edgfull_low.json')['edg60000'], J('p0_linchpin_edgfull.json')['edg60000']
    P1 = {k: J(f)[k] for f, k in (('p1_sota.json', 'otc'), ('p1_sota.json', 'alpha'), ('p1_sota_epn.json', 'epn30000'),
                                    ('p1_sota_edgfull.json', 'edg60000'))}
    share = lambda m: [100 * P1[d]['mean_dr'][m] / P1[d]['mean_dr']['counterfactual'] for d in P1]
    adapted = share('prbcd') + share('influence') + share('node_injection')
    t = [round(LF[f'low_B{b}']['flip_total_pct']) for b in (1, 3, 5)]
    q = [round(LF[f'low_B{b}']['flip_preexist_pct']) for b in (1, 3, 5)]
    e = C['edg60000']
    eg, ec = e['near_gate'], e['cells']
    pm = lambda c: f"${c['mean']:.0f}{{\\pm}}{c['std']:.0f}$"
    thousands = lambda n: f"{n:,}".replace(',', '{,}')
    checks += [
        (f"\\multirow{{2}}{{*}}{{Epn-d}} & Random & {pm(eg['random_total'])} & {pm(eg['random_prop'])} & {pm(ec['random_total'])} & {pm(ec['random_prop'])}\\\\", 'Table 2 dated random row'),
        (f"  & Optimized & {pm(eg['opt_total'])} & {pm(eg['opt_prop'])} & {pm(ec['opt_total'])} & {pm(ec['opt_prop'])}\\\\", 'Table 2 dated optimized row'),
        (f"$27{{,}}432$ / $19{{,}}837$ / ${thousands(sum(e['n_eligible'].values()))}$ in all", 'Table 2 caption n (all)'),
        (f"one optimized edge flips fewer targets propagation-only, ${pct(eg['opt_prop']['mean'])}\\%$ of low-reputation and ${pct(ec['opt_prop']['mean'])}\\%$ of all targets, and the naive rate is again higher in every model.", 'Sec 6 dated cohort'),
        (f"On the dated Epinions it lowers ${100 * e['shift']['all']['opt']['frac_negative']:.0f}\\%$ of targets, by a median of only ${-e['shift']['all']['opt']['median']:.2f}$. On all three graphs the median shift is negative in every model", 'Sec 6 dated shift'),
        (f"yet one optimized edge still flips ${pct(eg['opt_prop']['mean'])}\\%$ of low-reputation targets propagation-only (naive ${pct(eg['opt_total']['mean'])}\\%$).", 'intro dated'),
    ]
    naive_every = eg['total_above_prop_opt']['k_positive'] == eg['total_above_prop_opt']['n_nonzero'] == e['n_seeds']
    median_every = e['opt_prop_median_shift']['k_negative'] == e['n_seeds']
    if not (naive_every and median_every):
        print(f"FAIL: the dated cohort does not support 'every model' (naive {naive_every}, median {median_every})")
        checks.append(('<every-model claim unsupported>', 'dated every-model'))
    checks += [
        (f"Epn-d & Low & ${t[0]}$ & ${t[1]}$ & ${t[2]}$ & ${q[0]}$ & ${q[1]}$ & ${q[2]}$\\\\", 'Table 3 dated row'),
        (f"& $0.{round(1000 * LI['clean_perf']['AUC'])}$ & $0.{round(1000 * LI['clean_perf']['MCC'])}$", 'Table 1 dated AUC/MCC'),
        (f"$\\dr={LI['processed_mean_dr']:.2f}$ (label-$\\Delta={LI['processed_label_delta']:.2f}$), against ${LI['appended_mean_dr']:+.3f}$ when appended", 'Sec 3 dated'),
        (f"(${min(share('expert')):.0f}$ to ${max(share('expert')):.0f}\\%$ of the counterfactual's $\\dr$)", 'Expert range'),
        (f"The adapted attacks reach ${min(adapted):.0f}$ to ${max(adapted):.0f}\\%$, and Random reaches ${min(share('random')):.0f}$ to ${max(share('random')):.0f}\\%$.", 'adapted and random ranges'),
    ]
    bad = [(frag, name) for frag, name in checks if frag not in TEX]
    for frag, name in bad:
        print(f"FAIL [{name}]: {frag[:160]}")
    for marker in ('STATSPARAGRAPH', 'PENDINGOTC', 'PENDINGBOUND', 'OFFICIALROWS'):
        if marker in TEX:
            print(f"FAIL: placeholder {marker} still in the paper")
            bad.append((marker, 'placeholder'))
    print(f"{len(checks) - len([b for b in bad if b[1] != 'placeholder'])}/{len(checks)} number checks pass")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
