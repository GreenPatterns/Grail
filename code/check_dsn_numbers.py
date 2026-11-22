"""Check the numbers in the DSN paper's LaTeX source against the result JSONs.

Each check recomputes a value from its JSON, formats it the way the paper prints it, and asserts
that the formatted sentence fragment occurs in the paper (whitespace is normalized on both sides).
A failure means the prose and the data disagree, or the prose was reworded. In the second case
update the fragment here; never delete the check.

Run: GRAIL_PAPER_TEX=<path>/grail_dsn27.tex python3 code/check_dsn_numbers.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results', 'unified')
DATASETS = ('otc', 'alpha', 'edg60000')


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as handle:
        return json.load(handle)


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


def pct(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}"


def thousands(number: int) -> str:
    return f"{number:,}".replace(',', '{,}')


def pm(cell: dict) -> str:
    return f"${cell['mean']:.0f}{{\\pm}}{cell['std']:.0f}$"


def cohort_checks(C: dict) -> list:
    o, a, e = (C[d] for d in DATASETS)
    low = {d: C[d]['near_gate'] for d in DATASETS}
    cell = lambda d, k: pct(C[d]['cells'][k]['mean'])
    lo = lambda d, k: pct(low[d][k]['mean'])
    rows = []
    for d, name in zip(DATASETS, ('OTC', 'Alpha', 'Epinions-D')):
        g, c = C[d]['near_gate'], C[d]['cells']
        rows += [
            (f"\\multirow{{2}}{{*}}{{{name}}} & Random & {pm(g['random_total'])} & {pm(g['random_prop'])} & {pm(c['random_total'])} & {pm(c['random_prop'])}\\\\", f'cohort table, {name} random row'),
            (f"& Optimized & {pm(g['opt_total'])} & {pm(g['opt_prop'])} & {pm(c['opt_total'])} & {pm(c['opt_prop'])}\\\\", f'cohort table, {name} optimized row'),
        ]
    n_all = {d: sum(C[d]['n_eligible'].values()) for d in DATASETS}
    share = [100 * low[d]['n_pooled'] / n_all[d] for d in ('otc', 'alpha')]
    assert all(round(x) == 2 for x in share), f"low-reputation share is not about 2%: {share}"
    spread = {d: (min(C[d]['cells']['opt_prop']['per_seed'].values()), max(C[d]['cells']['opt_prop']['per_seed'].values())) for d in ('otc', 'alpha')}
    em = e['shift']['by_margin']
    low_med = sorted(-em[b]['opt']['median'] for b in ('<.05', '.05-.1'))
    ef = e['cells']['opt_prop']['by_margin']
    return rows + [
        (f"Under the propagation-only score, one random edge flips ${lo('otc', 'random_prop')}\\%$ of them on OTC and ${lo('alpha', 'random_prop')}\\%$ on Alpha, and one optimized edge flips ${lo('otc', 'opt_prop')}\\%$ and ${lo('alpha', 'opt_prop')}\\%$.", 'cohort low-reputation rates'),
        (f"One fake rating from a random account pushes ${lo('otc', 'random_prop')}\\%$ and ${lo('alpha', 'random_prop')}\\%$ of users just above the threshold below it, and ${cell('otc', 'random_prop')}\\%$ and ${cell('alpha', 'random_prop')}\\%$ of all eligible users.", 'abstract rates'),
        (f"For the optimized edges on low-reputation targets it reports ${lo('otc', 'opt_total')}\\%$ and ${lo('alpha', 'opt_total')}\\%$ instead of ${lo('otc', 'opt_prop')}\\%$ and ${lo('alpha', 'opt_prop')}\\%$", 'intro total rates'),
        (f"pushes below the threshold ${lo('otc', 'random_prop')}\\%$ (OTC) and ${lo('alpha', 'random_prop')}\\%$ (Alpha) of the \\emph{{low-reputation}} targets", 'intro random rates'),
        (f"who make up about $2\\%$ of the eligible targets, and ${cell('otc', 'random_prop')}\\%$ and ${cell('alpha', 'random_prop')}\\%$ of all eligible targets. An \\emph{{optimized}} source", 'intro whole cohort, random'),
        (f"reaches ${lo('otc', 'opt_prop')}\\%$ and ${lo('alpha', 'opt_prop')}\\%$, and ${cell('otc', 'opt_prop')}\\%$ and ${cell('alpha', 'opt_prop')}\\%$ (Section", 'intro optimized'),
        (f"one random edge flips ${cell('otc', 'random_prop')}\\%$ (OTC) and ${cell('alpha', 'random_prop')}\\%$ (Alpha) under the propagation-only score, and one optimized edge ${cell('otc', 'opt_prop')}\\%$ and ${cell('alpha', 'opt_prop')}\\%$.", 'whole cohort'),
        (f"the optimized rate ranges from ${spread['otc'][0]:.0f}\\%$ to ${spread['otc'][1]:.0f}\\%$ on OTC and from ${spread['alpha'][0]:.0f}\\%$ to ${spread['alpha'][1]:.0f}\\%$ on Alpha", 'spread across runs'),
        (f"${low['otc']['n_pooled']}$ of ${thousands(n_all['otc'])}$ target-run pairs on OTC and ${low['alpha']['n_pooled']}$ of ${thousands(n_all['alpha'])}$ on Alpha, about $2\\%$", 'low-reputation share'),
        (f"it reports ${lo('otc', 'opt_total')}\\%$ instead of ${lo('otc', 'opt_prop')}\\%$ on OTC, ${lo('alpha', 'opt_total')}\\%$ instead of ${lo('alpha', 'opt_prop')}\\%$ on Alpha, and ${lo('edg60000', 'opt_total')}\\%$ instead of ${lo('edg60000', 'opt_prop')}\\%$ on Epinions-D", 'gap between the scores'),
        (f"propagation-only reputation of ${100 * o['shift']['all']['opt']['frac_negative']:.1f}\\%$ (OTC) and ${100 * a['shift']['all']['opt']['frac_negative']:.1f}\\%$ (Alpha) of eligible targets, by a median of ${-o['shift']['all']['opt']['median']:.2f}$ and ${-a['shift']['all']['opt']['median']:.2f}$", 'shift share and median'),
        (f"On Epinions-D the edge lowers ${100 * e['shift']['all']['opt']['frac_negative']:.0f}\\%$ of targets, by a median of only ${-e['shift']['all']['opt']['median']:.2f}$ (${low_med[0]:.2f}$ to ${low_med[1]:.2f}$ for low-reputation targets); there, ${pct(ef['<.05']['mean'])}\\%$ of the targets within $0.05$ of the threshold flip, but only ${pct(ef['.05-.1']['mean'])}\\%$ of those between $0.05$ and $0.1$. On all three graphs the median shift is negative in every run.", 'Epinions-D shift'),
        (f"targets within $0.05$ of the threshold flip at ${pct(o['cells']['opt_prop']['by_margin']['<.05']['mean'])}\\%$ (OTC) and ${pct(a['cells']['opt_prop']['by_margin']['<.05']['mean'])}\\%$ (Alpha), and even targets with clean $R\\ge0.8$ flip at ${pct(o['cells']['opt_prop']['by_margin']['>=.3']['mean'])}\\%$ and ${pct(a['cells']['opt_prop']['by_margin']['>=.3']['mean'])}\\%$.", 'margin extremes'),
        (f"one optimized edge flips ${pct(o['by_indegree']['2-3']['opt_prop']['mean'])}\\%$ of OTC targets with two or three raters, which make up ${pct(100 * o['share_indegree_le3'])}\\%$ of the cohort, against ${pct(o['by_indegree']['>=31']['opt_prop']['mean'])}\\%$", 'raters'),
        (f"${pct(100 * o['share_indegree_le3'])}\\%$ of the OTC cohort has two or three", 'setup, share of few-rater targets'),
        (f"about $2\\%$ on each (on Epinions-D, ${thousands(low['edg60000']['n_pooled'])}$ of ${thousands(n_all['edg60000'])}$, about ${100 * low['edg60000']['n_pooled'] / n_all['edg60000']:.0f}\\%$)", 'Epinions-D pair counts'),
        (f"forward-chaining AUC of ${min(min(C[d]['auc'].values()) for d in ('otc', 'alpha')):.2f}$ to ${max(max(C[d]['auc'].values()) for d in ('otc', 'alpha')):.2f}$ across runs", 'AUC range across runs'),
        (f"one optimized edge flips ${lo('edg60000', 'opt_prop')}\\%$ of low-reputation and ${cell('edg60000', 'opt_prop')}\\%$ of all targets under the propagation-only score.", 'Epinions-D cohort'),
        (f"one in-window distrust edge from a random account flips ${lo('otc', 'random_prop')}\\%$ (OTC) and ${lo('alpha', 'random_prop')}\\%$ (Alpha) of low-reputation targets and ${cell('otc', 'random_prop')}\\%$ and ${cell('alpha', 'random_prop')}\\%$ of all eligible targets.", 'conclusion rates'),
    ]


def every_model_failures(C: dict) -> list:
    """Claims of the form 'in every run' need all ten runs to agree."""
    bad = []
    for d in DATASETS:
        gap = C[d]['near_gate']['total_above_prop_opt']
        if not gap['k_positive'] == gap['n_nonzero'] == C[d]['n_seeds']:
            bad.append((f'<total rate above propagation-only in every run: {d}>', 'every-model claim'))
        if C[d]['opt_prop_median_shift']['k_negative'] != C[d]['n_seeds']:
            bad.append((f'<median shift negative in every run: {d}>', 'every-model claim'))
    return bad


def placement_checks() -> list:
    lin = {'otc': load('p0_linchpin.json')['otc'], 'alpha': load('p0_linchpin.json')['alpha'],
           'edg60000': load('p0_linchpin_edgfull.json')['edg60000']}
    sweep = load('p0b_main.json')
    inw = [v['processed_mean_dr'] for v in lin.values()]
    assert max(v['appended_mean_dr'] for v in lin.values()) < 0.0105, "'about +0.01 or less' not supported"
    so, sa, ss = sweep['otc'], sweep['alpha'], sweep['epn30000']
    lam = [so['sweep_label_delta'][f's{i}'] for i in range(7)]
    drs = [so['sweep_dr'][f's{i}'] for i in range(7)]
    assert max(lam) == lam[6] and min(drs) == drs[6], 'OTC sweep is not largest at s6'
    return [
        (f"whereas the same edge inserted in-window shifts it by ${max(inw):.2f}$ to ${min(inw):.2f}$", 'intro in-window range'),
        (f"Out-of-window, the mean shift is ${lin['otc']['appended_mean_dr']:+.3f}$ on OTC, ${lin['alpha']['appended_mean_dr']:+.3f}$ on Alpha, and ${lin['edg60000']['appended_mean_dr']:+.3f}$ on Epinions-D", 'out-of-window shifts'),
        (f"In-window, the same edges shift reputation by ${lin['otc']['processed_mean_dr']:.2f}$, ${lin['alpha']['processed_mean_dr']:.2f}$, and ${lin['edg60000']['processed_mean_dr']:.2f}$, with $\\ls={lin['otc']['processed_label_delta']:.2f}$, ${lin['alpha']['processed_label_delta']:.2f}$, and ${lin['edg60000']['processed_label_delta']:.2f}$.", 'in-window shifts'),
        (f"on OTC, $\\ls$ lies between ${min(lam):.2f}$ and ${max(lam):.2f}$ and the shift between ${max(drs):.2f}$ and ${min(drs):.2f}$", 'OTC sweep range'),
        (f"the shift is ${ss['sweep_dr']['s6']:.2f}$ at $s_6$ and ${ss['sweep_dr']['appended']:.3f}$ out-of-window", 'Epinions-S sweep'),
        (f"a trust edge shifts reputation by ${so['sign_trust']:+.2f}$, a neutral edge by ${so['sign_neutral']:+.2f}$, and a distrust edge by ${so['sign_distrust']:+.2f}$ on OTC; on Alpha the three shifts are ${sa['sign_trust']:+.2f}$, ${sa['sign_neutral']:+.2f}$, and ${sa['sign_distrust']:+.2f}$", 'sign control'),
    ]


def statistics_checks(ST: dict) -> list:
    o, a = ST['otc'], ST['alpha']
    deltas = [ST[d][k]['cliffs_delta'] for d in ('otc', 'alpha') for k in ('appended_vs_inwindow', 'distrust_vs_trust')]
    return [
        (f"by a median of ${-o['appended_vs_inwindow']['median']:.2f}$ (OTC) and ${-a['appended_vs_inwindow']['median']:.2f}$ (Alpha), and the distrust label shifts reputation further down than the trust label by a median of ${-o['distrust_vs_trust']['median']:.2f}$ and ${-a['distrust_vs_trust']['median']:.2f}$, in every run", 'paired medians'),
        (f"(Cliff's $\\delta$ between ${min(deltas):.2f}$ and ${max(deltas):.2f}$;", 'Cliff delta range'),
        (f"sign-test $p={o['appended_vs_zero']['seed_sign_test_p']:.2f}$ and ${a['appended_vs_zero']['seed_sign_test_p']:.2f}$", 'out-of-window null p'),
        (f"(${o['appended_vs_inwindow']['n_pooled']}$ target-run pairs per dataset)", 'paired n'),
    ]


def strata_checks() -> list:
    flips = load('loo_flip_table.json')
    dated = load('loo_flip_edgfull_low.json')['edg60000']
    cell = lambda r, s, b, k: round(r[f'{s}_B{b}'][f'flip_{k}_pct'])
    row = lambda r, s: ' & '.join(f"${cell(r, s, b, k)}$" for k in ('total', 'preexist') for b in (1, 3, 5))
    checks = [(f"& {name} & {row(flips[d], s)}\\\\", f'strata table, {d} {s}')
              for d in ('otc', 'alpha') for s, name in (('high', 'High'), ('moderate', 'Mod.'), ('low', 'Low'))]
    o, a = flips['otc'], flips['alpha']
    reps = load('reputation_baselines_otc.json')['otc']
    plain_all, decayed_all = load('p0d_otcalpha.json'), load('p0d_decayed.json')
    plain = plain_all['otc']
    td = lambda d, cell_key, src: f"{100 * src[d][cell_key]['flip_rate_optimized']:.0f}"
    return checks + [
        (f"Epinions-D & Low & {row(dated, 'low')}\\\\", 'strata table, Epinions-D'),
        (f"one optimized edge flips ${cell(o, 'low', 1, 'preexist')}\\%$ (OTC) and ${cell(a, 'low', 1, 'preexist')}\\%$ (Alpha) of the targets under the propagation-only score", 'low stratum B=1'),
        (f"At $B=5$ the total rate reaches ${cell(o, 'low', 5, 'total')}\\%$ and ${cell(a, 'low', 5, 'total')}\\%$, whereas the propagation-only rate falls to ${cell(o, 'low', 5, 'preexist')}\\%$ and ${cell(a, 'low', 5, 'preexist')}\\%$", 'low stratum B=5'),
        (f"is ${cell(o, 'low', 1, 'total')}\\%$ in E4 (Table~\\ref{{tab:flip}}), ${reps['low|mean_opt|gdte_mean_total|B1']['flip_pct']:.0f}\\%$ in the low-stratum experiment of the supplement (Table~\\ref{{S-tab:s-repbase}}), and ${100 * plain['low_B1']['flip_rate_optimized']:.0f}\\%$ in E12", 'single-run variation'),
        (f"the mean propagation-only shift of the low stratum shrinks as well (OTC from ${o['low_B1']['dR_preexist']:.3f}$ to ${o['low_B5']['dR_preexist']:.3f}$, Alpha from ${a['low_B1']['dR_preexist']:.3f}$ to ${a['low_B5']['dR_preexist']:.3f}$)", 'low-stratum shift by budget'),
        (f"raises the total flip rate of one optimized edge on the low stratum from ${td('otc', 'low_B1', plain_all)}\\%$ to ${td('otc', 'low_B1', decayed_all)}\\%$ on OTC and from ${td('alpha', 'low_B1', plain_all)}\\%$ to ${td('alpha', 'low_B1', decayed_all)}\\%$ on Alpha, and that of five edges on the moderate stratum from ${td('otc', 'moderate_B5', plain_all)}\\%$ to ${td('otc', 'moderate_B5', decayed_all)}\\%$ and from ${td('alpha', 'moderate_B5', plain_all)}\\%$ to ${td('alpha', 'moderate_B5', decayed_all)}\\%$", 'time decay'),
    ]


def mechanism_checks(MK: dict, CS: dict) -> list:
    SR = load('stats_rigor.json')
    o, a = SR['otc'], SR['alpha']
    share = lambda r, b, k: f"{r[f'decomp_B{b}'][k]:.0f}"
    rows = []
    for d in ('otc', 'alpha'):
        for b in (1, 5):
            v = SR[d][f'decomp_B{b}']
            rows.append((f"& {b} & ${v['mean_total']:.3f}$ & ${v['mean_mechanical']:+.3f}$ & ${v['mean_self_edge']:.3f}$ (${v['self_share_pct']:.0f}\\%$) & ${v['mean_propagation']:.3f}$ (${v['prop_share_pct']:.0f}\\%$)\\\\", f'decomposition table, {d} B={b}'))
    co, ca = CS['otc'], CS['alpha']
    mo, ma = MK['otc']['B_query_edge_masked'], MK['alpha']['B_query_edge_masked']
    big = lambda r, f: pct(r['common_by_raters']['>10'][f'{f}|B1']['pct'])
    fn = lambda r, f, b: r['per_function'][f][b]['worst_pct']
    others = [fn(r, f, 'B5') for r in (co, ca) for f in ('fg', 'beta', 'fraction')]
    many = lambda r, f: r['by_raters'][f]['>=31']
    ratio = [many(r, 'gdte_mean_prop')['shift_B1_worst_median'] / min(many(r, f)['shift_B1_worst_median'] for f in ('fg', 'wilson', 'beta', 'fraction')) for r in (co, ca)]
    return rows + [
        (f"propagation accounts for a median of ${share(o, 1, 'prop_share_pct')}\\%$ (OTC) and ${share(a, 1, 'prop_share_pct')}\\%$ (Alpha) of a target's total shift, and self-conditioning for ${share(o, 1, 'self_share_pct')}\\%$ and ${share(a, 1, 'self_share_pct')}\\%$", 'decomposition shares B=1'),
        (f"the median self-conditioning share rises to ${share(o, 5, 'self_share_pct')}\\%$ and ${share(a, 5, 'self_share_pct')}\\%$", 'decomposition shares B=5'),
        (f"(${o['decomp_B1']['mean_mechanical']:+.3f}$ and ${a['decomp_B1']['mean_mechanical']:+.3f}$)", 'dilution'),
        (f"The pooled rate falls from ${mo['flip_preexist_pct']:.1f}\\%$ to ${mo['flip_masked_pct']:.1f}\\%$ on OTC (bootstrap $95\\%$ interval over runs $[{mo['flip_masked_seedboot_ci'][0]},{mo['flip_masked_seedboot_ci'][1]}]$) and stays at ${ma['flip_masked_pct']:.1f}\\%$ on Alpha ($[{ma['flip_masked_seedboot_ci'][0]:.1f},{ma['flip_masked_seedboot_ci'][1]}]$); the median masked shift is ${mo['dR_masked_median']:.2f}$ and ${ma['dR_masked_median']:.2f}$.", 'masked control'),
        (f"the masked rate is ${mo['flip_masked_per_seed_pct'][0]:.0f}\\%$ (OTC) and ${ma['flip_masked_per_seed_pct'][0]:.0f}\\%$ (Alpha)", 'masked control, seed 42'),
        (f"which leaves ${mo['n_pooled']}$ eligible target-run pairs on OTC and ${ma['n_pooled']}$ on Alpha", 'masked n'),
        (f"(${thousands(co['n_common'])}$ on OTC, ${thousands(ca['n_common'])}$ on Alpha)", 'common set n'),
        (f"Over all targets of that run with more than $30$ raters, one edge lowers GDTE's propagation-only reputation by a median of ${-many(co, 'gdte_mean_prop')['shift_B1_worst_median']:.3f}$ (OTC) and ${-many(ca, 'gdte_mean_prop')['shift_B1_worst_median']:.3f}$ (Alpha), ${min(ratio):.1f}$ to ${max(ratio):.1f}$ times", 'size of the shift'),
        (f"cuts this drop to ${-many(co, 'gdte_median_prop')['shift_B1_worst_median']:.3f}$ and ${-many(ca, 'gdte_median_prop')['shift_B1_worst_median']:.3f}$, close to the Wilson bound's ${-many(co, 'wilson')['shift_B1_worst_median']:.3f}$ and ${-many(ca, 'wilson')['shift_B1_worst_median']:.3f}$", 'median aggregation, drop'),
        (f"(``$>$10'': ${co['common_by_raters']['>10']['n']}$ and ${ca['common_by_raters']['>10']['n']}$, $B=1$), and median one-edge drop of all ${many(co, 'gdte_mean_prop')['n_targets']}$ and ${many(ca, 'gdte_mean_prop')['n_targets']}$ targets with more than $30$ raters", 'non-GNN table caption counts'),
        (f"whereas GDTE flips ${pct(fn(co, 'gdte_mean_total', 'B1'))}\\%$ (OTC) and ${pct(fn(ca, 'gdte_mean_total', 'B1'))}\\%$ (Alpha) under the total score and ${pct(fn(co, 'gdte_mean_prop', 'B1'))}\\%$ and ${pct(fn(ca, 'gdte_mean_prop', 'B1'))}\\%$ under the propagation-only score.", 'common set GDTE B=1'),
        (f"The Wilson bound flips more, ${pct(fn(co, 'wilson', 'B1'))}\\%$ and ${pct(fn(ca, 'wilson', 'B1'))}\\%$", 'common set Wilson B=1'),
        (f"(Wilson ${co['common_by_raters']['>10']['wilson|B1']['pct']:.1f}\\%$ and ${ca['common_by_raters']['>10']['wilson|B1']['pct']:.1f}\\%$)", 'Wilson, more than ten raters'),
        (f"while GDTE still flips ${big(co, 'gdte_mean_total')}\\%$ and ${big(ca, 'gdte_mean_total')}\\%$ under the total score and ${big(co, 'gdte_mean_prop')}\\%$ and ${big(ca, 'gdte_mean_prop')}\\%$ under the propagation-only score", 'GDTE, more than ten raters'),
        (f"(${pct(fn(co, 'gdte_median_prop', 'B1'))}\\%$ and ${pct(fn(ca, 'gdte_median_prop', 'B1'))}\\%$ under the propagation-only score, against ${pct(fn(co, 'gdte_mean_prop', 'B1'))}\\%$ and ${pct(fn(ca, 'gdte_mean_prop', 'B1'))}\\%$ for the mean)", 'median aggregation'),
        (f"The Wilson bound then flips ${pct(fn(co, 'wilson', 'B5'))}\\%$ of both common sets, more than GDTE under the total score (${pct(fn(co, 'gdte_mean_total', 'B5'))}\\%$ and ${pct(fn(ca, 'gdte_mean_total', 'B5'))}\\%$); FG, the Beta mean, and the trust fraction flip ${min(others):.0f}$ to ${max(others):.0f}\\%$. Under the propagation-only score GDTE flips ${pct(fn(co, 'gdte_mean_prop', 'B5'))}\\%$ and ${pct(fn(ca, 'gdte_mean_prop', 'B5'))}\\%$", 'common set B=5'),
    ]


def deployment_checks() -> list:
    PM, CR, RG = load('poison_matrix_s5.json'), load('causal_retrain_s5.json'), load('causal_regimes.json')
    ctl = lambda d, k: PM[d]['matrix']['controls'][k]

    def rate(v: dict, key: str) -> str:
        # The JSON stores rates rounded to one decimal. Rounding that value again can be off by
        # one (60.47 -> 60.5 -> 61), so recover the count and round the exact rate once.
        count = round(v[key] * v['n_eligible'] / 100)
        return f"{100 * count / v['n_eligible']:.0f}"

    rows = []
    for d in ('otc', 'alpha'):
        for key, name, b in (('inwindow_s6_B1', 'in', 1), ('inwindow_s6_B5', 'in', 5),
                             ('sliding_s7_B1', 'sliding', 1), ('sliding_s7_B5', 'sliding', 5)):
            v = RG[d][key]
            rows.append((f"& {name} & {b} & ${v['n_eligible']}$ & ${rate(v, 'flip_naive_pct')}$ & ${rate(v, 'flip_legit_pct')}$ & ${v['label_delta']:.2f}$\\\\", f'regime table, {d} {key}'))
        v = RG[d]['appended_B1']
        rows.append((f"& out & 1 & ${v['n_eligible']}$ & ${rate(v, 'flip_naive_pct')}$ & ${rate(v, 'flip_legit_pct')}$ &", f'regime table, {d} out'))
    ro, ra = RG['otc'], RG['alpha']
    legit = lambda r, key: rate(r[key], 'flip_legit_pct')
    assert all(round(PM[d]['matrix']['retrain_std_clean'], 2) == 0.05 for d in ('otc', 'alpha')), 'poisoning spread is not about 0.05'

    def retrain(d: str) -> str:
        c = CR[d]['causal_retrain']
        per = [100 * s['k_flip'] / s['n_eligible'] for s in c['per_seed'].values()]
        return f"(${c['k_flip']}/{c['n_eligible']}$ pooled; ${min(per):.0f}$ to ${max(per):.0f}\\%$ per run; mean shift ${c['mean_dR']:.2f}$)"

    return rows + [
        (f"the propagation-only flip rate is ${legit(ro, 'inwindow_s6_B1')}\\%$ in both cases on OTC, and ${legit(ra, 'sliding_s7_B1')}\\%$ against ${legit(ra, 'inwindow_s6_B1')}\\%$ on Alpha; with five edges it is ${legit(ro, 'sliding_s7_B5')}\\%$ against ${legit(ro, 'inwindow_s6_B5')}\\%$, and ${legit(ra, 'sliding_s7_B5')}\\%$ against ${legit(ra, 'inwindow_s6_B5')}\\%$.", 'sliding vs in-window'),
        (f"Within the sampling error of ${min(r[k]['n_eligible'] for r in (ro, ra) for k in r)}$ to ${max(r[k]['n_eligible'] for r in (ro, ra) for k in r)}$ targets", 'sliding eligible n'),
        (f"flip ${100 * CR['otc']['causal_retrain']['k_flip'] / CR['otc']['causal_retrain']['n_eligible']:.0f}\\%$ of the eligible OTC target-run pairs {retrain('otc')} and ${100 * CR['alpha']['causal_retrain']['k_flip'] / CR['alpha']['causal_retrain']['n_eligible']:.0f}\\%$ on Alpha {retrain('alpha')}", 'retraining'),
        (f"Distrust edges shift their mean total score by ${ctl('otc', 'dis_victim')['dR_vs_none']:.2f}$ (OTC) and ${ctl('alpha', 'dis_victim')['dR_vs_none']:.2f}$ (Alpha)", 'poisoning shift'),
        (f"and ${ctl('otc', 'dis_victim')['flip_pct']:.0f}\\%$ and ${ctl('alpha', 'dis_victim')['flip_pct']:.0f}\\%$ of the target-run pairs flip", 'poisoning flips'),
        (f"shift the score by ${ctl('otc', 'tru_victim')['dR_vs_none']:+.2f}$ and ${ctl('alpha', 'tru_victim')['dR_vs_none']:+.2f}$", 'poisoning trust control'),
        (f"by only ${ctl('otc', 'dis_unrelated')['dR_vs_none']:.2f}$ and ${ctl('alpha', 'dis_unrelated')['dR_vs_none']:.2f}$", 'poisoning placebo'),
        (f"are as damaging (${ctl('otc', 'dis_randsnap')['dR_vs_none']:.2f}$, ${ctl('alpha', 'dis_randsnap')['dR_vs_none']:.2f}$)", 'poisoning random snapshot'),
    ]


def source_checks() -> list:
    P1 = {k: load(f)[k] for f, k in (('p1_sota.json', 'otc'), ('p1_sota.json', 'alpha'),
                                       ('p1_sota_epn.json', 'epn30000'), ('p1_sota_edgfull.json', 'edg60000'))}
    share = lambda m, where='mean_dr': {d: 100 * P1[d][where][m] / P1[d]['mean_dr']['counterfactual'] for d in P1}
    adapted = [x for m in ('prbcd', 'influence', 'node_injection') for x in share(m).values()]
    expert, rand = share('expert').values(), share('random').values()
    out = share('expert', 'mean_dr_appended')
    three = [out[d] for d in ('otc', 'alpha', 'epn30000')]
    assert out['edg60000'] >= 100, f"heuristic out-of-window share on Epinions-D below 100%: {out['edg60000']}"
    worst = min(P1[d]['mean_dr_appended']['expert'] for d in P1)
    small = max(abs(P1[d]['mean_dr_appended'][m]) for d in P1 for m in ('random', 'counterfactual'))
    RR = load('random_source_sets.json')
    sets = [100 * RR[d]['C_random_distribution']['B5']['rand_dR_median_mean'] / RR[d]['C_random_distribution']['B5']['opt_dR_mean'] for d in ('otc', 'alpha')]
    return [
        (f"the heuristic reaches ${min(expert):.0f}$ to ${max(expert):.0f}\\%$ of the optimized shift", 'heuristic range'),
        (f"the adapted attacks ${min(adapted):.0f}$ to ${max(adapted):.0f}\\%$, and random sources ${min(rand):.0f}$ to ${max(rand):.0f}\\%$;", 'adapted and random ranges'),
        (f"fifty random source sets per target give ${min(sets):.0f}$ to ${max(sets):.0f}\\%$", 'random source sets'),
        (f"shift reputation by at most ${small:.3f}$ in magnitude", 'out-of-window, random and optimized'),
        (f"reaches ${min(three):.0f}$ to ${max(three):.0f}\\%$ of the optimized in-window shift on OTC, Alpha, and Epinions-S (a shift of up to ${worst:.2f}$)", 'heuristic out-of-window'),
        (f"five optimized edges shift OTC reputation by ${P1['otc']['mean_dr_appended']['counterfactual']:+.3f}$ out-of-window and by ${P1['otc']['mean_dr']['counterfactual']:.3f}$ in-window", 'evaluators'),
    ]


def defense_checks() -> list:
    D, SY = load('p3_defense_adaptive.json'), load('p3_sybil.json')
    kn = lambda d, cell, fn: D[d][cell][fn]['k_n']
    frac = lambda d, cell, fn: f"{kn(d, cell, fn)[0]}/{kn(d, cell, fn)[1]}"
    of = lambda d, cell, fn: f"${kn(d, cell, fn)[0]}$ of ${kn(d, cell, fn)[1]}$"
    pc = lambda d, cell, fn: f"{100 * kn(d, cell, fn)[0] / kn(d, cell, fn)[1]:.0f}"
    for d in ('otc', 'alpha'):   # random, established and new accounts flip equally many targets with one edge
        base = D[d]['low_random_B1']['mean']['flip_rate']
        assert base == D[d]['low_high_history_B1']['mean']['flip_rate'] == SY[d]['new_sybil_B1']['flip_rate'], d
    lh = 'low_low_history_B5'
    tpr5 = [load('detector_calib.json')[d]['detector']['test_disjoint']['tpr_at_5pct_fpr'] for d in ('otc', 'alpha')]
    return [
        (f"(${min(tpr5):.2f}$ to ${max(tpr5):.2f}$ at $5\\%$)", 'detector at 5% false positives'),
        (f"one distrust edge flips {of('otc', 'low_random_B1', 'mean')} targets on OTC and {of('alpha', 'low_random_B1', 'mean')} on Alpha whether its source", 'E13 one edge'),
        (f"flips ${kn('otc', 'low_low_history_B1', 'mean')[0]}$ and ${kn('alpha', 'low_low_history_B1', 'mean')[0]}$. We therefore weight", 'E13 low-history accounts'),
        (f"falls from ${pc('otc', lh, 'mean')}\\%$ (${frac('otc', lh, 'mean')}$) to ${pc('otc', lh, 'rater_weighted')}\\%$ (${frac('otc', lh, 'rater_weighted')}$) on OTC and from ${pc('alpha', lh, 'mean')}\\%$ (${frac('alpha', lh, 'mean')}$) to ${pc('alpha', lh, 'rater_weighted')}\\%$ (${frac('alpha', lh, 'rater_weighted')}$) on Alpha", 'E13 weighting'),
        (f"its shift stays at about ${D['otc']['low_high_history_B5']['rater_weighted']['dr']:.2f}$ (OTC) and ${D['alpha']['low_high_history_B5']['rater_weighted']['dr']:.2f}$ (Alpha), and it flips {of('otc', 'low_high_history_B5', 'rater_weighted')} and {of('alpha', 'low_high_history_B5', 'rater_weighted')} eligible targets, against {of('otc', 'low_high_history_B5', 'mean')} and {of('alpha', 'low_high_history_B5', 'mean')} without the defense", 'E13 adaptive attacker'),
        (f"one edge from a random account flips {of('otc', 'low_random_B1', 'rater_weighted')} eligible targets on OTC and {of('alpha', 'low_random_B1', 'rater_weighted')} on Alpha, against {of('otc', 'low_random_B1', 'mean')} and {of('alpha', 'low_random_B1', 'mean')} without the weighting.", 'E13 random account under the weighting'),
    ]

def dense_table_checks(CS: dict) -> list:
    """Rows of the non-GNN table (Table VII) and the source-selection shares of Section VI-F,
    rebuilt from the result files."""
    functions = [('gdte_mean_total', 'GDTE, total'), ('gdte_mean_prop', 'GDTE, prop.-only'), ('gdte_median_total', 'GDTE median, total'),
                 ('gdte_median_prop', 'GDTE median, prop.-only'), ('fg', 'FG'), ('wilson', 'Wilson bound'), ('beta', 'Beta mean'), ('fraction', 'Trust fraction')]
    checks = []
    for key, name in functions:
        cells = []
        for d in ('otc', 'alpha'):
            r = CS[d]
            cells += [f"${r['per_function'][key]['B1']['worst_pct']:.0f}$", f"${r['per_function'][key]['B5']['worst_pct']:.0f}$",
                      f"${r['common_by_raters']['>10'][key + '|B1']['pct']:.0f}$", f"${-r['by_raters'][key]['>=31']['shift_B1_worst_median']:.3f}$"]
        checks.append((f"{name} & " + ' & '.join(cells) + "\\\\", f'non-GNN table, {name}'))
    P1 = {k: load(f)[k] for f, k in (('p1_sota.json', 'otc'), ('p1_sota.json', 'alpha'),
                                       ('p1_sota_edgfull.json', 'edg60000'), ('p1_sota_epn.json', 'epn30000'))}
    # The per-strategy table moved to the supplement; the paper states ranges of shares of the optimized in-window shift.
    share = lambda d, key, col='mean_dr': 100 * P1[d][col][key] / P1[d]['mean_dr']['counterfactual']
    span = lambda xs: f"${min(xs):.0f}$ to ${max(xs):.0f}\\%$"
    adapted = [share(d, k) for d in P1 for k in ('prbcd', 'influence', 'node_injection')]
    checks.append((f"the heuristic reaches {span([share(d, 'expert') for d in P1])} of the optimized shift", 'source shares, heuristic'))
    checks.append((f"the adapted attacks {span(adapted)}, and random sources {span([share(d, 'random') for d in P1])}", 'source shares, adapted and random'))
    dilution = [share(d, 'expert', 'mean_dr_appended') for d in ('otc', 'alpha', 'epn30000')]
    checks.append((f"their dilution term alone reaches {span(dilution)} of the optimized in-window shift", 'source shares, heuristic out-of-window'))
    all_of_it = share('edg60000', 'expert', 'mean_dr_appended') >= 99.5
    checks.append(("and all of it on Epinions-D" if all_of_it else "<Epinions-D heuristic dilution share below 100%>", 'source shares, Epinions-D'))
    return checks


def main() -> int:
    tex_path = os.environ.get('GRAIL_PAPER_TEX', '')
    if not os.path.isfile(tex_path):
        sys.exit('set GRAIL_PAPER_TEX to the LaTeX source of the DSN paper (not part of this artifact)')
    with open(tex_path) as handle:
        tex = norm(handle.read())
    C = load('cohort_summary.json')
    checks = (cohort_checks(C) + placement_checks() + statistics_checks(load('stats_hier_s10.json')) + strata_checks()
              + mechanism_checks(load('masked_control_s10.json'), load('common_set_summary.json'))
              + deployment_checks() + source_checks() + defense_checks()
              + dense_table_checks(load('common_set_summary.json')))
    bad = [(frag, name) for frag, name in checks if norm(frag) not in tex] + every_model_failures(C)
    for frag, name in bad:
        print(f"FAIL [{name}]: {frag[:170]}")
    print(f"{len(checks) - len([b for b in bad if b[1] != 'every-model claim'])}/{len(checks)} number checks pass")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
