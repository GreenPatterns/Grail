"""Check the numbers in the DSN paper's LaTeX source against the result JSONs.

Each check recomputes a value from its JSON, formats it the way the paper prints it, and asserts
that the formatted sentence fragment occurs in the paper (whitespace is normalized on both sides).
A failure means the prose and the data disagree, or the prose was reworded. In the second case
update the fragment here; never delete the check.

Checks of tables and sentences that live in the supplement run against supplement.tex next to the
paper (or GRAIL_SUPP_TEX). The result files of the 2026-11-22 revision are read from results/.

Run: GRAIL_PAPER_TEX=<path>/grail_dsn27.tex python3 code/check_dsn_numbers.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mycode.dataset import get_heterogeneous_snapshot_index  # the model's own snapshot cut

RESULTS = os.path.join(HERE, '..', 'results', 'unified')
REVISION = os.path.join(HERE, '..', 'results')   # result files of the 2026-11-22 revision sit next to unified/
DATA = os.environ.get('GRAIL_DATA_ROOT', os.path.join(HERE, '..', 'data', 'cyberdata'))
DATASETS = ('otc', 'alpha', 'edg60000')
DATA_FILES = {'otc': 'bitcoinotc', 'alpha': 'bitcoinalpha', 'edg60000': 'edg60000'}
NUMBER_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as handle:
        return json.load(handle)


def load_rev(name: str) -> dict:
    with open(os.path.join(REVISION, name)) as handle:
        return json.load(handle)


def seed_ci(values) -> tuple:
    """The revision scripts' interval over runs (percentile bootstrap, 10,000 resamples of whole
    runs, RandomState(12345)), recomputed without their final rounding to four decimals."""
    v = np.asarray(values, float)
    means = v[np.random.RandomState(12345).randint(0, len(v), (10000, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_t_interval(on, off) -> tuple:
    """95% t-interval of the mean per-seed difference on - off (the two lists are paired by seed)."""
    d = np.asarray(on, float) - np.asarray(off, float)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean() - half), float(d.mean() + half)


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
        (f"one fake rating from a random account pushes below the threshold ${min((lo(d, 'random_prop') for d in ('otc', 'alpha', 'edg60000')), key=float)}\\%$ to ${max((lo(d, 'random_prop') for d in ('otc', 'alpha', 'edg60000')), key=float)}\\%$ of users just above it and ${min((cell(d, 'random_prop') for d in ('otc', 'alpha', 'edg60000')), key=float)}\\%$ to ${max((cell(d, 'random_prop') for d in ('otc', 'alpha', 'edg60000')), key=float)}\\%$ of all eligible users.", 'abstract rates'),
        (f"for these optimized sources it reports ${lo('otc', 'opt_total')}\\%$ and ${lo('alpha', 'opt_total')}\\%$ instead of ${lo('otc', 'opt_prop')}\\%$ and ${lo('alpha', 'opt_prop')}\\%$", 'intro total rates'),
        (f"one distrust rating from a random account flips ${lo('otc', 'random_prop')}\\%$ (OTC), ${lo('alpha', 'random_prop')}\\%$ (Alpha), and ${lo('edg60000', 'random_prop')}\\%$ (Epinions-D) of the \\emph{{low-reputation}} targets", 'intro random rates'),
        (f"those less than $0.1$ above the threshold (about $2\\%$ of eligible targets on the Bitcoin networks, ${100 * low['edg60000']['n_pooled'] / n_all['edg60000']:.0f}\\%$ on Epinions-D), and ${cell('otc', 'random_prop')}\\%$, ${cell('alpha', 'random_prop')}\\%$, and ${cell('edg60000', 'random_prop')}\\%$ of all eligible targets. An optimized source", 'intro whole cohort, random'),
        (f"flips ${lo('otc', 'opt_prop')}\\%$, ${lo('alpha', 'opt_prop')}\\%$, and ${lo('edg60000', 'opt_prop')}\\%$ of low-reputation and ${cell('otc', 'opt_prop')}\\%$, ${cell('alpha', 'opt_prop')}\\%$, and ${cell('edg60000', 'opt_prop')}\\%$ of all eligible targets (Section", 'intro optimized'),
        (f"one random edge flips ${cell('otc', 'random_prop')}\\%$ (OTC) and ${cell('alpha', 'random_prop')}\\%$ (Alpha) under the propagation-only score, and one optimized edge ${cell('otc', 'opt_prop')}\\%$ and ${cell('alpha', 'opt_prop')}\\%$.", 'whole cohort'),
        (f"the optimized rate ranges from ${spread['otc'][0]:.0f}\\%$ to ${spread['otc'][1]:.0f}\\%$ on OTC and from ${spread['alpha'][0]:.0f}\\%$ to ${spread['alpha'][1]:.0f}\\%$ on Alpha", 'spread across runs'),
        (f"${low['otc']['n_pooled']}$ of ${thousands(n_all['otc'])}$ target-run pairs on OTC and ${low['alpha']['n_pooled']}$ of ${thousands(n_all['alpha'])}$ on Alpha, about $2\\%$", 'low-reputation share'),
        (f"it reports ${lo('otc', 'opt_total')}\\%$ instead of ${lo('otc', 'opt_prop')}\\%$ on OTC, ${lo('alpha', 'opt_total')}\\%$ instead of ${lo('alpha', 'opt_prop')}\\%$ on Alpha, and ${lo('edg60000', 'opt_total')}\\%$ instead of ${lo('edg60000', 'opt_prop')}\\%$ on Epinions-D", 'gap between the scores'),
        (f"propagation-only reputation of ${100 * o['shift']['all']['opt']['frac_negative']:.1f}\\%$ (OTC) and ${100 * a['shift']['all']['opt']['frac_negative']:.1f}\\%$ (Alpha) of eligible targets, by a median of ${-o['shift']['all']['opt']['median']:.2f}$ and ${-a['shift']['all']['opt']['median']:.2f}$", 'shift share and median'),
        (f"On Epinions-D the edge lowers ${100 * e['shift']['all']['opt']['frac_negative']:.0f}\\%$ of targets, by a median of only ${-e['shift']['all']['opt']['median']:.2f}$ (${low_med[0]:.2f}$ to ${low_med[1]:.2f}$ for low-reputation targets); there, ${pct(ef['<.05']['mean'])}\\%$ of the targets within $0.05$ of the threshold flip, but only ${pct(ef['.05-.1']['mean'])}\\%$ of those between $0.05$ and $0.1$. On all three graphs the median shift is negative in every run.", 'Epinions-D shift'),
        (f"targets within $0.05$ of the threshold flip at ${pct(o['cells']['opt_prop']['by_margin']['<.05']['mean'])}\\%$ (OTC) and ${pct(a['cells']['opt_prop']['by_margin']['<.05']['mean'])}\\%$ (Alpha), and even targets with clean $R\\ge0.8$ flip at ${pct(o['cells']['opt_prop']['by_margin']['>=.3']['mean'])}\\%$ and ${pct(a['cells']['opt_prop']['by_margin']['>=.3']['mean'])}\\%$.", 'margin extremes'),
        (f"one optimized edge flips ${pct(o['by_indegree']['2-3']['opt_prop']['mean'])}\\%$ of OTC targets with two or three raters, which make up ${pct(100 * o['share_indegree_le3'])}\\%$ of the cohort, against ${pct(o['by_indegree']['>=31']['opt_prop']['mean'])}\\%$", 'raters'),
        # the setup's copy of this share was cut as a duplicate; the number stays in Section VI-C
        (f"two or three raters, which make up ${pct(100 * o['share_indegree_le3'])}\\%$ of the cohort", 'setup, share of few-rater targets'),
        (f"about $2\\%$ on each (on Epinions-D, ${thousands(low['edg60000']['n_pooled'])}$ of ${thousands(n_all['edg60000'])}$, about ${100 * low['edg60000']['n_pooled'] / n_all['edg60000']:.0f}\\%$)", 'Epinions-D pair counts'),
        (f"forward-chaining AUC of ${min(min(C[d]['auc'].values()) for d in ('otc', 'alpha')):.2f}$ to ${max(max(C[d]['auc'].values()) for d in ('otc', 'alpha')):.2f}$ across runs", 'AUC range across runs'),
        (f"one optimized edge flips ${lo('edg60000', 'opt_prop')}\\%$ of low-reputation and ${cell('edg60000', 'opt_prop')}\\%$ of all targets under the propagation-only score.", 'Epinions-D cohort'),
        (f"one in-window distrust edge from a random account flips ${lo('otc', 'random_prop')}\\%$ (OTC) and ${lo('alpha', 'random_prop')}\\%$ (Alpha) of low-reputation targets and ${cell('otc', 'random_prop')}\\%$ and ${cell('alpha', 'random_prop')}\\%$ of all eligible targets (Epinions-D: ${lo('edg60000', 'random_prop')}\\%$ and ${cell('edg60000', 'random_prop')}\\%$);", 'conclusion rates'),
    ]


def processed_distrust_share(d: str) -> float:
    """Share of distrust ratings in the processed window s0..s6, cut by the model's own code."""
    stem = os.path.join(DATA, DATA_FILES[d])
    end = get_heterogeneous_snapshot_index(10, stem + '.csv')[6] + 1
    rows = np.loadtxt(stem + '-rating.txt', dtype=np.int64, max_rows=end)
    assert len(rows) == end, f'{d}: rating file shorter than the processed window'
    return float(rows[:, 3].mean())


def epinions_checks(C: dict) -> list:
    """Why Epinions-D is less sensitive (Section VI-B): rater counts of the eligible targets (all
    target-run pairs of the complete cohort), distrust share of the processed ratings, accuracy."""
    medians = {}
    for d in DATASETS:
        raters = []
        for path in glob.glob(os.path.join(RESULTS, 'cohort', f'cohort_{d}_s*.json')):
            with open(path) as handle:
                raters += [row['indeg'] for row in json.load(handle)['rows']]
        medians[d] = float(np.median(raters))
    assert len(set(medians.values())) == 1, f"'four raters, as on the Bitcoin graphs' needs equal medians: {medians}"
    few = lambda d: pct(C[d]['by_indegree']['2-3']['opt_prop']['mean'])
    dis = {d: pct(100 * processed_distrust_share(d)) for d in DATASETS}
    auc = {d: C[d]['auc']['42'] for d in DATASETS}   # Table II reports the seed-42 run
    return [
        (f"its median eligible target has {NUMBER_WORDS[int(medians['edg60000'])]} raters, as on the Bitcoin graphs, yet one optimized edge flips only ${few('edg60000')}\\%$ of its targets with two or three raters (Alpha: ${few('alpha')}\\%$)", 'Epinions-D, rater counts'),
        (f"(${dis['edg60000']}\\%$ of the processed ratings, against ${dis['otc']}\\%$ and ${dis['alpha']}\\%$)", 'Epinions-D, distrust share'),
        (f"(AUC ${auc['edg60000']:.2f}$ against ${auc['otc']:.2f}$ and ${auc['alpha']:.2f}$)", 'Epinions-D, accuracy'),
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


PLACEMENT_SUPP = []   # placement checks whose sentences live in the supplement (filled by placement_checks)


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
    # The sweep ranges moved to Supplement F with the 2026-11-22 case-study revision (supplement checks).
    PLACEMENT_SUPP.extend([
        (f"on OTC, $\\ls$ lies between ${min(lam):.2f}$ and ${max(lam):.2f}$ and the shift between ${max(drs):.2f}$ and ${min(drs):.2f}$", 'OTC sweep range'),
        (f"the shift is ${ss['sweep_dr']['s6']:.2f}$ at $s_6$ and ${ss['sweep_dr']['appended']:.3f}$ out-of-window", 'Epinions-S sweep'),
    ])
    return [
        (f"whereas the same rating in-window shifts it by ${max(inw):.2f}$ to ${min(inw):.2f}$", 'intro in-window range'),
        (f"Out-of-window, the mean shift is ${lin['otc']['appended_mean_dr']:+.3f}$ on OTC, ${lin['alpha']['appended_mean_dr']:+.3f}$ on Alpha, and ${lin['edg60000']['appended_mean_dr']:+.3f}$ on Epinions-D", 'out-of-window shifts'),
        (f"In-window, the same edges shift reputation by ${lin['otc']['processed_mean_dr']:.2f}$, ${lin['alpha']['processed_mean_dr']:.2f}$, and ${lin['edg60000']['processed_mean_dr']:.2f}$, with $\\ls={lin['otc']['processed_label_delta']:.2f}$, ${lin['alpha']['processed_label_delta']:.2f}$, and ${lin['edg60000']['processed_label_delta']:.2f}$.", 'in-window shifts'),
        (f"a trust edge shifts reputation by ${so['sign_trust']:+.2f}$, a neutral edge by ${so['sign_neutral']:+.2f}$, and a distrust edge by ${so['sign_distrust']:+.2f}$ on OTC; on Alpha the three shifts are ${sa['sign_trust']:+.2f}$, ${sa['sign_neutral']:+.2f}$, and ${sa['sign_distrust']:+.2f}$", 'sign control'),
    ]


def statistics_checks(ST: dict) -> list:
    o, a = ST['otc'], ST['alpha']
    deltas = [ST[d][k]['cliffs_delta'] for d in ('otc', 'alpha') for k in ('appended_vs_inwindow', 'distrust_vs_trust')]
    return [
        (f"by a median of ${-o['appended_vs_inwindow']['median']:.2f}$ (OTC) and ${-a['appended_vs_inwindow']['median']:.2f}$ (Alpha), and the distrust label shifts reputation further down than the trust label by a median of ${-o['distrust_vs_trust']['median']:.2f}$ and ${-a['distrust_vs_trust']['median']:.2f}$, in every run", 'paired medians'),
        (f"where it is higher, between ${min(deltas):.2f}$ and ${max(deltas):.2f}$;", 'Cliff delta range'),
        (f"sign-test $p={o['appended_vs_zero']['seed_sign_test_p']:.2f}$ and ${a['appended_vs_zero']['seed_sign_test_p']:.2f}$", 'out-of-window null p'),
        (f"(${o['appended_vs_inwindow']['n_pooled']}$ target-run pairs per dataset)", 'paired n'),
    ]


def strata_checks() -> tuple:
    """Strata and budgets (E4, Table VI), single-run variation (Section V), and time decay (E12,
    Section VII and Supplement I). Returns (paper checks, supplement checks)."""
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
    rerun = load_rev('budget_ablation_otc.json')['e4_reproduction']['B1']['flip_tot_pct']   # rerun of E4, Supplement K
    td = lambda d, cell_key, src: f"{100 * src[d][cell_key]['flip_rate_optimized']:.0f}"
    assert all(decayed_all[d][c]['flip_rate_optimized'] >= plain_all[d][c]['flip_rate_optimized']
               for d in ('otc', 'alpha') for c in ('low_B1', 'moderate_B5')), "'time decay can help the attacker' not supported"
    # Shortening Section VII (2026-11-22, third review) moved the moderate-stratum cells of E12 to
    # Supplement I; the full sentence is checked there, the low-stratum cells in the paper.
    supp = [(f"raises the total flip rate of one optimized edge on the low stratum from ${td('otc', 'low_B1', plain_all)}\\%$ to ${td('otc', 'low_B1', decayed_all)}\\%$ on OTC and from ${td('alpha', 'low_B1', plain_all)}\\%$ to ${td('alpha', 'low_B1', decayed_all)}\\%$ on Alpha, and that of five edges on the moderate stratum from ${td('otc', 'moderate_B5', plain_all)}\\%$ to ${td('otc', 'moderate_B5', decayed_all)}\\%$ and from ${td('alpha', 'moderate_B5', plain_all)}\\%$ to ${td('alpha', 'moderate_B5', decayed_all)}\\%$", 'time decay')]
    return checks + [
        (f"Epinions-D & Low & {row(dated, 'low')}\\\\", 'strata table, Epinions-D'),
        (f"one optimized edge flips ${cell(o, 'low', 1, 'preexist')}\\%$ (OTC) and ${cell(a, 'low', 1, 'preexist')}\\%$ (Alpha) of the targets under the propagation-only score", 'low stratum B=1'),
        (f"At $B=5$ the total rate reaches ${cell(o, 'low', 5, 'total')}\\%$ and ${cell(a, 'low', 5, 'total')}\\%$, whereas the propagation-only rate falls to ${cell(o, 'low', 5, 'preexist')}\\%$ and ${cell(a, 'low', 5, 'preexist')}\\%$", 'low stratum B=5'),
        (f"is ${cell(o, 'low', 1, 'total')}\\%$ in E4 (Table~\\ref{{tab:flip}}), ${rerun:.0f}\\%$ in its rerun, ${reps['low|mean_opt|gdte_mean_total|B1']['flip_pct']:.0f}\\%$ in the low-stratum experiment of the supplement (Table~\\ref{{S-tab:s-repbase}}), and ${100 * plain['low_B1']['flip_rate_optimized']:.0f}\\%$ in E12", 'single-run variation'),
        (f"the mean propagation-only shift of the low stratum shrinks as well (OTC from ${o['low_B1']['dR_preexist']:.3f}$ to ${o['low_B5']['dR_preexist']:.3f}$, Alpha from ${a['low_B1']['dR_preexist']:.3f}$ to ${a['low_B5']['dR_preexist']:.3f}$)", 'low-stratum shift by budget'),
        (f"it raises the flip rate of one optimized edge on the low stratum from ${td('otc', 'low_B1', plain_all)}\\%$ to ${td('otc', 'low_B1', decayed_all)}\\%$ on OTC and from ${td('alpha', 'low_B1', plain_all)}\\%$ to ${td('alpha', 'low_B1', decayed_all)}\\%$ on Alpha.", 'time decay, low stratum'),
    ], supp


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
        (f"The pooled rate falls from ${mo['flip_preexist_pct']:.1f}\\%$ to ${mo['flip_masked_pct']:.1f}\\%$ on OTC (seed-clustered bootstrap $95\\%$ interval $[{mo['flip_masked_seedboot_ci'][0]},{mo['flip_masked_seedboot_ci'][1]}]$) and stays at ${ma['flip_masked_pct']:.1f}\\%$ on Alpha ($[{ma['flip_masked_seedboot_ci'][0]:.1f},{ma['flip_masked_seedboot_ci'][1]}]$); the median masked shift is ${mo['dR_masked_median']:.2f}$ and ${ma['dR_masked_median']:.2f}$.", 'masked control'),
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
        (f"At $B=5$ the Wilson bound flips ${pct(fn(co, 'wilson', 'B5'))}\\%$ of both common sets, more than GDTE under the total score (${pct(fn(co, 'gdte_mean_total', 'B5'))}\\%$ and ${pct(fn(ca, 'gdte_mean_total', 'B5'))}\\%$); FG, the Beta mean, and the trust fraction flip ${min(others):.0f}$ to ${max(others):.0f}\\%$. Under the propagation-only score GDTE flips ${pct(fn(co, 'gdte_mean_prop', 'B5'))}\\%$ and ${pct(fn(ca, 'gdte_mean_prop', 'B5'))}\\%$", 'common set B=5'),
    ]


def moved_regime_checks() -> list:
    """The first, single-run placement comparison of E8 (out, in and sliding rows). The paired
    ten-run test replaced it in the paper; the table and its sentences moved to Supplement J, so
    these checks run against the supplement."""
    RG, SP = load('causal_regimes.json'), load('sliding_placebo.json')

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
    assert max(RG[d][k]['label_delta'] for d in RG for k in ('appended_B1', 'appended_B5')) < 1e-7, "'a silent no-op (Lambda < 10^-7)' not supported"
    return rows + [
        (f"the propagation-only flip rate is ${legit(ro, 'inwindow_s6_B1')}\\%$ in both cases on OTC, and ${legit(ra, 'sliding_s7_B1')}\\%$ against ${legit(ra, 'inwindow_s6_B1')}\\%$ on Alpha; with five edges it is ${legit(ro, 'sliding_s7_B5')}\\%$ against ${legit(ro, 'inwindow_s6_B5')}\\%$, and ${legit(ra, 'sliding_s7_B5')}\\%$ against ${legit(ra, 'inwindow_s6_B5')}\\%$.", 'sliding vs in-window'),
        (f"Within the sampling error of ${min(r[k]['n_eligible'] for r in (ro, ra) for k in r)}$ to ${max(r[k]['n_eligible'] for r in (ro, ra) for k in r)}$ targets", 'sliding eligible n'),
        (f"shifts the clean reputation of these targets by a mean of ${SP['otc']['placebo_mean']:.2f}$ (OTC) and ${SP['alpha']['placebo_mean']:.2f}$ (Alpha)", 'sliding-window placebo'),
    ]


def prospective_checks() -> tuple:
    """Paired sliding-window test of E8 (Section VI-F, Table VIII) and its table of runs
    (Supplement J). Returns (paper checks, supplement checks)."""
    P = {d: load_rev(f'prospective_paired_{d}.json') for d in ('otc', 'alpha')}
    V = {d: P[d]['verdict'] for d in P}
    assert not any(V[d][f'B{b}']['useful_proxy'] for d in P for b in (1, 5)), "'Neither graph meets this criterion' no longer holds"
    assert all(V['otc'][f'B{b}']['flip_within_margin'] and not V['otc'][f'B{b}']['shift_within_margin']
               and not V['alpha'][f'B{b}']['flip_within_margin'] for b in (1, 5)), "'agree within the margin on OTC but not on Alpha' no longer holds"

    def run(d: str, b: int, key: str) -> tuple:
        """Mean over runs and interval of a per-run value, as the script stored it per run."""
        per = [m[key] for m in P[d]['summary'][f'B{b}']['per_seed'].values()]
        return (float(np.mean(per)),) + seed_ci(per)

    def per_run(rows: list, b: int) -> dict:
        """Per-run values recomputed from the per-target rows (the supplement table of runs)."""
        cells = {reg: [r[f'{reg}_B{b}'] for r in rows] for reg in ('inwindow_s6', 'sliding_s7')}
        flips = {reg: [c['R_prop'] < 0.5 for c in cs] for reg, cs in cells.items()}
        return {'n': len(rows), 'agree': float(np.mean([a == s for a, s in zip(flips['inwindow_s6'], flips['sliding_s7'])])),
                'clean_slide': float(np.mean([r['R0_slide'] - r['R0_in'] for r in rows])),
                **{f'{short}_{k}': v for reg, short in (('inwindow_s6', 'in'), ('sliding_s7', 'slide')) for k, v in (
                    ('flip', 100 * float(np.mean(flips[reg]))), ('dR', float(np.mean([c['R_prop'] - c['R0'] for c in cells[reg]]))))}}

    for d in P:
        for b in (1, 5):
            assert run(d, b, 'slide_dR_prop')[0] < run(d, b, 'in_dR_prop')[0], f'in-window no longer understates the drop: {d} B={b}'
    table = []
    for d, name in (('otc', 'OTC'), ('alpha', 'Alpha')):
        for b in (1, 5):
            f, s = run(d, b, 'diff_flip_prop'), run(d, b, 'diff_dR_prop')
            table.append((f"{name if b == 1 else ''} & {b} & ${run(d, b, 'in_flip_prop')[0]:.1f}$ & ${run(d, b, 'slide_flip_prop')[0]:.1f}$ & ${f[0]:+.1f}$ [${f[1]:.1f}$, ${f[2]:.1f}$] & ${s[0]:.3f}$ [${s[1]:.3f}$, ${s[2]:.3f}$]\\\\", f'paired table, {d} B={b}'))
    margin = P['otc']['prereg']['margins']
    lam = [P[d]['summary'][f'B{b}']['across_seeds'][k]['mean'] for d in P for b in (1, 5) for k in ('in_Lambda', 'slide_Lambda')]
    shift = {d: float(np.mean([s['mean_slide_shift_of_clean_R_on_targets'] for s in P[d]['per_seed']])) for d in P}
    pairs = {d: P[d]['summary']['B1']['n_target_seed_pairs'] for d in P}
    one = lambda d, k: run(d, 1, k)[0]
    paper = table + [
        (f"(${pairs['otc']}$ target-run pairs on OTC, ${pairs['alpha']}$ on Alpha)", 'paired test, pairs'),
        (f"lie within $\\pm{margin['flip_prop_pp']:.0f}$ percentage points for the propagation-only flip rate and within $\\pm{margin['mean_shift_prop']}$ for the mean propagation-only shift", 'paired test, criterion'),
        (f"with one edge, ${one('otc', 'in_dR_prop'):.3f}$ against ${one('otc', 'slide_dR_prop'):.3f}$ (OTC) and ${one('alpha', 'in_dR_prop'):.3f}$ against ${one('alpha', 'slide_dR_prop'):.3f}$ (Alpha)", 'paired test, mean shifts'),
        (f"raises these targets' clean score by ${shift['alpha']:.3f}$ on average (${shift['otc']:.3f}$ on OTC; not declared in advance;", 'paired test, clean score after the slide'),
    ]
    supp, runs = [], {d: [per_run(s['rows'], 1) for s in P[d]['per_seed']] for d in P}
    for d, name in (('otc', 'OTC'), ('alpha', 'Alpha')):
        for i, (s, m) in enumerate(zip(P[d]['per_seed'], runs[d])):
            supp.append((f"{name if i == 0 else ''} & {s['seed']} & ${m['n']}$ & ${m['in_flip']:.1f}$ & ${m['slide_flip']:.1f}$ & ${m['in_dR']:+.3f}$ & ${m['slide_dR']:+.3f}$ & ${m['clean_slide']:+.3f}$\\\\", f'paired runs table, {d} seed {s["seed"]}'))
    span = lambda d: (min(m['clean_slide'] for m in runs[d]), max(m['clean_slide'] for m in runs[d]))
    agree = {d: 100 * float(np.mean([m['agree'] for m in runs[d]])) for d in P}
    supp += [
        (f"by ${shift['otc']:+.3f}$ (OTC) and ${shift['alpha']:+.3f}$ (Alpha) on average over runs (per run ${span('otc')[0]:+.3f}$ to ${span('otc')[1]:+.3f}$ and ${span('alpha')[0]:+.3f}$ to ${span('alpha')[1]:+.3f}$)", 'paired test, clean score after the slide per run'),
        (f"agree on whether a target flips in ${agree['otc']:.0f}\\%$ (OTC) and ${agree['alpha']:.0f}\\%$ (Alpha) of the pairs", 'paired test, per-target agreement'),
        # moved from Section VI-F to Supplement J with the 2026-11-22 case-study revision
        (f"Under the total score one edge flips ${one('otc', 'in_flip_tot'):.1f}\\%$ in-window and ${one('otc', 'slide_flip_tot'):.1f}\\%$ sliding on OTC, and ${one('alpha', 'in_flip_tot'):.1f}\\%$ and ${one('alpha', 'slide_flip_tot'):.1f}\\%$ on Alpha; $\\ls$ is ${min(lam):.2f}$ to ${max(lam):.2f}$ throughout.", 'paired test, total score and label sensitivity'),
    ]
    return paper, supp


def retraining_checks() -> tuple:
    """E8 retraining and E9 poisoning controls rerun with both scores (retrain_prop.json), next to
    their earlier total-only runs. Returns (paper checks, supplement checks)."""
    R, CR, PM = load_rev('retrain_prop.json'), load('causal_retrain_s5.json'), load('poison_matrix_s5.json')

    def seeds(d: str, b: int) -> dict:
        """Per seed: eligible pairs, flips under both scores, mean shifts of the eligible pairs."""
        c, out = R[d]['causal_retrain'], {}
        for sd, raw in c['raw'].items():
            clean, att = raw['clean'], raw[f'B{b}']
            el = [str(v) for v in c['victims'] if clean[str(v)][0] >= 0.5]
            out[sd] = {'n': len(el), 'kt': sum(att[v][0] < 0.5 for v in el), 'kp': sum(att[v][1] < 0.5 for v in el),
                       'dt': float(np.mean([att[v][0] - clean[v][0] for v in el])), 'dp': float(np.mean([att[v][1] - clean[v][0] for v in el]))}
            js = c[f'B{b}']['per_seed'][sd]
            assert (out[sd]['n'], out[sd]['kt'], out[sd]['kp']) == (js['n_eligible'], js['k_flip_tot'], js['k_flip_prop']), (d, b, sd)
        return out

    S = {(d, b): seeds(d, b) for d in ('otc', 'alpha') for b in (1, 5)}
    tot = lambda d, b, k: sum(s[k] for s in S[(d, b)].values())
    rate = lambda d, b, k: f"{100 * tot(d, b, k) / tot(d, b, 'n'):.0f}"
    mean = lambda d, b, k: float(np.mean([s[k] for s in S[(d, b)].values()]))
    ci = lambda d, b: seed_ci([s['dp'] for s in S[(d, b)].values()])
    n = {tot(d, b, 'n') for d in ('otc', 'alpha') for b in (1, 5)}
    assert max(S[('alpha', 1)], key=lambda sd: S[('alpha', 1)][sd]['kp']) == '42', "'most of these flips come from one run (seed 42)' no longer holds"
    assert len(n) == 1, f'eligible pairs differ between datasets or budgets: {n}'
    assert all(ci(d, 5)[1] < 0 for d in ('otc', 'alpha')), "'the model's response survives retraining' (B=5) not supported"
    assert all(ci(d, 1)[0] < 0 < ci(d, 1)[1] for d in ('otc', 'alpha')), "'indistinguishable from zero' (B=1) not supported"
    old = {d: CR[d]['causal_retrain'] for d in ('otc', 'alpha')}
    assert all(old[d]['per_seed'][sd]['n_eligible'] == S[(d, 5)][sd]['n'] for d in old for sd in old[d]['per_seed']), 'the two retraining runs have different eligible pairs'

    def poison(d: str, c: str, score: int) -> float:
        """Mean shift over all 24 targets and five seeds against the clean retraining of the same seed."""
        raw = R[d]['poison_matrix']['raw']
        return float(np.mean([np.mean([raw[c][sd][v][score] - raw['none'][sd][v][0] for v in raw['none'][sd]]) for sd in raw['none']]))
    conditions = (('dis_victim', 'distrust on every target ($s_6$)'), ('tru_victim', 'trust on every target (label control)'),
                  ('dis_unrelated', 'distrust on one target (placebo)'), ('dis_randsnap', 'distrust, random processed snapshot'))
    first = lambda d, c: PM[d]['matrix']['controls'][c]['dR_vs_none']
    assert max(abs(first(d, c) - poison(d, c, 0)) for d in ('otc', 'alpha') for c, _ in conditions) < 0.005, "'total shifts within 0.005 of the first run' not supported"
    dis = {d: R[d]['poison_matrix']['controls']['dis_victim'] for d in ('otc', 'alpha')}
    pflip = lambda d: 100 * sum(s['k_flip_prop'] for s in dis[d]['per_seed'].values()) / dis[d]['pooled_n_eligible']
    assert len(S[('otc', 1)]) == len(S[('alpha', 1)]) == 5, "'five seeds' not supported"
    assert max(100 * tot(d, 1, 'kp') / tot(d, 1, 'n') for d in ('otc', 'alpha')) < 20, "'one edge rarely flips a target' not supported"
    # The retraining boundary (one edge against five) in the introduction and the conclusion
    # (2026-11-22, third review); the introduction sentence replaced the five-edge-only sentence.
    boundary = lambda b: f"${rate('otc', b, 'kp')}\\%$ and ${rate('alpha', b, 'kp')}\\%$"
    paper = [
        (f"After retraining, one fake rating rarely flips such a target (propagation-only: {boundary(1)}), but five do ({boundary(5)};", 'intro, retraining'),
        (f"after retraining (five seeds), one edge rarely flips such a target ({boundary(1)}), but five do ({boundary(5)}).", 'conclusion, retraining'),
        (f"(${n.pop()}$ eligible target-run pairs per dataset)", 'retraining, pairs'),
        (f"Five edges per target flip ${rate('otc', 5, 'kt')}\\%$ (OTC) and ${rate('alpha', 5, 'kt')}\\%$ (Alpha) of the pairs under the total score (mean shift ${mean('otc', 5, 'dt'):.2f}$ and ${mean('alpha', 5, 'dt'):.2f}$) and ${rate('otc', 5, 'kp')}\\%$ and ${rate('alpha', 5, 'kp')}\\%$ under the propagation-only score (${mean('otc', 5, 'dp'):.2f}$ and ${mean('alpha', 5, 'dp'):.2f}$), so the model's response survives retraining", 'retraining, five edges'),
        (f"one edge flips ${rate('otc', 1, 'kt')}\\%$ and ${rate('alpha', 1, 'kt')}\\%$ (total) and ${rate('otc', 1, 'kp')}\\%$ and ${rate('alpha', 1, 'kp')}\\%$ (propagation-only), yet its mean propagation-only shift is indistinguishable from zero", 'retraining, one edge'),
        (f"most of these flips come from one run (${S[('alpha', 1)]['42']['kp']}$ of the ${tot('alpha', 1, 'kp')}$ on Alpha in seed $42$)", 'retraining, one edge, flips concentrated in seed 42'),
        (f"A rerun with both scores gives propagation-only shifts of ${poison('otc', 'dis_victim', 1):.2f}$ and ${poison('alpha', 'dis_victim', 1):.2f}$ under distrust, ${poison('otc', 'tru_victim', 1):+.2f}$ and ${poison('alpha', 'tru_victim', 1):+.2f}$ with the edges labeled trust, and ${poison('otc', 'dis_unrelated', 1):.2f}$ and ${poison('alpha', 'dis_unrelated', 1):.2f}$ for the placebo, with ${pflip('otc'):.1f}\\%$ and ${pflip('alpha'):.1f}\\%$ propagation-only flips under distrust.", 'poisoning, propagation-only rerun'),
    ]
    supp = []
    for d, name in (('otc', 'OTC'), ('alpha', 'Alpha')):
        for i, sd in enumerate(S[(d, 5)]):
            a, b, o = S[(d, 5)][sd], S[(d, 1)][sd], old[d]['per_seed'][sd]
            supp.append((f"{name if i == 0 else ''} & {sd} & ${a['n']}$ & ${a['kt']}$ & ${a['kp']}$ & ${a['dt']:.3f}$ & ${a['dp']:.3f}$ & ${b['kt']}$ & ${b['kp']}$ & ${o['k_flip']}$ & ${o['mean_dR']:.3f}$\\\\", f'retraining table, {d} seed {sd}'))
        supp.append((f" & all & ${tot(d, 5, 'n')}$ & ${tot(d, 5, 'kt')}$ & ${tot(d, 5, 'kp')}$ & ${mean(d, 5, 'dt'):.3f}$ & ${mean(d, 5, 'dp'):.3f}$ & ${tot(d, 1, 'kt')}$ & ${tot(d, 1, 'kp')}$ & ${old[d]['k_flip']}$ & ${float(np.mean([o['mean_dR'] for o in old[d]['per_seed'].values()])):.3f}$\\\\", f'retraining table, {d} all runs'))
    for c, label in conditions:
        cells = ' & '.join(f"${first(d, c):+.3f}$ & ${poison(d, c, 0):+.3f}$ & ${poison(d, c, 1):+.3f}$" for d in ('otc', 'alpha'))
        supp.append((f"{label} & {cells}\\\\", f'poisoning table, {c}'))
    supp.append((f"propagation-only score flip ${pflip('otc'):.1f}\\%$ (OTC) and ${pflip('alpha'):.1f}\\%$ (Alpha) of the pairs", 'poisoning, propagation-only flips (supplement)'))
    # moved from Section VI-F to Supplement I with the 2026-11-22 case-study revision
    supp.append((f"the earlier run, scored with the total score only, agreed with the rerun (${100 * old['otc']['k_flip'] / old['otc']['n_eligible']:.0f}\\%$ and ${100 * old['alpha']['k_flip'] / old['alpha']['n_eligible']:.0f}\\%$ of the pairs flip at $B=5$;", 'retraining, earlier total-only run'))
    return paper, supp


def budget_checks() -> tuple:
    """Budget ablation of E4 (Sections VI-B and VI-C; Supplement K). The paper states its
    conclusions without numbers, so the assertions below guard them; the supplement states the
    numbers. Returns (paper checks, supplement checks)."""
    BA = {d: load_rev(f'budget_ablation_{d}.json') for d in ('otc', 'alpha')}
    rules = (('random', 'random (pool order)'), ('topB_tot', 'top $B$ by total shift'), ('topB_prop', 'top $B$ by prop.-only shift'),
             ('greedy_tot', 'greedy by total score'), ('greedy_prop', 'greedy by prop.-only score'))
    recs = {d: BA[d]['ablation']['targets'] for d in BA}

    def cell(d: str, rule: str, b: int) -> tuple:
        """Propagation-only flip rate (%) and mean propagation-only shift of one rule and budget."""
        x = [r['rules'][rule][b - 1] for r in recs[d]]
        return 100 * float(np.mean([v['R_prop'] < 0.5 for v in x])), float(np.mean([v['dR_prop'] for v in x]))

    def fit(d: str, own_s6: bool, predict) -> float:
        """R^2 of a prediction of the B-edge propagation-only shift (B >= 2) from the single-edge shifts."""
        obs, pred = [], []
        for r in recs[d]:
            if (r['s6_in'] > 0) != own_s6:
                continue
            for rule, _ in rules:
                for v in r['rules'][rule][1:]:
                    obs.append(v['dR_prop'])
                    pred.append(predict([r['singles'][str(s)]['dR_prop'] for s in v['srcs']]))
        obs, pred = np.asarray(obs), np.asarray(pred)
        return float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))

    e4 = lambda d, b, k: BA[d]['e4_reproduction'][f'B{b}'][k]
    e4_shift = lambda d, b: float(np.mean([r['R_prop'] - r['R0'] for r in BA[d]['e4_reproduction'][f'B{b}']['rows']]))
    assert e4('alpha', 5, 'flip_prop_pct') < e4('alpha', 1, 'flip_prop_pct') and e4('otc', 5, 'flip_prop_pct') >= e4('otc', 1, 'flip_prop_pct'), "'reproduced on Alpha but not on OTC' no longer holds"
    assert all(cell(d, 'random', 5)[0] >= cell(d, 'random', 1)[0] for d in BA), "'random sources show no fall' no longer holds"
    assert all(cell('alpha', r, 5)[0] < cell('alpha', r, 1)[0] for r in ('topB_prop', 'greedy_prop')), "'choosing by propagation-only effect does not remove it' no longer holds"
    mean_fit = {d: fit(d, False, np.mean) for d in BA}
    assert len({cell('alpha', r, 1)[0] for r, _ in rules[1:]}) == 1, "'every model-based rule falls from one rate' needs equal B=1 rates"
    assert min(mean_fit.values()) >= 0.975, f"'close to the mean of their single-edge shifts' not supported: {mean_fit}"
    terms = {d: [(v['rater_logit_mean'], v['tgt_logit']) for r in recs[d] for rule, _ in rules for v in r['rules'][rule]] for d in BA}
    share = {d: float(np.sum(np.abs([a for a, _ in terms[d]])) / np.sum(np.abs([t for _, t in terms[d]]))) for d in BA}
    by_cell = {d: max(abs(np.mean([r['rules'][rule][b]['rater_logit_mean'] for r in recs[d]])) / abs(np.mean([r['rules'][rule][b]['tgt_logit'] for r in recs[d]]))
                      for rule, _ in rules for b in range(5)) for d in BA}
    assert max(by_cell.values()) < 0.01, f"'on average less than 1%' not supported: {by_cell}"
    moved = [float(np.mean([r['rules'][rule][b]['rater_moved_frac'] for r in recs[d]])) for d in BA for rule, _ in rules for b in range(5)]
    no_s6 = {d: float(np.mean([r['s6_in'] == 0 for r in recs[d]])) for d in BA}
    supp = []
    for d, name in (('otc', 'OTC'), ('alpha', 'Alpha')):
        supp.append((f"{name} & E4 rerun, total & ${e4(d, 1, 'flip_tot_pct'):.0f}$ & & ${e4(d, 3, 'flip_tot_pct'):.0f}$ & & ${e4(d, 5, 'flip_tot_pct'):.0f}$ & & \\\\", f'budget table, {d} E4 rerun total'))
        supp.append((f" & E4 rerun, prop.-only & ${e4(d, 1, 'flip_prop_pct'):.0f}$ & & ${e4(d, 3, 'flip_prop_pct'):.0f}$ & & ${e4(d, 5, 'flip_prop_pct'):.0f}$ & ${e4_shift(d, 1):+.3f}$ & ${e4_shift(d, 5):+.3f}$\\\\", f'budget table, {d} E4 rerun prop'))
        for rule, label in rules:
            flips = ' & '.join(f"${cell(d, rule, b)[0]:.0f}$" for b in range(1, 6))
            supp.append((f" & {label} & {flips} & ${cell(d, rule, 1)[1]:+.3f}$ & ${cell(d, rule, 5)[1]:+.3f}$\\\\", f'budget table, {d} {rule}'))
    supp += [
        (f"reproduces the fall on Alpha (${e4('alpha', 1, 'flip_prop_pct'):.0f}$, ${e4('alpha', 3, 'flip_prop_pct'):.0f}$, and ${e4('alpha', 5, 'flip_prop_pct'):.0f}\\%$ at $B=1$, $3$, and $5$) but not on OTC (${e4('otc', 1, 'flip_prop_pct'):.0f}$, ${e4('otc', 3, 'flip_prop_pct'):.0f}$, and ${e4('otc', 5, 'flip_prop_pct'):.0f}\\%$)", 'budget, E4 rerun'),
        (f"On Alpha every model-based rule falls from ${min(cell('alpha', r, 1)[0] for r, _ in rules[1:]):.0f}\\%$ at $B=1$ to ${min(cell('alpha', r, 5)[0] for r, _ in rules[1:]):.0f}$ to ${max(cell('alpha', r, 5)[0] for r, _ in rules[1:]):.0f}\\%$ at $B=5$", 'budget, Alpha selection'),
        (f"on OTC none falls by more than ${max(cell('otc', r, 1)[0] - cell('otc', r, 5)[0] for r, _ in rules[1:]):.0f}$ points", 'budget, OTC selection'),
        (f"Random sources show no fall on either graph (OTC ${cell('otc', 'random', 1)[0]:.0f}$ to ${cell('otc', 'random', 5)[0]:.0f}\\%$, Alpha ${cell('alpha', 'random', 1)[0]:.0f}$ to ${cell('alpha', 'random', 5)[0]:.0f}\\%$)", 'budget, random sources'),
        (f"(${100 * no_s6['otc']:.0f}\\%$ of the OTC and ${100 * no_s6['alpha']:.0f}\\%$ of the Alpha targets)", 'budget, targets without other s6 ratings'),
        (f"($R^2={mean_fit['otc']:.2f}$ on OTC and ${mean_fit['alpha']:.2f}$ on Alpha, over all rules and budgets), less well by the shift of the first source alone (${fit('otc', False, lambda s: s[0]):.2f}$ and ${fit('alpha', False, lambda s: s[0]):.2f}$), and not at all by their sum ($R^2$ below ${np.ceil(max(fit(d, False, np.sum) for d in BA)):.0f}$)", 'budget, dilution fits'),
        (f"the mean of the single-edge shifts fits less well ($R^2={fit('otc', True, np.mean):.2f}$ and ${fit('alpha', True, np.mean):.2f}$)", 'budget, targets with other s6 ratings'),
        (f"the mean absolute raters' term is ${100 * share['otc']:.1f}\\%$ (OTC) and ${100 * share['alpha']:.1f}\\%$ (Alpha) of the mean absolute target term, and at most ${100 * max(by_cell.values()):.1f}\\%$ for any rule and budget", 'budget, raters term'),
        (f"only ${100 * min(moved):.0f}$ to ${100 * max(moved):.0f}\\%$ of the raters' embeddings move", 'budget, raters that move'),
    ]
    return [], supp


def deployment_checks() -> tuple:
    PM = load('poison_matrix_s5.json')
    ctl = lambda d, k: PM[d]['matrix']['controls'][k]
    assert all(round(PM[d]['matrix']['retrain_std_clean'], 2) == 0.05 for d in ('otc', 'alpha')), 'poisoning spread is not about 0.05'
    # The single-run placement table and the total-only retraining sentence left the paper with the
    # 2026-11-22 revision: moved_regime_checks() and retraining_checks() hold their numbers now.
    paper = [
        (f"Distrust edges shift the targets' mean total score by ${ctl('otc', 'dis_victim')['dR_vs_none']:.2f}$ (OTC) and ${ctl('alpha', 'dis_victim')['dR_vs_none']:.2f}$ (Alpha)", 'poisoning shift'),
        (f"by only ${ctl('otc', 'dis_unrelated')['dR_vs_none']:.2f}$ and ${ctl('alpha', 'dis_unrelated')['dR_vs_none']:.2f}$", 'poisoning placebo'),
    ]
    # Moved from Section VI-F to Supplement I with the 2026-11-22 case-study revision (first, total-only run).
    supp = [
        (f"shift the mean total score by ${ctl('otc', 'dis_victim')['dR_vs_none']:.2f}$ (OTC) and ${ctl('alpha', 'dis_victim')['dR_vs_none']:.2f}$ (Alpha), and ${ctl('otc', 'dis_victim')['flip_pct']:.0f}\\%$ and ${ctl('alpha', 'dis_victim')['flip_pct']:.0f}\\%$ of the target-run pairs flip", 'poisoning flips'),
        (f"shift the score by ${ctl('otc', 'tru_victim')['dR_vs_none']:+.2f}$ and ${ctl('alpha', 'tru_victim')['dR_vs_none']:+.2f}$", 'poisoning trust control'),
        (f"are as damaging (${ctl('otc', 'dis_randsnap')['dR_vs_none']:.2f}$, ${ctl('alpha', 'dis_randsnap')['dR_vs_none']:.2f}$)", 'poisoning random snapshot'),
    ]
    return paper, supp


def source_checks() -> tuple:
    """Source selection (E10, Section VI-G). Returns (paper checks, supplement checks)."""
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
    # The fifty-random-source-sets result (random_source_sets.json) moved to Supplement F with the
    # shortening of Section VI-F; it is no longer in the paper, so it has no fragment here.
    paper = [
        (f"the heuristic reaches ${min(expert):.0f}$ to ${max(expert):.0f}\\%$ of the optimized shift", 'heuristic range'),
        (f"the adapted attacks ${min(adapted):.0f}$ to ${max(adapted):.0f}\\%$, and random sources ${min(rand):.0f}$ to ${max(rand):.0f}\\%$.", 'adapted and random ranges'),
        (f"shift reputation by at most ${small:.3f}$ in magnitude", 'out-of-window, random and optimized'),
        (f"reaches ${min(three):.0f}$ to ${max(three):.0f}\\%$ of the optimized in-window shift on OTC, Alpha, and Epinions-S (a shift of up to ${worst:.2f}$)", 'heuristic out-of-window'),
    ]
    # The conclusion's E10 example (five optimized edges on OTC, out-of-window and in-window) was cut
    # to meet the page limit; the optimized row of the supplement's source-selection table states both numbers.
    supp = [(f"Optimized & ${P1['otc']['mean_dr']['counterfactual']:+.3f}$ & ${P1['otc']['mean_dr_appended']['counterfactual']:+.3f}$ &",
             'evaluators: E10 optimized row, OTC in and out')]
    paper_rerun, supp_rerun = rerun_source_checks(P1)
    return paper + paper_rerun, supp + supp_rerun


E10_STRATEGIES = (('counterfactual', 'Optimized'), ('expert', 'Heuristic'), ('ig', 'Integrated gradient'),
                  ('batch_trust', 'Batch gradient'), ('greedy', 'Greedy gradient'), ('random', 'Random'),
                  ('prbcd', 'PRBCD$^\\dagger$'), ('influence', 'Influence edit$^\\dagger$'), ('node_injection', 'Node injection$^\\dagger$'))
E10_ADAPTED = ('prbcd', 'influence', 'node_injection')


def rerun_source_checks(P1: dict) -> tuple:
    """E10 rerun with both scores (run_p1_prop.py, 2026-11-22): Section VI-G, Table III, and
    Supplement E with the bottom of Table S5. P1 is the published E10 run, keyed by dataset.
    Returns (paper checks, supplement checks)."""
    R = {d: load_rev(f'p1_sota_prop_{f}.json') for d, f in (('otc', 'otc'), ('alpha', 'alpha'), ('epn30000', 'epn'), ('edg60000', 'edgfull'))}
    names = [m for m, _ in E10_STRATEGIES]
    logged = []   # the published logs list E10's targets in order (OTC, then Alpha, in p1_sota.log)
    for log in ('p1_sota.log', 'p1_sota_epn.log', 'p1_sota_edgfull.log'):
        with open(os.path.join(RESULTS, log)) as handle:
            logged += [int(t) for t in re.findall(r'\[\d+/16\] t=(\d+)', handle.read())]
    assert logged == [r['t'] for d in R for r in R[d]['rows']], "'keeps E10's targets' not supported"
    assert {(R[d]['seed'], R[d]['C'], R[d]['B'], R[d]['n_targets']) for d in R} == {(42, 60, 5, 16)}, 'the rerun configuration differs from E10'
    assert max(abs(R[d]['mean_dr'][m] - P1[d]['mean_dr'][m]) for d in R for m in names) < 0.003, "'total shifts within 0.003' not supported"
    assert max(abs(R[d]['flips'][m]['all_tot'] - round(16 * P1[d]['flip_rate'][m])) for d in R for m in names) <= 1, "'differ by at most one target' not supported"
    fl = lambda d, m, k: R[d]['flips'][m][k]
    assert {fl(d, m, 'n_low_eligible') for d in R for m in names} == {8}, 'not eight eligible low-stratum targets'
    low_tot = [fl(d, m, 'low_tot') for d in R for m in names]
    low_prop = {d: [fl(d, m, 'low_prop') for m in names] for d in R}
    assert len(set(low_prop['otc'])) == 1 and max(low_prop['edg60000']) == 0, "'4 on OTC, none on Epinions-D' not supported"
    three = ('otc', 'alpha', 'epn30000')
    share = lambda ms: [R[d]['share_prop_pct'][m] for d in three for m in ms]
    span = lambda xs: f"${min(xs):.0f}$ to ${max(xs):.0f}\\%$"
    assert all(min(three, key=lambda d: R[d]['share_prop_pct'][m]) == 'alpha' for m in E10_ADAPTED + ('random',)), "'lowest shares on Alpha' not supported"
    dated = [R['edg60000']['mean_dr_prop'][m] for m in names]
    assert -0.0005 < max(dated) < 0, "'between -0.006 and 0.000' needs small negative shifts"
    own = min(100 * (R['edg60000']['mean_dr'][m] - R['edg60000']['mean_dr_prop'][m]) / R['edg60000']['mean_dr'][m] for m in names)
    opt = lambda d, k: f"${R[d][k]['counterfactual']:.3f}$"
    paper = [
        (f"under which each strategy flips at most ${max(max(v) for v in low_prop.values())}$ of the eight low-stratum targets, against ${min(low_tot)}$ to ${max(low_tot)}$ under the total score.", 'E10 rerun, flips under both scores'),
        (f"nine strategies, $C={R['otc']['C']}$ & ${R['otc']['B']}$ & T, P\\\\", 'experiments table, E10 scores'),
    ]
    rows = [(f"{name} & " + ' & '.join(f"${R[d]['mean_dr_prop'][m]:+.3f}$ & ${R[d]['share_prop_pct'][m]:.0f}$ & {fl(d, m, 'low_tot')}/{fl(d, m, 'low_prop')}"
                                    for d in ('otc', 'alpha', 'edg60000', 'epn30000')) + " & \\\\", f'source table, rerun row {m}')
            for m, name in E10_STRATEGIES]
    supp = rows + [
        (f"It retrains the seed-${R['otc']['seed']}$ model and keeps E10's ${R['otc']['n_targets']}$ targets, ${R['otc']['C']}$ candidates per target, and nine strategies.", 'E10 rerun, configuration'),
        ("Its total shifts lie within $0.003$ of those at the top of the table, and its flip counts under the total score differ by at most one of the $16$ targets.", 'E10 rerun, agreement with the published run'),
        (f"Five optimized edges shift the propagation-only score by {opt('otc', 'mean_dr_prop')} (OTC), {opt('alpha', 'mean_dr_prop')} (Alpha), {opt('epn30000', 'mean_dr_prop')} (Epinions-S), and {opt('edg60000', 'mean_dr_prop')} (Epinions-D), against total shifts of {opt('otc', 'mean_dr')}, {opt('alpha', 'mean_dr')}, {opt('epn30000', 'mean_dr')}, and {opt('edg60000', 'mean_dr')} in the rerun.", 'E10 rerun, optimized shifts'),
        (f"the heuristic reaches {span(share(['expert']))}, the adapted attacks {span(share(E10_ADAPTED))}, and random sources {span(share(['random']))}; the adapted attacks and random sources reach their lowest shares on Alpha.", 'E10 rerun, propagation-only shares'),
        (f"every strategy's mean propagation-only shift lies between ${min(dated):.3f}$ and $0.000$, so the own-edge terms make up at least ${int(np.floor(own))}\\%$ of each strategy's total shift", 'E10 rerun, Epinions-D'),
        (f"each strategy flips ${min(low_tot)}$ to ${max(low_tot)}$ under the total score but, under the propagation-only score, ${low_prop['otc'][0]}$ on OTC, at most ${max(low_prop['alpha'] + low_prop['epn30000'])}$ on Alpha and Epinions-S, and none on Epinions-D.", 'E10 rerun, low-stratum flips'),
    ]
    return paper, supp


def evolvegcn_h_checks() -> list:
    """Low-reputation targets of the EvolveGCN-H release (Section VI-H, Table IX)."""
    n = {d: load(f'evolvegcn_placement_egcn_h_{d}.json')[f'bitcoin{d}']['flip_low_B1']['n'] for d in ('otc', 'alpha')}
    assert n['otc'] == 0, f"'no such target on OTC' not supported: {n}"
    return [(f"EvolveGCN-H has no such target on OTC and {NUMBER_WORDS[n['alpha']]} on Alpha", 'EvolveGCN-H low-reputation targets')]


def defense_checks() -> tuple:
    D, SY = load('p3_defense_adaptive.json'), load('p3_sybil.json')
    kn = lambda d, cell, fn: D[d][cell][fn]['k_n']
    frac = lambda d, cell, fn: f"{kn(d, cell, fn)[0]}/{kn(d, cell, fn)[1]}"
    of = lambda d, cell, fn: f"${kn(d, cell, fn)[0]}$ of ${kn(d, cell, fn)[1]}$"
    pc = lambda d, cell, fn: f"{100 * kn(d, cell, fn)[0] / kn(d, cell, fn)[1]:.0f}"
    for d in ('otc', 'alpha'):   # random, established and new accounts flip equally many targets with one edge
        base = D[d]['low_random_B1']['mean']['flip_rate']
        assert base == D[d]['low_high_history_B1']['mean']['flip_rate'] == SY[d]['new_sybil_B1']['flip_rate'], d
    lh, hh = 'low_low_history_B5', 'low_high_history_B5'
    det = {d: load('detector_calib.json')[d]['detector'] for d in ('otc', 'alpha')}
    tpr = lambda k: [det[d]['test_disjoint'][f'tpr_at_{k}pct_fpr'] for d in det]
    auc, adapt = [det[d]['test_disjoint']['roc_auc'] for d in det], {d: det[d]['test_adaptive_symmetric']['roc_auc'] for d in det}
    share = lambda d, cell, fn: kn(d, cell, fn)[0] / kn(d, cell, fn)[1]
    assert all(share(d, hh, 'rater_weighted') >= share(d, hh, 'mean') - 0.02 for d in ('otc', 'alpha')), "'the adaptive attacker loses little' not supported"
    # Shortening Section VII (2026-11-22, third review) moved to Supplement I the E13 counts, the
    # adaptive attacker's counts and the detector's 5% operating point; their checks run there now.
    paper = [
        (f"lowers the flip rate of five edges from the least active accounts from ${pc('otc', lh, 'mean')}\\%$ to ${pc('otc', lh, 'rater_weighted')}\\%$ on OTC and from ${pc('alpha', lh, 'mean')}\\%$ to ${pc('alpha', lh, 'rater_weighted')}\\%$ on Alpha", 'E13 weighting, rates'),
        (f"with an AUC of ${min(auc):.2f}$ to ${max(auc):.2f}$, yet detects only ${min(tpr(1)):.2f}$ to ${max(tpr(1)):.2f}$ of the attack edges at a $1\\%$ false-positive rate; an attacker who obtains a reverse rating per attack edge cuts the AUC to ${adapt['otc']:.3f}$ (OTC) and ${adapt['alpha']:.3f}$ (Alpha).", 'detector'),
    ]
    # moved from Section VII to Supplement I with the 2026-11-22 case-study revision
    supp = [
        (f"against {of('otc', 'low_random_B1', 'mean')} and {of('alpha', 'low_random_B1', 'mean')} without the weighting", 'E13 one edge'),
        (f"one edge from a random account flips {of('otc', 'low_random_B1', 'rater_weighted')} eligible targets on OTC and {of('alpha', 'low_random_B1', 'rater_weighted')} on Alpha, against {of('otc', 'low_random_B1', 'mean')} and {of('alpha', 'low_random_B1', 'mean')} without the weighting.", 'E13 random account under the weighting'),
        (f"falls from ${pc('otc', lh, 'mean')}\\%$ (${frac('otc', lh, 'mean')}$) to ${pc('otc', lh, 'rater_weighted')}\\%$ (${frac('otc', lh, 'rater_weighted')}$) on OTC and from ${pc('alpha', lh, 'mean')}\\%$ (${frac('alpha', lh, 'mean')}$) to ${pc('alpha', lh, 'rater_weighted')}\\%$ (${frac('alpha', lh, 'rater_weighted')}$) on Alpha", 'E13 weighting'),
        (f"At $B=5$ it flips {of('otc', hh, 'rater_weighted')} and {of('alpha', hh, 'rater_weighted')} eligible targets, against {of('otc', hh, 'mean')} and {of('alpha', hh, 'mean')} without the defense", 'E13 adaptive attacker'),
        (f"At a false-positive rate of $1\\%$ it detects ${min(tpr(1)):.2f}$ to ${max(tpr(1)):.2f}$ of the attack edges (${min(tpr(5)):.2f}$ to ${max(tpr(5)):.2f}$ at $5\\%$;", 'detector at 5% false positives'),
    ]
    return paper, supp

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


def moved_detail_checks(C: dict, CS: dict) -> list:
    """Two details that the 2026-11-22 case-study revision moved from Sections VI-C and VI-E to
    Supplements C and D (supplement checks)."""
    drop = lambda d, b: -C[d]['shift']['by_margin'][b]['opt']['median']
    near = {d: sorted(drop(d, b) for b in ('<.05', '.05-.1')) for d in ('otc', 'alpha')}
    wil = lambda d: CS[d]['common_by_raters']['<=10']['wilson|B1']['pct']
    return [
        (f"the median propagation-only drop of the optimized edge is ${near['otc'][0]:.2f}$ to ${near['otc'][1]:.2f}$ on OTC and ${near['alpha'][0]:.2f}$ to ${near['alpha'][1]:.2f}$ on Alpha, against ${drop('otc', '>=.3'):.2f}$ and ${drop('alpha', '>=.3'):.2f}$ for margins above $0.3$", 'cohort, drop near the threshold'),
        (f"and ${wil('otc'):.0f}\\%$ (OTC) and ${wil('alpha'):.0f}\\%$ (Alpha) of the common targets with at most ten raters", 'common set, Wilson at most ten raters'),
    ]


CASE_DATA = (('otc', 'OTC'), ('alpha', 'Alpha'))
CASE_FEATURES = (('node2vec', 'node2vec'), ('release', 'random'))
CASE_DEGREES = (('train', 'training window'), ('test', 'test snapshot'))
CASE_AGGS = (('mean', 'off'), ('robust', 'on'))


def sci(x: float) -> str:
    """LaTeX form of x with a one-digit mantissa, e.g. 1.7\\times10^{-6}."""
    mantissa, exponent = f"{x:.1e}".split('e')
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def case_file(d: str, features: str, degree: str) -> dict:
    """One result file of the TrustGuard case study (E15)."""
    return load_rev(f"trustguard_casestudy_{d}{'_node2vec' if features == 'node2vec' else ''}{'_testdeg' if degree == 'test' else ''}.json")


def casestudy_checks() -> tuple:
    """TrustGuard case study (E15): Sections VI-A (with Table IV), I, IV, VI-H, X, and the abstract;
    Supplement L and its three tables. Values are means over the five seeds, as the paper prints them. Returns
    (paper checks, supplement checks)."""
    F = {(d, f, g): case_file(d, f, g) for d, _ in CASE_DATA for f, _ in CASE_FEATURES for g, _ in CASE_DEGREES}
    rep = lambda k, agg: F[k]['summary'][agg]['repro']
    ev = lambda d, f, agg: F[(d, f, 'train')]['summary'][agg]['evasion']   # the single-edge test runs in the training-window files
    mcc = lambda k, agg, cond: rep(k, agg)[f'MCC_best_{cond}']
    pub = {d: F[(d, 'node2vec', 'train')]['published_fig3_single_timeslot_MCC'] for d, _ in CASE_DATA}
    gain = {d: pub[d]['attack_with_defense'] - pub[d]['attack_without_defense'] for d in pub}
    effect = {k: mcc(k, 'robust', 'inwin_dis')['mean'] - mcc(k, 'mean', 'inwin_dis')['mean'] for k in F}
    gap = [pub[k[0]]['no_attack'] - mcc(k, agg, 'clean')['mean'] for k in F for agg, _ in CASE_AGGS]
    assert 0 < min(gap) and max(gap) < 0.03, f"'clean MCC reproduces within 0.03' not supported: {gap}"
    ci = {k: paired_t_interval(mcc(k, 'robust', 'inwin_dis')['per_seed'], mcc(k, 'mean', 'inwin_dis')['per_seed']) for k in F}
    assert all(hi < gain[k[0]] for k, (_, hi) in ci.items()), "'the 95% interval lies below the published gain in every cell' not supported"
    assert all(mcc(k, agg, 'inwin_dis')['mean'] < pub[k[0]]['attack_without_defense'] for k in F for agg, _ in CASE_AGGS), "'our attack lowers MCC further' not supported"
    gpu = [s['robust']['gpu_equivalence'] for J in F.values() for s in J['runs'].values() if 'gpu_equivalence' in s['robust']]
    assert len(gpu) == len(F) and not any(g['values_differing_gt_1e-5'] for g in gpu), 'GPU port check missing or failed'
    gpu_max = max(g['max_abs_diff'] for g in gpu)
    assert gpu_max <= float(f"{gpu_max:.1e}"), 'GPU bound rounds down'
    pm3 = lambda x: f"${x['mean']:.3f}\\pm{x['sd']:.3f}$"
    rng = lambda xs, nd=2, sign='': f"${min(xs):{sign}.{nd}f}$ to ${max(xs):{sign}.{nd}f}$"
    flip = lambda d, f, agg, src, sc: ev(d, f, agg)[f'{src}_{sc}_flip']['rate_pct']['mean']
    # Single-edge test: one in-window rating against low-reputation targets, fixed weights.
    big = ('alpha', 'node2vec', 'random')
    changes = [flip(d, f, 'robust', s, 'prop') - flip(d, f, 'mean', s, 'prop') for d, _ in CASE_DATA for f, _ in CASE_FEATURES
               for s in ('random', 'optimized') if (d, f, s) != big]
    drop = flip(*big[:2], 'mean', 'random', 'prop') - flip(*big[:2], 'robust', 'random', 'prop')
    assert drop > 2 * max(-min(changes), max(changes)), "'the only large reduction' not supported"
    lam_out = max(ev(d, f, agg)['Lambda_out_max'] for d, _ in CASE_DATA for f, _ in CASE_FEATURES for agg, _ in CASE_AGGS)
    assert lam_out < 3e-7, lam_out
    lam_in = {(d, f, agg): ev(d, f, agg)['Lambda_in']['mean'] for d, _ in CASE_DATA for f, _ in CASE_FEATURES for agg, _ in CASE_AGGS}
    low_lam = min(lam_in, key=lam_in.get)
    assert low_lam == ('alpha', 'node2vec', 'robust'), f'smallest in-window Lambda is not Alpha, node2vec, defense on: {low_lam}'
    rest = [v for k, v in lam_in.items() if k != low_lam]
    # Retraining: MCC changes and label swaps per placement; target embedding change.
    cells = [(k, agg) for k in F for agg, _ in CASE_AGGS]
    lam_mcc = lambda m: [rep(k, agg)[f'Lambda_MCC_best_{m}']['mean'] for k, agg in cells]
    dmcc = lambda c: [-rep(k, agg)[f'dMCC_best_{c}']['mean'] for k, agg in cells]
    reach = lambda c: [rep(k, agg)[f'targets_{c}']['reach_dz_mean']['mean'] for k, agg in cells]
    loss_only_dz = max(x for J in F.values() for s in J['runs'].values() for agg, _ in CASE_AGGS
                       for c in ('app_dis', 'app_tru') for x in s[agg]['repro'][c]['reach_dz'])
    assert loss_only_dz <= float(f"{loss_only_dz:.1e}"), loss_only_dz
    assert max(abs(rep(k, agg)['dMCC_best_app_tru']['mean']) for k, agg in cells) < 0.02, "'relabelling them undoes this' not supported"
    # Total score where the defense prunes the random rating (Alpha, node2vec, defense on).
    recs = [x for s in F[('alpha', 'node2vec', 'train')]['runs'].values() for x in s['robust']['evasion']['targets']]
    pruned = [x for x in recs if x['dz_in'] < 1e-5]
    assert max(x['dz_in'] for x in pruned) <= 1.4e-6 and min(x['dz_in'] for x in recs if x not in pruned) > 1e-5, 'no gap between pruned and moved embeddings'
    assert not any(x['dz_in'] < 1e-5 for s in F[('alpha', 'node2vec', 'train')]['runs'].values() for x in s['mean']['evasion']['targets']), 'an embedding stays unchanged with the defense off'
    d_tot = [x['random']['R_tot'] - x['R_clean'] for x in pruned]
    k_tot, k_prop = sum(x['random']['R_tot'] < 0.5 for x in pruned), sum(x['random']['R_prop'] < 0.5 for x in pruned)
    o_on, a_on = flip('otc', 'node2vec', 'robust', 'optimized', 'prop'), flip('alpha', 'node2vec', 'robust', 'optimized', 'prop')
    r_on = {d: flip(d, 'node2vec', 'robust', 'random', 'prop') for d, _ in CASE_DATA}
    assert min(flip(d, f, agg, 'optimized', 'prop') for d, _ in CASE_DATA for f, _ in CASE_FEATURES for agg, _ in CASE_AGGS) > 50, "'most users near the threshold stay exposed to one optimized rating' not supported"
    n_low = [ev(d, f, agg)['random_prop_flip']['pooled_n'] for d, _ in CASE_DATA for f, _ in CASE_FEATURES for agg, _ in CASE_AGGS]
    assert all(abs(flip(d, 'node2vec', 'robust', 'optimized', 'prop') - flip(d, 'node2vec', 'mean', 'optimized', 'prop')) <= 5 for d, _ in CASE_DATA), "'about as many as with it off' not supported"
    # The smaller attack (degree counted in the test snapshot) against the published drop (Section VI-A).
    small = {d: mcc((d, 'node2vec', 'test'), 'mean', 'clean')['mean'] - mcc((d, 'node2vec', 'test'), 'mean', 'inwin_dis')['mean'] for d, _ in CASE_DATA}
    pdrop = {d: pub[d]['no_attack'] - pub[d]['attack_without_defense'] for d in pub}
    assert small['otc'] > 2 * pdrop['otc'] and f"{small['alpha']:.2f}" == f"{pdrop['alpha']:.2f}", "'more on OTC, as published on Alpha' not supported"
    # Table IV (Section VI-A): node2vec features, the attack fixed in advance (training-window degree),
    # and the single-edge test, defense off and on. Its rows took over the numbers of two sentences of
    # Section VI-A ('case study, optimized flips', and the random-source rates of 'case study, random sources').
    n2v = lambda d: (d, 'node2vec', 'train')
    m3 = lambda d, agg, cond: f"${mcc(n2v(d), agg, cond)['mean']:.3f}$"
    fl = lambda d, agg, src: f"${flip(d, 'node2vec', agg, src, 'prop'):.0f}$ (${flip(d, 'node2vec', agg, src, 'tot'):.0f}$)"
    row = lambda name, cell: f"{name} & " + ' & '.join(cell(d, agg) for d, _ in CASE_DATA for agg, _ in CASE_AGGS) + "\\\\"
    on_off = lambda d: f"\\multicolumn{{2}}{{c}}{{${effect[n2v(d)]:+.3f}$ [${ci[n2v(d)][0]:+.3f}$, ${ci[n2v(d)][1]:+.3f}$]}}"
    table = [
        (row('Clean MCC', lambda d, agg: m3(d, agg, 'clean')), 'case table, clean MCC'),
        (row('Attacked MCC', lambda d, agg: m3(d, agg, 'inwin_dis')), 'case table, attacked MCC'),
        ("\\quad on$-$off & " + ' & '.join(on_off(d) for d, _ in CASE_DATA) + "\\\\", 'case table, on minus off'),
        (row('Flips, random', lambda d, agg: fl(d, agg, 'random')), 'case table, random sources'),
        (row('Flips, optimized', lambda d, agg: fl(d, agg, 'optimized')), 'case study, optimized flips'),
    ]
    paper = table + [
        (f"${min(n_low)}$ to ${max(n_low)}$ targets per model", 'case study, low-reputation targets per model'),
        (f"its defense changes the attacked MCC by ${min(effect.values()):+.3f}$ to ${max(effect.values()):+.3f}$, where its paper reports ${gain['otc']:+.3f}$ and ${gain['alpha']:+.3f}$, outside our $95\\%$ intervals.", 'intro, case study'),
        (f"with the defense on, one rating from a random account still \\emph{{flips}}, or pushes below the threshold, ${r_on['otc']:.0f}\\%$ (OTC) and ${r_on['alpha']:.0f}\\%$ (Alpha) of the users just above it, and an \\emph{{optimized}} source, the most damaging of $50$ candidates, ${o_on:.0f}\\%$ and ${a_on:.0f}\\%$ (propagation-only score, five seeds).", 'intro, case study flips'),
        (f"from ${pub['otc']['attack_without_defense']:.3f}$ to ${pub['otc']['attack_with_defense']:.3f}$ on OTC and from ${pub['alpha']['attack_without_defense']:.3f}$ to ${pub['alpha']['attack_with_defense']:.3f}$ on Alpha", 'case study, published gain'),
        (f"which matches the released function to within ${sci(gpu_max)}$", 'case study, GPU port'),
        (f"Clean MCC reproduces within $0.03$ of the published ${pub['otc']['no_attack']:.3f}$ (OTC) and ${pub['alpha']['no_attack']:.3f}$ (Alpha)", 'case study, clean MCC'),
        (f"On OTC, even our smaller attack lowers MCC by ${small['otc']:.2f}$ instead of the published ${pdrop['otc']:.2f}$, whereas on Alpha both lower it by ${small['alpha']:.2f}$", 'case study, smaller attack against the published drop'),
        (f"the defense changes the attacked MCC by ${min(effect.values()):+.3f}$ to ${max(effect.values()):+.3f}$, where the paper reports ${gain['otc']:+.3f}$ and ${gain['alpha']:+.3f}$; in every cell, the $95\\%$ interval of the per-seed change ($t$-interval, five seeds) lies below the published gain.", 'case study, defense effect'),
        ("Whatever the defense does to MCC, most users near the threshold stay exposed to one optimized rating.", 'case study, users near the threshold'),
        (f"On the published defense TrustGuard, one optimized rating pushes ${o_on:.0f}\\%$ to ${a_on:.0f}\\%$, a per-user fragility that aggregate accuracy hides.", 'abstract, TrustGuard rates'),
        (f"in the other seven comparisons, release features included, it changes the rate by {rng(changes, 0, '+')} points", 'case study, random sources'),
        (f"($\\ls<3\\times10^{{-7}}$), whereas in-window ones give $\\ls={min(rest):.2f}$ to ${max(rest):.2f}$, or ${lam_in[low_lam]:.2f}$ on Alpha with the defense on.", 'case study, label swap at fixed weights'),
        ("against less than $3\\times10^{-7}$ out-of-window", 'Check 2, out-of-window bound'),
        (f"relabeling the attack edges changes the MCC by {rng(lam_mcc('inwin'))}, and removing them moves the targets' embeddings by a distance $\\|\\Delta z\\|$ of ${min(reach('inwin_dis')):.1f}$ to ${max(reach('inwin_dis')):.1f}$ on average", 'case study, in-window under retraining'),
        (f"lower the MCC by {rng(dmcc('app_dis'))} (the published attack: {rng(dmcc('inwin_dis'))}), and relabeling them undoes this, although no target's embedding moves by more than ${sci(loss_only_dz)}$", 'case study, loss-only under retraining'),
        (f"prunes the random rating of ${len(pruned)}$ of the ${len(recs)}$ targets, leaving their embeddings unchanged ($\\|\\Delta z\\|\\le{sci(max(x['dz_in'] for x in pruned))}$)", 'case study, pruned ratings'),
        (f"the propagation-only score barely moves, whereas the total score moves by ${min(d_tot):+.2f}$ to ${max(d_tot):+.2f}$. The defense blocked the model's response for these targets, yet ${k_tot}$ of them still fall below the threshold through the attacker's own term, {'none' if k_prop == 0 else k_prop} under the propagation-only score.", 'case study, total score'),
        (f"close to the five-run rates of E15 with release features and the defense off (${flip('otc', 'release', 'mean', 'optimized', 'prop'):.0f}\\%$ and ${flip('alpha', 'release', 'mean', 'optimized', 'prop'):.0f}\\%$)", 'official releases, TrustGuard against E15'),
        # the release's own rates (E11, Table X), reworded with the 2026-11-22 E10 revision
        ("its propagation-only flip rates, " + ' and '.join(f"${load(f'p0b_trustguard_{d}.json')[d]['flip_low_B1']['flip_prop_pct']:.0f}\\%$ ({name})" for d, name in CASE_DATA) + ", are close to", 'official releases, TrustGuard release flip rates'),
    ]
    # moved from Section VI-A to Supplement L with the 2026-11-22 third-review revision (Table IV holds the means)
    moved = [(f"for the smaller attack on OTC, the attacked MCC is {pm3(mcc(('otc', 'node2vec', 'test'), 'mean', 'inwin_dis'))} with the defense off and {pm3(mcc(('otc', 'node2vec', 'test'), 'robust', 'inwin_dis'))} with it on", 'case study, OTC smaller attack')]
    return paper, casestudy_supplement(F, pub, pruned, recs) + moved


def casestudy_supplement(F: dict, pub: dict, pruned: list, recs: list) -> list:
    """Supplement L: the three tables of the case study and the numbers of its text."""
    rep = lambda k, agg: F[k]['summary'][agg]['repro']
    mcc = lambda k, agg, cond: rep(k, agg)[f'MCC_best_{cond}']
    pmt = lambda x: f"${x['mean']:.3f}{{\\pm}}{x['sd']:.3f}$"
    supp = []
    for d, name in CASE_DATA:
        p = pub[d]
        supp.append((f"{name} & \\multicolumn{{2}}{{l}}{{published}} & \\multicolumn{{2}}{{c}}{{${p['no_attack']:.3f}$}} & ${p['attack_without_defense']:.3f}$ & ${p['attack_with_defense']:.3f}$ & ${p['attack_with_defense'] - p['attack_without_defense']:+.3f}$ & \\\\", f'case MCC table, {d} published'))
        for f, fname in CASE_FEATURES:
            for g, gname in CASE_DEGREES:
                k = (d, f, g)
                eff = mcc(k, 'robust', 'inwin_dis')['mean'] - mcc(k, 'mean', 'inwin_dis')['mean']
                lo, hi = paired_t_interval(mcc(k, 'robust', 'inwin_dis')['per_seed'], mcc(k, 'mean', 'inwin_dis')['per_seed'])
                supp.append((f" & {fname} & {gname} & {pmt(mcc(k, 'mean', 'clean'))} & {pmt(mcc(k, 'robust', 'clean'))} & {pmt(mcc(k, 'mean', 'inwin_dis'))} & {pmt(mcc(k, 'robust', 'inwin_dis'))} & ${eff:+.3f}$ & [${lo:+.3f}$, ${hi:+.3f}$]\\\\", f'case MCC table, {d} {f} {g}'))
    for d, name in CASE_DATA:
        for f, fname in CASE_FEATURES:
            for g, gname in CASE_DEGREES:
                for agg, aname in CASE_AGGS:
                    r = rep((d, f, g), agg)
                    cells = [f"${r[f'dMCC_best_{c}']['mean']:+.3f}$" for c in ('inwin_dis', 'app_dis', 'graph_dis')]
                    cells += [f"${r[f'Lambda_MCC_best_{m}']['mean']:.3f}$" for m in ('inwin', 'app', 'graph')]
                    cells += [f"${r[f'targets_{c}']['reach_dz_mean']['mean']:.2f}$" for c in ('inwin_dis', 'graph_dis')]
                    supp.append((f"{name} & {fname} & {gname} & {aname} & " + ' & '.join(cells) + "\\\\", f'case retraining table, {d} {f} {g} {agg}'))
    for d, name in CASE_DATA:
        for f, fname in CASE_FEATURES:
            for agg, aname in CASE_AGGS:
                b = F[(d, f, 'train')]['summary'][agg]['evasion']
                cells = [f"${b['random_prop_flip']['pooled_n']}$"]
                for src in ('random', 'optimized'):
                    cells += [f"${b[f'{src}_{sc}_flip']['rate_pct']['mean']:.0f}$" for sc in ('prop', 'tot')]
                    cells += [f"${b[f'{src}_{sc}_shift']['mean']:+.3f}$" for sc in ('prop', 'tot')]
                cells.append(f"${b['Lambda_in']['mean']:.2f}$")
                supp.append((f"{name} & {fname} & {aname} & " + ' & '.join(cells) + "\\\\", f'case single-edge table, {d} {f} {agg}'))
    # Text of Supplement L.
    att = {(d, g): [v['n_attack_edges'] for v in F[(d, 'node2vec', g)]['attack'].values()] for d, _ in CASE_DATA for g, _ in CASE_DEGREES}
    tgt = {d: {v['n_targets'] for v in F[(d, 'node2vec', 'train')]['attack'].values()} for d, _ in CASE_DATA}
    assert all(len(t) == 1 for t in tgt.values()), tgt
    assert all(F[(d, f, g)]['attack'] == F[(d, 'node2vec', g)]['attack'] for d, _ in CASE_DATA for f, _ in CASE_FEATURES for g, _ in CASE_DEGREES), 'the two feature sets use different attacks'
    span = lambda xs: f"${thousands(min(xs))}$ to ${thousands(max(xs))}$"
    last = [rep(k, 'robust')['MCC_last_inwin_dis']['mean'] - rep(k, 'mean')['MCC_last_inwin_dis']['mean'] for k in F]
    last_noise = max(abs(x) for k in F for agg, _ in CASE_AGGS for x in rep(k, agg)['dMCC_last_clean2']['per_seed'])
    noise = max(abs(x) for k in F for agg, _ in CASE_AGGS for x in rep(k, agg)['dMCC_best_clean2']['per_seed'])
    l1 = lambda f, key: [F[(d, f, g)]['summary']['robust']['repro']['targets_inwin_dis'][key]['L1_in']['mean'] for d, _ in CASE_DATA for g, _ in CASE_DEGREES]
    assert round(100 * min(l1('release', 'coef_organic_share_zero') + l1('release', 'coef_share_zero')), 1) >= 99.9, "'prunes 99.9 to 100%' not supported"
    per_seed = [sum(x['dz_in'] < 1e-5 for x in s['robust']['evasion']['targets']) for s in F[('alpha', 'node2vec', 'train')]['runs'].values()]
    others = {(d, f): sum(x['dz_in'] < 1e-5 for s in F[(d, f, 'train')]['runs'].values() for x in s['robust']['evasion']['targets'])
              for d, _ in CASE_DATA for f, _ in CASE_FEATURES if (d, f) != ('alpha', 'node2vec')}
    n_ev = {(d, f): sum(len(s['robust']['evasion']['targets']) for s in F[(d, f, 'train')]['runs'].values()) for d, f in others}
    own = [x['random']['R_tot'] * (x['n_raters'] + 1) - x['random']['R_prop'] * x['n_raters'] for x in pruned]
    assert all(o < 0.5 for o, x in zip(own, pruned) if x['random']['R_tot'] < 0.5), "'whose attacker is predicted to distrust them' not supported"
    lam_pr = float(np.mean([x['Lambda_in'] for x in pruned]))
    runs = [s[agg] for J in F.values() for s in J['runs'].values() for agg, _ in CASE_AGGS if 'evasion' in s[agg]]
    assert all(r['evasion']['n_low_all'] == r['evasion']['n_low_evaluated'] for r in runs), "'a cap never reached' not supported"
    assert not any(x['dz_in'] < 1e-5 for r in runs if not r['robust_aggregation'] for x in r['evasion']['targets']), "'with the defense off this never happens' not supported"
    cells = [(k, agg) for k in F for agg, _ in CASE_AGGS]
    drop = lambda c: [-rep(k, agg)[f'dMCC_best_{c}']['mean'] for k, agg in cells]
    rng = lambda xs: f"${min(xs):.2f}$ to ${max(xs):.2f}$"
    swap = max(abs(rep(k, agg)['dMCC_best_app_tru']['mean']) for k, agg in cells)
    assert swap < 0.015, swap
    ev_all = [F[(d, f, 'train')]['summary'][agg]['evasion'] for d, _ in CASE_DATA for f, _ in CASE_FEATURES for agg, _ in CASE_AGGS]
    dz_out = max(b['dz_out_max'] for b in ev_all)
    assert max(b['Lambda_out_max'] for b in ev_all) < 3e-7 and dz_out <= float(f"{dz_out:.1e}"), dz_out
    supp += [
        (f"an edge appended after the window is a silent no-op ($\\ls<3\\times10^{{-7}}$, target-embedding change at most ${sci(dz_out)}$)", 'case study, appended edges'),
        (f"distrust edges lower the MCC by {rng(drop('inwin_dis'))} in-window, by {rng(drop('app_dis'))} loss-only, and by {rng(drop('graph_dis'))} encoder-only. Relabelled as trust, loss-only edges change the MCC by less than $0.015$", 'case study, retraining placements'),
        (f"adds {span(att[('otc', 'train')])} attack edges per seed on OTC (${tgt['otc'].pop()}$ targets) and {span(att[('alpha', 'train')])} on Alpha (${tgt['alpha'].pop()}$ targets); the test-snapshot reading adds {span(att[('otc', 'test')])} and {span(att[('alpha', 'test')])}", 'case study, attack sizes'),
        (f"Two identical clean retrainings differ by up to ${noise:.3f}$ at the selected epoch. At the last epoch instead of the selected one, the defense changes the attacked MCC by ${min(last):+.3f}$ to ${max(last):+.3f}$ (identical retrainings: up to ${last_noise:.3f}$)", 'case study, last epoch'),
        (f"prunes $99.9$ to $100\\%$ of all ratings in the first layer, the organic ones included; with node2vec features it prunes ${100 * min(l1('node2vec', 'coef_organic_share_zero')):.0f}$ to ${100 * max(l1('node2vec', 'coef_organic_share_zero')):.0f}\\%$ of the organic ratings and ${100 * min(l1('node2vec', 'coef_share_zero')):.0f}$ to ${100 * max(l1('node2vec', 'coef_share_zero')):.0f}\\%$ of the attack edges", 'case study, pruning by features'),
        (f"per seed ${'$, $'.join(str(k) for k in per_seed)}$", 'case study, pruned ratings per seed'),
        (f"OTC ${others[('otc', 'node2vec')]}$ of ${n_ev[('otc', 'node2vec')]}$; random features: OTC ${others[('otc', 'release')]}$ of ${n_ev[('otc', 'release')]}$, Alpha ${others[('alpha', 'release')]}$ of ${n_ev[('alpha', 'release')]}$", 'case study, pruned ratings elsewhere'),
        (f"the attacker's own predicted trust averages ${np.mean(own):.2f}$, so the total score rises on average (by ${np.mean([x['random']['R_tot'] - x['R_clean'] for x in pruned]):+.3f}$), yet it falls below the threshold for ${sum(x['random']['R_tot'] < 0.5 for x in pruned)}$ targets", 'case study, own term'),
        (f"the mean label sensitivity of these ${len(pruned)}$ ratings is ${lam_pr:.2f}$", 'case study, label sensitivity of pruned ratings'),
        (f"the propagation-only score barely moves (largest change ${max(abs(x['random']['R_prop'] - x['R_clean']) for x in pruned):.4f}$)", 'case study, pruned ratings, propagation-only score'),
    ]
    return supp


def main() -> int:
    tex_path = os.environ.get('GRAIL_PAPER_TEX', '')
    if not os.path.isfile(tex_path):
        sys.exit('set GRAIL_PAPER_TEX to the LaTeX source of the DSN paper (not part of this artifact)')
    supp_path = os.environ.get('GRAIL_SUPP_TEX', os.path.join(os.path.dirname(tex_path), 'supplement.tex'))
    if not os.path.isfile(supp_path):
        sys.exit('set GRAIL_SUPP_TEX to the LaTeX source of the supplement (default: supplement.tex next to the paper)')
    with open(tex_path) as handle:
        tex = norm(handle.read())
    with open(supp_path) as handle:
        supp_tex = norm(handle.read())
    C = load('cohort_summary.json')
    revision = (prospective_checks(), retraining_checks(), budget_checks(), casestudy_checks())
    checks = (cohort_checks(C) + epinions_checks(C) + placement_checks() + statistics_checks(load('stats_hier_s10.json')) + strata_checks()[0]
              + mechanism_checks(load('masked_control_s10.json'), load('common_set_summary.json'))
              + deployment_checks()[0] + source_checks()[0] + evolvegcn_h_checks() + defense_checks()[0]
              + dense_table_checks(load('common_set_summary.json')) + [c for paper, _ in revision for c in paper])
    supp_checks = moved_regime_checks() + moved_detail_checks(C, load('common_set_summary.json')) + PLACEMENT_SUPP + strata_checks()[1] + deployment_checks()[1] + source_checks()[1] + defense_checks()[1] + [c for _, supp in revision for c in supp]
    bad = [(frag, name) for frag, name in checks if norm(frag) not in tex]
    bad += [(frag, f'supplement: {name}') for frag, name in supp_checks if norm(frag) not in supp_tex]
    bad += every_model_failures(C)
    for frag, name in bad:
        print(f"FAIL [{name}]: {frag[:170]}")
    fails = lambda prefix: len([b for b in bad if b[1].startswith('supplement: ') == prefix and b[1] != 'every-model claim'])
    print(f"{len(checks) - fails(False)}/{len(checks)} number checks pass in the paper, "
          f"{len(supp_checks) - fails(True)}/{len(supp_checks)} in the supplement")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
