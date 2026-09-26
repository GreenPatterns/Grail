"""Print the supplement's reputation-baseline tables (LaTeX rows) from the paired run.

Source: results/unified/reputation_baselines_{otc,alpha,edgfull}.json
(experiments/run_reputation_baselines.py; the dated Epinions run is low stratum only,
with warm-started Fairness-Goodness). Every function is scored on the same low-stratum
targets and the same injections, so rows are directly comparable. Percentages are
recomputed from the eligible count, so a stored 54.5 (6/11) prints as 55, not 54.

  worst_case_rows : flip rate (%) at B = 1, 3, 5, the strongest of the four attackers
                    against each function (the figure's panel c shows B = 1)
  attacker_rows   : B = 1 flip rate (%) for each attacker separately

Run: python3 code/make_reputation_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, '..', 'results', 'unified')

DATASETS = ['otc', 'alpha', 'edg60000']
FILES = {'otc': 'otc', 'alpha': 'alpha', 'edg60000': 'edgfull'}
ATTACKERS = ['random', 'mean_opt', 'median_opt', 'fg_opt']
ROWS = [  # (function key, LaTeX label)
    ('gdte_mean_total', 'GDTE mean, total'),
    ('gdte_mean_prop', 'GDTE mean, prop.-only'),
    ('gdte_trimmed_total', 'GDTE trimmed, total'),
    ('gdte_trimmed_prop', 'GDTE trimmed, prop.-only'),
    ('gdte_median_total', 'GDTE median, total'),
    ('gdte_median_prop', 'GDTE median, prop.-only'),
    None,
    ('fg', 'Fairness--Goodness'),
    ('wilson', 'Wilson lower bound'),
    ('beta', 'Beta mean'),
    ('fraction', 'Trust fraction'),
]


def load(stratum='low'):
    out = {}
    for ds in DATASETS:
        path = os.path.join(RDIR, f'reputation_baselines_{FILES[ds]}.json')
        if os.path.exists(path):
            out[ds] = json.load(open(path))[ds]
    return out


def worst_case(cells, func, B, stratum='low'):
    """The attacker with the highest flip rate against func (ties: first listed)."""
    best = None
    for a in ATTACKERS:
        c = cells.get(f'{stratum}|{a}|{func}|B{B}')
        if c and c['flip_pct'] is not None and (best is None or c['flip_pct'] > best['flip_pct']):
            best = dict(c, attacker=a)
    return best


def exact_pct(c):
    """Flip rate recomputed from k/n (the JSON stores it rounded to one decimal)."""
    n = c['n_eligible']
    assert n < 1000, 'k cannot be recovered exactly from a 1-decimal percentage for n >= 1000'
    return 100.0 * round(c['flip_pct'] * n / 100.0) / n


def _fmt(c):
    return '--' if c is None or c['flip_pct'] is None else f"${int(exact_pct(c) + 0.5)}$"


def worst_case_rows(res):
    lines = []
    for row in ROWS:
        if row is None:
            lines.append(r'\midrule')
            continue
        func, label = row
        cells = []
        for ds in DATASETS:
            for B in (1, 3, 5):
                cells.append(_fmt(worst_case(res.get(ds, {}), func, B)))
        n = [worst_case(res[ds], func, 1)['n_eligible'] for ds in DATASETS if ds in res]
        lines.append(f"{label} & " + ' & '.join(cells) + f" & {'/'.join(map(str, n))}\\\\")
    return lines


def attacker_rows(res):
    lines = []
    for row in ROWS:
        if row is None:
            lines.append(r'\midrule')
            continue
        func, label = row
        cells = []
        for ds in DATASETS:
            for a in ATTACKERS:
                cells.append(_fmt(res.get(ds, {}).get(f'low|{a}|{func}|B1')))
        lines.append(f"{label} & " + ' & '.join(cells) + r'\\')
    return lines


def main():
    res = load()
    print('% worst case over attackers: OTC B1 B3 B5 | Alpha B1 B3 B5 | n eligible (OTC/Alpha)')
    print('\n'.join(worst_case_rows(res)))
    print('\n% B = 1 per attacker: OTC random mean_opt median_opt fg_opt | Alpha ...')
    print('\n'.join(attacker_rows(res)))
    for ds in DATASETS:
        if ds not in res:
            continue
        print(f"% {ds} worst-case B1 attackers: " + ', '.join(
            f"{f}={worst_case(res[ds], f, 1)['attacker']}" for f, _ in (r for r in ROWS if r)))


if __name__ == '__main__':
    main()
