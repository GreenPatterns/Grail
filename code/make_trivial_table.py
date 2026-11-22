"""Print the rows of Table S5 (source selection, E10) from JSON, and each strategy's share of
the optimized in-window shift, which Section VI-F states as ranges and Fig. S2 plots.

Sources (results/unified/), one run per dataset column:
  p1_sota.json      -> OTC, Alpha   (run_p1_efficient.py, GRAIL_P1_OUT=p1_sota.json)
  p1_sota_epn.json  -> Epinions-S (GRAIL_DATASETS=epn30000)
  p1_sota_edgfull.json -> Epinions-D (GRAIL_DATASETS=edg60000)

Three blocks are printed. (1) Compact rows, mean total shift (2 dp) / in-window flip rate
(integer %); the paper no longer has this table. (2) The rows of Table S5: mean total shift
in-window and out-of-window, and the in-window flip rate. (3) The shares. The JSON keys
`counterfactual` and `expert` are the optimized and the heuristic strategy. A missing dataset or
row prints "--". Rates use Python's round (half to even), so 10/16 prints 62 where Table S5
shows 63.

Run: python3 code/make_trivial_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, '..', 'results', 'unified')

COLUMNS = ['otc', 'alpha', 'epn', 'edt']
PUBLISHED = ['prbcd', 'influence', 'node_injection']
ROWS = [  # (json key, LaTeX label, cost)
    ('counterfactual', 'Optimized', '$60$f'),
    ('expert', 'Heuristic', '$0$'),
    ('ig', 'Integrated gradient', '$16$b'),
    ('batch_trust', 'Batch gradient', '$1$b'),
    ('greedy', 'Greedy gradient', '$5$b'),
    ('random', 'Random', '$0$'),
    None,  # \midrule; dagger rows: published attack families adapted to reputation
    ('prbcd', r'PRBCD$^\dagger$', '$30$b'),
    ('influence', r'Influence edit$^\dagger$', '$2$b'),
    ('node_injection', r'Node injection$^\dagger$', '$15$f'),
]


def _load(name):
    path = os.path.join(RDIR, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def load_results():
    """{column: run dict} for every dataset column that has a result file."""
    res = {k: v for k, v in _load('p1_sota.json').items() if k in ('otc', 'alpha')}
    epn = _load('p1_sota_epn.json')
    if 'epn30000' in epn:
        res['epn'] = epn['epn30000']
    edt = _load('p1_sota_edgfull.json')
    if 'edg60000' in edt:
        res['edt'] = edt['edg60000']
    return res


def cell(res, ds, key):
    r = res.get(ds)
    if not r or key not in r.get('mean_dr', {}):
        return '--'
    return f"${r['mean_dr'][key]:.2f}$/{round(100 * r['flip_rate'][key])}"


def latex_rows(res, columns=COLUMNS):
    out = []
    for row in ROWS:
        if row is None:
            out.append(r'\midrule')
            continue
        key, label, cost = row
        out.append(f"{label} & " + ' & '.join(cell(res, ds, key) for ds in columns) + f" & {cost}\\\\")
    return out


SUPP_COLUMNS = ['otc', 'alpha', 'edt', 'epn']   # column order of Table S5 (edt: Epinions-D, epn: Epinions-S)


def supplement_rows(res, columns=SUPP_COLUMNS):
    """Table S5: per dataset, in-window shift / out-of-window shift / in-window flip (%)."""
    out = []
    for row in ROWS:
        if row is None:
            out.append(r'\midrule')
            continue
        key, label, cost = row
        cells = []
        for ds in columns:
            r = res.get(ds)
            if not r or key not in r.get('mean_dr', {}):
                cells += ['--'] * 3
                continue
            app = r.get('mean_dr_appended', {}).get(key)
            cells += [f"${r['mean_dr'][key]:.3f}$", '--' if app is None else f"${app:+.3f}$",
                      f"${round(100 * r['flip_rate'][key])}$"]
        out.append(f"{label} & " + ' & '.join(cells) + f" & {cost}\\\\")
    return out


def shares(res, keys, columns=COLUMNS):
    """(min, max) of each key's shift as a % of the optimized shift, over available columns."""
    vals = [100 * res[ds]['mean_dr'][k] / res[ds]['mean_dr']['counterfactual']
            for ds in columns if ds in res for k in keys if k in res[ds]['mean_dr']]
    return (round(min(vals)), round(max(vals))) if vals else None


def main():
    res = load_results()
    print('% compact rows (not a table of the paper): mean dR / flip, columns ' + ' '.join(COLUMNS))
    print('\n'.join(latex_rows(res)))
    print('\n% Table S5: in dR / out dR / flip, columns ' + ' '.join(SUPP_COLUMNS))
    print('\n'.join(supplement_rows(res)))
    print('\n% shares of the optimized in-window dR (Section VI-F states the range over datasets)')
    for key in ('random', 'expert', 'prbcd', 'influence', 'node_injection'):
        print(f"% {key:15s} " + '  '.join(
            f"{ds}={100 * res[ds]['mean_dr'][key] / res[ds]['mean_dr']['counterfactual']:.0f}%"
            for ds in COLUMNS if ds in res and key in res[ds]['mean_dr']))
    print('% n_targets / n_eligible: ' + '  '.join(
        f"{ds}={res[ds]['n_targets']}/{res[ds]['n_eligible']}" for ds in COLUMNS if ds in res))


if __name__ == '__main__':
    main()
