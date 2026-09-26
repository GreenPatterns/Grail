"""
Re-draw Figure 2 (fig_forest, fig_dist) with TrueType fonts and fixed labels.

AAAI rejects PDFs whose graphics contain Type 3 fonts; the submitted panels had
them (matplotlib's default pdf.fonttype=3) and printed "(\\%)" and "OTC\\,hig"
literally. run_stats_rigor.py now sets pdf.fonttype=42 and plain-text labels.

  forest : drawn from the saved multiseed_flip.json (no GPU, no re-run).
  dist   : needs per-target rows, which stats_rigor.json does not store. They are
           re-collected with run_stats_rigor.collect (seed 42, same targets and
           pools) and cached in fig_dist_rows.json, so later re-plots need no GPU.

stats_rigor.json, the source of every number cited in the paper, is never written.

Run: GRAIL_FIGS=forest,dist python code/experiments/replot_stats_figs.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.insert(0, os.path.dirname(__file__))

import json

import numpy as np

import run_stats_rigor as rs

ROWS_JSON = os.path.join(rs.RDIR, 'fig_dist_rows.json')


def _dist_rows():
    if os.path.exists(ROWS_JSON):
        return json.load(open(ROWS_JSON))
    rows = []
    for ds in ('otc', 'alpha'):
        r, _ = rs.collect(ds)
        rows += r
    with open(ROWS_JSON, 'w') as f:
        json.dump(rows, f, indent=1)
    rs.log(f"  cached {len(rows)} per-target rows -> {ROWS_JSON}")
    return rows


def _anchor_report(rows):
    """Compare re-collected low-stratum medians/worst with the cited values."""
    cited = json.load(open(os.path.join(rs.RDIR, 'stats_rigor.json')))
    for ds in ('otc', 'alpha'):
        vals = [r['dr_legit_dis_B1'] for r in rows
                if r['ds'] == ds and r['stratum'] == 'low'
                and r.get('dr_legit_dis_B1') == r.get('dr_legit_dis_B1')]
        c = cited[ds]['distribution']['low']
        rs.log(f"  anchor {ds} low: median {np.median(vals):+.3f} (cited {c['median']:+.3f}), "
               f"worst {min(vals):+.3f} (cited {c['worst']:+.3f}), n={len(vals)} (cited {c['n']})")


def main():
    figs = os.environ.get('GRAIL_FIGS', 'forest,dist').split(',')
    if 'forest' in figs:
        rs.forest_figure(json.load(open(os.path.join(rs.RDIR, 'multiseed_flip.json'))))
        rs.log(f"  wrote {rs.FDIR}/fig_forest.pdf")
    if 'dist' in figs:
        rows = _dist_rows()
        _anchor_report(rows)
        rs.violin_figure(rows)
        rs.log(f"  wrote {rs.FDIR}/fig_dist.pdf")


if __name__ == "__main__":
    main()
