"""Build the dated Epinions subgraph edt<N> from Massa & Avesani's Extended Epinions.

Source: Network Repository `soc-epinions-trust-dir` (Rossi & Ahmed 2015), the Extended
Epinions release of Massa & Avesani (2005): 841,372 trust (+1) and distrust (-1)
statements among 131,828 users, each with a creation date (2001-01-10 to 2003-08-11;
68.8% carry the initial date 2001-01-10). SNAP's soc-sign-epinions has the same
statements without dates. Raw file: data/cyberdata/raw/soc-epinions-trust-dir.edges
(columns: source target sign unix_time; header line starts with '%').

Mirrors run_e10_scaling.make_epn_subsample (top-N nodes by total degree, induced
subgraph, nodes renumbered by first appearance, CSV and rating.txt row-aligned), with
two differences: the real timestamps are kept (epn<N> overwrote time with the row
index) and self-loops are dropped. Ratings are written as +10/-10 like epinions.csv.

Placeholder dates. 578,996 statements (68.8%) carry the earliest stamp, 2001-01-09
23:00 UTC; the next statement is six days later and the release averages about 200
statements per day afterwards (maximum 1,187). The first stamp therefore marks
statements created before Epinions started recording dates, whose true dates are
unknown. Kept as a timestamp, that block puts 75% of edt30000's edges into the first
of ten equal-span snapshots and leaves the last processed snapshot with 1.9% of the
window. With --genuine-dates the block is dropped before the top-N selection and the
files are written as edg<N> (every edge then carries a real creation date).

Run: python3 code/make_epinions_dated.py [N] [--genuine-dates]      (default N=30000)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "GRAIL_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "cyberdata"))
RAW = os.path.join(DATA_ROOT, "raw", "soc-epinions-trust-dir.edges")


def main(n_keep=30000, genuine_dates=False):
    E = np.loadtxt(RAW, comments="%", dtype=np.int64)
    df = pd.DataFrame(E, columns=["source", "target", "sign", "time"])
    n_raw, n_loops = len(df), int((df.source == df.target).sum())
    df = df[df.source != df.target]
    placeholder = int(df.time.min())
    n_placeholder = int((df.time == placeholder).sum())
    if genuine_dates:
        df = df[df.time > placeholder]
    prefix = "edg" if genuine_dates else "edt"
    deg = pd.concat([df.source, df.target]).value_counts()
    keep = set(deg.index[:n_keep])
    sub = df[df.source.isin(keep) & df.target.isin(keep)].copy()
    sub = sub.sort_values("time", kind="mergesort").reset_index(drop=True)
    nodes = pd.unique(pd.concat([sub.source, sub.target]))
    remap = {int(v): i for i, v in enumerate(nodes)}
    sub["source"] = sub.source.map(remap)
    sub["target"] = sub.target.map(remap)
    sub["rating"] = np.where(sub.sign > 0, 10, -10)

    csv_out = os.path.join(DATA_ROOT, f"{prefix}{n_keep}.csv")
    rt_out = os.path.join(DATA_ROOT, f"{prefix}{n_keep}-rating.txt")
    sub[["source", "target", "rating", "time"]].to_csv(csv_out, index=False)
    trust = (sub.sign > 0).astype(int)
    pd.DataFrame({"s": sub.source, "t": sub.target, "tr": trust, "di": 1 - trust}).to_csv(
        rt_out, sep=" ", header=False, index=False)

    t0 = int(df.time.min())
    stats = {
        "source": "Network Repository soc-epinions-trust-dir (Extended Epinions, Massa & Avesani 2005)",
        "raw_edges": n_raw, "raw_self_loops_dropped": n_loops, "top_n_by_degree": n_keep,
        "placeholder_time": placeholder, "raw_placeholder_statements": n_placeholder,
        "placeholder_block_dropped": bool(genuine_dates),
        "nodes": int(len(nodes)), "edges": int(len(sub)),
        "distrust_share": round(float((sub.sign < 0).mean()), 4),
        "initial_date_share": round(float((sub.time == t0).mean()), 4),
        "time_min": int(sub.time.min()), "time_max": int(sub.time.max()),
    }
    with open(os.path.join(DATA_ROOT, f"{prefix}{n_keep}-provenance.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 30000, genuine_dates="--genuine-dates" in sys.argv)
