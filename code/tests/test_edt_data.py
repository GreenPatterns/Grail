"""Validates the dated Epinions subgraph files (edt<N>.csv and edt<N>-rating.txt).

The files come from Massa & Avesani's Extended Epinions (trust and distrust
statements with creation dates; Network Repository soc-epinions-trust-dir), built
by code/make_epinions_dated.py. Unlike epn<N> (SNAP, whose "time" column is the
row index), edt<N> must carry the real Unix timestamps.

    pytest tests/test_edt_data.py        # ps env; no GPU needed
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
DATA_ROOT = os.environ.get(
    "GRAIL_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data", "cyberdata"),
)
N = 30000
# edt<N>: every statement, the placeholder-dated block included (69% share the first date);
# edg<N>: statements with a genuine creation date only (placeholder block dropped)
PREFIXES = ["edt", "edg"]
PLACEHOLDER_TIME = 979081200   # 2001-01-09 23:00 UTC, the stamp of 578,996 undated statements


def _paths(prefix):
    return (os.path.join(DATA_ROOT, f"{prefix}{N}.csv"), os.path.join(DATA_ROOT, f"{prefix}{N}-rating.txt"))


@pytest.fixture(scope="module", params=PREFIXES)
def data(request):
    csv, rating = _paths(request.param)
    assert os.path.exists(csv) and os.path.exists(rating), "run code/make_epinions_dated.py first"
    df = pd.read_csv(csv)
    rt = np.loadtxt(rating, dtype=np.int64)
    return df, rt, csv, request.param


def test_rows_align_between_csv_and_rating_file(data):
    df, rt, _, _ = data
    assert len(df) == len(rt)
    assert (df.source.to_numpy() == rt[:, 0]).all() and (df.target.to_numpy() == rt[:, 1]).all()


def test_labels_match_signs_and_are_one_hot(data):
    df, rt, _, _ = data
    assert set(np.unique(df.rating)) == {-10, 10}
    assert (rt[:, 2] + rt[:, 3] == 1).all()
    assert ((df.rating.to_numpy() > 0) == (rt[:, 2] == 1)).all()


def test_times_are_real_and_sorted(data):
    df = data[0]
    t = df.time.to_numpy()
    assert (np.diff(t) >= 0).all(), "edges must be time-sorted"
    assert t.min() > 9e8 and t.max() < 1.1e9, "expected Unix times in 2001-2003, not row indices"
    assert len(np.unique(t)) > 100


def test_no_self_loops_and_contiguous_ids(data):
    df = data[0]
    assert (df.source != df.target).all()
    ids = np.union1d(df.source, df.target)
    assert ids.min() == 0 and ids.max() == len(ids) - 1


def test_time_span_snapshots_are_ten_increasing_boundaries(data):
    from mycode.dataset import get_snapshot_index
    idx = get_snapshot_index(10, data[2], homogeneous_edges=False)
    assert len(idx) == 10
    assert all(a < b for a, b in zip(idx, idx[1:]))
    assert idx[-1] == len(data[0]) - 1


def test_genuine_dates_only_drops_the_placeholder_block():
    csv, _ = _paths("edg")
    df = pd.read_csv(csv)
    assert (df.time > PLACEHOLDER_TIME).all(), "edg<N> must hold no placeholder-dated statement"


def test_genuine_dates_give_non_degenerate_snapshots():
    from mycode.dataset import get_snapshot_index
    csv, _ = _paths("edg")
    idx = get_snapshot_index(10, csv, homogeneous_edges=False)
    first_share = (idx[0] + 1) / (idx[-1] + 1)
    assert first_share < 0.5, f"first snapshot holds {first_share:.0%} of the edges"
