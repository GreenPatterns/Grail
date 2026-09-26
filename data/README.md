# Data

`cyberdata/` holds the processed datasets the code reads. `env.sh` points the code at this folder
through `GRAIL_DATA_ROOT`. Every file was built from a public release, and the recipes below
rebuild each one from its source. The rebuilt files were checked against the bundled ones.

## Files

| Key (`GRAIL_DATASETS`) | Files | Statements | Users with an edge | Source |
|---|---|---|---|---|
| `otc` | `bitcoinotc.csv`, `bitcoinotc-rating.txt` | 35,592 | 5,881 | SNAP `soc-sign-bitcoinotc` |
| `alpha` | `bitcoinalpha.csv`, `bitcoinalpha-rating.txt` | 24,186 | 3,783 | SNAP `soc-sign-bitcoinalpha` |
| `epn30000` | `epn30000.csv`, `epn30000-rating.txt` | 689,307 | 29,923 | SNAP `soc-sign-epinions` |
| `edg60000` | `edg60000.csv`, `edg60000-rating.txt`, `edg60000-provenance.json` | 261,803 | 56,499 | Network Repository `soc-epinions-trust-dir` |
| `edg30000` | `edg30000.csv`, `edg30000-rating.txt`, `edg30000-provenance.json` | 237,794 | 28,080 | Network Repository `soc-epinions-trust-dir` |

`raw/soc-epinions-trust-dir.edges` is the raw dated Epinions release, and
`raw/soc-epinions-trust-dir.readme.html` is its acknowledgement page. They are included so that
`code/make_epinions_dated.py` can rebuild the dated graphs offline.

## Formats

- `<key>.csv` has the header `source,target,rating,time` and one rating per line. The model code
  reads only its times: `code/mycode/dataset.py` sorts them and cuts ten snapshots of equal time
  span (the equal-span mode that `code/mycode/mainz_protocol.py` selects). The snapshot
  boundaries are positions in time order, and they index the lines of the rating file.
- `<key>-rating.txt` has one line per rating, `source target trust distrust`, where
  `(trust, distrust)` is `(1, 0)` for a positive rating and `(0, 1)` otherwise. Its lines are in
  time order (ties keep the CSV's order), so line *i* is the *i*-th rating in time.
- Ratings are integers from −10 to 10 on the Bitcoin graphs and ±10 on the Epinions graphs.

## How each file was built

**Bitcoin-OTC and Bitcoin-Alpha.** The sources are `soc-sign-bitcoinotc.csv.gz` and
`soc-sign-bitcoinalpha.csv.gz` from https://snap.stanford.edu/data/ (columns source, target,
rating, time).

- `bitcoin<x>.csv` keeps SNAP's rows in file order, adds the header line (Alpha's header reads
  `target ` with a trailing space), rounds each time to whole seconds, and uses CRLF line endings.
  OTC's SNAP times carry fractions of a second; Alpha's are already whole.
- `bitcoin<x>-rating.txt` lists the ratings in stable time order. Each user id is replaced by its
  rank among the sorted original ids (0-based), and the separator is a tab. Alpha's CSV is not in
  time order; the code uses only its sorted times, so this does not matter.

```python
import numpy as np, pandas as pd
def build_bitcoin(snap_csv, name, header):
    s = pd.read_csv(snap_csv, header=None, names=['source', 'target', 'rating', 'time'])
    s['time'] = np.round(s.time).astype(np.int64)
    with open(f'{name}.csv', 'w', newline='') as f:
        f.write(header + '\r\n')
        for r in s.itertuples(index=False):
            f.write(f'{r.source},{r.target},{r.rating},{r.time}\r\n')
    t = s.sort_values('time', kind='mergesort')
    rank = {int(v): i for i, v in enumerate(np.unique(np.r_[s.source.values, s.target.values]))}
    with open(f'{name}-rating.txt', 'w', newline='') as f:
        for r in t.itertuples(index=False):
            trust = int(r.rating > 0)
            f.write(f'{rank[r.source]}\t{rank[r.target]}\t{trust}\t{1 - trust}\n')
build_bitcoin('soc-sign-bitcoinotc.csv.gz', 'bitcoinotc', 'source,target,rating,time')
build_bitcoin('soc-sign-bitcoinalpha.csv.gz', 'bitcoinalpha', 'source,target ,rating,time')
```

This recipe reproduces `bitcoinotc.csv`, `bitcoinalpha.csv` and `bitcoinalpha-rating.txt` byte
for byte. It reproduces `bitcoinotc-rating.txt` up to line terminators: the bundled file ends its
lines with CRLF and has no newline after the last line. The parsed contents are identical.

**SNAP Epinions, 30k-node subgraph (`epn30000`).** The source is `soc-sign-epinions.txt.gz` from
https://snap.stanford.edu/data/ (columns source, target, sign; no timestamps). Two steps build
the subgraph:

1. `epinions.csv` (not bundled) keeps SNAP's rows in file order. It relabels users in order of
   first appearance, reading each line's source before its target, sets `rating = 10 * sign`,
   and sets `time` to the row index, because SNAP gives no dates.
2. `make_epn_subsample(30000)` in `code/experiments/run_e10_scaling.py` keeps the subgraph
   induced by the 30,000 users of highest total degree. It relabels users by first appearance,
   sets `time` to the new row index, and writes both files.

```python
import numpy as np, pandas as pd
raw = pd.read_csv('soc-sign-epinions.txt.gz', sep='\t', comment='#', header=None,
                  names=['source', 'target', 'sign'])
first = pd.unique(np.column_stack([raw.source.values, raw.target.values]).ravel())
remap = {int(v): i for i, v in enumerate(first)}
pd.DataFrame({'source': raw.source.map(remap), 'target': raw.target.map(remap),
              'rating': 10 * raw.sign, 'time': np.arange(len(raw))}).to_csv('epinions.csv', index=False)
```

```bash
# with epinions.csv in an empty folder DIR (the function skips files that already exist):
GRAIL_DATA_ROOT=DIR PYTHONPATH=code:code/common python -c \
  "from experiments.run_e10_scaling import make_epn_subsample; make_epn_subsample(30000)"
```

The recipe reproduces both `epn30000` files byte for byte. Its intermediate `epinions.csv`
matches the development copy up to line endings.

**Dated Epinions (`edg60000`, `edg30000`).** The source is the Extended Epinions release of
Massa and Avesani (2005), distributed by the Network Repository as `soc-epinions-trust-dir`
(https://networkrepository.com/soc-epinions-trust-dir.php; columns source, target, sign, Unix
time). It holds 841,372 statements, and 578,996 of them (68.8%) carry the release's first date, a
placeholder for statements made before dates were recorded. `code/make_epinions_dated.py` drops
that block with `--genuine-dates`, keeps the subgraph induced by the N users of highest total
degree, sorts by time and relabels users by first appearance:

```bash
source env.sh
python code/make_epinions_dated.py 60000 --genuine-dates   # edg60000: all genuinely dated statements
python code/make_epinions_dated.py 30000 --genuine-dates   # edg30000: the earlier 30k subgraph
python code/make_epinions_dated.py 30000                   # edt30000: placeholder block kept; only a data test reads it
```

With N = 60000 the subgraph keeps every user with a dated statement, so `edg60000` is the complete
dated graph. These commands reproduce all six `edg` files byte for byte, including the
provenance files, which record the counts above.

## Licenses and citation

- **SNAP datasets** (Bitcoin-OTC, Bitcoin-Alpha, Epinions): the SNAP pages state no license. Cite
  Kumar et al., "Edge Weight Prediction in Weighted Signed Networks" (ICDM 2016) for the Bitcoin
  graphs and Leskovec et al. (2010) for Epinions, as the SNAP pages request.
- **Network Repository `soc-epinions-trust-dir`**: Creative Commons Attribution-ShareAlike
  (https://networkrepository.com/policy.php). The raw file and the derived `edg60000`,
  `edg30000` (and `edt30000`) files are redistributed under the same license. Cite Rossi and
  Ahmed, "The Network Data Repository with Interactive Graph Analytics and Visualization" (AAAI
  2015), and Massa and Avesani (2005).

## Checksums (SHA-256)

```
37ee60d8bf2080b6fa82174458359bbf4e9de53676d24266f0fb65d7fd8e76fb  bitcoinotc.csv
309dda5fad163a746b6bb2746cf6175a46f461fcb2d85e0042f17d150eb2abce  bitcoinotc-rating.txt
69b3bae235f91dfa784a251baf85888e1c1eff0b2dfcb05a4903a9cf184f6a81  bitcoinalpha.csv
4484de2d10fadd60f80f2667f93980acd4a7ade4a0b664dfcb1c2b08455c9d69  bitcoinalpha-rating.txt
e245d2a3250744d5ed02f870e12d8937bf8240e8f8e4cac777eb046069d64c54  epn30000.csv
ea9340997e576296f28894a68cc4f9d37d6fce5adde0671bce8c8d6039f448e4  epn30000-rating.txt
1fbd1c36397580875c204629449a5846eb3f6c8ce548155865a67aecbae1fe3d  edg60000.csv
0ec71570fedd8ed5ecf600003f214ab3c6a9daf7a8e6cff0e7f0780ae870abe0  edg60000-rating.txt
c8cbde4e48509231e0a0c25fd47749819f6e69426d5ba101bed92f0ca38258a0  edg60000-provenance.json
d0db91134c81b07e03e6e03f7844ca9ecd98c951da44c109d86a5604bb27f2d5  edg30000.csv
49699835ecbee07b9f03fd3b4c4b40cec004d53c113962a8edda1b66b9e8eea0  edg30000-rating.txt
9b982d9583e755243d009c5d7ab42ec969ce8eea95021c256e9330d34cd3f0be  edg30000-provenance.json
69396e0f6a46a4b8fde43836a3084864a938db63f68b597e395c2516859d2d01  raw/soc-epinions-trust-dir.edges
```
