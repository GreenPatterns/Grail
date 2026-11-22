# Artifact: Silent No-Ops and Self-Conditioning

This folder holds the code, figures, result files and supplement of the anonymous submission
*Silent No-Ops and Self-Conditioning: Two Evaluation Artifacts in Robustness Tests of Temporal
Trust GNNs*. It also holds the datasets the code reads and the instructions for running every
experiment.

You can check the paper at three depths:

| Depth | What you run | Time | Hardware |
|---|---|---|---|
| 1. Stored results | Regenerate every summary, table row and figure from the stored result files (Section 4) | about 1 minute | CPU |
| 2. Placement demo | Show that an appended attack edge is a silent, label-independent no-op, while the same edge inside the window is not (Section 5) | about 1 minute | 1 GPU |
| 3. Full reruns | Rerun any experiment behind a table, figure or number (Sections 6 and 7) | minutes to hours per experiment | 1 GPU |

All commands below run from the root of this artifact, the folder that holds this README.

## 1. Contents

```
README.md                 this file
supplement.pdf            the paper's online supplement (appendices A to I, Tables S1 to S12)
env.sh                    sets GRAIL_DATA_ROOT and PYTHONPATH; source it before every command
requirements.txt          main environment (Python 3.10, PyTorch 2.4, PyG 2.6)
requirements-dysat.txt    separate environment for the official DySAT release (Python 3.7, TensorFlow 1.15)
code/
  mycode/                 GDTE model (dgten.py, SL.py, s3r.py), training (gcn3.py), reputation and
                          attack oracle (trust_influence.py), reputation baselines, adapted attacks
  common/                 shared helpers: training entry point, target selection, paths
  experiments/            one script per experiment (run_*.py) and the figure scripts (make_*_figure.py)
  analyze_*.py, make_*_table.py, make_tables.py, make_stats_paragraph.py
                          summaries and LaTeX table rows computed from the result files
  silent_noop_demo.py     the placement demo of Section 5
  check_dsn_numbers.py    checks 114 numbers in the paper's LaTeX source against the result files
  make_epinions_dated.py  builds the dated Epinions graphs from the raw release
  tests/                  unit tests (pytest)
data/
  README.md               sources, licenses, file formats and how every file was built
  cyberdata/              the processed datasets the code reads, plus raw/ (dated Epinions release)
figures/                  the figures (Fig. 1 to 3 of the paper, Fig. S1 and S2 of the supplement); fig_system_src.tex is the source of Fig. 1
results/
  README.md               maps every result file to the paper item it backs
  unified/                every result file the paper takes a number from (74 JSON files and
                          their run logs)
```

The paper numbers its sections and tables with Roman numerals (Section VI-B, Table IV) and its
figures with Arabic numerals (Fig. 3). The supplement has appendices A to I (Supplement C),
Tables S1 to S12 and Fig. S1 and S2. This README uses the same numbering, and the experiment IDs E1 to E14 of the
paper's Table III.

**Names in the code and the result files.** The code and the result files predate the paper's
final terms. Read them as follows:

| In code and result files | In the paper |
|---|---|
| `naive`, `total` | total score |
| `legit`, `preexist`, `prop` | propagation-only score |
| `appended` | out-of-window placement |
| `processed`, `inwindow` | in-window placement |
| `label_delta`, label-Δ | label sensitivity Λ |
| `near_gate`, gate | low-reputation targets, threshold |
| `counterfactual` | optimized source |
| `expert` | the model-free heuristic |
| `victim` | target |
| `edg60000`, `edgfull` | Epinions-D |
| `epn30000`, `epn` | Epinions-S |
| `T`, `train_time_slots` | number of processed snapshots W |

## 2. Requirements

**Hardware.** A Linux machine with one NVIDIA GPU. The experiments ran on NVIDIA RTX A6000 and
L40S GPUs (48 GB). The Bitcoin experiments use about 3 to 5 GB of GPU memory. The two Epinions
graphs were only run on 48 GB GPUs, and smaller GPUs are untested for them. Steps 1 and 2 of the
table above need no more than a CPU and a small GPU respectively. The DySAT test runs on CPU.

**Software.** Two conda environments:

- *Main environment:* Python 3.10.14, PyTorch 2.4.0 with CUDA 12.4, PyTorch Geometric 2.6.1,
  torch-scatter 2.1.2 and the pinned packages in `requirements.txt`. Every script except the DySAT
  test runs here.
- *DySAT environment:* Python 3.7.16 with TensorFlow 1.15.5, networkx 1.11 and the pinned packages
  in `requirements-dysat.txt`. It is needed only for the official DySAT test (Section 7).

**Optional.** The figure scripts use the Liberation Sans font; without it, matplotlib falls back
to its default font and the figures differ slightly in appearance. Rebuilding Fig. 1 from
`figures/fig_system_src.tex` needs a LaTeX installation with TikZ.

## 3. Setup

```bash
conda create -n grail python=3.10.14 -y && conda activate grail
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 pyg_lib==0.4.0 \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install -r requirements.txt

source env.sh                                    # prints GRAIL_DATA_ROOT
python code/make_epinions_dated.py 30000         # builds edt30000, which one data test reads
python -m pytest code/tests -q                   # 43 tests; about 20 seconds on one GPU
```

`env.sh` sets `GRAIL_DATA_ROOT` to `data/cyberdata` and puts `code/` and `code/common/` on
`PYTHONPATH`. Source it in every new shell. Without a GPU, `test_fast_s6.py` and
`test_seed_propagation.py` skip themselves.

## 4. Regenerate the tables and figures from the stored results (CPU, about 1 minute)

```bash
source env.sh
python code/analyze_cohort.py                    # rewrites cohort_summary.json from the 30 cohort files
python code/analyze_common_set.py                # rewrites common_set_summary.json
python code/make_tables.py                    # rows of Table IV, Table S1, the official-release and common-set tables
python code/make_trivial_table.py                # Table S5; its shares are Table IX and Fig. S2
python code/make_reputation_table.py             # Tables S3 and S4
python code/make_stats_paragraph.py              # the Section VI-A paired statistics and Table S6
python code/experiments/make_dsn_figures.py figures   # Fig. 2, 3, S1 and S2 -> figures/fig_{placement,cohort,functions,sources}.pdf
```

The summaries and figures are written over the shipped copies in `results/unified/` and
`figures/`. On the machine that produced the results, the two summaries come out identical to
the shipped files, and the figures show the same data as the shipped PDFs. The table rows print
as LaTeX, for comparison with the paper's tables; they use the names of the code (see the table
of names above), so `naive` in a printed row is the paper's total score.
`make_dsn_figures.py` draws the panels with the functions of `make_evidence_figure.py` and
`make_cohort_figure.py`, which on their own write a combined three-panel version of Fig. 2, S2
and S1 (placement, sources, functions) (`figures/fig_evidence.pdf`) with the older labels.

`code/check_dsn_numbers.py` checks 114 numbers in the paper's text against the result files. It
reads the paper's LaTeX source, which is not part of this artifact:
`GRAIL_PAPER_TEX=<paper>.tex python code/check_dsn_numbers.py`.

## 5. Placement demo (one GPU, about 1 minute)

```bash
source env.sh
GRAIL_DATASETS=otc python code/silent_noop_demo.py
```

The demo trains GDTE on Bitcoin-OTC and scores one injected edge per target at two placements:
appended past the processed window, and inside the last processed snapshot. For each placement
it reports the mean reputation shift and the label sensitivity Λ = |ΔR(distrust) − ΔR(trust)|,
which the script prints as label-Δ. A run on one RTX A6000 printed:

```
=== Silent-No-Op Diagnostic on otc (n=8 targets) ===
   appended: mean ΔR = +0.0011   label-Δ = 0.000000
  processed: mean ΔR = -0.2078   label-Δ = 0.282234

  appended placement is a SILENT, LABEL-INDEPENDENT no-op: True
  processed placement is LABEL-SENSITIVE and damaging:       True
```

label-Δ is exactly zero at the appended placement in every run, because the injected label never
reaches the encoder (Proposition 1). The processed row varies slightly between runs, because GPU
training is not bit-deterministic; a second run printed −0.2093 and 0.283811. The in-window shift
is the total shift; Section III-B of the paper separates its propagation-only part.

## 6. Rerun the experiments

**Conventions.**

- Run every script from the artifact root after `source env.sh`, and choose the GPU with
  `CUDA_VISIBLE_DEVICES`.
- `GRAIL_DATASETS` selects the datasets: `otc` and `alpha` (Bitcoin-OTC and Bitcoin-Alpha),
  `edg60000` (dated Epinions, the complete genuinely dated graph) and `epn30000` (SNAP Epinions,
  30k-node subgraph).
- Scripts write JSON to `results/unified/` and log to standard output. `GRAIL_OUT` (or the
  script's own output variable, shown below) names the output file. The commands below overwrite
  the shipped file of the same name, so copy `results/` first if you want to compare.
- `GRAIL_SEEDS` lists the training seeds, which are genuine: each seed trains an independent model.
- GPU training is not bit-deterministic, so a rerun reproduces a result up to training noise,
  not digit for digit. The spread across the ten cohort models (Table S1) shows the size of that
  noise.
- Times are the wall-clock spans recorded in the shipped run logs, rounded. They were measured on
  shared GPUs and exclude any time spent before a script's first log line.

**Main paper.**

| Paper item | Command (prefix each with `source env.sh`) | Output in `results/unified/` | Logged time |
|---|---|---|---|
| Table IV, Fig. 3, abstract, introduction and conclusion rates (E3) | `GRAIL_DATASETS=otc GRAIL_SEEDS=42,1,2,3,4,5,6,7,8,9 python code/experiments/run_cohort_seeds.py`, then the same with `alpha` and `edg60000`; then `python code/analyze_cohort.py`, `python code/make_tables.py cohort` and `python code/experiments/make_cohort_figure.py` | `cohort/cohort_{otc,alpha,edg60000}_s{42,1..9}.json`, `cohort_summary.json` | per seed: OTC 0.3 h, Alpha 0.2 h, dated Epinions 0.9 h |
| Table II, clean AUC and MCC of the Bitcoin graphs | printed at the start of every seed-42 log, e.g. `loo_flip_table.log` | (log line) | |
| Table II, Epinions-S row | one call of `experiment_common.train_model('epn30000')`; no script writes it | `epn_clean.log` (the matching `epn_clean.json` holds null values) | 0.2 h |
| Table II, Epinions-D row | from the Section VI-A run on `edg60000` below (`clean_perf`, `train_edges`) and `data/cyberdata/edg60000-provenance.json` | `p0_linchpin_edgfull.json` | |
| Table V (strata and budgets, E4) | `GRAIL_DATASETS=otc,alpha python code/experiments/run_loo_flip_table.py`; `GRAIL_DATASETS=edg60000 GRAIL_STRATA=low GRAIL_OUT=loo_flip_edgfull_low.json python code/experiments/run_loo_flip_table.py`; the same with `GRAIL_STRATA=moderate,high GRAIL_OUT=loo_flip_edgfull_modhigh.json` | `loo_flip_table.json`, `loo_flip_edgfull_low.json`, `loo_flip_edgfull_modhigh.json` | 0.1 h; 0.3 h; 0.5 h |
| Table VIII (deployment regimes, E8): out, in and sliding rows | `GRAIL_DATASETS=otc,alpha python code/experiments/run_causal_regimes.py` | `causal_regimes.json` | 0.05 h |
| Section VI-E, retraining (E8); Table S11 | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4 GRAIL_NVICTIMS=24 GRAIL_POISON_PARTS=causal GRAIL_POISON_OUT=causal_retrain_s5.json python code/experiments/run_poison_matrix.py` | `causal_retrain_s5.json` | 0.1 h |
| Table VI (decomposition, E5); Section VI-B rank shift | `GRAIL_DATASETS=otc,alpha python code/experiments/run_stats_rigor.py` | `stats_rigor.json` (`decomp_B1`, `decomp_B5`, `decision_cost`) | 0.1 h |
| Table X (official releases, E11) | Section 7 below; then `python code/make_tables.py official_compact` | see Section 7 | |
| Fig. 2 (label sensitivity by snapshot, E1); Section VI-A sign controls | `GRAIL_DATASETS=otc,alpha,epn30000 GRAIL_OUT=p0b_main.json python code/experiments/run_p0b_verify.py`; `GRAIL_DATASETS=edg60000 GRAIL_OUT=p0b_edgfull.json python code/experiments/run_p0b_verify.py`; `GRAIL_DATASETS=otc,alpha python code/experiments/run_p2_arch_signature.py`; TrustGuard as in Section 7; then `python code/experiments/make_evidence_figure.py` | `p0b_main.json`, `p0b_edgfull.json`, `p2_arch_signature.json`, `p0b_trustguard_{otc,alpha}.json` | edg60000: 0.15 h |
| Table IX, Fig. S2, Section VI-F, Table S5 (source selection, E10) | `GRAIL_DATASETS=otc,alpha GRAIL_P1_OUT=p1_sota.json python code/experiments/run_p1_efficient.py`; the same with `GRAIL_DATASETS=epn30000 GRAIL_P1_OUT=p1_sota_epn.json` and with `GRAIL_DATASETS=edg60000 GRAIL_P1_OUT=p1_sota_edgfull.json`; then `python code/make_trivial_table.py` | `p1_sota.json`, `p1_sota_epn.json`, `p1_sota_edgfull.json` | 1.1 h; 2.5 h; 0.65 h |
| Table VII, Fig. S1, Section VI-D, Table S2 (common set, E7) | `GRAIL_DATASETS=otc python code/experiments/run_common_set.py`, then with `alpha`; then `python code/analyze_common_set.py` and `python code/make_tables.py common` | `common_set_{otc,alpha}.json`, `common_set_summary.json` | 0.85 h; 0.5 h |
| Section VII, time-decayed mean | `GRAIL_DATASETS=otc,alpha GRAIL_OUT=p0d_otcalpha.json python code/experiments/run_p0d_harden.py`; the same with `GRAIL_REP_MODE=decayed GRAIL_OUT=p0d_decayed.json` | `p0d_otcalpha.json`, `p0d_decayed.json` | |
| Section VI-A, in-window against out-of-window (E1) | `GRAIL_DATASETS=otc,alpha python code/experiments/run_p0_linchpin.py`; `GRAIL_DATASETS=edg60000 GRAIL_P0_OUT=p0_linchpin_edgfull.json python code/experiments/run_p0_linchpin.py`; Epinions-S from `p0b_main.json` (Fig. 2 row) | `p0_linchpin.json`, `p0_linchpin_edgfull.json` | edg60000: 0.65 h |
| Section VI-A, paired contrasts over ten models (E2); Table S6 | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4,5,6,7,8,9 GRAIL_OUT=stats_hier_s10.json python code/experiments/run_stats_hier.py`; then `python code/make_stats_paragraph.py` | `stats_hier_s10.json` | 1.25 h |
| Section VI-B, other thresholds | `GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_margin.py` | `p3_margin.json` (`by_threshold`) | |
| Section VI-C, query-edge-masked control (E6); Table S7 | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4,5,6,7,8,9 GRAIL_BLOCKS=B GRAIL_OUT=masked_control_s10.json python code/experiments/run_masked_and_random_sets.py` | `masked_control_s10.json` | 0.8 h |
| Supplement D, low-stratum run of the reputation functions; Tables S3 and S4 | `GRAIL_DATASETS=otc GRAIL_OUT=reputation_baselines_otc.json python code/experiments/run_reputation_baselines.py`, then the same for `alpha`; `GRAIL_DATASETS=edg60000 GRAIL_STRATA=low GRAIL_FG_WARM=1 GRAIL_OUT=reputation_baselines_edgfull.json python code/experiments/run_reputation_baselines.py` (`GRAIL_FG_WARM=1` starts each attacked Fairness-Goodness solve from the clean fixed point: same fixed point, fewer iterations); then `python code/make_reputation_table.py` | `reputation_baselines_{otc,alpha,edgfull}.json` | 0.1 h each; 0.25 h |
| Section V, calibration (Brier, ECE); Section VII detector; Table S12 | `GRAIL_DATASETS=otc,alpha python code/experiments/run_detector_calib.py` | `detector_calib.json` (`calibration_oos`, `detector`) | 0.05 h |
| Section VI-E, poisoning controls (E9); Table S10 | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4 GRAIL_NVICTIMS=24 GRAIL_POISON_OUT=poison_matrix_s5.json python code/experiments/run_poison_matrix.py` | `poison_matrix_s5.json` | 0.2 h |
| Section VII, rater-history weighting and the adaptive attacker (E12) | `GRAIL_DATASETS=otc,alpha GRAIL_OUT=p3_defense_adaptive.json python code/experiments/run_p3_defense.py`; `GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_sybil.py` | `p3_defense_adaptive.json`, `p3_sybil.json` | 0.15 h |

**Supplement only.**

| Supplement item | Command or source | Output in `results/unified/` |
|---|---|---|
| Supplement B, the earlier 30k dated subgraph | `GRAIL_DATASETS=edg30000 GRAIL_P0_OUT=p0_linchpin_edg.json python code/experiments/run_p0_linchpin.py`; `GRAIL_DATASETS=edg30000 GRAIL_OUT=p0b_edg.json python code/experiments/run_p0b_verify.py` | `p0_linchpin_edg.json`, `p0b_edg.json` |
| Supplement C, Table S1 (per-model cohort) | from the cohort files; `python code/make_tables.py cohort_per_model` | `cohort/*.json` |
| Section VI-F and Supplement F, random source sets (78 to 94%, 88% quartile) | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42 GRAIL_BLOCKS=C python code/experiments/run_masked_and_random_sets.py` (the block uses the seed-42 model only) | block `C_random_distribution` of `random_source_sets.json` |
| Section VI-E and Supplement F, sliding-window placebo | a one-off run; no current script writes this file | `sliding_placebo.json` |
| Supplement F, regression on the margin | the Table VI run | `stats_rigor.json` (`dose_response`) |
| Supplement H, Table S8 (TrustGuard sweep) | Section 7 | `p0b_trustguard_{otc,alpha}.json` |
| Supplement H, Table S9 (label-blind releases) | Section 7; then `python code/make_tables.py official_supplement` | `evolvegcn_placement_*.json`, `dysat_placement_*.json` |

Two cautions about the shipped files. Blocks `A_fixed_cohort` and `B_query_edge_masked` of
`random_source_sets.json` come from an older run whose seeds did not produce independent models;
`cohort/` and `masked_control_s10.json` replace them, and the paper cites neither block. The
`stats_rigor.log` and `stats_rigor_regen.log` runs both wrote `stats_rigor.json`.

## 7. Official releases

The placement test runs on four official releases, each cloned at the commit we used. The clones
are not included; each release keeps its own license.

| Release | Clone | Command | Output in `results/unified/` |
|---|---|---|---|
| TrustGuard (Wang et al., TDSC 2024) | `git clone https://github.com/Jieerbobo/TrustGuard && git -C TrustGuard checkout 5458c0e240fa5a48f449a11dde4ba3d1d53227fc` | `TRUSTGUARD_DIR=<clone>/code GRAIL_DATASETS=otc GRAIL_OUT=p0b_trustguard_otc.json python code/experiments/run_trustguard_placement.py`, then the same with `alpha` | `p0b_trustguard_{otc,alpha}.json` |
| EvolveGCN-O and EvolveGCN-H (Pareja et al., AAAI 2020) | `git clone https://github.com/IBM/EvolveGCN && git -C EvolveGCN checkout 90869062bbc98d56935e3d92e1d9b1b4c25be593`; create its input files from the bundled data: `tail -n +2 data/cyberdata/bitcoinotc.csv > <clone>/data/soc-sign-bitcoinotc.csv` and the same for `bitcoinalpha` (these are the files our runs used) | `EVOLVEGCN_DIR=<clone> EGCN_MODEL=egcn_o GRAIL_DATASETS=bitcoinotc GRAIL_OUT=evolvegcn_placement_egcn_o_otc.json python code/experiments/run_evolvegcn_placement.py`; repeat for `bitcoinalpha` and for `EGCN_MODEL=egcn_h` | `evolvegcn_placement_egcn_{o,h}_{otc,alpha}.json` |
| DySAT (Sankar et al., WSDM 2020) | `git clone https://github.com/aravindsankar28/DySAT && git -C DySAT checkout 777f290dcef38c28c390ba4642ce45621e7cedb4`; use one clone per dataset to run both at once | in the DySAT environment: `DYSAT_DIR=<clone> GRAIL_DATASETS=otc GRAIL_OUT=dysat_placement_otc.json python code/experiments/run_dysat_placement.py`, then the same with `alpha` | `dysat_placement_{otc,alpha}.json` |
| SignedGCN (PyTorch Geometric's implementation) | none; ships with PyTorch Geometric | `GRAIL_DATASETS=otc,alpha python code/experiments/run_official_signedgcn.py` | `official_signedgcn.json` |

Each release runs with its own trainer and configuration. The harness scripts apply only the
compatibility shims described in their docstrings, such as PyYAML's `Loader` argument for
EvolveGCN and a list-returning `map` for DySAT's Python 2 code. EvolveGCN and DySAT never read a
rating's sign, so their diagnostic is the change of the target's embedding rather than label-Δ.
The EvolveGCN and DySAT harnesses restore the epoch that the release's own selection rule picks,
not the last epoch. The shipped training logs (`evolvegcn_egcn_*.log`, `dysat_{otc,alpha}.log`) hold each selected
epoch's test score.

## 8. Data

`data/cyberdata/` holds the processed files the code reads. `data/README.md` gives each file's
source, license and exact construction. In short:

| Dataset | Key | Source |
|---|---|---|
| Bitcoin-OTC | `otc` | SNAP `soc-sign-bitcoinotc` (Kumar et al., ICDM 2016) |
| Bitcoin-Alpha | `alpha` | SNAP `soc-sign-bitcoinalpha` (Kumar et al., ICDM 2016) |
| Epinions-S (SNAP Epinions, 30k-node subgraph) | `epn30000` | SNAP `soc-sign-epinions` (Leskovec et al. 2010), no timestamps |
| Epinions-D (dated Epinions, the complete genuinely dated graph) | `edg60000` | Network Repository `soc-epinions-trust-dir`, Extended Epinions (Massa and Avesani 2005), CC BY-SA |
| Epinions-D, earlier 30k subgraph | `edg30000` | as above |

## 9. Scope notes

- `code/` also holds scripts from earlier versions of this project. Scripts not named in
  Sections 4 to 8 back no number in the paper. The exception is `run_e10_scaling.py`, whose
  `make_epn_subsample` function built `epn30000` (see `data/README.md`).
- Single-model tables are one draw of a training procedure that is not bit-deterministic on GPU.
  Each experiment script trains its own model, so two scripts that use seed 42 obtain slightly
  different models (Section V of the paper, run-to-run variation). The complete cohort over ten
  models gives the spread across models.
- Every experiment uses public datasets only.
