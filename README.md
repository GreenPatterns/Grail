# Artifact: Silent No-Ops and Self-Conditioning

This folder holds the code, figures, result files and supplement of the anonymous submission
*Silent No-Ops and Self-Conditioning: Two Evaluation Artifacts in Robustness Tests of Temporal
Trust GNNs*. It also holds the datasets the code reads and the instructions for running every
experiment. The model under test is GDTE (GNN for Dynamic Trust Evaluation), the paper's
temporal trust GNN.

You can check the paper at three depths:

| Depth | What you run | Time | Hardware |
|---|---|---|---|
| 1. Stored results | Regenerate every summary, table row and figure from the stored result files (Section 4) | about 1 minute | CPU |
| 2. Placement demo | Show that an attack edge appended after the processed window (out-of-window) is a silent, label-independent no-op, while the same edge inside the window (in-window) is not (Section 5) | about 1 minute (about 9 on a CPU) | 1 GPU, or a CPU |
| 3. Full reruns | Rerun any experiment behind a table, figure or number (Sections 6 and 7) | minutes to hours per experiment | 1 GPU |

All commands below run from the root of this artifact, the folder that holds this README.

## 1. Contents

```text
README.md                 this file
supplement.pdf            the paper's online supplement (appendices A to L, Tables S1 to S18, Fig. S1 and S2)
env.sh                    sets GRAIL_DATA_ROOT and PYTHONPATH; source it before every command
requirements.txt          main environment (Python 3.10, PyTorch 2.4, PyG 2.6)
requirements-dysat.txt    separate environment for the official DySAT release (Python 3.7, TensorFlow 1.15)
code/
  mycode/                 GDTE (class DGTEN in dgten.py, with SL.py and s3r.py), training (gcn3.py),
                          reputation and attack scoring (trust_influence.py), reputation
                          baselines, adapted attacks
  common/                 shared helpers: training entry point, target selection, paths
  experiments/            one script per experiment (run_*.py), the figure scripts
                          (make_dsn_figures.py and the two panel modules it uses), and two
                          helpers of the case study E15 (report_trustguard_casestudy.py prints
                          its tables; trustguard_robust_gpu.py runs TrustGuard's robust
                          aggregation on the GPU)
  analyze_*.py, make_*_table.py, make_tables.py, make_stats_paragraph.py
                          summaries and LaTeX table rows computed from the result files
  silent_noop_demo.py     the placement demo of Section 5
  check_dsn_numbers.py    checks 119 numbers in the paper's LaTeX source and 130 in the
                          supplement's against the result files
  make_epinions_dated.py  builds Epinions-D and the other dated Epinions graphs from the raw release
  tests/                  unit tests (pytest)
data/
  README.md               sources, licenses, file formats and how every file was built
  cyberdata/              the processed datasets the code reads, plus raw/ (dated Epinions release)
figures/                  Fig. 3 and 4 of the paper and Fig. S1 and S2 of the supplement (written by
                          make_dsn_figures.py); fig_problem.pdf, with its TikZ source
                          fig_problem_src.tex, is Fig. 1, and fig_system.pdf, from fig_system_src.tex, is Fig. 2; and
                          fig_evidence.pdf is an older combined figure that is not in the paper
results/
  README.md               maps every result file to the paper item and experiment it backs
  unified/                the result files of the paper's first version (74 JSON files and
                          their run logs)
  *.json, *.log           the result files of the four revision experiments and their logs: the
                          case study E15, the paired sliding-window test (E8), retraining with
                          both scores (E8, E9) and the budget ablation (E4)
```

The paper numbers its sections and tables with Roman numerals (Section VI-B, Table IV), its
figures and equations with Arabic numerals (Fig. 4, Eq. 1), and its experiments E1 to E15 (its
Table III). The supplement has appendices A to L (Supplement C), Tables S1 to S18 and Fig. S1
and S2. This README uses the same numbering.

**Names in the code and the result files.** The code and the result files predate the paper's
final terms. Docstrings, the printed table rows and the demo use the paper's terms; JSON keys,
variable names and log lines keep the code's names. Read them as follows:

| In code and result files | In the paper |
|---|---|
| `naive`, `total` | total score |
| `legit`, `preexist`, `prop`, leakage-free | propagation-only score |
| `appended` (in older logs also "appended (paper)", the placement of the first, flawed evaluation) | out-of-window placement |
| `processed`, `inwindow` | in-window placement |
| `label_delta`, label-Δ | label sensitivity Λ |
| `near_gate`, gate | low-reputation targets (clean score in [0.5, 0.6)), threshold |
| `counterfactual` | optimized source; in method names, an exact shift |
| `expert` | the model-free heuristic |
| `victim`, `GRAIL_NVICTIMS` | target, number of targets |
| `mechanical`, `self_edge` | dilution, self-conditioning (Eq. 5, Table VI) |
| TM-A, TM-B | frozen embeddings, re-embedding |
| seed, model (one trained model) | run |
| SOTA baselines | adapted published attacks |
| `DGTEN` (class in `mycode/dgten.py`) | GDTE, the paper's simplified variant of DGTEN |
| `edg60000`, `edgfull`, `edt` (column key in `make_trivial_table.py`) | Epinions-D |
| `epn30000`, `epn` | Epinions-S |
| `T`, `train_time_slots` | number of processed snapshots W |

## 2. Requirements

**Hardware.** A Linux machine with one NVIDIA GPU. The experiments ran on NVIDIA RTX A6000 and
L40S GPUs (48 GB). The Bitcoin experiments use about 3 to 5 GB of GPU memory. The two Epinions
graphs were only run on 48 GB GPUs, and smaller GPUs are untested for them. Step 1 of the table
above needs only a CPU; step 2 runs on a small GPU or, more slowly, on a CPU. The DySAT test
runs on CPU.

**Software.** Two conda environments:

- *Main environment:* Python 3.10.14, PyTorch 2.4.0 with CUDA 12.4, PyTorch Geometric 2.6.1,
  torch-scatter 2.1.2 and the pinned packages in `requirements.txt`. Every script except the DySAT
  test runs here.
- *DySAT environment:* Python 3.7.16 with TensorFlow 1.15.5, networkx 1.11 and the pinned packages
  in `requirements-dysat.txt`. It is needed only for the official DySAT test (Section 7).

**Optional.** The figure scripts use the Liberation Sans font; without it, matplotlib falls back
to its default font and the figures differ slightly in appearance. Rebuilding
`figures/fig_problem.pdf` (Fig. 1) from `figures/fig_problem_src.tex` needs a LaTeX installation with TikZ.

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
`test_seed_propagation.py` skip themselves, and 36 tests run.

## 4. Regenerate the tables and figures from the stored results (CPU, about 1 minute)

```bash
source env.sh
python code/analyze_cohort.py                    # rewrites cohort_summary.json from the 30 cohort files
python code/analyze_common_set.py                # rewrites common_set_summary.json
python code/make_tables.py                       # Table IV, Table S1, official-release rows, common-set lines
python code/make_tables.py official_compact official_supplement   # the numbers of Table IX; Table S9
python code/make_trivial_table.py                # Table S5; its shares are in Section VI-G and Fig. S2
python code/make_reputation_table.py             # Tables S3 and S4
python code/make_stats_paragraph.py              # the Section VI-B paired statistics (E2) and Table S6
python code/experiments/make_dsn_figures.py figures   # Fig. 3, 4, S1 and S2 -> figures/fig_{placement,cohort,functions,sources}.pdf
python code/experiments/report_trustguard_casestudy.py   # E15: the values of Tables S16 to S18 per result file (--tables: two summary tables)
```

The summaries and figures are written over the shipped copies in `results/unified/` and
`figures/`. On the machine that produced the results, the two summaries come out identical to
the shipped files, and the figures show the same data as the shipped PDFs. The table rows print
as LaTeX with the paper's row labels, and a comment line names the table of each block; the
case-study report prints plain text.
Python rounds halves to even, so a rate of exactly 62.5% (10 of 16 targets) prints as 62 where
Table S5 shows 63.

`make_dsn_figures.py` draws the panels with the functions of `make_evidence_figure.py` and
`make_cohort_figure.py` and relabels them with the paper's terms. Run on their own, those two
scripts write versions with the older labels: `make_evidence_figure.py` writes a combined
three-panel figure (`figures/fig_evidence.pdf`, not in the paper), and `make_cohort_figure.py`
writes over `figures/fig_cohort.pdf`, the paper's Fig. 4. Rerun `make_dsn_figures.py` afterwards.

`code/check_dsn_numbers.py` checks 119 numbers in the paper's text and 130 in the supplement's
against the result files, those of the revision experiments in `results/` included. It reads the
LaTeX sources of the paper and the supplement, which are not part of this artifact:
`GRAIL_PAPER_TEX=<paper>.tex python code/check_dsn_numbers.py` takes the supplement from
`supplement.tex` next to the paper, or from the file that `GRAIL_SUPP_TEX` names, and prints
`119/119 number checks pass in the paper, 130/130 in the supplement`.

## 5. Placement demo (one GPU, about 1 minute; a CPU takes about 9 minutes)

```bash
source env.sh
GRAIL_DATASETS=otc python code/silent_noop_demo.py
```

The demo trains GDTE on Bitcoin-OTC and scores one distrust edge from a random source for each
of eight strata targets, at two placements: out-of-window, appended after the processed window,
and in-window, at the end of the last processed snapshot s6. For each placement it reports the
mean total shift ΔR and the label sensitivity Λ, the mean of |ΔR(distrust) − ΔR(trust)| (Eq. 3).
A run on CPU (`CUDA_VISIBLE_DEVICES=""`) printed:

```text
=== Silent-No-Op Diagnostic on otc (n=8 targets) ===
  out-of-window: mean ΔR = +0.0030   Λ = 0.000000
      in-window: mean ΔR = -0.2566   Λ = 0.255307

  out-of-window edge is a silent, label-independent no-op: True
  in-window edge is label-sensitive and damaging:          True

  => A 'robust' verdict from an out-of-window test is a silent no-op (Artifact 1).
```

Λ is exactly zero out-of-window in every run, because the attack edge's label never reaches the
encoder (Remark 1). The in-window row varies between runs, because training is not
bit-deterministic: two earlier runs on one RTX A6000 printed an in-window ΔR of −0.2078 and
−0.2093 with Λ = 0.282234 and 0.283811 (out-of-window ΔR +0.0011, Λ = 0). The in-window shift is
the total shift; Section III-B of the paper separates its propagation-only part.

## 6. Rerun the experiments

**Conventions.**

- Run every script from the artifact root after `source env.sh`, and choose the GPU with
  `CUDA_VISIBLE_DEVICES`.
- `GRAIL_DATASETS` selects the datasets: `otc` and `alpha` (Bitcoin-OTC and Bitcoin-Alpha),
  `edg60000` (Epinions-D, every statement with a real date) and `epn30000` (Epinions-S, the
  30k-user subgraph of the SNAP release).
- Scripts write JSON to `results/unified/` and log to standard output. `GRAIL_OUT` (or the
  script's own output variable, shown below) names the output file. The four revision scripts
  (`run_trustguard_casestudy.py`, `run_prospective_paired.py`, `run_retrain_prop.py` and
  `run_budget_ablation.py`) write to `results/` itself, marked `../` in the table below;
  `GRAIL_OUT_DIR` (for E15, `--out`) redirects them. The commands below overwrite the shipped
  file of the same name, so copy `results/` first if you want to compare.
- `GRAIL_SEEDS` lists the training seeds, which are genuine: each seed trains an independent
  model, a run of the paper.
- GPU training is not bit-deterministic, so a rerun reproduces a result up to training noise,
  not digit for digit. The spread across the ten cohort runs (Table S1) shows the size of that
  noise.
- Times are the wall-clock spans recorded in the shipped run logs, rounded. They were measured on
  shared GPUs and exclude any time spent before a script's first log line.

**Main paper.**

| Paper item | Command (prefix each with `source env.sh`) | Output in `results/unified/` | Logged time |
|---|---|---|---|
| Table IV, Fig. 4, Table S1 (complete cohort, E3); the cohort rates of the abstract and Sections I, V, VI-C, VI-G, IX and X | `GRAIL_DATASETS=otc GRAIL_SEEDS=42,1,2,3,4,5,6,7,8,9 python code/experiments/run_cohort_seeds.py`, then the same with `alpha` and `edg60000`; then `python code/analyze_cohort.py`, `python code/make_tables.py cohort cohort_per_model` and `python code/experiments/make_dsn_figures.py figures` | `cohort/cohort_{otc,alpha,edg60000}_s{42,1..9}.json`, `cohort_summary.json` | per run: OTC 0.3 h, Alpha 0.2 h, Epinions-D 0.9 h |
| Table II, AUC and MCC of OTC and Alpha | the `MCC=..., AUC=...` line that `train_model` prints after training, in every seed-42 log, e.g. `loo_flip_table.log` | (log line) | |
| Table II, Train column of OTC, Alpha and Epinions-S | the Fig. 3 runs below | `p0b_main.json` (`n_edges`) | |
| Table II, AUC and MCC of Epinions-S | one call of `experiment_common.train_model('epn30000')`; no script writes it | `epn_clean.log` (the matching `epn_clean.json` holds null values) | 0.2 h |
| Table II, Epinions-D row | from the Section VI-B run on `edg60000` below (`clean_perf`, `train_edges`) and `data/cyberdata/edg60000-provenance.json` | `p0_linchpin_edgfull.json` | |
| Table V (strata and budgets, E4) | `GRAIL_DATASETS=otc,alpha python code/experiments/run_loo_flip_table.py`; `GRAIL_DATASETS=edg60000 GRAIL_STRATA=low GRAIL_OUT=loo_flip_edgfull_low.json python code/experiments/run_loo_flip_table.py`; the same with `GRAIL_STRATA=moderate,high GRAIL_OUT=loo_flip_edgfull_modhigh.json` | `loo_flip_table.json`, `loo_flip_edgfull_low.json`, `loo_flip_edgfull_modhigh.json` | 0.1 h; 0.3 h; 0.5 h |
| Supplement K and Table S15 (budget ablation of E4); Section VI-C, the fall of the propagation-only flip rate with the budget; Section VI-D, the raters' share of the change; Section V, the rerun's 64% | `GRAIL_DATASETS=otc,alpha python code/experiments/run_budget_ablation.py` | `../budget_ablation_{otc,alpha}.json` | 0.2 h per dataset |
| Table VIII, Sections I and VI-F (paired sliding-window test over ten runs, E8; the script writes the criterion fixed in advance into the output file before any outcome); Supplement J and Table S14 | `GRAIL_DATASETS=otc,alpha python code/experiments/run_prospective_paired.py` | `../prospective_paired_{otc,alpha}.json` | OTC 0.2 h, Alpha 0.1 h |
| Section VI-F, frozen embeddings (E8, out-of-window rows); Supplement J and Table S13 (the first, single-run placement comparison) | `GRAIL_DATASETS=otc,alpha python code/experiments/run_causal_regimes.py` | `causal_regimes.json` | 0.05 h |
| Section VI-F, the earlier total-score retraining run (E8); Supplement I and Table S11, columns "Earlier" | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4 GRAIL_NVICTIMS=24 GRAIL_POISON_PARTS=causal GRAIL_POISON_OUT=causal_retrain_s5.json python code/experiments/run_poison_matrix.py` | `causal_retrain_s5.json` | 0.1 h |
| Sections I and VI-F, retraining (E8) and poisoning controls (E9) with both scores; Table S10, columns T' and P'; Table S11 except the columns "Earlier" | `GRAIL_DATASETS=otc,alpha python code/experiments/run_retrain_prop.py` | `../retrain_prop.json` | 0.3 h |
| Table VI (decomposition, E5); Supplement F, rank shift and regression on the margin | `GRAIL_DATASETS=otc,alpha python code/experiments/run_stats_rigor.py` | `stats_rigor.json` (`decomp_B1`, `decomp_B5`, `decision_cost`, `dose_response`) | 0.1 h |
| Table IX (official releases, E11) | Section 7 below; then `python code/make_tables.py official_compact` | see Section 7 | |
| Section VI-A, Supplement L and Tables S16 to S18 (TrustGuard case study, E15); its mentions in Sections I, IV, VI-H and X | with the TrustGuard clone of Section 7: `TRUSTGUARD_DIR=<clone>/code python code/experiments/run_trustguard_casestudy.py --dataset otc`, the same with `--features node2vec`, and both again with `--attack-degree test --max-low 0`; then these four runs with `--dataset alpha`; then `python code/experiments/report_trustguard_casestudy.py`. The script runs robust aggregation through `trustguard_robust_gpu.py`, and `python code/experiments/trustguard_robust_gpu.py` checks that port against its copy of the released function on random graphs | `../trustguard_casestudy_{otc,alpha}{,_node2vec}{,_testdeg}.json` | 0.6 to 1.4 h per file |
| Fig. 3 (label sensitivity by snapshot, E1 and E11); Section VI-B, sweep and sign control; Supplement F, sweep ranges | `GRAIL_DATASETS=otc,alpha,epn30000 GRAIL_OUT=p0b_main.json python code/experiments/run_p0b_verify.py`; `GRAIL_DATASETS=edg60000 GRAIL_OUT=p0b_edgfull.json python code/experiments/run_p0b_verify.py`; `GRAIL_DATASETS=otc,alpha python code/experiments/run_p2_arch_signature.py`; TrustGuard as in Section 7; then `python code/experiments/make_dsn_figures.py figures` | `p0b_main.json`, `p0b_edgfull.json`, `p2_arch_signature.json`, `p0b_trustguard_{otc,alpha}.json` | edg60000: 0.15 h |
| Section VI-G, Fig. S2, Table S5 (source selection, E10); the Section X example | `GRAIL_DATASETS=otc,alpha GRAIL_P1_OUT=p1_sota.json python code/experiments/run_p1_efficient.py`; the same with `GRAIL_DATASETS=epn30000 GRAIL_P1_OUT=p1_sota_epn.json` and with `GRAIL_DATASETS=edg60000 GRAIL_P1_OUT=p1_sota_edgfull.json`; then `python code/make_trivial_table.py` | `p1_sota.json`, `p1_sota_epn.json`, `p1_sota_edgfull.json` | 1.1 h; 2.5 h; 0.65 h |
| Table VII, Fig. S1, Section VI-E, Table S2 (common set, E7) | `GRAIL_DATASETS=otc python code/experiments/run_common_set.py`, then with `alpha`; then `python code/analyze_common_set.py` and `python code/make_tables.py common` | `common_set_{otc,alpha}.json`, `common_set_summary.json` | 0.85 h; 0.5 h |
| Section VII, time-decayed mean (E12); the plain-mean run also gives Section V's single-run example | `GRAIL_DATASETS=otc,alpha GRAIL_OUT=p0d_otcalpha.json python code/experiments/run_p0d_harden.py`; the same with `GRAIL_REP_MODE=decayed GRAIL_OUT=p0d_decayed.json` | `p0d_otcalpha.json`, `p0d_decayed.json` | |
| Section VI-B, in-window against out-of-window (E1) | `GRAIL_DATASETS=otc,alpha python code/experiments/run_p0_linchpin.py`; `GRAIL_DATASETS=edg60000 GRAIL_P0_OUT=p0_linchpin_edgfull.json python code/experiments/run_p0_linchpin.py`; Epinions-S from `p0b_main.json` (Fig. 3 row) | `p0_linchpin.json`, `p0_linchpin_edgfull.json` | edg60000: 0.65 h |
| Section VI-B, paired contrasts over ten runs (E2); Table S6 | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4,5,6,7,8,9 GRAIL_OUT=stats_hier_s10.json python code/experiments/run_stats_hier.py`; then `python code/make_stats_paragraph.py` | `stats_hier_s10.json` | 1.25 h |
| Supplement F, other thresholds (cited in Section VI-C) | `GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_margin.py` | `p3_margin.json` (`by_threshold`) | |
| Section VI-D, query-edge-masked control (E6); Table S7 | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4,5,6,7,8,9 GRAIL_BLOCKS=B GRAIL_OUT=masked_control_s10.json python code/experiments/run_masked_and_random_sets.py` | `masked_control_s10.json` | 0.8 h |
| Supplement D, low-stratum experiment of the reputation functions; Tables S3 and S4; the examples of Sections V and VI-E | `GRAIL_DATASETS=otc GRAIL_OUT=reputation_baselines_otc.json python code/experiments/run_reputation_baselines.py`, then the same for `alpha`; `GRAIL_DATASETS=edg60000 GRAIL_STRATA=low GRAIL_FG_WARM=1 GRAIL_OUT=reputation_baselines_edgfull.json python code/experiments/run_reputation_baselines.py` (`GRAIL_FG_WARM=1` starts each attacked Fairness and Goodness solve from the clean fixed point: same fixed point, fewer iterations); then `python code/make_reputation_table.py` | `reputation_baselines_{otc,alpha,edgfull}.json` | 0.1 h each; 0.25 h |
| Section V and Supplement F, calibration (Brier, ECE); Section VII, structural detector (E14); Table S12 | `GRAIL_DATASETS=otc,alpha python code/experiments/run_detector_calib.py` | `detector_calib.json` (`calibration_oos`, `detector`) | 0.05 h |
| Section VI-F and Supplement I, poisoning controls (E9, total score); Table S10, column T | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42,1,2,3,4 GRAIL_NVICTIMS=24 GRAIL_POISON_OUT=poison_matrix_s5.json python code/experiments/run_poison_matrix.py` | `poison_matrix_s5.json` | 0.2 h |
| Section VII, rater-history weighting, the adaptive attacker and newly created accounts (E13); Supplement I, random accounts under the weighting | `GRAIL_DATASETS=otc,alpha GRAIL_OUT=p3_defense_adaptive.json python code/experiments/run_p3_defense.py`; `GRAIL_DATASETS=otc,alpha python code/experiments/run_p3_sybil.py` | `p3_defense_adaptive.json`, `p3_sybil.json` | 0.15 h |

**Supplement only.**

| Supplement item | Command or source | Output in `results/unified/` |
|---|---|---|
| Supplement B, the dated 30k-user subgraph | `GRAIL_DATASETS=edg30000 GRAIL_P0_OUT=p0_linchpin_edg.json python code/experiments/run_p0_linchpin.py`; `GRAIL_DATASETS=edg30000 GRAIL_OUT=p0b_edg.json python code/experiments/run_p0b_verify.py` | `p0_linchpin_edg.json`, `p0b_edg.json` |
| Supplement C, Table S1 (complete cohort per run) | from the cohort files; `python code/make_tables.py cohort_per_model` | `cohort/*.json` |
| Supplement F, random source sets (78 to 94%, 88% quartile) | `GRAIL_DATASETS=otc,alpha GRAIL_SEEDS=42 GRAIL_BLOCKS=C python code/experiments/run_masked_and_random_sets.py` (the block uses the seed-42 model only) | block `C_random_distribution` of `random_source_sets.json` |
| Supplement J, sliding-window placebo of the first, single run | a one-off run; no current script writes this file | `sliding_placebo.json` |
| Supplement F, regression on the margin | the Table VI run | `stats_rigor.json` (`dose_response`) |
| Supplement H, Table S8 (TrustGuard sweep) | Section 7 | `p0b_trustguard_{otc,alpha}.json` |
| Supplement H, Table S9 (label-blind releases) | Section 7; then `python code/make_tables.py official_supplement` | `evolvegcn_placement_*.json`, `dysat_placement_*.json` |

Two cautions about the shipped files. Blocks `A_fixed_cohort` and `B_query_edge_masked` of
`random_source_sets.json` come from an older run whose seeds did not produce independent models;
`cohort/` and `masked_control_s10.json` replace them, and the paper cites neither block. The
`stats_rigor.log` and `stats_rigor_regen.log` runs both wrote `stats_rigor.json`.

## 7. Official releases

The placement test of E11 (Table IX, Tables S8 and S9) runs on four official releases, each
cloned at the commit we used; the case study E15 (Section 6) uses the same TrustGuard clone. The
clones are not included; each release keeps its own license.

| Release | Clone | Command | Output in `results/unified/` |
|---|---|---|---|
| TrustGuard (Wang et al., TDSC 2024) | `git clone https://github.com/Jieerbobo/TrustGuard && git -C TrustGuard checkout 5458c0e240fa5a48f449a11dde4ba3d1d53227fc` | `TRUSTGUARD_DIR=<clone>/code GRAIL_DATASETS=otc GRAIL_OUT=p0b_trustguard_otc.json python code/experiments/run_trustguard_placement.py`, then the same with `alpha` | `p0b_trustguard_{otc,alpha}.json` |
| EvolveGCN-O and EvolveGCN-H (Pareja et al., AAAI 2020) | `git clone https://github.com/IBM/EvolveGCN && git -C EvolveGCN checkout 90869062bbc98d56935e3d92e1d9b1b4c25be593`; create its input files from the bundled data: `tail -n +2 data/cyberdata/bitcoinotc.csv > <clone>/data/soc-sign-bitcoinotc.csv` and the same for `bitcoinalpha` (these are the files our runs used) | `EVOLVEGCN_DIR=<clone> EGCN_MODEL=egcn_o GRAIL_DATASETS=bitcoinotc GRAIL_OUT=evolvegcn_placement_egcn_o_otc.json python code/experiments/run_evolvegcn_placement.py`; repeat for `bitcoinalpha` and for `EGCN_MODEL=egcn_h` | `evolvegcn_placement_egcn_{o,h}_{otc,alpha}.json` |
| DySAT (Sankar et al., WSDM 2020) | `git clone https://github.com/aravindsankar28/DySAT && git -C DySAT checkout 777f290dcef38c28c390ba4642ce45621e7cedb4`; use one clone per dataset to run both at once | in the DySAT environment: `DYSAT_DIR=<clone> GRAIL_DATASETS=otc GRAIL_OUT=dysat_placement_otc.json python code/experiments/run_dysat_placement.py`, then the same with `alpha` | `dysat_placement_{otc,alpha}.json` |
| SignedGCN (PyTorch Geometric's implementation) | none; ships with PyTorch Geometric | `GRAIL_DATASETS=otc,alpha python code/experiments/run_official_signedgcn.py` | `official_signedgcn.json` |

Each release runs with its own trainer and configuration. The harness scripts apply only the
compatibility steps listed in their docstrings and in Supplement H, such as PyYAML's `Loader`
argument for EvolveGCN and a list-returning `map` for DySAT's Python 2 code. The case-study
script lists its deviations from the TrustGuard release in its docstring and in Supplement L.
EvolveGCN and DySAT never read a rating's sign, so their diagnostic is the change of the
target's embedding rather than Λ. The EvolveGCN and DySAT harnesses restore the epoch that the release's own selection rule
picks, not the last epoch. The shipped training logs (`evolvegcn_egcn_*.log`,
`dysat_{otc,alpha}.log`) hold each selected epoch's test score.

## 8. Data

`data/cyberdata/` holds the processed files the code reads. `data/README.md` gives each file's
source, license and exact construction. In short:

| Dataset | Key | Source |
|---|---|---|
| Bitcoin-OTC | `otc` | SNAP `soc-sign-bitcoinotc` (Kumar et al., ICDM 2016) |
| Bitcoin-Alpha | `alpha` | SNAP `soc-sign-bitcoinalpha` (Kumar et al., ICDM 2016) |
| Epinions-S (SNAP Epinions, 30k-user subgraph) | `epn30000` | SNAP `soc-sign-epinions` (Leskovec et al. 2010), no timestamps |
| Epinions-D (dated Epinions, every statement with a real date) | `edg60000` | Network Repository `soc-epinions-trust-dir`, Extended Epinions (Massa and Avesani 2005), CC BY-SA |
| Dated 30k-user subgraph (Supplement B check only) | `edg30000` | as above |

## 9. Scope notes

- `code/` also holds scripts from earlier versions of this project. Scripts not named in
  Sections 4 to 8 back no number in the paper. The E1 to E10 in their names, docstrings and log
  lines (`run_e1_*.py` to `run_e10_*.py`, `run_all.py`, `summarize_results.py` and the `run_E*`
  functions of `common/experiment_common.py`) are IDs of that earlier version, not the paper's
  E1 to E15. The exception is `run_e10_scaling.py`, whose `make_epn_subsample` function built
  `epn30000` (see `data/README.md`).
- Single-run tables are one draw of a training procedure that is not bit-deterministic on GPU.
  Each experiment script trains its own run, so two scripts that use seed 42 obtain slightly
  different models (Section V of the paper, single-run results). The complete cohort over ten
  runs gives the spread across runs.
- Every experiment uses public datasets only.
