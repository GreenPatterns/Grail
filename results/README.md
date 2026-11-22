# Result files

`unified/` holds every result file the paper takes a number from: 74 JSON files and the run
logs of most of them (a log shares its JSON file's name, and `cohort/*.log` are the cohort
jobs' logs). The scripts are in `code/experiments/` unless the name starts with `analyze_`.
Section 6 of the top-level README gives the command behind each file. Numbering follows the
paper: Roman numerals for sections and tables, Arabic numerals for figures and equations, E1 to
E14 for the experiments of its Table III, and letters, S1 to S12 and Fig. S1 and S2 for the
appendices, tables and figures of the supplement. A dash in the experiment column marks a run
outside Table III. The top-level README maps the names used in these files to the terms of the
paper.

| File(s) | Experiment | Paper item | Written by |
|---|---|---|---|
| `cohort/cohort_{otc,alpha,edg60000}_s{42,1..9}.json` | E3 | Table IV, Fig. 4, Table S1 (one file per dataset and run); the AUC range of Section IX | `run_cohort_seeds.py` |
| `cohort_summary.json` | E3 | Table IV, Fig. 4; the cohort rates of the abstract and Sections I, V, VI-B, VI-F and X | `analyze_cohort.py` |
| `loo_flip_table.json` | E4 | Table V, Bitcoin rows; Section V, single-run example | `run_loo_flip_table.py` |
| `loo_flip_edgfull_low.json` | E4 | Table V, Epinions-D row | `run_loo_flip_table.py` |
| `loo_flip_edgfull_modhigh.json` | E4 | Section VI-B and Supplement D: no Epinions-D target of the moderate or high stratum flips | `run_loo_flip_table.py` |
| `causal_regimes.json` | E8 | Table VIII (out, in and sliding rows); Section VI-E, sliding window | `run_causal_regimes.py` |
| `causal_retrain_s5.json` | E8 | Section VI-E, retraining; Table S11 | `run_poison_matrix.py` (part `causal`) |
| `stats_rigor.json` | E5 | Table VI (`decomp_B1`, `decomp_B5`); Supplement F, rank shift (`decision_cost`) and regression on the margin (`dose_response`) | `run_stats_rigor.py` |
| `p0b_trustguard_otc.json` | E11 | Table IX, Fig. 3, Table S8 (TrustGuard, OTC) | `run_trustguard_placement.py` |
| `p0b_trustguard_alpha.json` | E11 | Table IX, Fig. 3, Table S8 (TrustGuard, Alpha) | `run_trustguard_placement.py` |
| `official_signedgcn.json` | E11 | Table IX (SignedGCN) | `run_official_signedgcn.py` |
| `evolvegcn_placement_egcn_{o,h}_{otc,alpha}.json` | E11 | Table IX, Table S9 (EvolveGCN-O and -H) | `run_evolvegcn_placement.py` |
| `dysat_placement_{otc,alpha}.json` | E11 | Table IX, Table S9 (DySAT) | `run_dysat_placement.py` |
| `p0b_main.json` | E1 | Fig. 3 (OTC, Alpha, Epinions-S); Section VI-A, sweep and sign control; Table II, Train column (`n_edges`) | `run_p0b_verify.py` |
| `p0b_edgfull.json` | E1 | Fig. 3 (Epinions-D) | `run_p0b_verify.py` |
| `p2_arch_signature.json` | E11 | Fig. 3 and Section VI-G (the two further models of our own) | `run_p2_arch_signature.py` |
| `p1_sota.json` | E10 | Section VI-F, Fig. S2, Table S5 (OTC, Alpha); Section VII example | `run_p1_efficient.py` |
| `p1_sota_epn.json` | E10 | Section VI-F, Fig. S2, Table S5 (Epinions-S) | `run_p1_efficient.py` |
| `p1_sota_edgfull.json` | E10 | Section VI-F, Fig. S2, Table S5 (Epinions-D) | `run_p1_efficient.py` |
| `common_set_{otc,alpha}.json` | E7 | Table VII, Fig. S1, Section VI-D, Table S2 (per-target rows) | `run_common_set.py` |
| `common_set_summary.json` | E7 | Table VII, Fig. S1, Section VI-D, Table S2 | `analyze_common_set.py` |
| `p0d_otcalpha.json` | E12 | Section VII, time decay (plain mean); Section V, single-run example | `run_p0d_harden.py` |
| `p0d_decayed.json` | E12 | Section VII, time decay (time-decayed mean) | `run_p0d_harden.py` (`GRAIL_REP_MODE=decayed`) |
| `p0_linchpin.json` | E1 | Section VI-A, in-window against out-of-window (OTC, Alpha); Section I | `run_p0_linchpin.py` |
| `p0_linchpin_edgfull.json` | E1 | Section VI-A (Epinions-D); Table II, Epinions-D row (`clean_perf`, `train_edges`) | `run_p0_linchpin.py` |
| `stats_hier_s10.json` | E2 | Section VI-A, paired contrasts; Table S6 | `run_stats_hier.py` |
| `p3_margin.json` | – | Supplement F, other thresholds (`by_threshold`; cited in Section VI-B) | `run_p3_margin.py` |
| `masked_control_s10.json` | E6 | Section VI-C, query-edge-masked control; Table S7 | `run_masked_and_random_sets.py` (`GRAIL_BLOCKS=B`) |
| `reputation_baselines_{otc,alpha,edgfull}.json` | – | Supplement D, low-stratum experiment; Tables S3 and S4; the examples of Sections V and VI-D | `run_reputation_baselines.py` |
| `detector_calib.json` | E14 | Section VII, structural detector, and Table S12 (`detector`); Section V and Supplement F, calibration (`calibration_oos`) | `run_detector_calib.py` |
| `poison_matrix_s5.json` | E9 | Section VI-E, poisoning controls; Table S10; Supplement I, spread of the clean reputation across seeds (`retrain_std_clean`) | `run_poison_matrix.py` (part `matrix`) |
| `p3_defense_adaptive.json` | E13 | Section VII, rater-history weighting and the adaptive attacker | `run_p3_defense.py` |
| `p3_sybil.json` | E13 | Section VII, newly created accounts | `run_p3_sybil.py` |
| `random_source_sets.json` | – | Section VI-F and Supplement F, random source sets: block `C_random_distribution` only (blocks A and B are superseded; the paper cites neither) | `run_masked_and_random_sets.py` (`GRAIL_BLOCKS=C`) |
| `sliding_placebo.json` | E8 | Section VI-E and Supplement F, sliding-window placebo | a one-off run; no current script writes it |
| `p0_linchpin_edg.json` | E1 (check) | Supplement B, the dated 30k-user subgraph | `run_p0_linchpin.py` |
| `p0b_edg.json` | E1 (check) | Supplement B, the dated 30k-user subgraph | `run_p0b_verify.py` |
| `epn_clean.json` | – | none: holds null values; Table II's Epinions-S AUC and MCC come from `epn_clean.log` | one call of `train_model('epn30000')` |

## Naming and notes

- `edgfull` and `edg60000` denote Epinions-D, the complete graph of the genuinely dated
  statements; `edg` alone denotes the earlier dated 30k-user subgraph (Supplement B only).
- `epn30000` or `epn` denotes the SNAP Epinions 30k-user subgraph, which has no timestamps and
  which the paper calls Epinions-S.
- `_s10` and `_s5` mark results pooled over ten or five runs (independently trained models).
- In `detector_calib.json`, the field `held_out_snapshots` reads `s7-s9`, but the calibration
  edges are those of snapshot s7 whose two endpoints appear in the processed window (1,046 on
  OTC, 883 on Alpha); the paper describes them so.
- In `poison_matrix_s5.json`, the placebo control `dis_unrelated` (Table S10) puts the five
  distrust edges on the first of the 24 targets (`victims[0]` in `run_poison_matrix.py`). Its
  `dR_vs_none` and `flip_pct` average over all 24 targets and five seeds, that target included.
- Table II's AUC and MCC of the Bitcoin graphs are printed after training in every seed-42 log
  (the `MCC=..., AUC=...` line), for example `loo_flip_table.log`.
- Log lines keep the code's names (top-level README, Section 1). In `p0_linchpin_edgfull.log`
  and `p0_linchpin_edg.log`, "appended (paper)" is the out-of-window placement, which the first,
  flawed evaluation used.
- Paths inside the logs are anonymized: `<ps-env>` and `<dysat-env>` stand for the two Python
  environments and `<clones>` for the folder holding the official releases.
