# Result files

`unified/` holds every result file the paper takes a number from: 74 JSON files and the run
logs of most of them (a log shares its JSON file's name, and `cohort/*.log` are the cohort
jobs' logs). The scripts are in `code/experiments/` unless the name starts with `analyze_`.
Section 6 of the top-level README gives the command behind each file. Numbering follows the
paper: Roman numerals for sections and tables, E1 to E14 for the experiments of its Table III,
and letters and S1 to S12 for the appendices and tables of the supplement. The top-level README
maps the names used in these files to the terms of the paper.

| File(s) | Paper item | Written by |
|---|---|---|
| `cohort/cohort_{otc,alpha,edg60000}_s{42,1..9}.json` | Table IV, Fig. 3, Table S1 (one file per dataset and model) | `run_cohort_seeds.py` |
| `cohort_summary.json` | Table IV, Fig. 3, abstract, Sections I, VI-B and X rates | `analyze_cohort.py` |
| `loo_flip_table.json` | Table V, Bitcoin rows | `run_loo_flip_table.py` |
| `loo_flip_edgfull_low.json` | Table V, Epinions-D, low stratum | `run_loo_flip_table.py` |
| `loo_flip_edgfull_modhigh.json` | Table V caption, Epinions-D, moderate and high strata | `run_loo_flip_table.py` |
| `causal_regimes.json` | Table VIII: out, in and sliding rows | `run_causal_regimes.py` |
| `causal_retrain_s5.json` | Section VI-E retraining; Table S11 | run_poison_matrix.py (causal part) |
| `stats_rigor.json` | Table VI (decomp_B1, decomp_B5); Section VI-B rank shift (decision_cost); Supplement F regression (dose_response) | `run_stats_rigor.py` |
| `p0b_trustguard_otc.json` | Table IX, Fig. 2, Table S8 (TrustGuard, OTC) | `run_trustguard_placement.py` |
| `p0b_trustguard_alpha.json` | Table IX, Fig. 2, Table S8 (TrustGuard, Alpha) | `run_trustguard_placement.py` |
| `official_signedgcn.json` | Table IX (SignedGCN) | `run_official_signedgcn.py` |
| `evolvegcn_placement_egcn_{o,h}_{otc,alpha}.json` | Table IX, Table S9 (EvolveGCN-O and -H) | `run_evolvegcn_placement.py` |
| `dysat_placement_{otc,alpha}.json` | Table IX, Table S9 (DySAT) | `run_dysat_placement.py` |
| `p0b_main.json` | Fig. 2 (OTC, Alpha, Epinions-S); Section VI-A sweep and sign controls | `run_p0b_verify.py` |
| `p0b_edgfull.json` | Fig. 2 (Epinions-D) | `run_p0b_verify.py` |
| `p2_arch_signature.json` | Fig. 2 and Section VI-G (the two further encoders of our own) | `run_p2_arch_signature.py` |
| `p1_sota.json` | Section VI-F, Fig. S2, Table S5 (OTC, Alpha) | `run_p1_efficient.py` |
| `p1_sota_epn.json` | Section VI-F, Fig. S2, Table S5 (Epinions-S) | `run_p1_efficient.py` |
| `p1_sota_edgfull.json` | Section VI-F, Fig. S2, Table S5 (Epinions-D) | `run_p1_efficient.py` |
| `common_set_{otc,alpha}.json` | Table VII, Fig. S1, Section VI-D, Table S2 (per-target rows) | `run_common_set.py` |
| `common_set_summary.json` | Table VII, Fig. S1, Section VI-D, Table S2 | `analyze_common_set.py` |
| `p0d_otcalpha.json` | Section VII, time decay (mean reputation) | `run_p0d_harden.py` |
| `p0d_decayed.json` | Section VII, time decay (time-decayed mean) | run_p0d_harden.py (GRAIL_REP_MODE=decayed) |
| `p0_linchpin.json` | Section VI-A, in-window against out-of-window (OTC, Alpha) | `run_p0_linchpin.py` |
| `p0_linchpin_edgfull.json` | Section VI-A (Epinions-D); Table II Epinions-D row (clean_perf, train_edges) | `run_p0_linchpin.py` |
| `stats_hier_s10.json` | Section VI-A paired contrasts; Table S6 | `run_stats_hier.py` |
| `p3_margin.json` | Section VI-B, other thresholds (by_threshold) | `run_p3_margin.py` |
| `masked_control_s10.json` | Section VI-C query-edge-masked control; Table S7 | run_masked_and_random_sets.py (GRAIL_BLOCKS=B) |
| `reputation_baselines_{otc,alpha,edgfull}.json` | Supplement D low-stratum run; Tables S3 and S4 | `run_reputation_baselines.py` |
| `detector_calib.json` | Section V calibration (calibration_oos); Section VII detector and Table S12 (detector) | `run_detector_calib.py` |
| `poison_matrix_s5.json` | Section VI-E poisoning controls; Table S10 | `run_poison_matrix.py` |
| `p3_defense_adaptive.json` | Section VII rater-history weighting and the adaptive attacker | `run_p3_defense.py` |
| `p3_sybil.json` | Section VII, fresh vs. established accounts | `run_p3_sybil.py` |
| `random_source_sets.json` | Section VI-F and Supplement F random source sets: block C_random_distribution only (blocks A and B are superseded; the paper cites neither) | run_masked_and_random_sets.py (GRAIL_BLOCKS=C) |
| `sliding_placebo.json` | Section VI-E and Supplement F sliding-window placebo | a one-off run; no current script writes it |
| `p0_linchpin_edg.json` | Supplement B, the earlier 30k dated subgraph | `run_p0_linchpin.py` |
| `p0b_edg.json` | Supplement B, the earlier 30k dated subgraph | `run_p0b_verify.py` |
| `epn_clean.json` | none: holds null values; the Table II Epinions-S row comes from epn_clean.log | one call of train_model('epn30000') |

## Naming

- `edgfull` and `edg60000` denote the complete genuinely dated Epinions graph, which the paper
  calls Epinions-D; `edg` alone denotes the earlier 30k-user dated subgraph (Supplement B only).
- `epn30000` or `epn` denotes the SNAP Epinions 30k-node subgraph, which has no timestamps and
  which the paper calls Epinions-S.
- `_s10` and `_s5` mark results pooled over ten or five independently trained models.
- In `detector_calib.json`, the field `held_out_snapshots` reads `s7-s9`, but the calibration
  edges are those of snapshot s7 whose two endpoints appear in the processed window (1,046 on
  OTC, 883 on Alpha); the paper describes them so.
- Table II's clean AUC and MCC of the Bitcoin graphs are printed at the start of every seed-42
  log, for example `loo_flip_table.log`.
- Paths inside the logs are anonymized: `<ps-env>` and `<dysat-env>` stand for the two Python
  environments and `<clones>` for the folder holding the official releases.
