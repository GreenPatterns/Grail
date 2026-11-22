# Result files

`unified/` holds the result files of the first DSN version that the paper takes a number from: 74
JSON files and the run logs of most of them (a log shares its JSON file's name, and
`cohort/*.log` are the cohort jobs' logs). The result files of the revision experiments of
2026-11-22 (and their logs) sit next to this README, in `results/` itself; those already in the
paper are the last rows of the table, and `code/check_dsn_numbers.py` reads them from there. The scripts are in
`code/experiments/` unless the name starts with `analyze_`. Section 6 of the top-level README gives the command behind each file
of `unified/`; the docstring of each revision script gives its command. Numbering follows the
paper: Roman numerals for sections and tables, Arabic numerals for figures and equations, E1 to
E15 for the experiments of its Table III, and letters, S1 to S18 and Fig. S1 and S2 for the
appendices, tables and figures of the supplement. A dash in the experiment column marks a run
outside Table III. The top-level README maps the names used in these files to the terms of the
paper.

| File(s) | Experiment | Paper item | Written by |
|---|---|---|---|
| `cohort/cohort_{otc,alpha,edg60000}_s{42,1..9}.json` | E3 | Table IV, Fig. 4, Table S1 (one file per dataset and run); the AUC range of Section IX | `run_cohort_seeds.py` |
| `cohort_summary.json` | E3 | Table IV, Fig. 4; the cohort rates of the abstract and Sections I, V, VI-C, VI-G and X | `analyze_cohort.py` |
| `loo_flip_table.json` | E4 | Table V, Bitcoin rows; Section V, single-run example | `run_loo_flip_table.py` |
| `loo_flip_edgfull_low.json` | E4 | Table V, Epinions-D row | `run_loo_flip_table.py` |
| `loo_flip_edgfull_modhigh.json` | E4 | Section VI-C and Supplement D: no Epinions-D target of the moderate or high stratum flips | `run_loo_flip_table.py` |
| `causal_regimes.json` | E8 | Table S13 and Supplement J (first, single-run placement comparison; Section VI-F cites its out-of-window rows) | `run_causal_regimes.py` |
| `causal_retrain_s5.json` | E8 | Supplement I, the earlier total-only retraining run (70% and 96%; cited in Section VI-F); Table S11, columns "Earlier" | `run_poison_matrix.py` (part `causal`) |
| `stats_rigor.json` | E5 | Table VI (`decomp_B1`, `decomp_B5`); Supplement F, rank shift (`decision_cost`) and regression on the margin (`dose_response`) | `run_stats_rigor.py` |
| `p0b_trustguard_otc.json` | E11 | Table IX, Fig. 3, Table S8 (TrustGuard, OTC) | `run_trustguard_placement.py` |
| `p0b_trustguard_alpha.json` | E11 | Table IX, Fig. 3, Table S8 (TrustGuard, Alpha) | `run_trustguard_placement.py` |
| `official_signedgcn.json` | E11 | Table IX (SignedGCN) | `run_official_signedgcn.py` |
| `evolvegcn_placement_egcn_{o,h}_{otc,alpha}.json` | E11 | Table IX, Table S9 (EvolveGCN-O and -H) | `run_evolvegcn_placement.py` |
| `dysat_placement_{otc,alpha}.json` | E11 | Table IX, Table S9 (DySAT) | `run_dysat_placement.py` |
| `p0b_main.json` | E1 | Fig. 3 (OTC, Alpha, Epinions-S); Section VI-B, sign control; Supplement F, sweep ranges; Table II, Train column (`n_edges`) | `run_p0b_verify.py` |
| `p0b_edgfull.json` | E1 | Fig. 3 (Epinions-D) | `run_p0b_verify.py` |
| `p2_arch_signature.json` | E11 | Fig. 3 and Section VI-H (the two further models of our own) | `run_p2_arch_signature.py` |
| `p1_sota.json` | E10 | Section VI-G, Fig. S2, Table S5 (OTC, Alpha); the example of Section X | `run_p1_efficient.py` |
| `p1_sota_epn.json` | E10 | Section VI-G, Fig. S2, Table S5 (Epinions-S) | `run_p1_efficient.py` |
| `p1_sota_edgfull.json` | E10 | Section VI-G, Fig. S2, Table S5 (Epinions-D) | `run_p1_efficient.py` |
| `common_set_{otc,alpha}.json` | E7 | Table VII, Fig. S1, Section VI-E, Table S2 (per-target rows) | `run_common_set.py` |
| `common_set_summary.json` | E7 | Table VII, Fig. S1, Section VI-E, Table S2 | `analyze_common_set.py` |
| `p0d_otcalpha.json` | E12 | Section VII, time decay (plain mean); Section V, single-run example | `run_p0d_harden.py` |
| `p0d_decayed.json` | E12 | Section VII, time decay (time-decayed mean) | `run_p0d_harden.py` (`GRAIL_REP_MODE=decayed`) |
| `p0_linchpin.json` | E1 | Section VI-B, in-window against out-of-window (OTC, Alpha); Section I | `run_p0_linchpin.py` |
| `p0_linchpin_edgfull.json` | E1 | Section VI-B (Epinions-D); Table II, Epinions-D row (`clean_perf`, `train_edges`) | `run_p0_linchpin.py` |
| `stats_hier_s10.json` | E2 | Section VI-B, paired contrasts; Table S6 | `run_stats_hier.py` |
| `p3_margin.json` | – | Supplement F, other thresholds (`by_threshold`; cited in Section VI-C) | `run_p3_margin.py` |
| `masked_control_s10.json` | E6 | Section VI-D, query-edge-masked control; Table S7 | `run_masked_and_random_sets.py` (`GRAIL_BLOCKS=B`) |
| `reputation_baselines_{otc,alpha,edgfull}.json` | – | Supplement D, low-stratum experiment; Tables S3 and S4; the examples of Sections V and VI-E | `run_reputation_baselines.py` |
| `detector_calib.json` | E14 | Section VII, structural detector, and Table S12 (`detector`); Section V and Supplement F, calibration (`calibration_oos`) | `run_detector_calib.py` |
| `poison_matrix_s5.json` | E9 | Section VI-F and Supplement I, poisoning controls (total score); Table S10, column T; Supplement I, spread of the clean reputation across seeds (`retrain_std_clean`) | `run_poison_matrix.py` (part `matrix`) |
| `p3_defense_adaptive.json` | E13 | Section VII, rater-history weighting and the adaptive attacker; Supplement I, random accounts under the weighting | `run_p3_defense.py` |
| `p3_sybil.json` | E13 | Section VII, newly created accounts | `run_p3_sybil.py` |
| `random_source_sets.json` | – | Supplement F, random source sets: block `C_random_distribution` only (blocks A and B are superseded; the paper cites neither) | `run_masked_and_random_sets.py` (`GRAIL_BLOCKS=C`) |
| `sliding_placebo.json` | E8 | Supplement J, sliding-window placebo of the first, single run | a one-off run; no current script writes it |
| `p0_linchpin_edg.json` | E1 (check) | Supplement B, the dated 30k-user subgraph | `run_p0_linchpin.py` |
| `p0b_edg.json` | E1 (check) | Supplement B, the dated 30k-user subgraph | `run_p0b_verify.py` |
| `epn_clean.json` | – | none: holds null values; Table II's Epinions-S AUC and MCC come from `epn_clean.log` | one call of `train_model('epn30000')` |
| `../prospective_paired_{otc,alpha}.json` | E8 | Table VIII and Section VI-F (paired sliding-window test over ten runs, its pre-declared criterion, stored in the file before any outcome, and the verdict); Section I; Supplement J (total score) and Table S14 | `run_prospective_paired.py` |
| `../retrain_prop.json` | E8, E9 | Section VI-F, retraining and poisoning controls rerun with both scores; Section I; Table S10 (columns T' and P') and Table S11 | `run_retrain_prop.py` |
| `../budget_ablation_{otc,alpha}.json` | E4 | Section VI-C (the fall of the propagation-only flip rate with the budget), Section VI-D (raters' share of the change), Section V (the rerun's 64%); Supplement K and Table S15 | `run_budget_ablation.py` |
| `../trustguard_casestudy_{otc,alpha}{,_node2vec}{,_testdeg}.json` | E15 | Section VI-A (TrustGuard case study: reproduction of its Fig. 3 MCC, label swap and placement under retraining, single-edge test with robust aggregation off and on); Sections I, IV (Check 2 bound), VI-H (the five-run rates) and X; Supplement L and Tables S16 to S18 | `run_trustguard_casestudy.py` (`--features node2vec`, `--attack-degree test`); `report_trustguard_casestudy.py --tables` prints the summaries |

## Naming and notes

- `edgfull` and `edg60000` denote Epinions-D, the complete graph of the genuinely dated
  statements; `edg` alone denotes the earlier dated 30k-user subgraph (Supplement B only).
- `epn30000` or `epn` denotes the SNAP Epinions 30k-user subgraph, which has no timestamps and
  which the paper calls Epinions-S.
- `_s10` and `_s5` mark results pooled over ten or five runs (independently trained models).
- In `trustguard_casestudy_*.json` (E15), `_node2vec` marks the node2vec input features (otherwise the release's random features) and `_testdeg` the attack size counted in the test snapshot (otherwise in the training window). The keys `mean` and `robust` are robust aggregation off and on; the conditions `inwin`, `app` and `graph` are the in-window, loss-only and encoder-only placements, `clean2` is the identical retraining, and `evasion` is the single-edge test. Each file covers seeds 40 to 44 and has a log of the same name.
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
