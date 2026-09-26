# Result files

`unified/` holds every result file the paper takes a number from: 74 JSON files and the run
logs of most of them (a log shares its JSON file's name, and `cohort/*.log` are the cohort
jobs' logs). The scripts are in `code/experiments/` unless the name starts with `analyze_`.
Section 6 of the top-level README gives the command behind each file. Numbering follows the
SaTML paper: Roman numerals for sections and main tables, letters for appendices, S1 to S12
for appendix tables.

| File(s) | Paper item | Written by |
|---|---|---|
| `cohort/cohort_{otc,alpha,edg60000}_s{42,1..9}.json` | Table II, Fig. 1 flip labels, Fig. 3, Table S1 (one file per dataset and model) | `run_cohort_seeds.py` |
| `cohort_summary.json` | Table II, Fig. 3, abstract, Sections I, VI and XIV rates | `analyze_cohort.py` |
| `loo_flip_table.json` | Table III, Bitcoin rows | `run_loo_flip_table.py` |
| `loo_flip_edgfull_low.json` | Table III, dated Epinions, low stratum | `run_loo_flip_table.py` |
| `loo_flip_edgfull_modhigh.json` | Table III, dated Epinions, moderate and high strata | `run_loo_flip_table.py` |
| `causal_regimes.json` | Table IV: appended, in-window and sliding rows | `run_causal_regimes.py` |
| `causal_retrain_s5.json` | Table IV retrain rows; Table S11 | run_poison_matrix.py (causal part) |
| `stats_rigor.json` | Table V (decomp_B1, decomp_B5); Section VI rank shift (decision_cost); Appendix G regression (dose_response) | `run_stats_rigor.py` |
| `p0b_trustguard_otc.json` | Table VI, Fig. 2a, Table S8 (TrustGuard, OTC) | `run_trustguard_placement.py` |
| `p0b_trustguard_alpha.json` | Table VI, Fig. 2a, Table S8 (TrustGuard, Alpha) | `run_trustguard_placement.py` |
| `official_signedgcn.json` | Table VI (SignedGCN) | `run_official_signedgcn.py` |
| `evolvegcn_placement_egcn_{o,h}_{otc,alpha}.json` | Table VI, Table S9 (EvolveGCN-O and -H) | `run_evolvegcn_placement.py` |
| `dysat_placement_{otc,alpha}.json` | Table VI, Table S9 (DySAT) | `run_dysat_placement.py` |
| `p0b_main.json` | Fig. 2a (OTC, Alpha, SNAP Epinions); Section III, SNAP Epinions | `run_p0b_verify.py` |
| `p0b_edgfull.json` | Fig. 2a (dated Epinions) | `run_p0b_verify.py` |
| `p2_arch_signature.json` | Fig. 2a (the two in-house builds) | `run_p2_arch_signature.py` |
| `p1_sota.json` | Fig. 2b, Section IX, Table S5 (OTC, Alpha) | `run_p1_efficient.py` |
| `p1_sota_epn.json` | Fig. 2b, Section IX, Table S5 (SNAP Epinions) | `run_p1_efficient.py` |
| `p1_sota_edgfull.json` | Fig. 2b, Section IX, Table S5 (dated Epinions) | `run_p1_efficient.py` |
| `common_set_{otc,alpha}.json` | Fig. 2c, Section VIII, Table S2 (per-target rows) | `run_common_set.py` |
| `common_set_summary.json` | Fig. 2c, Section VIII, Table S2 | `analyze_common_set.py` |
| `p0d_otcalpha.json` | Section II, time decay (mean reputation) | `run_p0d_harden.py` |
| `p0d_decayed.json` | Section II, time decay (decayed reputation) | run_p0d_harden.py (GRAIL_REP_MODE=decayed) |
| `p0_linchpin.json` | Section III, in-window vs. appended (OTC, Alpha) | `run_p0_linchpin.py` |
| `p0_linchpin_edgfull.json` | Section III (dated Epinions); Table I dated Epinions row (clean_perf, train_edges) | `run_p0_linchpin.py` |
| `stats_hier_s10.json` | Section VI paired statistics; Table S6 | `run_stats_hier.py` |
| `p3_margin.json` | Section VI, other gates (by_threshold) | `run_p3_margin.py` |
| `masked_control_s10.json` | Section VIII query-edge-masked control; Table S7 | run_reviewer_rebuttal.py (GRAIL_BLOCKS=B) |
| `reputation_baselines_{otc,alpha,edgfull}.json` | Section VIII label-only reputation scores; Tables S3 and S4 | `run_reputation_baselines.py` |
| `detector_calib.json` | Section VIII calibration (calibration_oos); Section XI detector and Table S12 (detector) | `run_detector_calib.py` |
| `poison_matrix_s5.json` | Section X poisoning placebo; Table S10 | `run_poison_matrix.py` |
| `p3_defense_adaptive.json` | Section XI rater-history weighting and the adaptive attacker | `run_p3_defense.py` |
| `p3_sybil.json` | Section XI, fresh vs. existing accounts | `run_p3_sybil.py` |
| `reviewer_rebuttal.json` | Appendix G random source draws: block C_random_distribution only (blocks A and B are superseded; the paper cites neither) | run_reviewer_rebuttal.py (GRAIL_BLOCKS=C) |
| `sliding_placebo.json` | Appendix G sliding-window placebo | a one-off run; no current script writes it |
| `p0_linchpin_edg.json` | Appendix C, the earlier 30k dated subgraph | `run_p0_linchpin.py` |
| `p0b_edg.json` | Appendix C, the earlier 30k dated subgraph | `run_p0b_verify.py` |
| `epn_clean.json` | none: holds null values; the Table I SNAP Epinions row comes from epn_clean.log | one call of train_model('epn30000') |

## Naming

- `edgfull` and `edg60000` denote the complete genuinely dated Epinions graph, the one the paper
  uses; `edg` alone denotes the earlier 30k-user dated subgraph (Appendix C only).
- `epn30000` or `epn` denotes the SNAP Epinions 30k-node subgraph, which has no timestamps.
- `_s10` and `_s5` mark results pooled over ten or five independently trained models.
- Table I's clean AUC and MCC of the Bitcoin graphs are printed at the start of every seed-42
  log, for example `loo_flip_table.log`.
- Paths inside the logs are anonymized: `<ps-env>` and `<dysat-env>` stand for the two Python
  environments and `<clones>` for the folder holding the official releases.
