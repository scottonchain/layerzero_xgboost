# Corrected run versus the pre-fix reference (b7a9a3f)

| Notebook | Corrected run at | Stored fields compared | Differ | Max abs numeric difference |
|---|---|---|---|---|
| 00_data_pipeline | 7e7130d | 33 | 0 | 0 |
| 01_ablation_gini | da08630 | 82 | 64 | 3 |
| 02_hyperparameter_search | 03cea83 | 20 | 12 | 962 |
| 03_xgboost_sybil | 622ca58 | 84 | 76 | 480 |
| 04_lightgbm_sybil | 71a009e | 83 | 74 | 962 |
| 05_logistic_regression_sybil | 9cd7e6b | 19 | 7 | 0.09482 |
| 06_cross_ensemble_sybil | 93a79f5 | 95 | 87 | 962 |
| 07_sensitivity_label_vintage | 8b09009 | 209 | 151 | 384 |
| 08_ablation_families | deef156 | 1,520 | 1,059 | 384 |
| 09_shap_importance | 38c3093 | 378 | 330 | 384 |
| 11_split_comparison | 2b23ba4 | 129 | 89 | 384 |
| 12_figures | 41a76ca | 77 | 4 | 465 |
| 13_random_forest_sybil | 2adac1c | 384 | 308 | 384 |
| 14_entity_level_recall | 8188899 | 4,905 | 1,469 | 131 |
| 15_split_tree_report | afd06fd | 1,101 | 504 | 336 |
| 16_label_robustness_initial_list | 7baad94 | 572 | 473 | 740 |
| 17_models_10_splits | 2f50786 | 2,794 | 1,370 | 1019 |
| 18_mimicry_stress | 2ac94cb | 806 | 565 | 384 |
| 19_ablation_by_category | 52b3fc4 | 3,207 | 1,448 | 384 |
| 20_temporal_holdout | dc4d551 | 352 | 106 | 1236 |
| 22_report_dependence | 1f3da83 | 2,005 | 660 | 1331 |
| 23_revision_figures | a2f163c | 54 | 2 | 0 |
| 24_mimicry_controls | 5b83115 | 7,401 | 3,252 | 384 |
| 25_paper_tables | 3179d43 | 88 | 1 | 12 |
| 99_tie_out_revision | 4761590 | 1,149 | 609 | 1236 |

## Headline model results (test, validation-selected threshold)

| Notebook | Model | F1 pre-fix | F1 corrected | AP pre-fix | AP corrected | AUROC pre-fix | AUROC corrected |
|---|---|---|---|---|---|---|---|
| 03_xgboost_sybil | XGBoost (tuned, 3-seed) | 0.7200 | 0.7159 | 0.7645 | 0.7643 | 0.9657 | 0.9695 |
| 04_lightgbm_sybil | LightGBM (tuned, 3-seed) | 0.7188 | 0.7149 | 0.7696 | 0.7693 | 0.9683 | 0.9696 |
| 05_logistic_regression_sybil | Logistic Regression (C=0.01, L1) | 0.2412 | 0.2400 | 0.1656 | 0.1668 | 0.8368 | 0.8376 |
| 06_cross_ensemble_sybil | XGBoost (tuned, 3-seed) | 0.7200 | 0.7159 | 0.7645 | 0.7643 | 0.9657 | 0.9695 |
| 06_cross_ensemble_sybil | LightGBM (tuned, 3-seed) | 0.7188 | 0.7149 | 0.7696 | 0.7693 | 0.9683 | 0.9696 |
| 06_cross_ensemble_sybil | Cross-Ensemble | 0.7173 | 0.7165 | 0.7699 | 0.7702 | 0.9683 | 0.9703 |
| 13_random_forest_sybil | Random Forest (tuned, 3-seed) | 0.7112 | 0.7115 | 0.7679 | 0.7689 | 0.9707 | 0.9709 |

## Notebook 99 (corrected run): 84 of 84 checks pass; all_passed = True; multiple_platforms_cited = False
