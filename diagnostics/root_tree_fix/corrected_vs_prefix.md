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

## Headline model results (test, validation-selected threshold)

| Notebook | Model | F1 pre-fix | F1 corrected | AP pre-fix | AP corrected | AUROC pre-fix | AUROC corrected |
|---|---|---|---|---|---|---|---|
| 03_xgboost_sybil | XGBoost (tuned, 3-seed) | 0.7200 | 0.7159 | 0.7645 | 0.7643 | 0.9657 | 0.9695 |
| 04_lightgbm_sybil | LightGBM (tuned, 3-seed) | 0.7188 | 0.7149 | 0.7696 | 0.7693 | 0.9683 | 0.9696 |
| 05_logistic_regression_sybil | Logistic Regression (C=0.01, L1) | 0.2412 | 0.2400 | 0.1656 | 0.1668 | 0.8368 | 0.8376 |
| 06_cross_ensemble_sybil | XGBoost (tuned, 3-seed) | 0.7200 | 0.7159 | 0.7645 | 0.7643 | 0.9657 | 0.9695 |
| 06_cross_ensemble_sybil | LightGBM (tuned, 3-seed) | 0.7188 | 0.7149 | 0.7696 | 0.7693 | 0.9683 | 0.9696 |
| 06_cross_ensemble_sybil | Cross-Ensemble | 0.7173 | 0.7165 | 0.7699 | 0.7702 | 0.9683 | 0.9703 |
