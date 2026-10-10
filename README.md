# LayerZero Sybil Detection: ML Pipeline

> Reproduces and extends: *Sybil Detection on Public Blockchains via XGBoost and Gas Provision Network Analysis* (Imig et al., 2025)

Trains four classifiers (XGBoost, LightGBM, Random Forest, Logistic Regression) and a cross-model ensemble on 434,786 Ethereum addresses to identify Sybil wallets in the LayerZero airdrop of June 2024. Every number below is produced by a notebook in this repository. `10_tie_out.ipynb` checks the core pipeline through notebook 09, and `99_tie_out_revision.ipynb` checks the revision notebooks and their dependencies. The corrected publication rerun completed on x86_64 Linux and `99` passed all checks.

---

## Benchmark results

Stratified group split on gas provision trees (no tree spans train and test). Test partition:
130,435 addresses, 5,463 Sybil (4.19 %). Hyperparameters, thresholds and the blend weight are chosen
on validation; test labels are used only for this table.

| Model | Precision | Recall | F1 | AUROC | AP | FPR |
|---|---|---|---|---|---|---|
| XGBoost (3 seeds) | 0.710 | 0.722 | 0.716 | 0.969 | 0.764 | 0.0129 |
| LightGBM (3 seeds) | 0.703 | 0.728 | 0.715 | 0.970 | 0.769 | 0.0135 |
| Cross-ensemble | 0.710 | 0.723 | 0.717 | 0.970 | 0.770 | 0.0129 |
| Random forest (3 seeds) | 0.723 | 0.701 | 0.712 | 0.971 | 0.769 | 0.0118 |
| Logistic regression | 0.156 | 0.522 | 0.240 | 0.838 | 0.167 | 0.1236 |

Sources: `03`–`06` and `13`. AP is average precision; FPR is the false-positive rate among
non-Sybils at the model's validation-selected threshold. These are the corrected x86_64 Linux results.

- **Split noise.** Over 10 group splits, LightGBM test F1 is 0.709 ± 0.008.

**Over 10 group splits** (`17_models_10_splits.ipynb`, one model seed per fit, corrected x86_64 Linux rerun).
Mean ± SD of test metrics; p: corrected resampled t-test (Nadeau and Bengio 2003) against LightGBM,
Holm-adjusted per metric.

| Model | F1 | p | AP | p | AUROC | p |
|---|---|---|---|---|---|---|
| LightGBM | 0.709 ± 0.008 | — | 0.761 ± 0.007 | — | 0.969 ± 0.002 | — |
| Cross-ensemble | 0.710 ± 0.008 | 1.000 | 0.763 ± 0.006 | 0.351 | 0.970 ± 0.002 | 0.043 |
| XGBoost | 0.708 ± 0.007 | 1.000 | 0.754 ± 0.007 | 0.234 | 0.969 ± 0.001 | 1.000 |
| Random forest | 0.702 ± 0.005 | 1.000 | 0.754 ± 0.008 | 0.234 | 0.969 ± 0.001 | 1.000 |
| Logistic regression | 0.234 ± 0.004 | <0.001 | 0.163 ± 0.008 | <0.001 | 0.834 ± 0.003 | <0.001 |

No F1 difference among the tree models is significant after correction. LightGBM's AP advantage over
XGBoost and Random Forest is also not significant after the corrected resampled test and Holm adjustment
(p = 0.234 for each). The weighted average has a small AUROC advantage over LightGBM
(+0.0012 ± 0.0004, Holm-adjusted p = 0.043), but not a significant F1 or AP advantage.

On the seed-42 split, the validation-selected XGBoost weight in the 3-seed cross-ensemble is 0.56.
Across the ten one-seed fits in `17`, selected XGBoost weights range from 0.30 to 0.76.

### Note on leakage with simple random split

If a simple random split is used, rather than a stratified group split, wallets from a gas provision tree can fall on both
sides of the split. In the seed-42 random split, 599 of 5,463 test Sybils share a gas provision tree with a training
Sybil; this cannot occur under the group split. Under the corrected pipeline (`11_split_comparison.ipynb`):

| Model | F1 random (leaky) | F1 group | Δ F1 | AP random | AP group | Δ AP |
|---|---|---|---|---|---|---|
| XGBoost (3 seeds) | 0.738 | 0.716 | −0.0218 | 0.797 | 0.764 | −0.0326 |
| LightGBM (3 seeds) | 0.741 | 0.715 | −0.0262 | 0.804 | 0.769 | −0.0348 |
| Cross-ensemble | 0.741 | 0.717 | −0.0249 | 0.805 | 0.770 | −0.0350 |
| Logistic regression | 0.236 | 0.240 | +0.0041 | 0.162 | 0.167 | +0.0049 |

The tree-split effect is much smaller than the report-dependence effect below. The loss is concentrated
in the interactor categories: for LightGBM, F1 changes from 0.847 to 0.453 for IxI and from 0.544 to 0.271 for IxE.

### Robustness checks (group split; 10 splits, SHAP on one)

| Check | Notebook | Corrected result |
|---|---|---|
| Corrected `gini_coefficient`, rule fixed in advance | `01` | Removing it lowered **validation** F1 on 6 of 10 splits (8 required): removed. Sign-test p = 0.377. The test set was not used |
| Labels known before the snapshot vs current labels | `07` | Test F1 change −0.003 ± 0.007 (mean ± SD) |
| All provision-network features removed | `08` | All-features F1 0.709; without all provision-network features 0.708. The incremental F1 is +0.0002 (all minus reduced) |
| SHAP by feature family | `09` | `n_l0_source_contracts` is the top feature overall; `l0_tx_time_span` is a close second and is top for IxL. Provision-network families account for about 15.3% of total mean absolute SHAP |
| Random forest on 10 group splits | `13`, `17` | Test F1 0.702 ± 0.005 vs LightGBM 0.709 ± 0.008; mean difference −0.006 ± 0.008. Corrected Holm-adjusted F1 p = 1.000 |
| Entity-level recall (LightGBM, validation threshold) | `14` | Gas provision trees: any-hit 0.776; bounty reports: any-hit 0.707, majority-hit 0.565; 48.5% of reports keep at least half their ZRO allocation unflagged. 36.8% of test-Sybil allocation is unflagged. Predictions reproduce the committed corrected x86 results |
| Split grouped by tree and bounty report | `15` | Under the tree split, 5,406 of 5,463 test Sybils share a bounty report with a training Sybil (random split: 5,434). Grouping by tree and report: LightGBM test F1 0.250 ± 0.027 vs 0.709 ± 0.008 under the tree split. One component holds 45.4% of Sybils, so this union split is superseded by the paired design in `22` |
| LayerZero initial Sybil list | `16` | 34,545 interactors (8.3% of bounty-only negatives) are on the initial list. At the validation threshold, 705 of 1,683 apparent false positives (41.9%) are on it; counting them as Sybil gives precision 0.827 instead of 0.703. Residual FPR is 978 / 114,657 = 0.853% if initial-list addresses are treated as positives |
| Mimicry controls | `24` | Profile copy at k=5: top SHAP features 0.039 recall, wallet-local 0.038, Ethereum-local 0.153, funding-graph 0.519, random 5 of 62 0.423 mean. Replacing all 30 funding-graph features gives 0.452 recall |
| Mimicry stress test (fixed model) | `18` | Replacing the 5 most important features cuts recall from 0.728 to 0.012 (marginal) or 0.039 (profile copy). Tentative cost classes; this is a fragility measure, not an attack simulation |
| Provision-network features by category | `19` | ΔF1 (all minus without all provision-network families): +0.0002 overall, +0.0028 IxL, −0.0328 IxI, −0.0534 IxE; none is statistically significant after correction |
| Report-mates in training | `22` | Same test Sybils and matched training size: LightGBM seen vs held-out F1 0.593 vs 0.363 (Δ +0.230), AP 0.600 vs 0.305 (Δ +0.294), AUROC 0.971 vs 0.895 (Δ +0.076); seen is higher in all 15 fold-repeats. Corrected p < 0.002 for all three headline metrics |
| Temporal holdout | `20` | Latest 30%: temporal F1 0.166, recall 0.093, AUROC 0.888, versus same-cohort reference F1 0.735, recall 0.697, AUROC 0.983. Late-cohort Sybil rate is 1.02% vs 5.55% in the early cohort |

The corrected publication rerun completed through `99_tie_out_revision`; all provenance and named-number checks passed.

---

## Evaluation protocol

- **Split.** 49 % train, 21 % validation, 30 % test, stratified by class. The primary split is a
  **stratified group split**: wallets are grouped by gas provision tree (identified by the root of
  the unlabeled part of their gas provision chain), and no tree spans two partitions (`make_splits` and `_group_partition` in
  [`sybil_pipeline.py`](sybil_pipeline.py)). The address-level random split
  used in the original paper is run only to measure the leakage in the original evaluation; it is
  not a reported result.
- **Test isolation.** Hyperparameters (`02_hyperparameter_search`), decision thresholds, and the
  ensemble blend weight are all chosen on the validation set. Test labels are used only for the
  final report.
- **Temporal cutoff.** The LayerZero snapshot table ends at 2024-05-01 23:59:59 UTC. Ethereum
  transaction features are cut at 2024-05-01 00:00 UTC by their source queries; provision-network
  edges at or after 2024-05-02 00:00 UTC are dropped in code before any feature is computed.
- **Labeled addresses.** Known services (exchanges, bridges, protocol contracts) end a provision
  chain. They come from a fixed rule with no hand selection: public label datasets, then an
  Etherscan check of every other address that funded at least 50 LayerZero interactors (see
  [Labeled addresses](#labeled-addresses)). Sybil labels are never used to choose them.
- **Features.** 62 features. The 63 of the submitted paper were chosen manually by the authors;
  `gini_coefficient` was removed by a test whose rule was fixed before it ran (`01_ablation_gini`).
- **Imbalance.** The Sybil class is upsampled to 1:1 in the training partition only, after the split.

The open work items for the current revision are in [`docs/REVISION_LEAKAGE.md`](docs/REVISION_LEAKAGE.md).

---

## Repository layout

```
layerzero_xgboost/
│
├── data/                                   ← Git LFS; run `git lfs pull` after cloning
│   ├── 20240915_final_sybil_list/          ← fcfs_list.csv (ground-truth labels)
│   ├── 20241013_hildobby_cex_evms/         ← CEX address list (source for the labeled list)
│   ├── 20241104_layer0_sybil_features/     ← l0_features_*.csv (×5) + source query
│   ├── 20241114_gas_provision/             ← gas provision network + source query
│   ├── 20241117_tree_features/             ← original precomputed tree features + featurization notebook
│   ├── 20241214_labeled_addresses/         ← labeled entity addresses (2024) + build script
│   ├── 20250208_cex_dex_indegree/          ← cex_dex_features_in_*.csv (×5) + source query
│   ├── 20260128_dune_spellbook_labels/     ← Dune Spellbook CEX, DEX and bridge lists (pinned commits)
│   └── 20260930_etherscan_service_labels/  ← Etherscan check of funders of 50+ interactors + build script
│
├── output/                                 ← created on first run (gitignored)
│   ├── master_df.parquet                   ← full feature table (434,786 rows, all computed features + labels)
│   ├── splits.npz                          ← train/val/test arrays and partition indices
│   ├── feature_list.json                   ← ordered list of the 62 model features
│   └── pred_*.parquet                      ← validation and test predictions from 03–06
│
├── results/                                ← metrics per notebook, with the code commit
├── review_support/                         ← supporting files for the reviewer response (Blockscout checks; LayerZero's initial list)
├── docs/REVISION_LEAKAGE.md                ← revision work items and findings log
├── legacy/                                 ← original 2025 notebook (Windows paths; reference only)
│
├── sybil_pipeline.py                       ← shared code: features, gas provision trees, splits, training, metrics
├── 00_data_pipeline.ipynb                  ← builds the feature table; leakage check
├── 01_ablation_gini.ipynb                  ← keep-or-drop test for gini_coefficient (fixes the feature set)
├── 02_hyperparameter_search.ipynb          ← selects hyperparameters on validation
├── 03_xgboost_sybil.ipynb                  ← XGBoost (3-seed ensemble)
├── 04_lightgbm_sybil.ipynb                 ← LightGBM (3-seed ensemble)
├── 05_logistic_regression_sybil.ipynb      ← Logistic Regression baseline
├── 06_cross_ensemble_sybil.ipynb           ← Cross-model ensemble (blends 03 and 04)
├── 07_sensitivity_label_vintage.ipynb      ← labels known before the snapshot vs current labels
├── 08_ablation_families.ipynb              ← feature-family ablation
├── 09_shap_importance.ipynb                ← SHAP importance by family and taxonomy category
├── 10_tie_out.ipynb                        ← provenance checks and every reported number
├── 11_split_comparison.ipynb               ← leakage under the original random split (not a reported result)
├── 12_figures.ipynb                        ← every figure in the paper, written to figures/
├── 13_random_forest_sybil.ipynb            ← tuned Random Forest baseline (validation-only search)
├── 14_entity_level_recall.ipynb            ← recall by gas provision tree, bounty report and reporter
├── 15_split_tree_report.ipynb              ← report-level overlap; split grouped by tree and bounty report
├── 16_label_robustness_initial_list.ipynb  ← labels against LayerZero's initial Sybil list
├── 17_models_10_splits.ipynb               ← every model over 10 splits; significance tests; cost
├── 18_mimicry_stress.ipynb                 ← mimicry stress test of the fixed model
├── 19_ablation_by_category.ipynb           ← provision-network features removed, per category
├── 20_temporal_holdout.ipynb               ← train on earlier cohorts, test on later ones
├── 22_report_dependence.ipynb              ← test Sybils with and without report-mates in training
├── 23_revision_figures.ipynb               ← revision figures (figures/rev_*) and tables (tables/) from results/
├── 24_mimicry_controls.ipynb               ← mimicry controls: single feature, Ethereum-local, funding-graph only, random k
├── 25_paper_tables.ipynb                   ← remaining paper tables and named numbers for the text, from results/
├── 99_tie_out_revision.ipynb               ← provenance checks and every cited number for 13–22 (runs last)
├── DATA.md                                 ← what each data file is, where it came from, terms
├── CITATION.cff                            ← citation metadata (draft)
├── figures/                                ← PNGs named as in the manuscript's \includegraphics; rev_* (PDF and PNG) from 23
├── tables/                                 ← LaTeX booktabs fragments from 23 (\input in the manuscript)
├── requirements.txt
└── README.md
```

---

## Prerequisites

Python 3.11 or later (required by pandas 3).

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions used to produce the results. Model libraries are pinned
because default hyperparameters change between releases.

**Hardware.** The corrected publication rerun used x86_64 Linux, Intel Xeon @ 2.80 GHz,
4 CPU cores, Python 3.11.15, and the pinned package set in `requirements.txt`; no GPU is used.
XGBoost and LightGBM use four threads. A cloud-VM restart changed only the guest kernel build during the
run; CPU model/core count, Python packages, data hashes and thread counts were unchanged. As a cross-boundary
check, post-restart notebook `19` reproduced the pre-restart `08` all-features F1 on all ten splits to
within 1e-12, and `17` independently reproduced `08`'s LightGBM arm.

> **RAM note.** The labeled-addresses file (`20241214_labeled_addresses.csv`) contains 9 million
> entries. The pipeline streams it and retains only the addresses that appear in the provision
> network (a few thousand), so peak RAM stays low. Do **not** load the full file into memory.

---

## Step-by-step instructions

### 1. Fetch data files

The input files are stored with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

The notebooks expect the layout above, with `DATA_DIR = './data'` relative to the notebook.

### 2. Run the notebooks in order

Interactively (`jupyter lab`), or headless:

```bash
for nb in [0-9][0-9]_*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb" --inplace \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1
done
```

The glob runs `00` to `25` and then `99` in numerical order, which is the dependency order. Every notebook
uses the stratified group split except `11`, which also trains the models on the original random
split to measure the leakage. Each notebook rebuilds the feature table itself through
`sybil_pipeline.build_master_df`, so after `02_hyperparameter_search` has written
`results/02_hyperparameter_search.json`, the model notebooks can be run independently; `06` also
needs the predictions saved by `03` and `04`. `01` and `02` fix the feature set and the
hyperparameters; they must be rerun whenever `sybil_pipeline.py` changes, or `10` rejects the
results that depend on them. `10` checks every result from `00` to `09`; `11` and `12` run after it.
`13` to `25` answer later reviewer requests and are not checked by `10`; `99_tie_out_revision` checks them
instead and runs last. `23` reads only `results/` and regenerates the revision's figures and tables in seconds;
rerun `23`, `25` and then `99` after any rerun of the notebooks they read. They rebuild the feature table themselves, except `14`, which reads the predictions `03`,
`04` and `06` saved in `output/`; `17` needs `13`'s result, and `16` reads `review_support/initial_list/`
(Git LFS).

Runtimes on 4 cores from the corrected x86_64 publication rerun (`02` searched from scratch):

| Notebook | Runtime |
|---|---|
| `00_data_pipeline` | 2.6 min |
| `01_ablation_gini` | 20.5 min |
| `02_hyperparameter_search` | 115.4 min |
| `03_xgboost_sybil` | 5.9 min |
| `04_lightgbm_sybil` | 4.0 min |
| `05_logistic_regression_sybil` | 2.2 min |
| `06_cross_ensemble_sybil` | 10 s |
| `07_sensitivity_label_vintage` | 21.2 min |
| `08_ablation_families` | 76.9 min |
| `09_shap_importance` | 116.0 min |
| `10_tie_out` | 7 s |
| `11_split_comparison` | 9.1 min |
| `12_figures` | 4.6 min |
| `13_random_forest_sybil` | 216.9 min |
| `14_entity_level_recall` | 1.1 min |
| `15_split_tree_report` | 17.7 min |
| `16_label_robustness_initial_list` | 25.2 min |
| `17_models_10_splits` | 102.9 min |
| `18_mimicry_stress` | 5.4 min |
| `19_ablation_by_category` | 36.9 min |
| `20_temporal_holdout` | 11.5 min |
| `22_report_dependence` | 66.3 min |
| `23_revision_figures` | 12 s |
| `24_mimicry_controls` | 14.8 min |
| `25_paper_tables` | 1.2 min |
| `99_tie_out_revision` | 8 s |

The serial notebook execution time was about 14.6 hours, excluding the interrupted first attempt at `16`.

### What each notebook does
### What each notebook does

- **`00_data_pipeline`**: loads the L0 features, the provision network (with the snapshot cutoff),
  the labeled addresses, and the CEX/DEX in-degree; computes provider, tree, and chain features;
  checks the recomputed tree features against the original precomputed file; assigns taxonomy
  categories and gas provision trees; reports row and gas-provision-tree overlap between partitions under
  both split methods; exports `output/`.
- **`01_ablation_gini`**: trains LightGBM with and without the corrected `gini_coefficient` on 10
  group splits and applies the rule fixed in advance (keep it only if removing it lowers validation
  F1 on at least 8 of 10). Uses no test labels for the decision. It runs before the search because
  it decides the feature set.
- **`02_hyperparameter_search`**: full grids for XGBoost (learning rate, depth, row and column
  subsampling, then `min_child_weight`) and LightGBM (leaves, learning rate, subsampling, L2),
  selected by validation F1. Never loads test labels.
- **`03`–`05`**: train the selected XGBoost and LightGBM configurations as 3-seed ensembles (seeds
  42, 123, 456) and the L1 logistic-regression baseline; report test metrics at the
  validation-selected threshold, feature importance, and an operating-point table.
- **`06`**: blends the XGBoost and LightGBM probabilities saved by `03` and `04` (no retraining),
  `w × P(XGB) + (1 − w) × P(LGBM)`, with `w` chosen on validation F1; reports agreement between the
  two models.
- **`07`**: retrains on labels restricted to list entries known before the snapshot and compares
  with the current labels over 10 group splits; also reports the Sybil share among wallets funded by
  exchanges added to the list before the snapshot, between the snapshot and the Sybil list, and after.
- **`08`**: removes each feature family, and keeps each family alone, over 10 group splits. Reported
  only; it changes no modeling choice.
- **`09`**: mean |SHAP| per feature, summed by family, overall and per taxonomy category (test
  partition, group split).
- **`10`**: verifies provenance and prints every table the paper reports.
- **`11`**: trains the same models, with the same hyperparameters and shared code, on the original
  address-level random split, and reports the difference from the group split overall and by
  taxonomy category. A measurement of the leakage, not a reported result.
- **`12`**: draws every figure in the paper from the saved predictions, results and feature table
  (group split), and trains one XGBoost model (seed 42, selected hyperparameters) for the learning
  curve; it checks the curve against `02` and `03` and the plotted AP and AUROC against `results/`.
  The depth distribution uses all addresses.
- **`13`**: a Random Forest baseline selected like `02` (24 configurations, validation only, test
  deleted before any fit), trained as a 3-seed ensemble, and run on the 10 group splits of `08`.
- **`14`**: evaluates the saved test predictions of `03`, `04` and `06` per entity (gas provision
  tree, bounty report, reporter): any-, majority- and full-hit recall, within-entity recall, and the
  ZRO allocation left on unflagged addresses. Trains nothing. It records whether the predictions in
  `output/` reproduce the committed results.
- **`15`**: counts test Sybils whose bounty report, GitHub issue or reporter also has a training
  Sybil, then builds groups joining gas provision trees and bounty reports, checks partition rates and
  sizes before any training (and warns when a group holds more than 5 % of Sybils), and trains
  XGBoost and LightGBM on that split and LightGBM on 10 such splits against the tree split.
- **`16`**: counts interactors on LayerZero's initial Sybil list (`review_support/initial_list/`), the test
  false positives that are on it, and retrains with them removed from training and validation (and, as an
  exploratory variant, labeled Sybil); 10 splits for the first.
- **`17`**: every model on the 10 group splits on one machine, with corrected resampled t-tests (Holm-adjusted)
  against LightGBM, and the training and scoring cost on the seed-42 split.
- **`18`**: replaces the top-k features (by SHAP) of test Sybils with values from non-Sybil training wallets,
  marginally or as a whole profile, and scores the fixed model; also restricted to tentatively cheap features.
- **`19`**: LightGBM with and without the provision-network families on 10 splits, per taxonomy category.
- **`20`**: a temporal holdout: trains on wallets whose tree first used LayerZero before a cut-off and tests on
  the later 20, 30 or 40 %, against the tree-split model on the same rows.
- **`22`**: scores the same test Sybils with and without their bounty-report mates in training (equal training
  size, same validation and test rows; 5 folds of the union components of `15` × 3 repeats; LightGBM and XGBoost),
  with a dose-response by number of report-mates.
- **`23`**: draws the revision figures and writes the revision tables from the committed `results/` (see
  [Paper figures and tables](#paper-figures-and-tables)); no data, no model.
- **`24`**: controls for the mimicry test of `18` (same model, threshold and donors; asserts it reproduces `18`): every
  feature replaced alone, a rule-based Ethereum-local subset (Ethereum-side counts and values), the 30 funding-graph
  features, and random subsets of k features. Writes `figures/rev_mimicry_controls` and two tables.
- **`25`**: renders the remaining paper tables (labels, taxonomy, dependence levels, per-category performance, ablations,
  entity level, temporal holdout, the union split, the searches) and `tables/rev_text_numbers.json`, every scalar the
  text cites with its source; reads only `results/`, the Sybil list and the master table; fits no model.
- **`99`** (was `21`): provenance of `13`–`25` and the notebooks they depend on, their stored hyperparameters, `14`'s inputs
  and the platforms; prints every number cited from them. It fails until `14` is rerun on predictions that
  reproduce the committed `03`, `04` and `06`.

---

## Paper figures and tables

Generated by `23_revision_figures.ipynb` from `results/` only; each figure is saved as `.pdf` (vector) and `.png`.
`results/23_revision_figures.json` records the source results, their commits and platforms.

| File | Content | Source |
|---|---|---|
| `figures/rev_leakage_hierarchy` | A: address vs gas-provision-tree split (seed 42); B: report-mates seen vs held out (15 fold-repeats) | `11`, `22` (corrected x86_64 run) |
| `figures/rev_report_dependence` | A: dose-response by number of report-mates; B: by category | `22` |
| `figures/rev_label_effects` | A: test false positives on LayerZero's initial list; B: clean negatives | `16` |
| `figures/rev_mimicry` | Recall against top-k features replaced | `18` |
| `figures/rev_shap_by_category` | Top 15 features, share of mean \|SHAP\| per category | `09` |
| `figures/rev_mimicry_controls` | A: recall against k by attack scope (profile copy); B: 15 most sensitive single features | `24` |
| `tables/rev_models_10split.tex` | LightGBM, XGBoost, Random Forest and LR over 10 splits, mean ± SD, corrected p vs LightGBM | `17` |
| `tables/rev_ensemble_appendix.tex`, `tables/rev_ensemble_weight_stats.json` | Weighted average vs LightGBM and XGBoost (appendix); blend weights | `17`, `06` |
| `tables/rev_report_dependence.tex` | Seen vs held out, LightGBM and XGBoost | `22` |
| `tables/rev_label_robustness.tex` | Initial-list counts, false positives, clean negatives, relabeling | `16` |
| `tables/rev_operating_points.tex` | Operating points with false positives on the initial list | `16` |
| `tables/rev_mimicry.tex` | Recall at k = 1, 3, 5, 10 per attack | `18` |
| `tables/rev_mimicry_controls.tex` | Recall at k = 1, 3, 5, 10 per attack scope, including random k | `24` |
| `tables/rev_single_feature_sensitivity.tex` | 15 features whose replacement alone lowers recall most | `24` |

Tables from `25_paper_tables.ipynb` (same rules; `results/25_paper_tables.json` records sources and platforms):

| File | Content | Source |
|---|---|---|
| `tables/rev_data_labels.tex` | The two LayerZero label sets among the interactors | `00`, `16`, Sybil list |
| `tables/rev_taxonomy.tex` | Interactors and Sybils per category | `00` |
| `tables/rev_levels.tex` | Test Sybils sharing a tree or a report with training, three levels | `15`, `22` |
| `tables/rev_per_category.tex` | LightGBM and XGBoost per category over 10 splits | `17` |
| `tables/rev_family_ablation.tex` | Feature families removed, 10 splits | `08` |
| `tables/rev_provision_ablation_by_category.tex` | Provision-network families removed, per category | `19` |
| `tables/rev_entity_level.tex` | Entity-level outcomes; `14` verified that its predictions reproduce the committed corrected results | `14` |
| `tables/rev_temporal_holdout.tex` | Temporal holdout at 20, 30, 40 % late cohorts | `20` |
| `tables/rev_union_split_appendix.tex` | The superseded union split (appendix) | `15` |
| `tables/rev_search_top.tex` | Top configurations of the three searches (appendix) | `02`, `13`, search CSVs |
| `tables/rev_text_numbers.json` | 215 named scalars for the running text, each with source and platform | as listed in the file |

## Input data files

| File | Rows | Date cutoff | Description |
|---|---|---|---|
| `l0_features_*.csv` (×5) | 434,793 raw; 434,786 after cleaning | L0 snapshot table (ends 2024-05-01 23:59:59 UTC); Ethereum transactions ≤ 2024-05-01 00:00 UTC | Transaction, bridge, and timing features per LayerZero interactor. Source: Flipside `external.layerzero.fact_transactions_snapshot` and `ethereum.core.fact_transactions`. |
| `20241114_1633_layer0_provision_network_000000000000.csv` | 604,864 | None in the query; edges ≥ 2024-05-02 00:00 UTC dropped in code (296 edges) | First ETH transfer into each address, traced upward from the interactors. Source: BigQuery `crypto_ethereum.traces`. |
| `20241214_labeled_addresses.csv` | 9,054,104 | Label lists compiled Oct–Dec 2024 | Known entities (CEXs, DEXs, contracts, named accounts). The 44 addresses its `readme.txt` adds by hand are excluded. |
| Dune Spellbook lists (`data/20260128_dune_spellbook_labels/`) | CEX 4,957; DEX 73; bridges 136 (current versions) | Pinned commits; each row has the date it was added | CEX, DEX and bridge addresses on Ethereum. A `presnapshot` version (entries added by 2024-05-01) is the sensitivity in `07`. |
| `service_labels.csv` (`data/20260930_etherscan_service_labels/`) | 36 | Etherscan read 2026-09-30 and 2026-10-01 | Every other funder of 50+ interactors, with its Etherscan tag and the rule's decision; 10 labeled. |
| `20241117_graph_and_tree_features.csv` | 434,111 | Built from the unfiltered network | Original precomputed provider and tree features. Now used only as a regression check; the pipeline recomputes these features. |
| `cex_dex_features_in_*.csv` (×5) | 434,793 total | Transfers ≤ 2024-05-01 | Distinct CEX and DEX addresses that sent ETH to each interactor. Labels from Flipside `dim_labels`. |
| `fcfs_list.csv` | 151,784 | Sybil list snapshot 2024-09-15 | LayerZero Foundation's final Sybil list. Ground-truth labels. |
| `review_support/initial_list/initial_list.csv` | 803,093 | Published 18 May 2024; archived 24 May 2024 | LayerZero's initial Sybil list, from the deleted `LayerZero-Labs/sybil-report` repository, as archived by the [Wayback Machine](https://web.archive.org/web/*/https://github.com/LayerZero-Labs/sybil-report/raw/main/*). Not a label; used only by `16`. Provenance: [`review_support/initial_list/README.md`](review_support/initial_list/README.md). |

---

## Key design decisions

**Why a group split?**
Provider and tree features are relational: every wallet in a gas provision tree shares the same
`provider_*`, `tree_size`, `branching_factor`, and similar values. Under an address-level random
split, wallets from one gas provision tree land in both train and test, so test performance partly
measures recognition of trees already seen in training. The group split keeps each tree in one
partition, and `make_splits` in [`sybil_pipeline.py`](sybil_pipeline.py) asserts that no tree spans two partitions. Labeled
entities are not part of any tree, so a wallet funded directly by one (an exchange, for example) is
the root of its own tree, and a CEX hot wallet never merges its customers into one giant tree.

**Why upsample only the training set?**
Upsampling before splitting would put duplicated minority-class rows into validation and test,
inflating recall and F1. The pipeline splits first, then upsamples the training partition to 1:1.

**Why recompute the tree features?**
The original precomputed file was built from the provision network without a date filter, with
inputs outside this repository, and listed one wallet twice. `sybil_pipeline.provision_features`
ports the same featurization, applies the snapshot cutoff, and runs from repository data;
`00_data_pipeline` reports how the recomputed values compare with the original file.

<a id="labeled-addresses"></a>**Labeled addresses.**
A labeled address ends a provision chain, so it shapes the provider, chain and tree features of every
wallet it funded. The set is built by one rule, applied to every address: (1) public label datasets,
namely the 2024 labeled-address file (the union of six public lists) without its 44 hand additions, plus Dune Spellbook's CEX,
DEX and bridge lists; (2) an Etherscan check of every other address that funded at least 50
LayerZero interactors, labeled when its public name tag identifies a shared service (exchange hot
wallet, bridge or relayer, protocol contract, other named service). Exchange deposit addresses,
personal ENS names and untagged addresses are not labeled. `data/20260930_etherscan_service_labels/etherscan_lookups.csv`
records every lookup with its date and URL, so each row can be re-checked.

**Why stream labeled addresses instead of loading all 9M?**
The full set uses about 1.2 GB as a Python set. Only a few thousand of them appear in the provision
network, so the pipeline streams the file and keeps those, with no change in the computed features.

**LightGBM bagging.**
`subsample_freq=1` must be set for `subsample < 1` to have any effect; the default of 0 disables
bagging. `reg_lambda` defaults to 0 in LightGBM (1 in XGBoost); the search covers both values.

---

## Troubleshooting

**`FileNotFoundError: results/02_hyperparameter_search.json`**
Run `02_hyperparameter_search.ipynb` before `03` to `11`.

**`FileNotFoundError` on data files**
Run `git lfs pull`, and check that `DATA_DIR` points to the `data/` folder.

**`AssertionError` in `10_tie_out` or `11_split_comparison`**
A result was produced by code that differs from the current commit, or from a dirty working
tree. Rerun the notebooks it names.

**Results differ from the paper's original numbers**
The revision changes the evaluation (group split, validation-only selection, snapshot cutoff,
duplicate removed, fixed thread count). `11_split_comparison` reports the random-split numbers
under the same corrected pipeline for comparison.

---

Source code and labeled dataset archive:
[github.com/paven86/layerzero_xgboost](https://github.com/paven86/layerzero_xgboost)
