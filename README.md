# LayerZero Sybil Detection: ML Pipeline

> Reproduces and extends: *Sybil Detection on Public Blockchains via XGBoost and Gas Provision Network Analysis* (Imig et al., 2025)

Trains three classifiers (XGBoost, LightGBM, Logistic Regression) and a cross-model ensemble on 434,786 Ethereum addresses to identify Sybil wallets in the LayerZero airdrop of June 2024. Every number below is produced by a notebook in this repository; `06_split_comparison.ipynb` checks that each one traces to the code in the current commit.

---

## Benchmark results

Stratified group split on funding clusters (no cluster spans train and test). Test partition:
130,435 addresses, 5,463 Sybil (4.19 %). Hyperparameters, thresholds and the blend weight are chosen
on validation; test labels are used only for this table.

| Model | Precision | Recall | F1 | AUROC | AP | FPR |
|---|---|---|---|---|---|---|
| XGBoost (3 seeds) | 0.720 | 0.720 | 0.720 | 0.966 | 0.764 | 0.0123 |
| LightGBM (3 seeds) | 0.727 | 0.711 | 0.719 | 0.968 | 0.770 | 0.0117 |
| Cross-ensemble | 0.697 | 0.739 | 0.717 | 0.968 | 0.770 | 0.0140 |
| Logistic regression | 0.150 | 0.617 | 0.241 | 0.837 | 0.166 | 0.1529 |

Source: `06_split_comparison.ipynb`. AP is average precision; FPR is the false-positive rate among
non-Sybils at the model's threshold.

- **Split noise.** Over 10 group splits, LightGBM test F1 has SD 0.006 (`09`). Differences between
  XGBoost, LightGBM and the ensemble are smaller than that.
- **Ensemble.** The validation-selected XGBoost weight is 0.06; the ensemble is essentially LightGBM.

### Note on leakage with simple random split

If a simple random split is used, rather than a stratified group split, wallets from a funding tree can fall on both
sides of the split.  This causes leakage where the model is tested on the same actors it saw in training: In a sample random split, 599 of 5,463 test
Sybils share a funding cluster with a training Sybil.  This cannot occur under the group
split. This shows the leakage in the event of a simple random split:

| Model | Random split (original, leaky) | Group split | Δ F1 |
|---|---|---|---|
| XGBoost (3 seeds) | 0.741 | 0.720 | −0.021 |
| LightGBM (3 seeds) | 0.742 | 0.719 | −0.023 |
| Cross-ensemble | 0.743 | 0.717 | −0.025 |
| Logistic regression | 0.234 | 0.241 | +0.008 |

Δ is computed before rounding. The tree models also lose 0.033 AP. Logistic regression, which cannot
memorize clusters, does not lose anything. The loss sits in the clustered categories (LightGBM F1:
IxI 0.85 to 0.47, IxE 0.54 to 0.29); IxL wallets, which are singletons, are unchanged (0.74 to 0.75).
The random-split runs are kept only to measure this.

### Robustness checks (group split, 10 splits)

| Check | Notebook | Result |
|---|---|---|
| Corrected `gini_coefficient`, rule fixed in advance | `08` | Removing it lowered validation F1 on 3 of 10 splits (8 required): removed |
| Labels known before the snapshot vs current labels | `07` | Test F1 −0.005 ± 0.007 (mean ± SD) |
| All provision-network features removed | `09` | Test F1 −0.008; transaction features alone carry most of the signal |
| SHAP by feature family | `10` | LayerZero transactions first, then Ethereum transactions, gas provider, funding tree, funding chain |

---

## Evaluation protocol

- **Split.** 49 % train, 21 % validation, 30 % test, stratified by class. The primary split is a
  **stratified group split**: wallets are grouped by funding cluster (the root of the unlabeled part
  of their gas provision chain), and no group spans two partitions. The address-level random split
  used in the original paper is run only to measure the leakage in the original evaluation; it is
  not a reported result.
- **Test isolation.** Hyperparameters (`05_hyperparameter_search`), decision thresholds, and the
  ensemble blend weight are all chosen on the validation set. Test labels are used only for the
  final report.
- **Temporal cutoff.** The LayerZero snapshot table ends at 2024-05-01 23:59:59 UTC. Ethereum
  transaction features are cut at 2024-05-01 00:00 UTC by their source queries; provision-network
  edges at or after 2024-05-02 00:00 UTC are dropped in code before any feature is computed.
- **Labeled addresses.** Known services (exchanges, bridges, protocol contracts) end a funding
  chain. They come from a fixed rule with no hand selection: public label datasets, then an
  Etherscan check of every other address that funded at least 50 LayerZero interactors (see
  [Labeled addresses](#labeled-addresses)). Sybil labels are never used to choose them.
- **Features.** 62 features. The 63 of the submitted paper were chosen manually by the authors;
  `gini_coefficient` was removed by a test whose rule was fixed before it ran (`08_ablation_gini`).
- **Imbalance.** The Sybil class is upsampled to 1:1 in the training partition only, after the split.

The open work items for the current revision are in [`docs/REVISION_LEAKAGE.md`](docs/REVISION_LEAKAGE.md).

---

## Repository layout

```
layerzero_xgboost/
│
├── data/                                   ← Git LFS; run `git lfs pull` after cloning
│   ├── 20240915_final_sybil_list/          fcfs_list.csv (ground-truth labels)
│   ├── 20241013_hildobby_cex_evms/         CEX address list (source for the labeled list)
│   ├── 20241104_layer0_sybil_features/     l0_features_*.csv (×5) + source query
│   ├── 20241114_gas_provision/             gas provision network + source query
│   ├── 20241117_tree_features/             original precomputed tree features + featurization notebook
│   ├── 20241214_labeled_addresses/         labeled entity addresses (2024) + build script
│   ├── 20250208_cex_dex_indegree/          cex_dex_features_in_*.csv (×5) + source query
│   ├── 20260128_dune_spellbook_labels/     Dune Spellbook CEX, DEX and bridge lists (pinned commits)
│   └── 20260930_etherscan_service_labels/  Etherscan check of funders of 50+ interactors
│
├── output/                                 ← created on first run (gitignored)
│   ├── master_df.parquet                   ← full feature table (434,786 rows, all computed features + labels)
│   ├── splits.npz                          ← train/val/test arrays and partition indices
│   ├── feature_list.json                   ← ordered list of the 62 model features
│   └── pred_*.parquet                      ← test predictions from 01–04
│
├── results/                                ← metrics per notebook and split, with the code commit
│
├── sybil_pipeline.py                       ← shared pipeline: features, funding groups, splits
├── 00_data_pipeline.ipynb                  ← ① builds the feature table; leakage check
├── 08_ablation_gini.ipynb                  ← ② keep-or-drop test for gini_coefficient (fixes the feature set)
├── 05_hyperparameter_search.ipynb          ← ③ selects hyperparameters on validation
├── 01_xgboost_sybil.ipynb                  ← ④ XGBoost (3-seed ensemble)
├── 02_lightgbm_sybil.ipynb                 ← ④ LightGBM (3-seed ensemble)
├── 03_logistic_regression_sybil.ipynb      ← ④ Logistic Regression baseline
├── 04_cross_ensemble_sybil.ipynb           ← ④ Cross-model ensemble (XGB + LGBM)
├── 07_sensitivity_label_vintage.ipynb      ← ⑤ labels known before the snapshot vs current labels
├── 09_ablation_families.ipynb              ← ⑤ feature-family ablation
├── 10_shap_importance.ipynb                ← ⑤ SHAP importance by family and taxonomy category
├── 06_split_comparison.ipynb               ← ⑥ tie-out: provenance checks and every reported number
├── review_support/                         ← labeled-address evidence: hand additions, Etherscan lookups
├── docs/REVISION_LEAKAGE.md                ← revision work items and findings log
├── legacy/                                 ← original 2025 notebook (Windows paths; reference only)
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

**Hardware.** 4 CPU cores and 8 GB RAM are enough; no GPU is used. The model notebooks fix the
XGBoost and LightGBM thread count at 4 (`sp.N_JOBS`), because XGBoost's `hist` algorithm gives
slightly different trees with different thread counts. LightGBM also runs with `force_col_wise` and
`deterministic` (`sp.LGBM_REPRO`): otherwise it picks its histogram method by a timing test at
startup and sums in thread order, and its trees change between runs. With these settings, results
reproduce across runs and machines.

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
run() { SPLIT_METHOD=$2 jupyter nbconvert --to notebook --execute "$1" --inplace \
          --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1; }
run 00_data_pipeline.ipynb group
run 08_ablation_gini.ipynb group
run 05_hyperparameter_search.ipynb group
for m in group random; do
  for nb in 01_xgboost_sybil 02_lightgbm_sybil 03_logistic_regression_sybil 04_cross_ensemble_sybil; do
    run $nb.ipynb $m
  done
done
for nb in 07_sensitivity_label_vintage 09_ablation_families 10_shap_importance 06_split_comparison; do
  run $nb.ipynb group
done
```

`SPLIT_METHOD` selects `group` (default) or `random`. Each model notebook rebuilds the feature
table itself through `sybil_pipeline.build_master_df`, so notebooks can be run independently once
`05_hyperparameter_search` has written `results/05_hyperparameter_search_group.json`. The
committed notebook outputs are the `group` runs; the `random` runs are recorded in `results/`.
`06` runs last because it checks every other notebook's results.

Runtimes on 4 cores for the committed run: `00` 1.5 min; `08` 12 min; `05` 82 min; per split,
`01` 5 min, `02` 4 min, `03` 2 min, `04` 8 min; `07` 22 min; `09` 92 min; `10` 43 min; `06`
under 1 min. About 5 hours in total.

### What each notebook does

- **`00_data_pipeline`**: loads the L0 features, the provision network (with the snapshot cutoff),
  the labeled addresses, and the CEX/DEX in-degree; computes provider, tree, and chain features;
  checks the recomputed tree features against the original precomputed file; assigns taxonomy
  categories and funding groups; reports row and funding-group overlap between partitions under
  both split methods; exports `output/`.
- **`08_ablation_gini`**: trains LightGBM with and without the corrected `gini_coefficient` on 10
  group splits and applies the rule fixed in advance (keep it only if removing it lowers validation
  F1 on at least 8 of 10). Uses no test labels for the decision. It runs before the search because
  it decides the feature set.
- **`05_hyperparameter_search`**: full grids for XGBoost (learning rate, depth, row and column
  subsampling, then `min_child_weight`) and LightGBM (leaves, learning rate, subsampling, L2),
  selected by validation F1. Never loads test labels.
- **`01`–`03`**: train the selected XGBoost and LightGBM configurations as 3-seed ensembles (seeds
  42, 123, 456) and the L1 logistic-regression baseline; report test metrics at the
  validation-selected threshold, feature importance, and an operating-point table.
- **`04`**: blends the XGBoost and LightGBM probabilities, `w × P(XGB) + (1 − w) × P(LGBM)`, with
  `w` chosen on validation F1; reports agreement between the two models.
- **`07`**: retrains on labels restricted to list entries known before the snapshot and compares
  with the current labels over 10 group splits; also reports the Sybil share among wallets funded by
  exchanges added to the list before the snapshot, between the snapshot and the Sybil list, and after.
- **`09`**: removes each feature family, and keeps each family alone, over 10 group splits. Reported
  only; it changes no modeling choice.
- **`10`**: mean |SHAP| per feature, summed by family, overall and per taxonomy category (test
  partition, group split).
- **`06`**: verifies provenance and prints every table the paper reports.

---

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

---

## Key design decisions

**Why a group split?**
Provider and tree features are relational: every wallet in a funding cluster shares the same
`provider_*`, `tree_size`, `branching_factor`, and similar values. Under an address-level random
split, wallets from one cluster land in both train and test, so test performance partly measures
recognition of clusters already seen in training. The group split keeps each cluster in one
partition, and `sybil_pipeline.make_splits` asserts that no group spans two partitions. Wallets
funded directly by a labeled entity (an exchange, for example) are their own group, so a CEX hot
wallet never merges its customers into one giant group.

**Why upsample only the training set?**
Upsampling before splitting would put duplicated minority-class rows into validation and test,
inflating recall and F1. The pipeline splits first, then upsamples the training partition to 1:1.

**Why recompute the tree features?**
The original precomputed file was built from the provision network without a date filter, with
inputs outside this repository, and listed one wallet twice. `sybil_pipeline.provision_features`
ports the same featurization, applies the snapshot cutoff, and runs from repository data;
`00_data_pipeline` reports how the recomputed values compare with the original file.

<a id="labeled-addresses"></a>**Labeled addresses.**
A labeled address ends a funding chain, so it shapes the provider, chain and tree features of every
wallet it funded. The set is built by one rule, applied to every address: (1) public label datasets,
namely the 2024 consolidated label file without its 44 hand additions, plus Dune Spellbook's CEX,
DEX and bridge lists; (2) an Etherscan check of every other address that funded at least 50
LayerZero interactors, labeled when its public name tag identifies a shared service (exchange hot
wallet, bridge or relayer, protocol contract, other named service). Exchange deposit addresses,
personal ENS names and untagged addresses are not labeled. `review_support/etherscan_lookups.csv`
records every lookup with its date and URL, so each row can be re-checked.

**Why stream labeled addresses instead of loading all 9M?**
The full set uses about 1.2 GB as a Python set. Only a few thousand of them appear in the provision
network, so the pipeline streams the file and keeps those, with no change in the computed features.

**LightGBM bagging.**
`subsample_freq=1` must be set for `subsample < 1` to have any effect; the default of 0 disables
bagging. `reg_lambda` defaults to 0 in LightGBM (1 in XGBoost); the search covers both values.

---

## Troubleshooting

**`FileNotFoundError: results/05_hyperparameter_search_group.json`**
Run `05_hyperparameter_search.ipynb` before 01, 02, and 04.

**`FileNotFoundError` on data files**
Run `git lfs pull`, and check that `DATA_DIR` points to the `data/` folder.

**`AssertionError` in `06_split_comparison`**
A result was produced by code that differs from the current commit, or from a dirty working
tree. Rerun the notebooks it names.

**Results differ from the paper's original numbers**
The revision changes the evaluation (group split, validation-only selection, snapshot cutoff,
duplicate removed, fixed thread count). `06_split_comparison` reports the random-split numbers
under the same corrected pipeline for comparison.

---

Source code and labeled dataset archive:
[github.com/paven86/layerzero_xgboost](https://github.com/paven86/layerzero_xgboost)
