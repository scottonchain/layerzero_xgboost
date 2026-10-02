# LayerZero Sybil Detection: ML Pipeline

> Reproduces and extends: *Sybil Detection on Public Blockchains via XGBoost and Gas Provision Network Analysis* (Imig et al., 2025)

Trains three classifiers (XGBoost, LightGBM, Logistic Regression) and a cross-model ensemble on 434,786 Ethereum addresses to identify Sybil wallets in the LayerZero airdrop of June 2024. Every number below is produced by a notebook in this repository; `10_tie_out.ipynb` checks that each one traces to the code in the current commit.

---

## Benchmark results

Stratified group split on gas provision trees (no tree spans train and test). Test partition:
130,435 addresses, 5,463 Sybil (4.19 %). Hyperparameters, thresholds and the blend weight are chosen
on validation; test labels are used only for this table.

| Model | Precision | Recall | F1 | AUROC | AP | FPR |
|---|---|---|---|---|---|---|
| XGBoost (3 seeds) | 0.720 | 0.720 | 0.720 | 0.966 | 0.764 | 0.0123 |
| LightGBM (3 seeds) | 0.727 | 0.711 | 0.719 | 0.968 | 0.770 | 0.0117 |
| Cross-ensemble | 0.697 | 0.739 | 0.717 | 0.968 | 0.770 | 0.0140 |
| Logistic regression | 0.150 | 0.617 | 0.241 | 0.837 | 0.166 | 0.1529 |

Source: `10_tie_out.ipynb`. AP is average precision; FPR is the false-positive rate among
non-Sybils at the model's threshold.

- **Split noise.** Over 10 group splits, LightGBM test F1 has SD 0.006 (`08`). Differences between
  XGBoost, LightGBM and the ensemble are smaller than that.
- **Ensemble.** The validation-selected XGBoost weight is 0.06; the ensemble is essentially LightGBM.

### Note on leakage with simple random split

If a simple random split is used, rather than a stratified group split, wallets from a gas provision tree can fall on both
sides of the split.  This causes leakage where the model is tested on the same actors it saw in training: In a sample random split, 599 of 5,463 test
Sybils share a gas provision tree with a training Sybil.  This cannot occur under the group
split. This shows the leakage in the event of a simple random split (`11_split_comparison.ipynb`):

| Model | F1 random (original, leaky) | F1 group | Δ F1 | AP random | AP group | Δ AP |
|---|---|---|---|---|---|---|
| XGBoost (3 seeds) | 0.741 | 0.720 | −0.0210 | 0.798 | 0.764 | −0.0331 |
| LightGBM (3 seeds) | 0.742 | 0.719 | −0.0230 | 0.802 | 0.770 | −0.0328 |
| Cross-ensemble | 0.743 | 0.717 | −0.0253 | 0.803 | 0.770 | −0.0327 |
| Logistic regression | 0.234 | 0.241 | +0.0077 | 0.160 | 0.166 | +0.0056 |

XGBoost and LightGBM lose similar Average Precision; the cross-ensemble tracks LightGBM closely. Logistic regression has low performance and is included for completeness. The loss sits in the interactor categories where they would be expected in the event of leakage, e.g. LightGBM F1: IxI 0.85 to 0.47, IxE 0.54 to 0.29.

### Robustness checks (group split; 10 splits, SHAP on one)

| Check | Notebook | Result |
|---|---|---|
| Corrected `gini_coefficient`, rule fixed in advance | `01` | Removing it lowered **validation** F1 on 3 of 10 splits (8 required): removed. The test set was not used |
| Labels known before the snapshot vs current labels | `07` | Test F1 −0.005 ± 0.007 (mean ± SD) |
| All provision-network features removed | `08` | Test F1 −0.008; transaction features alone carry most of the signal |
| SHAP by feature family | `09` | LayerZero transactions first, then Ethereum transactions, gas provider, gas provision tree, provision chain |

---

## Evaluation protocol

- **Split.** 49 % train, 21 % validation, 30 % test, stratified by class. The primary split is a
  **stratified group split**: wallets are grouped by gas provision tree (identified by the root of
  the unlabeled part of their gas provision chain), and no tree spans two partitions. The address-level random split
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
│   └── 20260930_etherscan_service_labels/  ← Etherscan check of funders of 50+ interactors
│
├── output/                                 ← created on first run (gitignored)
│   ├── master_df.parquet                   ← full feature table (434,786 rows, all computed features + labels)
│   ├── splits.npz                          ← train/val/test arrays and partition indices
│   ├── feature_list.json                   ← ordered list of the 62 model features
│   └── pred_*.parquet                      ← validation and test predictions from 03–06
│
├── results/                                ← metrics per notebook, with the code commit
├── review_support/                         ← labeled-address evidence: hand additions, Etherscan and Blockscout lookups
├── docs/REVISION_LEAKAGE.md                ← revision work items and findings log
├── code_review/                            ← plain-Python copies of the notebooks, for review only
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
for nb in [0-9][0-9]_*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb" --inplace \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1
done
```

The glob runs `00` to `11` in numerical order, which is the dependency order. Every notebook
uses the stratified group split except `11`, which also trains the models on the original random
split to measure the leakage. Each notebook rebuilds the feature table itself through
`sybil_pipeline.build_master_df`, so after `02_hyperparameter_search` has written
`results/02_hyperparameter_search.json`, the model notebooks can be run independently; `06` also
needs the predictions saved by `03` and `04`. `01` and `02` fix the feature set and the
hyperparameters; they must be rerun whenever `sybil_pipeline.py` changes, or `10` rejects the
results that depend on them. `10` checks every result from `00` to `09`; `11` runs last.

Runtimes on 4 cores, from the last full run (before the notebooks were renumbered): `00` 1.5 min;
`01` 12 min; `02` 82 min; `03` 5 min; `04` 4 min; `05` 2 min; `06` seconds; `07` 22 min; `08`
92 min; `09` 43 min; `10` under 1 min; `11` about 11 min. About 5 hours in total.

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
Provider and tree features are relational: every wallet in a gas provision tree shares the same
`provider_*`, `tree_size`, `branching_factor`, and similar values. Under an address-level random
split, wallets from one gas provision tree land in both train and test, so test performance partly
measures recognition of trees already seen in training. The group split keeps each tree in one
partition, and `sybil_pipeline.make_splits` asserts that no tree spans two partitions. Labeled
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
