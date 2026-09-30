# LayerZero Sybil Detection — ML Pipeline

> Reproduces and extends: *Sybil Detection on Public Blockchains via XGBoost and Gas Provision Network Analysis* (Imig et al., 2025)

Trains three classifiers (XGBoost, LightGBM, Logistic Regression) and a cross-model ensemble on 434k Ethereum addresses to identify Sybil wallets in the LayerZero airdrop of June 2024. The full pipeline — from raw CSV files to evaluation plots — runs in under 30 minutes on a standard laptop CPU.

---

## Benchmark results

| Model | F1 | AUROC | Notes |
|---|---|---|---|
| XGBoost baseline | 0.720 | 0.970 | Default params, no tuning |
| LightGBM baseline | 0.723 | 0.969 | Default params, no tuning |
| Logistic Regression | 0.235 | 0.834 | Linear boundary; lower bound |
| **XGBoost tuned** | **0.735** | **0.974** | lr=0.1, depth=10, sub=0.7 |
| **LightGBM tuned** | **0.739** | **0.976** | lr=0.05, leaves=511, L2=1 |
| **Cross-ensemble** | **0.741** | **0.976** | 40 % XGB + 60 % LGBM |

Test set: 130,437 addresses · 4.19 % Sybil prevalence · threshold selected on validation set.

---

## Repository layout

```
layerzero_xgboost/
│
├── data/                                   ← Git LFS; run `git lfs pull` after cloning
│   ├── 20240915_final_sybil_list/          fcfs_list.csv (ground-truth labels)
│   ├── 20241013_hildobby_cex_evms/         CEX address list
│   ├── 20241104_layer0_sybil_features/     l0_features_*.csv (×5)
│   ├── 20241114_gas_provision/             gas provision network + query (readme.txt)
│   ├── 20241117_tree_features/             graph and tree features + featurization notebook
│   ├── 20241214_labeled_addresses/         labeled entity addresses
│   └── 20250208_cex_dex_indegree/          cex_dex_features_in_*.csv (×5)
│
├── output/                                 ← created on first run (gitignored)
│   ├── master_df.parquet                   ← full feature matrix (434k × 63 + labels)
│   ├── splits.npz                          ← train/val/test NumPy arrays
│   └── feature_list.json                   ← ordered list of 63 feature names
│
├── 00_data_pipeline.ipynb                  ← ① run first: builds master_df + splits
├── 01_xgboost_sybil.ipynb                  ← ② XGBoost (tuned 3-seed ensemble)
├── 02_lightgbm_sybil.ipynb                 ← ③ LightGBM (tuned 3-seed ensemble)
├── 03_logistic_regression_sybil.ipynb      ← ④ Logistic Regression baseline
├── 04_cross_ensemble_sybil.ipynb           ← ⑤ Cross-model ensemble (XGB + LGBM)
├── docs/REVISION_LEAKAGE.md                ← open work items for the current revision
├── legacy/                                 ← original 2025 notebook (Windows paths; reference only)
├── requirements.txt
└── README.md
```

---

## Prerequisites

### Python version
Python 3.11 or later (required by pandas 3).

### Packages

```bash
pip install -r requirements.txt
```

Minimum verified versions:

| Package | Min version |
|---|---|
| xgboost | 1.7 |
| lightgbm | 3.3 |
| scikit-learn | 1.0 |
| pandas | 1.4 |
| numpy | 1.21 |
| pyarrow | 8.0 (for parquet export) |

---

## Hardware requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| CPU | 4 cores | 8 cores |
| Disk | 2 GB free | 5 GB free |
| GPU | not needed | not needed |

> **RAM note.** The labeled-addresses file (`20241214_labeled_addresses.csv`) contains 9 million
> entries. The pipeline uses a streaming approach that retains only the ~3,700 addresses
> that appear as gas providers, so peak RAM stays under 2 GB. Do **not** load the full
> file into memory manually.

---

## Step-by-step instructions

### 1 — Fetch data files

The input files are stored with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

The notebooks expect the layout above, with `DATA_DIR = './data'` relative to the
notebook. If your data lives elsewhere, edit `DATA_DIR` in each notebook's config cell.

### 2 — Launch Jupyter

```bash
cd layerzero_xgboost/
jupyter lab          # or: jupyter notebook
```

### 3 — Run the data pipeline (required first)

Open **`00_data_pipeline.ipynb`** and run all cells top to bottom (`Kernel → Restart & Run All`).

**What it does (35 cells, ~25 seconds):**

| Step | Cell | Time |
|---|---|---|
| Load 5 L0 feature files | Step 1 | ~4 s |
| Build gas provision graph | Step 2 | ~3 s |
| Stream labeled anchors | Step 3 | ~5 s |
| Merge tree topology features | Step 4 | ~2 s |
| Merge CEX/DEX in-degree | Step 5 | ~1 s |
| Apply Sybil labels | Step 6 | ~1 s |
| Chain traversal | Step 7 | ~2 s |
| Quality checks + EDA | — | ~3 s |
| Split + upsample + export | — | ~4 s |

**Outputs written to `output/`:**
- `master_df.parquet` — full feature matrix
- `splits.npz` — `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`
- `feature_list.json` — 63 feature names in order

### 4 — Run the model notebooks

Each notebook is **self-contained**: it re-runs the full data pipeline internally.
If you have already completed Step 3, you can optionally skip the data-loading cells
by loading the pre-built splits instead (see the *Loading Pre-Built Splits* section
at the bottom of `00_data_pipeline.ipynb`).

#### `01_xgboost_sybil.ipynb` — ~4 minutes

Trains a 3-seed XGBoost ensemble using tuned parameters found by grid search.

```
lr=0.1  ·  max_depth=10  ·  subsample=0.7  ·  colsample_bytree=0.8
n_estimators=5000 (early stopping at ~475 rounds per seed)
Seeds: 42, 123, 456
```

Outputs: test metrics · feature importance plot · operating point table.

#### `02_lightgbm_sybil.ipynb` — ~4 minutes

Trains a 3-seed LightGBM ensemble.

```
lr=0.05  ·  num_leaves=511  ·  subsample=0.7  ·  subsample_freq=1
colsample_bytree=0.8  ·  reg_lambda=1
n_estimators=5000 (early stopping at ~340 rounds per seed)
Seeds: 42, 123, 456
```

> **LightGBM-specific parameters** that differ from XGBoost defaults:
> - `num_leaves` (not `max_depth`) controls model capacity
> - `subsample_freq=1` **must** be set alongside `subsample < 1.0` — otherwise
>   bagging is silently disabled
> - `reg_lambda` defaults to 0 in LightGBM (vs 1 in XGBoost); set to 1 explicitly

#### `03_logistic_regression_sybil.ipynb` — ~30 seconds

Linear baseline with L1 regularisation and StandardScaler. Included to confirm
the Sybil decision boundary is non-linear (F1 ≈ 0.235 vs ≥ 0.735 for tree models).

```
C=0.01  ·  penalty='l1'  ·  solver='liblinear'
```

#### `04_cross_ensemble_sybil.ipynb` — ~8 minutes

Trains all 6 models (3 XGB seeds + 3 LGBM seeds), then combines probability
scores with a weighted average:

```
final_score = 0.40 × P(Sybil | XGBoost) + 0.60 × P(Sybil | LightGBM)
```

Includes blend-weight sensitivity analysis and agreement zone breakdown.

---

## Expected runtimes (4-core CPU)

| Notebook | Data loading | Model training | Total |
|---|---|---|---|
| 00 Pipeline | 25 s | — | ~35 s |
| 01 XGBoost | 25 s | ~3.5 min | ~4 min |
| 02 LightGBM | 25 s | ~3.5 min | ~4 min |
| 03 LR | 25 s | 5 s | ~30 s |
| 04 Cross-ensemble | 25 s | ~7.5 min | ~8 min |
| **Total (fresh)** | | | **~17 min** |

If splits are pre-loaded from `splits.npz`, remove the ~25 s data loading from each model notebook.

---

## Input data files — descriptions

| File | Rows | Description |
|---|---|---|
| `l0_features_*.csv` (×5) | 434,788 total | Transaction, bridge, and timing features for each LayerZero interactor address. Snapshot: May 1 2024. Source: Flipside Crypto `fact_transactions_snapshot`. |
| `20241114_1633_layer0_provision_network_000000000000.csv` | 604,864 | Gas provision graph: which address first sent ETH to each interactor, enabling on-chain transactions. Source: BigQuery public blockchain dataset (recursive CTE). |
| `20241214_labeled_addresses.csv` | 9,054,105 | Known entities: centralized exchanges, decentralized exchanges, contracts, and other named accounts. Used to identify labeled anchors in provision chains. |
| `20241117_graph_and_tree_features.csv` | 434,111 | Pre-computed structural metrics on each address's gas provision subtree (fan-out, Gini coefficient, branching factor, tree depth, etc.). |
| `cex_dex_features_in_*.csv` (×5) | 434,789 total | Count of incoming transactions from centralized and decentralized exchanges per address. |
| `fcfs_list.csv` | 151,784 | LayerZero Foundation's final Sybil list (snapshot: Sept 15 2024). Ground-truth labels. Columns: `address`, reward addresses, forum links, timestamp, ZRO allocation. |

---

## Key design decisions

**Why upsample only the training set?**
Upsampling before splitting would leak duplicated minority-class rows into the
validation and test sets, inflating recall and F1. The pipeline splits first, then
upsamples the training set to 1:1 balance. Validation and test sets keep the
original ~4.2 % Sybil rate, so reported metrics reflect real-world conditions.

**Why stream labeled addresses instead of loading all 9M?**
The full set uses ~1.2 GB as a Python set. Only ~3,700 of the 202,899 unique
gas provider addresses appear in the labeled file. The pipeline streams the file
line-by-line and retains only those 3,700 entries, reducing memory to < 1 MB with
no change in the computed features.

**Why does LightGBM need `subsample_freq=1`?**
LightGBM's bagging is controlled by two parameters: `subsample` (the fraction) and
`subsample_freq` (how often to apply it, in rounds). The default `subsample_freq=0`
disables bagging entirely regardless of the `subsample` value. Setting
`subsample_freq=1` applies bagging every round, which is the behavior XGBoost uses
by default. Without this flag, `subsample=0.7` has no effect.

**Why `reg_lambda=1` for LightGBM?**
LightGBM defaults to `reg_lambda=0` (no L2 regularisation). XGBoost defaults to
`reg_lambda=1`. The grid search found that adding `reg_lambda=1` to LightGBM
improves AUROC by +0.001 and brings calibration in line with XGBoost. Omitting it
matches the LightGBM baseline (AUROC 0.969); including it reaches the tuned result
(AUROC 0.976).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'xgboost'`**
```bash
pip install xgboost lightgbm scikit-learn pyarrow
```

**`FileNotFoundError` on data files**
Check that `DATA_DIR` in the notebook config cell points to the folder containing
all input files. File names must match exactly (case-sensitive on Linux/macOS).

**`KeyError` or `AssertionError: Missing features`**
The quality-check cell in `00_data_pipeline.ipynb` will list which of the 63
features are absent. This usually means a data file is missing or has different
column names. Check that all 6 file groups are present.

**Memory error during labeled-address streaming**
Do not load `20241214_labeled_addresses.csv` with `pd.read_csv`. The pipeline reads
it line-by-line. If you see a memory error, ensure you are running the notebook
cells in order — the streaming cell must run before any cell that references
`labeled_anchors`.

**LightGBM training never converges (runs all 5,000 rounds)**
The default `learning_rate=0.1` with `num_leaves=31` (LightGBM defaults) produces
very slow improvement that never triggers early stopping. The tuned parameters
(`lr=0.05`, `num_leaves=511`) converge in ~340 rounds. If you are experimenting
with different parameters, set `learning_rate ≥ 0.05` or increase
`early_stopping_rounds` to 100+.

**Results differ slightly from the benchmark table**
Minor numeric differences are expected due to:
- Dataset size: this pipeline builds 434,787 addresses vs the paper's 425,044
  (a 2.3 % difference attributable to different source-data snapshots)
- XGBoost/LightGBM version differences in default hyperparameters
- The 3-seed ensemble reduces but does not eliminate random variance

---

Source code and labeled dataset archive:
[github.com/paven86/layerzero_xgboost](https://github.com/paven86/layerzero_xgboost)
