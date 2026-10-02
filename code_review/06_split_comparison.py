# %% [markdown]
# # Split Comparison and Tie-Out — Random vs Group
#
# Collects the numbers the paper reports and checks that they trace to the code in this commit.
# Trains nothing: it reads `results/*.json` (written by notebooks 00 to 10) and
# `output/pred_*.parquet` (test predictions written by 01 to 04).
#
# - **random**: address-level stratified split, the original paper's evaluation.
# - **group**: stratified group split on funding groups (see `00_data_pipeline`), so no funding
#   cluster spans train and test. The gap between the two measures relational leakage.
#
# **Run order**: `00` → `08` → `05` → `01`–`04` with `SPLIT_METHOD=group` and again with
# `SPLIT_METHOD=random` → `07`, `09`, `10` → this notebook.
#
# **Checks, before any number is shown**
# 1. Every expected results file exists and none was produced from a dirty working tree.
# 2. For each result, the code it depends on is identical between the commit that produced it and
#    the current commit: `sybil_pipeline.py`, `requirements.txt`, and `data/` byte for byte, and the
#    code cells of its own notebook (and, for 01/02/04/07/09/10, of the search notebook). Saved outputs and
#    markdown may differ.
# 3. 01, 02, and 04 used exactly the hyperparameters selected by `05_hyperparameter_search`.
# 4. The individual XGBoost and LightGBM metrics inside 04 equal those from 01 and 02.

# %%
import json, os, subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

pd.set_option('display.width', 220)
RESULTS_DIR, OUTPUT_DIR = './results', './output'
METHODS = ['random', 'group']
NOTEBOOK_FILE = {'00_data_pipeline': '00_data_pipeline.ipynb', '01_xgboost': '01_xgboost_sybil.ipynb',
                 '02_lightgbm': '02_lightgbm_sybil.ipynb', '03_logistic_regression': '03_logistic_regression_sybil.ipynb',
                 '04_cross_ensemble': '04_cross_ensemble_sybil.ipynb',
                 '05_hyperparameter_search': '05_hyperparameter_search.ipynb',
                 '07_sensitivity_label_vintage': '07_sensitivity_label_vintage.ipynb',
                 '08_ablation_gini': '08_ablation_gini.ipynb', '09_ablation_families': '09_ablation_families.ipynb',
                 '10_shap_importance': '10_shap_importance.ipynb'}
MODELS = ['01_xgboost', '02_lightgbm', '03_logistic_regression', '04_cross_ensemble']

def load(name, method):
    path = os.path.join(RESULTS_DIR, f'{name}_{method}.json')
    assert os.path.exists(path), f'Missing {path}: run {NOTEBOOK_FILE[name]} with SPLIT_METHOD={method}'
    with open(path) as f:
        return json.load(f)

ROBUSTNESS = ['07_sensitivity_label_vintage', '08_ablation_gini', '09_ablation_families', '10_shap_importance']
R = {(nb, 'group'): load(nb, 'group') for nb in ['00_data_pipeline', '05_hyperparameter_search'] + ROBUSTNESS}
for nb in MODELS:
    for m in METHODS:
        R[nb, m] = load(nb, m)

import nbformat

def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True)

def code_cells(ref, path):
    # code-cell sources of a notebook at a git ref (None = working tree); outputs and markdown ignored
    text = git('show', f'{ref}:{path}').stdout if ref else open(path).read()
    return [c.source for c in nbformat.reads(text, as_version=4).cells if c.cell_type == 'code']

head = git('rev-parse', '--short', 'HEAD').stdout.strip()
print(f'Current commit: {head}\n')
rows = []
for (nb, m), r in R.items():
    commit = r['code_commit']
    assert not commit.endswith('-dirty'), f'{nb} ({m}) ran from a dirty working tree'
    files_same = git('diff', '--quiet', commit, 'HEAD', '--',
                     'sybil_pipeline.py', 'requirements.txt', 'data').returncode == 0
    nbs = [NOTEBOOK_FILE[nb]] + ([NOTEBOOK_FILE['05_hyperparameter_search']]
                                 if nb in ('01_xgboost', '02_lightgbm', '04_cross_ensemble', '07_sensitivity_label_vintage',
                                           '09_ablation_families', '10_shap_importance') else [])
    code_same = all(code_cells(commit, f) == code_cells(None, f) for f in nbs)
    rows.append(dict(notebook=nb, split=m, commit=commit, dependencies_unchanged=files_same and code_same))
prov = pd.DataFrame(rows)
print(prov.to_string(index=False))
assert prov['dependencies_unchanged'].all(), 'Some results were produced by different code: rerun them'
print('\nAll results trace to the code in this commit ✓')

# %%
# ── Hyperparameters used = hyperparameters selected on validation ──
HP = R['05_hyperparameter_search', 'group']
for m in METHODS:
    assert R['01_xgboost', m]['params'] == HP['xgb_selected'], ('01', m)
    assert R['02_lightgbm', m]['params'] == HP['lgbm_selected'], ('02', m)
    assert R['04_cross_ensemble', m]['params'] == dict(xgb=HP['xgb_selected'], lgbm=HP['lgbm_selected']), ('04', m)
print('XGBoost  (selected on validation):', HP['xgb_selected'])
print('LightGBM (selected on validation):', HP['lgbm_selected'])
print('01, 02, 04 used these under both splits ✓')

# ── 04 individual models = 01 / 02 ────────────────────────────
for m in METHODS:
    ens = {r['name']: r for r in R['04_cross_ensemble', m]['models']}
    for nb in ['01_xgboost', '02_lightgbm']:
        single = R[nb, m]['models'][0]
        for k in ['threshold', 'precision', 'recall', 'f1', 'auroc', 'ap']:
            assert abs(ens[single['name']][k] - single[k]) < 1e-9, (m, nb, k)
print('04 individual models match 01 and 02 under both splits ✓')

# %% [markdown]
# ## Dataset and partitions (paper Tables 1 and 5)

# %%
D = R['00_data_pipeline', 'group']
print(f"Addresses: {D['n_addresses']:,}  |  Sybil: {D['n_sybil']:,} ({D['n_sybil']/D['n_addresses']*100:.2f}%)")
print(f"Provision edges dropped by the snapshot cutoff: {D['provision_edges_after_cutoff']}")
fg = D['funding_groups']
print(f"Funding groups: {fg['n']:,}  |  multi-wallet: {fg['multi_wallet']:,}  |  largest: {fg['largest']:,} wallets"
      f"  |  Sybils in multi-wallet groups: {fg['sybils_in_multi']:,}\n")

cat = pd.DataFrame(D['categories']).T
cat['pct_of_total'] = (cat['n'] / cat['n'].sum() * 100).round(2)
cat['sybil_rate_pct'] = (cat['sybil'] / cat['n'] * 100).round(2)
print('Taxonomy categories:'); print(cat.to_string(), '\n')

for m in METHODS:
    print(f'Partitions ({m} split, before upsampling):')
    print(pd.DataFrame(D['splits'][m]).T.to_string(), '\n')

# %% [markdown]
# ## Relational leakage in each split

# %%
leak = pd.DataFrame(D['leakage']).T[['test_rows', 'test_sybils',
                                      'test_rows_sharing_group_with_train',
                                      'test_sybils_sharing_group_with_train_sybil']]
print(leak.to_string())

# %% [markdown]
# ## Test-set metrics, random vs group split
#
# Thresholds are the F1-maximising thresholds on each split's own validation set; the blend weight
# in 04 is also chosen on validation. AP is average precision (area under the precision-recall
# curve). FPR is the false-positive rate among true non-Sybils at the model's threshold.

# %%
PRED_COL = {'XGBoost (tuned, 3-seed)': ('01_xgboost', 'prob_xgb'),
            'LightGBM (tuned, 3-seed)': ('02_lightgbm', 'prob_lgbm'),
            'Logistic Regression (C=0.01, L1)': ('03_logistic_regression', 'prob_lr'),
            'Cross-Ensemble': ('04_cross_ensemble', 'prob_ens')}

def result(name, m):
    nb, _ = PRED_COL[name]
    return next(x for x in R[nb, m]['models'] if x['name'] == name)

def preds(name, m):
    nb, col = PRED_COL[name]
    p = pd.read_parquet(os.path.join(OUTPUT_DIR, f'pred_{nb}_{m}.parquet'))
    return p[['addr', 'category', 'y']].assign(prob=p[col])

rows = []
for name in PRED_COL:
    row = {'Model': name}
    for m in METHODS:
        r = result(name, m); p = preds(name, m)
        yhat = (p['prob'] >= r['threshold']).astype(int)
        assert abs(f1_score(p['y'], yhat) - r['f1']) < 1e-9, (name, m)   # predictions match saved metrics
        tn, fp, fn, tp = confusion_matrix(p['y'], yhat).ravel()
        for k in ['precision', 'recall', 'f1', 'auroc', 'ap', 'threshold']:
            row[(m, k)] = r[k]
        row[(m, 'fpr')] = fp / (fp + tn)
        row[(m, 'fp')] = int(fp)
    rows.append(row)
table = pd.DataFrame(rows).set_index('Model')
table.columns = pd.MultiIndex.from_tuples(table.columns)
for k in ['f1', 'auroc', 'ap']:
    table[('Δ group−random', k)] = table[('group', k)] - table[('random', k)]
print(table.round(4).to_string())

# %%
# ── Blend weight and model agreement (04), per split ──────────
for m in METHODS:
    r = R['04_cross_ensemble', m]
    print(f"{m:<6}: XGB weight = {r['xgb_weight']:.2f} (LightGBM {1 - r['xgb_weight']:.2f})  |  "
          f"disagreement at 0.5: {r['disagreement_n']:,} ({r['disagreement_pct']:.2f}%)  |  "
          f"rounds XGB {r['rounds']['xgb']}  LGBM {r['rounds']['lgbm']}")

# %% [markdown]
# ## Per-category metrics (paper Table 6)
#
# Each model at its own validation-selected threshold, evaluated within each taxonomy category of
# the test set.

# %%
rows = []
for name in PRED_COL:
    for m in METHODS:
        r = result(name, m); p = preds(name, m)
        for c, g in p.groupby('category'):
            yhat = (g['prob'] >= r['threshold']).astype(int)
            tn, fp, fn, tp = confusion_matrix(g['y'], yhat, labels=[0, 1]).ravel()
            rows.append({'Model': name, 'split': m, 'category': c, 'n': len(g), 'sybil': int(g['y'].sum()),
                         'precision': precision_score(g['y'], yhat, zero_division=0),
                         'recall': recall_score(g['y'], yhat, zero_division=0),
                         'f1': f1_score(g['y'], yhat, zero_division=0),
                         'auroc': roc_auc_score(g['y'], g['prob']) if g['y'].nunique() == 2 else np.nan,
                         'fpr': fp / (fp + tn) if (fp + tn) else np.nan})
per_cat = pd.DataFrame(rows)
print(per_cat.pivot_table(index=['Model', 'category'], columns='split',
                          values=['n', 'sybil', 'precision', 'recall', 'f1', 'auroc', 'fpr']).round(4).to_string())

# %% [markdown]
# ## Hyperparameter search (paper Tables 7 and 8; validation only)

# %%
for model in ['xgboost', 'lightgbm']:
    t = pd.read_csv(os.path.join(RESULTS_DIR, f'05_search_{model}.csv'))
    print(f'{model}: {len(t)} configurations; top 5 by validation F1')
    print(t.sort_values(['val_f1', 'val_ap'], ascending=False).head(5).round(4).to_string(index=False), '\n')
print('Selected on validation:', {k: v for k, v in HP.items() if k.endswith('_selected') or k.endswith('_selected_val')})

# %% [markdown]
# ## Robustness checks (07 to 10; group split, 10 split seeds unless noted)
#
# - `08`: keep-or-drop test for the corrected `gini_coefficient` (rule fixed before the run; validation only).
# - `07`: labels as known before the snapshot vs current labels (sensitivity for the labeling rule).
# - `09`: each feature family removed, and each family alone (report only).
# - `10`: mean |SHAP| by feature family and taxonomy category (test partition, split seed 42).

# %%
G = R['08_ablation_gini', 'group']
print(f"Gini (08): removing it lowered validation F1 on {G['splits_lowered']} of {len(G['per_split'])} splits "
      f"(one-sided sign test p = {G['sign_test_p']:.3f}; mean change {G['mean_f1_change']:+.4f}). Decision: {G['decision']}")
V = R['07_sensitivity_label_vintage', 'group']
print(f"\nLabel vintage (07): labeled addresses in the network, current {V['labeled_current']:,}, "
      f"pre-snapshot {V['labeled_presnapshot']:,}; wallets with any feature changed {V['wallets_changed']:,}")
print('Pre-snapshot minus current, mean ± SD over splits: ' +
      ', '.join(f"{k} {V['diff_mean'][k]:+.4f} ± {V['diff_sd'][k]:.4f}" for k in V['diff_mean']))
F = pd.DataFrame(R['09_ablation_families', 'group']['summary']).T
F = F[['n_features_mean', 'val_f1_mean', 'test_f1_mean', 'test_f1_std', 'test_ap_mean', 'test_auroc_mean',
       'test_f1_change_vs_all_mean']]
print('\nFeature families (09), mean and SD over splits:'); print(F.round(4).to_string())
S = pd.DataFrame(R['10_shap_importance', 'group']['mean_abs_shap'])
print('\nSum of mean |SHAP| by family (10):')
print(S.groupby('family')[['All', 'IxL', 'IxI', 'IxE']].sum().sort_values('All', ascending=False).round(4).to_string())

# %%
# ── Markdown table for the README / paper ─────────────────────
lines = ['| Model | Split | Precision | Recall | F1 | AUROC | AP | FPR |', '|---|---|---|---|---|---|---|---|']
for model, row in table.iterrows():
    for m in METHODS:
        lines.append(f"| {model} | {m} | {row[(m, 'precision')]:.3f} | {row[(m, 'recall')]:.3f} | "
                     f"{row[(m, 'f1')]:.3f} | {row[(m, 'auroc')]:.3f} | {row[(m, 'ap')]:.3f} | {row[(m, 'fpr')]:.4f} |")
print('\n'.join(lines))
print(f'\nSource: results/ checked against commit {head}')
