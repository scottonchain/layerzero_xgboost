# %% [markdown]
# # Tie-Out — Every Reported Number
#
# Collects the numbers the paper reports and checks that they trace to the code in this commit. Trains
# nothing: it reads `results/*.json` and `output/pred_*` written by notebooks `00` to `09`. Everything here
# is on the stratified group split; the original random split appears only in `11_split_comparison`.
#
# **Run order**: `00` → `01` → … → `09` → this notebook → `11_split_comparison`.
#
# **Checks, before any number is shown**
# 1. Every expected results file exists and none was produced from a dirty working tree.
# 2. For each result, the code it depends on is identical between the commit that produced it and the current
#    commit: `sybil_pipeline.py`, `requirements.txt` and `data/` byte for byte, and the code cells of its own
#    notebook (plus the search notebook, and for `06` the notebooks whose predictions it reads)
#    (`sp.provenance`). Saved outputs and markdown may differ.
# 3. `03`, `04` and `06` used exactly the hyperparameters selected by `02_hyperparameter_search`.
# 4. The individual XGBoost and LightGBM metrics in `06` equal those of `03` and `04`.
# 5. Each model's F1 recomputed from its saved test predictions equals the saved metric.

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sybil_pipeline as sp
pd.set_option('display.width', 220)

NOTEBOOKS = ['00_data_pipeline', '01_ablation_gini', '02_hyperparameter_search', '03_xgboost_sybil',
             '04_lightgbm_sybil', '05_logistic_regression_sybil', '06_cross_ensemble_sybil',
             '07_sensitivity_label_vintage', '08_ablation_families', '09_shap_importance']
R = {nb: sp.load_results(nb) for nb in NOTEBOOKS}
prov = sp.provenance(NOTEBOOKS)
print(prov.to_string(index=False))
assert prov['dependencies_unchanged'].all(), 'Some results were produced by different code: rerun them'
print('\nAll results trace to the code in this commit ✓')

# %%
# ── Hyperparameters used = hyperparameters selected on validation ──
HP = R['02_hyperparameter_search']
assert R['03_xgboost_sybil']['params'] == HP['xgb_selected']
assert R['04_lightgbm_sybil']['params'] == HP['lgbm_selected']
assert R['06_cross_ensemble_sybil']['params'] == dict(xgb=HP['xgb_selected'], lgbm=HP['lgbm_selected'])
print('XGBoost  (selected on validation):', HP['xgb_selected'])
print('LightGBM (selected on validation):', HP['lgbm_selected'])
print('03, 04, 06 used these ✓')

# ── 06 individual models = 03 / 04 ──
ens = {r['name']: r for r in R['06_cross_ensemble_sybil']['models']}
for nb in ['03_xgboost_sybil', '04_lightgbm_sybil']:
    single = R[nb]['models'][0]
    for k in ['threshold', 'precision', 'recall', 'f1', 'auroc', 'ap']:
        assert abs(ens[single['name']][k] - single[k]) < 1e-9, (nb, k)
print('06 individual models match 03 and 04 ✓')

# %% [markdown]
# ## Dataset and partitions (paper Tables 1 and 5)

# %%
D = R['00_data_pipeline']
print(f"Addresses: {D['n_addresses']:,}  |  Sybil: {D['n_sybil']:,} ({D['n_sybil']/D['n_addresses']*100:.2f}%)")
print(f"Provision edges dropped by the snapshot cutoff: {D['provision_edges_after_cutoff']}")
fg = D['provision_trees']
print(f"Gas provision trees: {fg['n']:,}  |  multi-wallet: {fg['multi_wallet']:,}  |  largest: {fg['largest']:,} wallets"
      f"  |  Sybils in multi-wallet trees: {fg['sybils_in_multi']:,}\n")
cat = pd.DataFrame(D['categories']).T
cat['pct_of_total'] = (cat['n'] / cat['n'].sum() * 100).round(2)
cat['sybil_rate_pct'] = (cat['sybil'] / cat['n'] * 100).round(2)
print('Taxonomy categories:'); print(cat.to_string(), '\n')
print('Partitions (before upsampling):'); print(pd.DataFrame(D['splits']).T.to_string())
L = D['leakage']
print(f"\nRelational leakage (row overlap is asserted 0): test rows sharing a gas provision tree with train "
      f"{L['test_rows_sharing_tree_with_train']:,} of {L['test_rows']:,}; test Sybils sharing one with a train Sybil "
      f"{L['test_sybils_sharing_tree_with_train_sybil']:,} of {L['test_sybils']:,}")

# %% [markdown]
# ## Test-set metrics (paper Table 3)
#
# Thresholds are the F1-maximizing thresholds on validation; the blend weight in `06` is also chosen on
# validation. AP is average precision (area under the precision-recall curve). FPR is the false-positive
# rate among true non-Sybils at the model's threshold.

# %%
MODELS = {'XGBoost (tuned, 3-seed)': ('03_xgboost_sybil', 'prob_xgb'),
          'LightGBM (tuned, 3-seed)': ('04_lightgbm_sybil', 'prob_lgbm'),
          'Logistic Regression (C=0.01, L1)': ('05_logistic_regression_sybil', 'prob_lr'),
          'Cross-Ensemble': ('06_cross_ensemble_sybil', 'prob_ens')}

def result(name):
    nb, _ = MODELS[name]
    return next(x for x in R[nb]['models'] if x['name'] == name)

def preds(name):
    nb, col = MODELS[name]
    p = sp.load_predictions(nb, 'test')
    return p[['addr', 'category', 'y']].assign(prob=p[col])

rows = []
for name in MODELS:
    r, p = result(name), preds(name)
    yhat = (p['prob'] >= r['threshold']).astype(int)
    assert abs(f1_score(p['y'], yhat) - r['f1']) < 1e-9, name   # predictions match saved metrics
    tn, fp, fn, tp = confusion_matrix(p['y'], yhat).ravel()
    rows.append(dict(Model=name, **{k: r[k] for k in ['precision', 'recall', 'f1', 'auroc', 'ap', 'threshold']},
                     fpr=fp / (fp + tn), fp=int(fp)))
table = pd.DataFrame(rows).set_index('Model')
print(table.round(4).to_string())
E = R['06_cross_ensemble_sybil']
print(f"\nBlend: XGB weight = {E['xgb_weight']:.2f} (LightGBM {1 - E['xgb_weight']:.2f})  |  "
      f"disagreement at 0.5: {E['disagreement_n']:,} ({E['disagreement_pct']:.2f}%)  |  "
      f"rounds XGB {E['rounds']['xgb']}  LGBM {E['rounds']['lgbm']}")

# %% [markdown]
# ## Per-category metrics (paper Table 6)
#
# Each model at its own validation-selected threshold, evaluated within each taxonomy category of the test set.

# %%
rows = []
for name in MODELS:
    r, p = result(name), preds(name)
    for c, g in p.groupby('category'):
        yhat = (g['prob'] >= r['threshold']).astype(int)
        tn, fp, fn, tp = confusion_matrix(g['y'], yhat, labels=[0, 1]).ravel()
        rows.append({'Model': name, 'category': c, 'n': len(g), 'sybil': int(g['y'].sum()),
                     'precision': precision_score(g['y'], yhat, zero_division=0),
                     'recall': recall_score(g['y'], yhat, zero_division=0),
                     'f1': f1_score(g['y'], yhat, zero_division=0),
                     'auroc': roc_auc_score(g['y'], g['prob']) if g['y'].nunique() == 2 else np.nan,
                     'fpr': fp / (fp + tn) if (fp + tn) else np.nan})
print(pd.DataFrame(rows).set_index(['Model', 'category']).round(4).to_string())

# %% [markdown]
# ## Hyperparameter search (paper Tables 7 and 8; validation only)

# %%
for model in ['xgboost', 'lightgbm']:
    t = pd.read_csv(f'results/02_search_{model}.csv')
    print(f'{model}: {len(t)} configurations; top 5 by validation F1')
    print(t.sort_values(['val_f1', 'val_ap'], ascending=False).head(5).round(4).to_string(index=False), '\n')

# %% [markdown]
# ## Robustness checks (group split, 10 split seeds unless noted)
#
# - `01`: keep-or-drop test for the corrected `gini_coefficient` (rule fixed in advance; validation only).
# - `07`: labels as known before the snapshot vs current labels.
# - `08`: each feature family removed, and each family alone (report only).
# - `09`: mean |SHAP| by feature family and taxonomy category (test partition, split seed 42).

# %%
G = R['01_ablation_gini']
print(f"Gini (01): removing it lowered validation F1 on {G['splits_lowered']} of {len(G['per_split'])} splits "
      f"(one-sided sign test p = {G['sign_test_p']:.3f}; mean change {G['mean_f1_change']:+.4f}). Decision: {G['decision']}")
V = R['07_sensitivity_label_vintage']
print(f"\nLabel vintage (07): labeled addresses in the network, current {V['labeled_current']:,}, "
      f"pre-snapshot {V['labeled_presnapshot']:,}; wallets with any feature changed {V['wallets_changed']:,}")
print('Pre-snapshot minus current, mean ± SD over splits: ' +
      ', '.join(f"{k} {V['diff_mean'][k]:+.4f} ± {V['diff_sd'][k]:.4f}" for k in V['diff_mean']))
F = pd.DataFrame(R['08_ablation_families']['summary']).T
F = F[['n_features_mean', 'val_f1_mean', 'test_f1_mean', 'test_f1_std', 'test_ap_mean', 'test_auroc_mean',
       'test_f1_change_vs_all_mean']]
print('\nFeature families (08), mean and SD over splits:'); print(F.round(4).to_string())
Sh = pd.DataFrame(R['09_shap_importance']['mean_abs_shap'])
print('\nSum of mean |SHAP| by family (09):')
print(Sh.groupby('family')[['All', 'IxL', 'IxI', 'IxE']].sum().sort_values('All', ascending=False).round(4).to_string())

# %%
# ── Markdown table for the README / paper ─────────────────────
lines = ['| Model | Precision | Recall | F1 | AUROC | AP | FPR |', '|---|---|---|---|---|---|---|']
for model, row in table.iterrows():
    lines.append(f"| {model} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
                 f"{row['auroc']:.3f} | {row['ap']:.3f} | {row['fpr']:.4f} |")
print('\n'.join(lines))
print(f"\nSource: results/ checked against commit {sp.CODE_COMMIT}")
