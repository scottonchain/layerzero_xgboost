# %% [markdown]
# # XGBoost Sybil Detector — LayerZero
#
# XGBoost, 3-seed ensemble (seeds 42, 123, 456), 5,000-round ceiling with early stopping (50 rounds) on
# validation log-loss, on the stratified group split.
#
# Training, evaluation and helper code live in `sybil_pipeline.py` (`sp.fit_xgb`, `sp.evaluate`), the same
# code `11_split_comparison` runs on the original random split.
#
# **Results**: test metrics in section 3, saved to `results/03_xgboost_sybil.json`.

# %% [markdown]
# ## Setup

# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sybil_pipeline as sp
warnings.filterwarnings('ignore')

DATA_DIR = './data'
FEATS = sp.FEATS   # 62 model features (defined in sybil_pipeline.py)

# %% [markdown]
# ## 1. Data and split
#
# Stratified group split on funding groups (49 % train, 21 % validation, 30 % test); the Sybil class is
# upsampled to 1:1 in the training partition only.

# %%
# ── Feature table (steps 1-7; see 00_data_pipeline) and the stratified group split ──
df, tfm, labeled_anchors = sp.build_master_df(DATA_DIR)
S = sp.make_splits(df, FEATS, method='group', seed=sp.SEED)   # asserts no funding group spans two partitions
print(sp.split_summary(df, S).to_string(index=False))
print(f"Train after upsampling: {len(S['X_train']):,} rows ({S['y_train'].mean()*100:.1f}% Sybil)\n")
leak = sp.leakage_report(df, S)

# %% [markdown]
# ## 2. Train (3 seeds, early stopping on validation)
#
# Hyperparameters selected on validation by `02_hyperparameter_search`.

# %%
XGB_SELECTED, LGBM_SELECTED = sp.load_selected_params()
PARAMS = XGB_SELECTED
print('Tuned hyperparameters:', PARAMS)
m = sp.fit_xgb(S, PARAMS)

# %% [markdown]
# ## 3. Evaluate on the test set
#
# Threshold: the F1-maximizing threshold on validation.

# %%
res = sp.evaluate('XGBoost (tuned, 3-seed)', S['y_val'], m['val'], S['y_test'], m['test'])
sp.print_metrics(res)
sp.plot_curves([res])

# %% [markdown]
# ## 4. Feature importances
#
# Total gain: the reduction in training loss summed over every split on the feature, normalized to sum to 1
# for each model and averaged over the three seeds (`sp.gain_importance`). Gain credits correlated features
# unevenly; `09_shap_importance` reports SHAP values and sums them by feature family.

# %%
imp_df = sp.gain_importance(m['models'], FEATS)
print('Top 20 features (normalized total gain, mean over 3 seeds):')
print(imp_df.head(20).to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 7))
top20 = imp_df.head(20)
ax.barh(top20['Feature'][::-1], top20['NormGain'][::-1], color='#1565C0')
ax.set(xlabel='Normalized total gain (mean over 3 seeds)', title='XGBoost — Top 20 Feature Importances')
ax.grid(axis='x', alpha=0.25)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Operating points
#
# Precision, recall and counts on the test set at fixed thresholds and at the threshold chosen on validation
# (whose row equals the reported test metrics).

# %%
print(sp.operating_points(S['y_test'], m['test'], res['threshold']).round(4).to_string(index=False))

# %% [markdown]
# ## Save results
#
# Scalar metrics go to `results/03_xgboost_sybil.json` with the code commit, so every number in the paper can be traced
# to one run. Validation and test probabilities go to `output/` for `06_cross_ensemble_sybil` and `10_tie_out`.

# %%
sp.save_results([res], '03_xgboost_sybil', leakage=leak,
                extra=dict(params=PARAMS, rounds=m['rounds'],
                           importance=imp_df.set_index('Feature')['NormGain'].to_dict()))
sp.save_predictions(df, S, '03_xgboost_sybil', prob_xgb=m)
