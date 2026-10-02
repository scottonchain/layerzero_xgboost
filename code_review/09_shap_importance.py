# %% [markdown]
# # Feature Importance — SHAP by Taxonomy Category
#
# SHAP values (TreeExplainer) for the selected LightGBM configuration trained on the primary group split
# (seed 42, model seed 42), computed on the test partition. Mean |SHAP| ranks features overall and within
# each taxonomy category (IxL, IxI, IxE). SHAP splits credit among correlated features consistently,
# unlike gain-based importance; this replaces the gain-based Table 11.

# %%
import json, time, warnings
import numpy as np
import pandas as pd
import sybil_pipeline as sp
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
DATA_DIR, SEED = './data', sp.SEED
import shap, lightgbm as lgb
import matplotlib.pyplot as plt
_, LGBM_SELECTED = sp.load_selected_params()
df, _, _ = sp.build_master_df(DATA_DIR, verbose=False)
S = sp.make_splits(df, sp.FEATS, method='group', seed=SEED)
m = sp.lgbm_model(LGBM_SELECTED, SEED)   # same settings as 04_lightgbm_sybil
m.fit(S['X_train'], S['y_train'], eval_set=[(S['X_val'], S['y_val'])],
      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
X = S['X_test']
sv = shap.TreeExplainer(m).shap_values(X)
sv = sv[1] if isinstance(sv, list) else sv
cat = df.loc[S['idx_test'], 'category'].to_numpy()
imp = pd.DataFrame({'All': np.abs(sv).mean(0)}, index=X.columns)
for c in ['IxL', 'IxI', 'IxE']:
    imp[c] = np.abs(sv[cat == c]).mean(0)
fam = {f: k for k, v in sp.FEATURE_FAMILIES.items() for f in v}
imp['family'] = imp.index.map(fam)
imp = imp.sort_values('All', ascending=False)
print(imp.head(25).round(4).to_string())
print('\nMean |SHAP| by family:'); print(imp.groupby('family')[['All', 'IxL', 'IxI', 'IxE']].sum().round(4).to_string())

# %%
top = imp.head(20).index
fig, ax = plt.subplots(figsize=(8, 7))
plt.sca(ax)
shap.summary_plot(sv[:, [list(X.columns).index(f) for f in top]], X[top], show=False, plot_size=None)
plt.title('SHAP values, top 20 features (test partition, group split)')
plt.tight_layout(); plt.show()

# %%
sp.save_results([], '09_shap_importance', extra=dict(
    mean_abs_shap=imp.reset_index().rename(columns={'index': 'feature'}).to_dict('records'), params=LGBM_SELECTED))
