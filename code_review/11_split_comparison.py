# %% [markdown]
# # Split Comparison — Leakage in the Original Random Split (not a benchmark result)
#
# The original paper split addresses at random. Wallets in the same funding group (the wallets that share the
# root of an unlabeled funding chain) share their provider and tree feature values, so under a random split
# test wallets can match training wallets exactly on those features (relational leakage). Reviewer 2 raised
# this.
#
# This notebook measures that leakage. It trains the same models, with the same hyperparameters (selected on
# the group split by `02`), on the original address-level random split, using the same code as the model
# notebooks (`sp.fit_xgb`, `sp.fit_lgbm`, `sp.fit_lr`, `sp.select_blend_weight`, `sp.evaluate`), and compares
# the results with the group-split results of `03`–`06`. The reported results are the group-split ones; the
# random split is run only here.
#
# Runs last: it reads `03`–`06`'s results and test predictions and checks that they trace to this commit.

# %% [markdown]
# ## Setup

# %%
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import sybil_pipeline as sp
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)

GROUP_NBS = ['03_xgboost_sybil', '04_lightgbm_sybil', '05_logistic_regression_sybil', '06_cross_ensemble_sybil']
prov = sp.provenance(GROUP_NBS)
print(prov.to_string(index=False))
assert prov['dependencies_unchanged'].all(), 'Group-split results were produced by different code: rerun them'

# %% [markdown]
# ## 1. Relational leakage under each split

# %%
df, tfm, labeled_anchors = sp.build_master_df('./data', verbose=False)
S = {m: sp.make_splits(df, sp.FEATS, method=m, seed=sp.SEED) for m in ['random', 'group']}
leak = {}
for m in ['random', 'group']:
    leak[m] = sp.leakage_report(df, S[m]); print()
print(pd.DataFrame(leak).T[['test_rows', 'test_sybils', 'test_rows_sharing_group_with_train',
                            'test_sybils_sharing_group_with_train_sybil']].to_string())

# %% [markdown]
# ## 2. Train the same models on the random split
#
# Same hyperparameters as the group-split models; thresholds and the blend weight chosen on the random
# split's own validation partition.

# %%
XGB_SELECTED, LGBM_SELECTED = sp.load_selected_params()
Sr = S['random']
xgb = sp.fit_xgb(Sr, XGB_SELECTED)
lgbm = sp.fit_lgbm(Sr, LGBM_SELECTED)
lr = sp.fit_lr(Sr)
w, _ = sp.select_blend_weight(Sr['y_val'], xgb['val'], lgbm['val'])
ens = {part: w * xgb[part] + (1 - w) * lgbm[part] for part in ['val', 'test']}
print(f'Blend weight on the random split: XGB {w:.2f}')
RANDOM = {name: sp.evaluate(name, Sr['y_val'], p['val'], Sr['y_test'], p['test']) for name, p in [
    ('XGBoost (tuned, 3-seed)', xgb), ('LightGBM (tuned, 3-seed)', lgbm),
    ('Logistic Regression (C=0.01, L1)', lr), ('Cross-Ensemble', ens)]}

# %% [markdown]
# ## 3. Random vs group split (test set)

# %%
GROUP = {}
for nb in GROUP_NBS:
    for r in sp.load_results(nb)['models']:
        GROUP.setdefault(r['name'], r)          # 06 repeats 03/04's single models; they are identical
rows = []
for name, r in RANDOM.items():
    g = GROUP[name]
    row = {'Model': name}
    for k in ['f1', 'auroc', 'ap']:
        row[(k, 'random')], row[(k, 'group')], row[(k, 'Δ')] = r[k], g[k], g[k] - r[k]
    rows.append(row)
comp = pd.DataFrame(rows).set_index('Model')
comp.columns = pd.MultiIndex.from_tuples(comp.columns)
print(comp.round(4).to_string())

# %% [markdown]
# ## 4. Where the leakage sits: F1 by taxonomy category
#
# Share of each taxonomy category in multi-wallet funding groups, then test F1 under each split. IxL wallets
# are their own group unless they fund other wallets.

# %%
PROB = {'XGBoost (tuned, 3-seed)': ('03_xgboost_sybil', 'prob_xgb', xgb),
        'LightGBM (tuned, 3-seed)': ('04_lightgbm_sybil', 'prob_lgbm', lgbm)}
cat_r = df.loc[Sr['idx_test'], 'category'].to_numpy()
y_r = np.asarray(Sr['y_test'])
rows = []
for name, (nb, col, m) in PROB.items():
    pg = sp.load_predictions(nb, 'test')
    for c in sorted(set(cat_r)):
        kr, kg = cat_r == c, pg['category'].to_numpy() == c
        f1r = f1_score(y_r[kr], (m['test'][kr] >= RANDOM[name]['threshold']).astype(int), zero_division=0)
        f1g = f1_score(pg['y'][kg], (pg[col][kg] >= GROUP[name]['threshold']).astype(int), zero_division=0)
        rows.append(dict(Model=name, category=c, f1_random=f1r, f1_group=f1g, delta=f1g - f1r))
by_cat = pd.DataFrame(rows)
size = df['funding_group'].map(df['funding_group'].value_counts())
print((size > 1).groupby(df['category']).agg(['sum', 'size', 'mean']).rename(
    columns={'sum': 'in_multi_wallet_groups', 'size': 'wallets', 'mean': 'share'}).round(3).to_string(), '\n')
print(by_cat.round(4).to_string(index=False))

# %% [markdown]
# ## Save results

# %%
sp.save_results(list(RANDOM.values()), '11_split_comparison', leakage=leak, extra=dict(
    random_split_xgb_weight=w, comparison={n: {k: comp.loc[n, k].to_dict() for k in ['f1', 'auroc', 'ap']} for n in comp.index},
    f1_by_category=by_cat.to_dict('records'), params=dict(xgb=XGB_SELECTED, lgbm=LGBM_SELECTED)))
