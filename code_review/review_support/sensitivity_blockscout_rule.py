# %% [markdown]
# # Robustness — Automated Labeling Rule (Blockscout tags)
#
# The primary labeling rule uses public label lists plus an Etherscan check of every other funder of at least 50
# LayerZero interactors (tracker A12). This notebook replaces the Etherscan step with a rule that has no
# interactor threshold and no human step: every address that funded at least 2 addresses is labeled if
# Blockscout's tag service gives it a service category tag (`build_blockscout_rule.py`, rule fixed before this
# run). The public lists are the same. Everything else (features, funding groups, splits, the selected LightGBM
# configuration) follows the primary pipeline; `sybil_pipeline.py` is not modified.
#
# Comparison: LightGBM on the same 10 group-split seeds as `07`, one model seed, primary vs automated labels.

# %%
import os, json, warnings
os.chdir('..')                      # run from the repo root so the pipeline's relative paths resolve
import numpy as np, pandas as pd
import sybil_pipeline as sp
warnings.filterwarnings('ignore'); pd.set_option('display.width', 200)
_, LGBM_SELECTED = sp.load_selected_params()
cur, _, anchors_cur = sp.build_master_df('./data', verbose=False)
_orig = sp.data_paths
def _alt(data_dir='./data'):
    p = _orig(data_dir); p['etherscan_services'] = 'review_support/blockscout_rule.csv'; return p
sp.data_paths = _alt
try:
    alt, _, anchors_alt = sp.build_master_df('./data', verbose=False)
finally:
    sp.data_paths = _orig
print(f'Labeled addresses in the network: primary {len(anchors_cur):,}, automated {len(anchors_alt):,} '
      f'(only primary {len(anchors_cur - anchors_alt)}, only automated {len(anchors_alt - anchors_cur)})')
b, a = cur.set_index('addr'), alt.set_index('addr').loc[cur['addr']]
changed = pd.Series(False, index=b.index)
for f in sp.FEATS:
    changed |= ~np.isclose(b[f].astype(float), a[f].astype(float), rtol=1e-9, atol=1e-12)
print(f'Wallets with any feature changed: {changed.sum():,} ({int(b.loc[changed, "sybil"].sum())} Sybil)')
print(pd.crosstab(b.loc[changed, 'category'], a.loc[changed, 'category']).to_string())

# %% [markdown]
# ## Primary vs automated labels (LightGBM, 10 group splits, 1 model seed)

# %%
rows = []
for s in sp.SPLIT_SEEDS:
    for name, d in [('primary', cur), ('automated', alt)]:
        S = sp.make_splits(d, sp.FEATS, method='group', seed=s)
        rows.append({'split_seed': s, 'labels': name, **sp.lgbm_fit_eval(S, sp.FEATS, LGBM_SELECTED)})
res = pd.DataFrame(rows)
p = res.pivot(index='split_seed', columns='labels', values=['val_f1', 'test_f1', 'test_ap', 'test_auroc'])
for m in ['val_f1', 'test_f1', 'test_ap', 'test_auroc']:
    p[(m, 'diff')] = p[(m, 'automated')] - p[(m, 'primary')]
print(p.round(4).to_string())
summary = p.xs('diff', axis=1, level=1).agg(['mean', 'std'])
print('\nAutomated minus primary, mean and SD over 10 splits:'); print(summary.round(4).to_string())

# %%
sp.save_results([], 'review_blockscout_rule', 'group', out_dir='review_support', extra=dict(
    labeled_primary=len(anchors_cur), labeled_automated=len(anchors_alt),
    only_primary=sorted(anchors_cur - anchors_alt), only_automated=sorted(anchors_alt - anchors_cur),
    wallets_changed=int(changed.sum()), per_split=res.to_dict('records'),
    diff_mean=summary.loc['mean'].to_dict(), diff_sd=summary.loc['std'].to_dict(), params=LGBM_SELECTED))
