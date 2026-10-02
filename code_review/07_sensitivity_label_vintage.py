# %% [markdown]
# # Sensitivity — Label Vintage
#
# The labeled set uses current public label lists plus Etherscan service tags
# (`sp.stream_labeled_anchors`, tracker A12). Some labels were published after the LayerZero snapshot
# (2024-05-01). A label records which entity controls an address, a fixed fact; later lists identify
# existing entities rather than record later behavior. This notebook tests whether the choice matters:
#
# 1. **Vintage comparison.** Rebuild every feature, the taxonomy, and the funding groups with labels known
#    before the snapshot only (`label_vintage='presnapshot'`: pre-snapshot Spellbook commits, CEX entries
#    added on or before 2024-05-01, no Etherscan step), and compare LightGBM on 10 group splits.
# 2. **Independence from the Sybil list.** Compare the Sybil share of wallets funded by CEX addresses added
#    to the list before vs after the LayerZero Sybil list was published (2024-09-15). Similar shares are
#    consistent with labels curated independently of the Sybil outcome.

# %%
import json, time, warnings
import numpy as np
import pandas as pd
import sybil_pipeline as sp
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
DATA_DIR, SEED = './data', sp.SEED
_, LGBM_SELECTED = sp.load_selected_params()
cur, tfm, anchors_cur = sp.build_master_df(DATA_DIR, verbose=False, label_vintage='current')
pre, _, anchors_pre = sp.build_master_df(DATA_DIR, verbose=False, label_vintage='presnapshot')
print(f'Labeled addresses in network: current {len(anchors_cur):,}, pre-snapshot {len(anchors_pre):,}')
b, a = cur.set_index('addr'), pre.set_index('addr').loc[cur['addr']]
changed = pd.Series(False, index=b.index)
for f in sp.FEATS:
    changed |= ~np.isclose(b[f].astype(float), a[f].astype(float), rtol=1e-9, atol=1e-12)
print(f'Wallets with any feature changed: {changed.sum():,} ({int(b.loc[changed, "sybil"].sum())} Sybil)')
print(pd.crosstab(b.loc[changed, 'category'], a.loc[changed, 'category']).to_string())

# %% [markdown]
# ## 1. Vintage comparison (LightGBM, 10 group splits, 1 model seed)

# %%
rows = []
for s in sp.SPLIT_SEEDS:
    for name, d in [('current', cur), ('presnapshot', pre)]:
        S = sp.make_splits(d, sp.FEATS, method='group', seed=s)
        rows.append({'split_seed': s, 'labels': name, **sp.lgbm_fit_eval(S, sp.FEATS, LGBM_SELECTED)})
res = pd.DataFrame(rows)
p = res.pivot(index='split_seed', columns='labels', values=['val_f1', 'test_f1', 'test_ap', 'test_auroc'])
for m in ['val_f1', 'test_f1', 'test_ap', 'test_auroc']:
    p[(m, 'diff')] = p[(m, 'presnapshot')] - p[(m, 'current')]
print(p.round(4).to_string())
summary = p.xs('diff', axis=1, level=1).agg(['mean', 'std'])
print('\nPre-snapshot minus current, mean and SD over 10 splits:'); print(summary.round(4).to_string())

# %% [markdown]
# **Result.** Pre-snapshot labels change test F1 by −0.005 ± 0.007 and validation F1 by +0.009 ± 0.012
# (mean ± SD over the 10 splits). The two move in opposite directions and both lie within the split-to-split
# spread of the full model (test F1 SD 0.006 in `09`). Funding groups depend on the labels, so the partitions
# also differ slightly between the two label sets.

# %% [markdown]
# ## 2. CEX labels added before vs after the Sybil list was published

# %%
cex = pd.read_csv('data/20260128_dune_spellbook_labels/cex_evms_addresses_current_9f61b0dd2.csv', dtype=str)
fund = sp.load_provision(sp.data_paths(DATA_DIR)['gas_prov'])
fund = fund[fund['activated_address'].isin(set(cur['addr']))]
fund = fund.merge(cur[['addr', 'sybil']], left_on='activated_address', right_on='addr')
fund = fund.merge(cex[['address', 'added_date']], left_on='gas_provider', right_on='address')
fund['period'] = np.where(fund['added_date'] <= '2024-05-01', 'added before snapshot',
                  np.where(fund['added_date'] <= '2024-09-15', 'added between snapshot and Sybil list',
                           'added after Sybil list'))
added = fund.groupby('period').agg(funders=('gas_provider', 'nunique'), wallets=('addr', 'size'),
                                   sybil_share=('sybil', 'mean'))
added['sybil_share'] = (added['sybil_share'] * 100).round(2)
print(added.to_string())

# %% [markdown]
# **Result.** The shares are not similar, and the difference runs the other way from what labeling informed by
# the Sybil outcome would produce: wallets funded by CEX addresses added after the snapshot have lower Sybil
# shares (2.40 % between the snapshot and the Sybil list, 0.92 % after it) than wallets funded by addresses
# listed before the snapshot (5.30 %). The later additions fund 2,168 wallets, 0.5 % of the data.

# %%
sp.save_results([], '07_sensitivity_label_vintage', 'group', extra=dict(
    labeled_current=len(anchors_cur), labeled_presnapshot=len(anchors_pre), wallets_changed=int(changed.sum()),
    per_split=res.to_dict('records'), diff_mean=summary.loc['mean'].to_dict(), diff_sd=summary.loc['std'].to_dict(),
    cex_added_period=added.reset_index().to_dict('records'), params=LGBM_SELECTED))
