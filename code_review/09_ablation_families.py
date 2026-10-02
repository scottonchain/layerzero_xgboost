# %% [markdown]
# # Ablation — Feature Families (report only)
#
# Which groups of features carry the signal? Five families (`sp.FEATURE_FAMILIES`): LayerZero transactions,
# Ethereum transactions, gas provider, funding tree, funding chain. Two views on 10 stratified group splits
# with the selected LightGBM configuration (1 model seed per fit):
#
# - **Without the family**: what it adds on top of all others.
# - **Family alone**: how much signal it carries by itself.
# - **Without all provision-network families** (gas provider, funding tree, funding chain together): what
#   the gas provision network adds on top of the transaction features.
#
# Whole families are removed so correlated features cannot mask each other. Nothing is selected from these
# results; they are reported only. The "All features" row also gives the split-to-split spread of the full
# model.

# %%
import json, time, warnings
import numpy as np
import pandas as pd
import sybil_pipeline as sp
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
DATA_DIR, SEED = './data', sp.SEED
_, LGBM_SELECTED = sp.load_selected_params()
FAM = {k: [f for f in v if f in sp.FEATS] for k, v in sp.FEATURE_FAMILIES.items()}
df, _, _ = sp.build_master_df(DATA_DIR, verbose=False)
configs = [('All features', sp.FEATS)]
configs += [(f'Without {k}', [f for f in sp.FEATS if f not in v]) for k, v in FAM.items()]
configs += [(f'{k} only', v) for k, v in FAM.items()]
NETWORK = ['Gas provider', 'Funding tree', 'Funding chain']   # the provision-network families together
configs += [('Without all provision-network families', [f for f in sp.FEATS if not any(f in FAM[k] for k in NETWORK)])]
rows = []
for s in sp.SPLIT_SEEDS:
    S = sp.make_splits(df, sp.FEATS, method='group', seed=s)
    for name, feats in configs:
        rows.append({'split_seed': s, 'config': name, 'n_features': len(feats), **sp.lgbm_fit_eval(S, feats, LGBM_SELECTED)})
    print('split', s, 'done', flush=True)
res = pd.DataFrame(rows)
order = [c for c, _ in configs]
summ = res.groupby('config')[['n_features', 'val_f1', 'test_f1', 'test_ap', 'test_auroc']].agg(['mean', 'std'])
summ = summ.loc[order]
base = res[res.config == 'All features'].set_index('split_seed')['test_f1']
res['test_f1_change'] = res.apply(lambda x: x.test_f1 - base[x.split_seed], axis=1)
summ[('test_f1_change_vs_all', 'mean')] = res.groupby('config')['test_f1_change'].mean().loc[order]
print(summ.round(4).to_string())

# %%
sp.save_results([], '09_ablation_families', 'group', extra=dict(
    families=FAM, per_split=res.to_dict('records'),
    summary={c: {f'{m}_{s}': float(summ.loc[c, (m, s)]) for m, s in summ.columns} for c in order},
    params=LGBM_SELECTED))
