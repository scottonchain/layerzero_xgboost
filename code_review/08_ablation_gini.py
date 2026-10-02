# %% [markdown]
# # Ablation — Corrected `gini_coefficient` (keep or drop)
#
# The submitted `gini_coefficient` was identically zero; the revision corrects it (`sp._gini`). It is the only
# feature whose definition changed, so it alone gets a keep-or-drop test.
#
# **Pre-specified rule** (tracker, commit 72b2128, recorded before this run): train LightGBM with and without
# the feature on 10 stratified group splits; retain it only if removing it lowers **validation** F1 on at
# least 8 of the 10 splits (one-sided sign test, p ≈ 0.055). The test set is not used for the decision.
#
# The LightGBM configuration is the one selected by the last completed validation-only search
# (commit 82148f1): `num_leaves=511, learning_rate=0.05, subsample=0.7, reg_lambda=1`. This notebook runs
# before the final search, because the search depends on the feature set it decides.

# %%
import json, time, warnings
import numpy as np
import pandas as pd
import sybil_pipeline as sp
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
DATA_DIR, SEED = './data', sp.SEED
from scipy.stats import binomtest
PARAMS = dict(num_leaves=511, learning_rate=0.05, subsample=0.7, reg_lambda=1.0)
ALL63 = list(sp.FEATS_SUBMITTED)   # the submitted 63, in the submitted order
WITHOUT = [f for f in ALL63 if f != 'gini_coefficient']
df, _, _ = sp.build_master_df(DATA_DIR, verbose=False)
rows = []
for s in sp.SPLIT_SEEDS:
    S = sp.make_splits(df, ALL63, method='group', seed=s)
    w = sp.lgbm_fit_eval(S, ALL63, PARAMS); wo = sp.lgbm_fit_eval(S, WITHOUT, PARAMS)
    rows.append(dict(split_seed=s, val_f1_with=w['val_f1'], val_f1_without=wo['val_f1'],
                     val_ap_with=w['val_ap'], val_ap_without=wo['val_ap']))
    print(s, round(w['val_f1'], 4), round(wo['val_f1'], 4), flush=True)
r = pd.DataFrame(rows)
r['f1_change_when_removed'] = r.val_f1_without - r.val_f1_with
r['ap_change_when_removed'] = r.val_ap_without - r.val_ap_with
print(r.round(4).to_string(index=False))

# %%
k = int((r.f1_change_when_removed < 0).sum())
p = binomtest(k, len(r), 0.5, alternative='greater').pvalue
KEEP = k >= 8
print(f'Removing gini_coefficient lowered validation F1 on {k} of {len(r)} splits '
      f'(mean change {r.f1_change_when_removed.mean():+.4f}; one-sided sign test p = {p:.3f}).')
print('Decision:', 'RETAIN gini_coefficient' if KEEP else 'REMOVE gini_coefficient (62 features)')
sp.save_results([], '08_ablation_gini', 'group', extra=dict(
    rule='retain if removing lowers validation F1 on >= 8 of 10 group splits', splits_lowered=k,
    sign_test_p=p, mean_f1_change=float(r.f1_change_when_removed.mean()),
    mean_ap_change=float(r.ap_change_when_removed.mean()), decision='retain' if KEEP else 'remove',
    per_split=r.to_dict('records'), params=PARAMS))
