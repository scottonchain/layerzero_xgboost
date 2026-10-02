# %% [markdown]
# # Logistic Regression Sybil Detector — LayerZero
#
# Baseline linear classifier, the paper's configuration, not tuned: L1 penalty, `C=0.01`, `liblinear`, on
# standardized features (`sp.LR_PARAMS`, `sp.fit_lr`). Same group split as the tree models.
#
# **Results**: test metrics in section 3, saved to `results/05_logistic_regression_sybil.json`.

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
# Stratified group split on gas provision trees (49 % train, 21 % validation, 30 % test); the Sybil class is
# upsampled to 1:1 in the training partition only.

# %%
# ── Feature table (steps 1-7; see 00_data_pipeline) and the stratified group split ──
df, tfm, labeled_anchors = sp.build_master_df(DATA_DIR)
S = sp.make_splits(df, FEATS, method='group', seed=sp.SEED)   # asserts no gas provision tree spans two partitions
print(sp.split_summary(df, S).to_string(index=False))
print(f"Train after upsampling: {len(S['X_train']):,} rows ({S['y_train'].mean()*100:.1f}% Sybil)\n")
leak = sp.leakage_report(df, S)

# %% [markdown]
# ## 2. Train

# %%
print('Logistic regression:', sp.LR_PARAMS)
m = sp.fit_lr(S)

# %% [markdown]
# ## 3. Evaluate on the test set
#
# Threshold: the F1-maximizing threshold on validation.

# %%
res = sp.evaluate('Logistic Regression (C=0.01, L1)', S['y_val'], m['val'], S['y_test'], m['test'])
sp.print_metrics(res)
sp.plot_curves([res])

# %% [markdown]
# ## 4. Non-zero coefficients (L1 sparsity)

# %%
coef = m['model'].named_steps['lr'].coef_[0]
nz_df = pd.DataFrame([(f, c) for f, c in zip(FEATS, coef) if c != 0], columns=['Feature', 'Coefficient'])
nz_df = nz_df.reindex(nz_df['Coefficient'].abs().sort_values(ascending=False).index)
print(f'Non-zero coefficients: {len(nz_df)} / {len(FEATS)}')
print(nz_df.to_string(index=False))
fig, ax = plt.subplots(figsize=(10, max(5, len(nz_df) * 0.35)))
colors = ['#B71C1C' if v > 0 else '#1565C0' for v in nz_df['Coefficient']]
ax.barh(nz_df['Feature'][::-1], nz_df['Coefficient'][::-1], color=colors[::-1])
ax.axvline(0, color='black', lw=0.8, alpha=0.5)
ax.set(xlabel='Coefficient (red=positive/Sybil, blue=negative/Clean)',
       title=f'Logistic Regression — {len(nz_df)} Non-zero Coefficients (L1)')
ax.grid(axis='x', alpha=0.2)
plt.tight_layout(); plt.show()

# %% [markdown]
# Logistic regression scores well below the gradient-boosted models on F1, although its AUROC is moderate;
# it is included as a linear baseline, not as a deployment choice.

# %% [markdown]
# ## Save results
#
# Scalar metrics go to `results/05_logistic_regression_sybil.json` with the code commit, so every number in the paper can be traced
# to one run. Validation and test probabilities go to `output/` for `06_cross_ensemble_sybil` and `10_tie_out`.

# %%
sp.save_results([res], '05_logistic_regression_sybil', leakage=leak, extra=dict(params=sp.LR_PARAMS))
sp.save_predictions(df, S, '05_logistic_regression_sybil', prob_lr=m)
