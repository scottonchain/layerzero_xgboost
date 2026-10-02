# %% [markdown]
# # Logistic Regression Sybil Detector — LayerZero
#
# Baseline linear classifier (the paper's LR configuration; not tuned).
#
# **Configuration**: `C=0.01`, L1 regularisation, `liblinear` solver, `StandardScaler`
#
# **Results**: test metrics are printed in section 6 and saved to
# `results/03_logistic_regression_<split>.json`.
#
# > LR serves as a lower bound confirming that the Sybil decision boundary is non-linear. The high AUROC relative to F1 reflects that the model ranks addresses well but cannot separate them sharply at any single threshold.

# %% [markdown]
# ## 1. Setup

# %%
# Install required packages (run once)
# # !pip install xgboost lightgbm scikit-learn pandas numpy matplotlib

# %%
import os, time, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    brier_score_loss, average_precision_score
)

import sybil_pipeline as sp

warnings.filterwarnings('ignore')
SEED = sp.SEED

# %%
# ── Set this to the folder containing all data files ──────────
DATA_DIR = './data'   # <-- adjust if needed

# 'group'  = stratified group split by funding cluster (revision default)
# 'random' = address-level split used in the original paper
SPLIT_METHOD = os.environ.get('SPLIT_METHOD', 'group')

FEATS = sp.FEATS   # 62 model features (defined in sybil_pipeline.py)
print(f'Split method: {SPLIT_METHOD}  |  {len(FEATS)} features')

# %% [markdown]
# ## 2. Data Loading

# %%
# ── Build the master feature table (steps 1-7; see 00_data_pipeline) ──
df, tfm, labeled_anchors = sp.build_master_df(DATA_DIR)

# %% [markdown]
# ## 3. Train / Validation / Test Split

# %%
# ── Train / Val / Test split + minority upsampling ────────────
# 49 / 21 / 30 partitions. With SPLIT_METHOD='group', no funding group
# spans two partitions (asserted inside sp.make_splits).
S = sp.make_splits(df, FEATS, method=SPLIT_METHOD, seed=SEED)
X_train, y_train = S['X_train'], S['y_train']
X_val,   y_val   = S['X_val'],   S['y_val']
X_test,  y_test  = S['X_test'],  S['y_test']

print(sp.split_summary(df, S).to_string(index=False))
print(f'Train after upsampling: {len(X_train):,} rows ({y_train.mean()*100:.1f}% Sybil)\n')
leak = sp.leakage_report(df, S)


# %% [markdown]
# ## 4. Evaluation Helpers

# %%
# ── Evaluation helpers ────────────────────────────────────────
def optimal_threshold(y_true, probs):
    """Find threshold that maximises F1 on the given set."""
    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    return float(thresholds[np.argmax(f1[:-1])])

def evaluate(name, val_probs, test_probs, y_val, y_test, threshold=None):
    """Evaluate model at optimal val threshold, report test metrics."""
    thr = threshold if threshold is not None else optimal_threshold(y_val, val_probs)
    y_pred = (test_probs >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    prc, rec, _ = precision_recall_curve(y_test, test_probs)
    results = dict(
        name      = name,
        threshold = thr,
        precision = precision_score(y_test, y_pred, zero_division=0),
        recall    = recall_score(y_test, y_pred,    zero_division=0),
        f1        = f1_score(y_test, y_pred,         zero_division=0),
        auroc     = auc(fpr, tpr),
        ap        = average_precision_score(y_test, test_probs),
        brier     = brier_score_loss(y_test, test_probs),
        fpr=fpr, tpr=tpr, prc=prc, rec_c=rec,
    )
    print(f'  Threshold : {thr:.4f}')
    print(f'  Precision : {results["precision"]:.4f}')
    print(f'  Recall    : {results["recall"]:.4f}')
    print(f'  F1        : {results["f1"]:.4f}')
    print(f'  AUROC     : {results["auroc"]:.4f}')
    print(f'  Avg Prec  : {results["ap"]:.4f}')
    print(f'  Brier     : {results["brier"]:.4f}')
    print(classification_report(y_test, y_pred, digits=4))
    return results

def plot_curves(results_list):
    """Plot ROC and PR curves for one or more models."""
    colors = ['#1565C0','#2E7D32','#B71C1C','#6A1B9A','#E65100']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r, c in zip(results_list, colors):
        axes[0].plot(r['fpr'], r['tpr'], color=c, lw=2,
                     label=f"{r['name']}  AUC={r['auroc']:.4f}")
        axes[1].plot(r['rec_c'], r['prc'], color=c, lw=2,
                     label=f"{r['name']}  AP={r['ap']:.4f}")
    axes[0].plot([0,1],[0,1],'k--',alpha=0.3,lw=1)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    axes[1].legend(fontsize=9, loc='upper right'); axes[1].grid(alpha=0.2)
    plt.tight_layout(); plt.show()


# %% [markdown]
# ## 5. Train Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Parameters match paper's baseline ─────────────────────────
# LR is a linear model trained on the same balanced, upsampled data.
# StandardScaler is mandatory: gradient-based solvers are scale-sensitive.
# C=0.01 (strong L1 regularisation) induces sparsity; only ~20-30 of
# the features receive non-zero weights, improving interpretability.
LR_PARAMS = dict(
    C          = 0.01,
    penalty    = 'l1',
    solver     = 'liblinear',   # required for L1
    max_iter   = 1000,
    random_state = SEED,
)

print('Training Logistic Regression (C=0.01, L1, liblinear)...')
t0 = time.time()
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr',     LogisticRegression(**LR_PARAMS)),
])
lr_pipe.fit(X_train, y_train)
print(f'Done in {time.time()-t0:.1f}s')

lr_val_probs  = lr_pipe.predict_proba(X_val)[:,1]
lr_test_probs = lr_pipe.predict_proba(X_test)[:,1]

# %% [markdown]
# ## 6. Evaluate on Test Set

# %%
print('=== Logistic Regression — Test Set Results ===')
lr_results = evaluate('Logistic Regression (C=0.01, L1)',
                       lr_val_probs, lr_test_probs, y_val, y_test)
plot_curves([lr_results])

# %% [markdown]
# ## 7. Non-Zero Coefficients

# %%
# ── Non-zero coefficients (L1 sparsity) ───────────────────────
coef  = lr_pipe.named_steps['lr'].coef_[0]
nz    = [(FEATS[i], coef[i]) for i in range(len(FEATS)) if coef[i] != 0]
nz_df = pd.DataFrame(nz, columns=['Feature', 'Coefficient'])
nz_df = nz_df.reindex(nz_df['Coefficient'].abs().sort_values(ascending=False).index)

print(f'Non-zero coefficients: {len(nz_df)} / {len(FEATS)}')
print(nz_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, max(5, len(nz_df)*0.35)))
colors = ['#B71C1C' if v > 0 else '#1565C0' for v in nz_df['Coefficient']]
ax.barh(nz_df['Feature'][::-1], nz_df['Coefficient'][::-1], color=colors[::-1])
ax.axvline(0, color='black', lw=0.8, alpha=0.5)
ax.set(xlabel='Coefficient (red=positive/Sybil, blue=negative/Clean)',
       title=f'Logistic Regression — {len(nz_df)} Non-zero Coefficients (L1)')
ax.grid(axis='x', alpha=0.2)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Note on LR performance
#
# Logistic Regression scores well below the gradient-boosted models on F1 (see section 6 and
# `06_split_comparison`), although its AUROC is moderate. This confirms the Sybil decision boundary is **non-linear**:
# the behavioral signals (timing, gas topology, cross-chain breadth) interact in ways
# that a linear classifier cannot fully capture.
#
# LR is included as a baseline, not a practical deployment choice for this task.

# %% [markdown]
# ## Save Results
#
# Scalar metrics go to `results/<notebook>_<split>.json` with the code commit, so every number in the paper can be traced to one run.

# %%
sp.save_results([lr_results], '03_logistic_regression', SPLIT_METHOD, leakage=leak,
                extra=dict(params={k: v for k, v in LR_PARAMS.items()}))

# ── Test predictions (read by 06_split_comparison for per-category metrics) ──
os.makedirs('output', exist_ok=True)
pred = pd.DataFrame({'addr': df.loc[S['idx_test'], 'addr'].to_numpy(),
                     'category': df.loc[S['idx_test'], 'category'].to_numpy(),
                     'y': y_test.to_numpy(), 'prob_lr': lr_test_probs})
pred.to_parquet(f'output/pred_03_logistic_regression_{SPLIT_METHOD}.parquet', index=False)
print(f'Saved output/pred_03_logistic_regression_{SPLIT_METHOD}.parquet ({len(pred):,} rows)')
