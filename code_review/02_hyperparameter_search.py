# %% [markdown]
# # Hyperparameter Search — Validation Set Only
#
# Selects the XGBoost and LightGBM configurations used by `03_xgboost_sybil`, `04_lightgbm_sybil`, and the
# robustness notebooks (`07`, `08`, `09`). The model settings that are not tuned come from `sp.xgb_model` and
# `sp.lgbm_model`, the same definitions the model notebooks use.
#
# **Protocol**
# - Split: the stratified **group** split (`sp.make_splits(..., method='group')`), the revision's
#   primary evaluation. The same selected configuration is reused in `11_split_comparison`.
# - The test partition is removed before any model is fit; this notebook never sees test labels.
# - Each configuration is trained once (seed 42) on the upsampled training partition, with early
#   stopping on validation log-loss (patience 50, ceiling 5,000 rounds).
# - **Selection criterion**: validation F1 at the F1-maximising validation threshold. Ties are broken
#   by validation average precision (AP).
#
# **Search space** (full grids over the ranges stated in the paper)
#
# | Model | Stage | Grid | Configs |
# |---|---|---|---|
# | XGBoost | 1 | `learning_rate` {0.3, 0.1, 0.05} × `max_depth` {6, 8, 10, 12} × `subsample` {1.0, 0.7, 0.5} × `colsample_bytree` {1.0, 0.8} | 72 |
# | XGBoost | 2 | `min_child_weight` {1, 5, 10} at the stage-1 best | +2 |
# | LightGBM | — | `num_leaves` {31, 63, 127, 255, 511, 1023} × `learning_rate` {0.1, 0.05} × `subsample` {1.0, 0.7} (`subsample_freq=1`) × `reg_lambda` {0, 1}; `colsample_bytree` 0.8 | 48 |
#
# Results are written after every configuration to `results/02_search_<model>.csv`, so an
# interrupted run resumes where it stopped. The selected configurations go to
# `results/02_hyperparameter_search.json`, which the model notebooks read.

# %%
import os, time, json, itertools, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, log_loss
import lightgbm as lgb_lib
import sybil_pipeline as sp
warnings.filterwarnings('ignore')

DATA_DIR = './data'
RESULTS_DIR = './results'
SEED = sp.SEED
os.makedirs(RESULTS_DIR, exist_ok=True)

df, tfm, labeled_anchors = sp.build_master_df(DATA_DIR)
S = sp.make_splits(df, sp.FEATS, method='group', seed=SEED)   # selection is done on the group split
X_train, y_train = S['X_train'], S['y_train']
X_val,   y_val   = S['X_val'],   S['y_val']
for k in ['X_test', 'y_test', 'idx_test']:
    del S[k]                    # test partition is not available in this notebook
print(sp.split_summary(df, {**S, 'idx_test': []}).iloc[:2].to_string(index=False))


# %%
def val_scores(probs):
    thr = sp.best_f1_threshold(y_val, probs)
    return dict(val_f1=f1_score(y_val, (probs >= thr).astype(int)),
                val_ap=average_precision_score(y_val, probs),
                val_logloss=log_loss(y_val, probs), val_threshold=thr)

def run_grid(name, configs, fit):
    """Fit every config not already in results/02_search_<name>.csv; return the full table."""
    path = os.path.join(RESULTS_DIR, f'02_search_{name}.csv')
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    keys = list(configs[0])
    seen = set(map(tuple, done[keys].astype(str).values)) if len(done) else set()
    for i, cfg in enumerate(configs, 1):
        if tuple(str(cfg[k]) for k in keys) in seen:
            continue
        t0 = time.time()
        probs, rounds = fit(cfg)
        row = {**cfg, **val_scores(probs), 'rounds': rounds, 'seconds': round(time.time() - t0, 1)}
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(path, index=False)
        print(f"[{i}/{len(configs)}] {cfg}  val_F1={row['val_f1']:.4f}  AP={row['val_ap']:.4f}  "
              f"rounds={rounds}  {row['seconds']}s", flush=True)
    for k in keys:                          # restore dtypes after CSV round-trip
        done[k] = done[k].astype(type(configs[0][k]))
    return done

def best(table):
    return table.sort_values(['val_f1', 'val_ap'], ascending=False).iloc[0]

INT_PARAMS = {'max_depth', 'min_child_weight', 'num_leaves'}

def as_params(row, keys):
    # results-table row -> hyperparameter dict with the right Python types
    return {k: int(row[k]) if k in INT_PARAMS else float(row[k]) for k in keys}


# %% [markdown]
# ## XGBoost — stage 1

# %%
def fit_xgb(cfg):
    m = sp.xgb_model(cfg, SEED)   # fixed settings as in the model notebooks
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return m.predict_proba(X_val)[:, 1], int(m.best_iteration)

xgb_grid = [dict(learning_rate=lr, max_depth=d, subsample=s, colsample_bytree=c, min_child_weight=1)
            for lr, d, s, c in itertools.product([0.3, 0.1, 0.05], [6, 8, 10, 12],
                                                 [1.0, 0.7, 0.5], [1.0, 0.8])]
xgb1 = run_grid('xgboost', xgb_grid, fit_xgb)
print(xgb1.sort_values('val_f1', ascending=False).head(10).round(4).to_string(index=False))

# %% [markdown]
# ## XGBoost — stage 2 (`min_child_weight`)

# %%
b1 = best(xgb1)
xgb_grid2 = xgb_grid + [{**as_params(b1, ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree']),
                         'min_child_weight': w} for w in [5, 10]]
xgb_all = run_grid('xgboost', xgb_grid2, fit_xgb)
xgb_best = best(xgb_all)
XGB_SELECTED = as_params(xgb_best, ['learning_rate', 'max_depth', 'subsample',
                                    'colsample_bytree', 'min_child_weight'])
print('Selected XGBoost config:', XGB_SELECTED)
print(xgb_best[['val_f1', 'val_ap', 'rounds']].to_string())


# %% [markdown]
# ## LightGBM

# %%
def fit_lgbm(cfg):
    m = sp.lgbm_model(cfg, SEED)   # fixed settings as in the model notebooks
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb_lib.early_stopping(50, verbose=False), lgb_lib.log_evaluation(-1)])
    return m.predict_proba(X_val)[:, 1], int(m.best_iteration_)

lgbm_grid = [dict(num_leaves=nl, learning_rate=lr, subsample=s, reg_lambda=l2)
             for nl, lr, s, l2 in itertools.product([31, 63, 127, 255, 511, 1023], [0.1, 0.05],
                                                    [1.0, 0.7], [0.0, 1.0])]
lgbm_all = run_grid('lightgbm', lgbm_grid, fit_lgbm)
lgbm_best = best(lgbm_all)
LGBM_SELECTED = as_params(lgbm_best, ['num_leaves', 'learning_rate', 'subsample', 'reg_lambda'])
print('Selected LightGBM config:', LGBM_SELECTED)
print(lgbm_all.sort_values('val_f1', ascending=False).head(10).round(4).to_string(index=False))


# %% [markdown]
# ## Marginal effect of each hyperparameter (validation F1)
#
# For each hyperparameter value: the best validation F1 over all configs with that value. This is
# the ablation view of the grid.

# %%
def marginal(table, params):
    out = []
    for p in params:
        g = table.groupby(p)['val_f1'].max()
        out += [dict(param=p, value=v, best_val_f1=round(f, 4)) for v, f in g.items()]
    return pd.DataFrame(out)

print('XGBoost (stage 1 grid):')
print(marginal(xgb1, ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree']).to_string(index=False))
print('\nLightGBM:')
print(marginal(lgbm_all, ['num_leaves', 'learning_rate', 'subsample', 'reg_lambda']).to_string(index=False))

# %% [markdown]
# ## Save selected configurations

# %%
sp.save_results([], '02_hyperparameter_search', extra=dict(
    selection='max validation F1 at validation-optimal threshold; ties by validation AP; seed 42',
    xgb_selected=XGB_SELECTED, lgbm_selected=LGBM_SELECTED,
    xgb_selected_val=dict(val_f1=float(xgb_best.val_f1), val_ap=float(xgb_best.val_ap), rounds=int(xgb_best.rounds)),
    lgbm_selected_val=dict(val_f1=float(lgbm_best.val_f1), val_ap=float(lgbm_best.val_ap), rounds=int(lgbm_best.rounds)),
    n_configs=dict(xgboost=len(xgb_all), lightgbm=len(lgbm_all)),
))
