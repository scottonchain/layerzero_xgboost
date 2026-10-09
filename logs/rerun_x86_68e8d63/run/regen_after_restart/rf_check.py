"""Re-score the first Random Forest search configuration after the restart, exactly as 13 does (seed 42,
300 trees, n_jobs 4, validation only), and compare with the row written before the restart."""
import sys, time, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, log_loss
import sybil_pipeline as sp
df, _, _ = sp.build_master_df('./data', verbose=False)
S = sp.make_splits(df, sp.FEATS, method='group', seed=sp.SEED)
for k in ['X_test', 'y_test', 'idx_test']: del S[k]
cfg = dict(max_features='sqrt', min_samples_leaf=1, max_depth=None, max_samples=None)
t0 = time.time()
m = RandomForestClassifier(n_estimators=300, criterion='gini', bootstrap=True, n_jobs=sp.N_JOBS, random_state=sp.SEED, **cfg)
m.fit(S['X_train'], S['y_train']); p = m.predict_proba(S['X_val'])[:, 1]
thr = sp.best_f1_threshold(S['y_val'], p)
now = dict(val_f1=f1_score(S['y_val'], (p >= thr).astype(int)), val_ap=average_precision_score(S['y_val'], p),
           val_auroc=roc_auc_score(S['y_val'], p), val_logloss=log_loss(S['y_val'], np.clip(p, 1e-15, 1 - 1e-15)), val_threshold=thr)
before = pd.read_csv('logs/rerun_x86_68e8d63/run/13_search_random_forest.partial_before_restart.csv').iloc[0]
print(f'seconds {time.time()-t0:.1f}')
for k, v in now.items(): print(f'{k:14s} before {before[k]!r:24} after {v!r:24} diff {v - before[k]:.3g}')
