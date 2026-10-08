"""Does this host reproduce 13's Random Forest (3 seeds, seed-42 group split) saved by the 2.10 GHz host?
Replicates 13's final-fit cell with its selected configuration from results/13."""
import json, sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
sys.path.insert(0, '.')
import sybil_pipeline as sp
r13 = json.load(open('results/13_random_forest_sybil.json'))
p = r13['params']; print('results/13 params:', p, 'platform:', r13['platform'])
cfg = {k: p[k] for k in ['max_features', 'min_samples_leaf', 'max_depth', 'max_samples']}
df, _, _ = sp.build_master_df('./data', verbose=False)
S = sp.make_splits(df, sp.FEATS, method='group', seed=sp.SEED)
val_p, test_p = [], []
for seed in sp.MODEL_SEEDS:
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=300, criterion='gini', bootstrap=True, n_jobs=sp.N_JOBS, random_state=seed, **cfg)
    m.fit(S['X_train'], S['y_train'])
    val_p.append(m.predict_proba(S['X_val'])[:, 1]); test_p.append(m.predict_proba(S['X_test'])[:, 1])
    print(f'seed {seed}: {time.time()-t0:.0f}s', flush=True)
ok = True
for part, new in [('val', np.mean(val_p, axis=0)), ('test', np.mean(test_p, axis=0))]:
    ref = pd.read_parquet(f'output/pred_13_random_forest_sybil_{part}.parquet')
    col = [c for c in ref.columns if c.startswith('prob')][0]
    exact = np.array_equal(ref[col].values, new)
    print(f'{part}: {len(ref):,} rows; bit-identical to the 2.10 GHz host: {exact}; max |diff| {np.abs(ref[col].values-new).max():.3g}')
    ok &= exact
print('RF REPRODUCED' if ok else 'RF NOT REPRODUCED'); sys.exit(0 if ok else 1)
