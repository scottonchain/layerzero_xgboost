"""Stage 2: fixed-configuration sensitivity analysis (not the retuned paper result).

Same 62 features, same committed LightGBM hyperparameters (results/02), the seed-42 tree split, one LightGBM
model seed (42), identical train/validation/test rows and the same 1:1 upsampling. The original and the corrected
feature table are fitted one after the other in this process; early stopping and the threshold come from
validation independently for each fit.
Usage: python diagnostics/root_tree_fix/stage2_fixed_config.py   (from the repository root)
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
import sybil_pipeline as sp
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

D = 'diagnostics/root_tree_fix'
def log(*a):
    msg = ' '.join(str(x) for x in a); print(msg, flush=True)
    with open('/home/user/fix_diag/progress.log', 'a') as f: f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {msg}\n")

old = pd.read_parquet(f'{D}/cache/master_old.parquet'); new = pd.read_parquet(f'{D}/cache/master_new.parquet')
meta = {t: json.load(open(f'{D}/cache/master_{t}.json')) for t in ('old', 'new')}
S = {'old': sp.make_splits(old, sp.FEATS, 'group', sp.SEED), 'new': sp.make_splits(new, sp.FEATS, 'group', sp.SEED)}
for k in ('idx_train', 'idx_val', 'idx_test'):
    assert np.array_equal(np.asarray(S['old'][k]), np.asarray(S['new'][k])), k
for k in ('X_train', 'X_val', 'X_test'):
    assert np.array_equal(S['old'][k].index.to_numpy(), S['new'][k].index.to_numpy()), k
for k in ('y_train', 'y_val', 'y_test'):
    assert np.array_equal(S['old'][k].to_numpy(), S['new'][k].to_numpy()), k
log('[stage2] splits verified identical (row ids, labels, upsampled training index)')
_, LGBM_P = sp.load_selected_params()
cfg = dict(model_seeds=[42], split_seed=sp.SEED, lgbm_params=LGBM_P, n_features=len(sp.FEATS), n_jobs=sp.N_JOBS, lgbm_repro=sp.LGBM_REPRO)
key = hashlib.sha256(json.dumps(dict(cfg, old=meta['old']['code_sha256'], new=meta['new']['code_sha256'], inputs=meta['new']['inputs_sha256']),
                                sort_keys=True, default=str).encode()).hexdigest()

fits = {}
for tag in ('old', 'new'):
    ck = f'{D}/cache/stage2_fit_{tag}.npz'
    if os.path.exists(ck) and str(np.load(ck, allow_pickle=True)['key']) == key:
        z = np.load(ck, allow_pickle=True); fits[tag] = dict(val=z['val'], test=z['test'], rounds=int(z['rounds']), seconds=float(z['seconds']))
        log(f'[stage2] {tag}: checkpoint reused'); continue
    t0 = time.time(); log(f'[stage2] fitting {tag} ...')
    r = sp.fit_lgbm(S[tag], LGBM_P, seeds=(42,), verbose=True)
    sec = time.time() - t0
    fits[tag] = dict(val=r['val'], test=r['test'], rounds=r['rounds'][0], seconds=sec)
    np.savez(ck, key=key, val=r['val'], test=r['test'], rounds=r['rounds'][0], seconds=sec)
    log(f'[stage2] {tag}: rounds {r["rounds"][0]}, {sec:.0f}s')

y_val, y_test = S['old']['y_val'].to_numpy(), S['old']['y_test'].to_numpy()
te_idx = np.asarray(S['old']['idx_test'])
cat_test = new.loc[te_idx, 'category'].to_numpy()
def overall(tag):
    f = fits[tag]; thr = sp.best_f1_threshold(y_val, f['val']); yh = (f['test'] >= thr).astype(int)
    yv = (f['val'] >= thr).astype(int)
    return dict(threshold=thr, f1=f1_score(y_test, yh), ap=average_precision_score(y_test, f['test']), auroc=roc_auc_score(y_test, f['test']),
                precision=precision_score(y_test, yh, zero_division=0), recall=recall_score(y_test, yh), val_f1=f1_score(y_val, yv),
                val_ap=average_precision_score(y_val, f['val']), early_stopping_rounds=f['rounds'], fit_seconds=round(f['seconds'], 1),
                flagged=int(yh.sum()))
O = {t: overall(t) for t in ('old', 'new')}
R = dict(label='fixed-configuration sensitivity analysis (not the retuned paper result)', config=cfg, key=key,
         old_code_sha256=meta['old']['code_sha256'], new_code_sha256=meta['new']['code_sha256'], inputs_sha256=meta['new']['inputs_sha256'],
         n_val=int(len(y_val)), n_test=int(len(y_test)), test_sybils=int(y_test.sum()),
         overall={m: dict(old=O['old'][m], new=O['new'][m], diff=O['new'][m] - O['old'][m]) for m in O['old']})

# by category
byc = {}
for tag in ('old', 'new'):
    byc[tag] = sp.metrics_by_category(y_test, fits[tag]['test'], cat_test, O[tag]['threshold']).set_index('category')
R['by_category'] = {}
for c in byc['old'].index:
    row = dict(n=int(byc['old'].loc[c, 'n']), sybils=int(byc['old'].loc[c, 'sybil']))
    for m in ('precision', 'recall', 'f1', 'auroc'):
        if m in byc['old'].columns:
            row[m] = dict(old=float(byc['old'].loc[c, m]), new=float(byc['new'].loc[c, m]), diff=float(byc['new'].loc[c, m] - byc['old'].loc[c, m]))
    R['by_category'][c] = row

# affected rows / roots
feat_chg = (old.loc[te_idx, sp.FEATS].astype(float).to_numpy() != new.loc[te_idx, sp.FEATS].astype(float).to_numpy()).any(axis=1)
root = new.loc[te_idx, 'depth'].to_numpy() == 0
sing = new.loc[te_idx, 'tree_size'].to_numpy() == 1
pred = {t: (fits[t]['test'] >= O[t]['threshold']).astype(int) for t in ('old', 'new')}
def rec(mask, t):
    m = mask & (y_test == 1)
    return dict(sybils=int(m.sum()), recall=float(pred[t][m].mean()) if m.sum() else None, mean_prob=float(fits[t]['test'][m].mean()) if m.sum() else None)
groups = {'affected_sybil_roots': feat_chg & root, 'affected_sybil_roots_with_descendants': feat_chg & root & ~sing,
          'affected_sybil_singleton_roots': feat_chg & root & sing, 'unaffected_sybils': ~feat_chg}
R['affected'] = {g: dict(old=rec(m, 'old'), new=rec(m, 'new')) for g, m in groups.items()}
R['test_rows_with_changed_inputs'] = dict(total=int(feat_chg.sum()), sybil=int((feat_chg & (y_test == 1)).sum()), non_sybil=int((feat_chg & (y_test == 0)).sum()))

# prediction flips
fl = pred['old'] != pred['new']
R['flips'] = dict(total=int(fl.sum()), pct_of_test=float(100 * fl.mean()), to_flagged=int((fl & (pred['new'] == 1)).sum()), to_unflagged=int((fl & (pred['new'] == 0)).sum()),
                  sybil=int((fl & (y_test == 1)).sum()), non_sybil=int((fl & (y_test == 0)).sum()),
                  in_changed_input_rows=int((fl & feat_chg).sum()), in_unchanged_input_rows=int((fl & ~feat_chg).sum()),
                  sybil_missed_to_caught=int((fl & (y_test == 1) & (pred['new'] == 1)).sum()), sybil_caught_to_missed=int((fl & (y_test == 1) & (pred['new'] == 0)).sum()),
                  false_pos_added=int((fl & (y_test == 0) & (pred['new'] == 1)).sum()), false_pos_removed=int((fl & (y_test == 0) & (pred['new'] == 0)).sum()))
R['note_on_flips_outside_changed_rows'] = ('Rows whose inputs did not change can still flip: the two fits are different models (different trees, '
                                           'early-stopping round and threshold).')
json.dump(R, open(f'{D}/stage2_fixed_config.json', 'w'), indent=1, default=float)

L = ['# Stage 2: fixed-configuration sensitivity analysis (LightGBM, seed 42, 62 features, committed hyperparameters)', '',
     'Not the retuned paper result. Old = b7a9a3f features, corrected = 663e026 features; identical rows and procedure.', '',
     '| Metric | Original | Corrected | Difference |', '|---|---|---|---|']
for m in ('f1', 'ap', 'auroc', 'precision', 'recall', 'threshold'):
    o = R['overall'][m]; L.append(f"| {m} | {o['old']:.4f} | {o['new']:.4f} | {o['diff']:+.4f} |")
L += ['', f"Early-stopping rounds: {O['old']['early_stopping_rounds']} -> {O['new']['early_stopping_rounds']}. Test rows {R['n_test']:,} ({R['test_sybils']:,} Sybils).", '',
      '| Category | n | Sybils | Recall old | new | F1 old | new |', '|---|---|---|---|---|---|---|']
for c, r in R['by_category'].items():
    L.append(f"| {c} | {r['n']:,} | {r['sybils']:,} | {r['recall']['old']:.4f} | {r['recall']['new']:.4f} | {r['f1']['old']:.4f} | {r['f1']['new']:.4f} |")
L += ['', 'Affected Sybil roots (test): ' + json.dumps(R['affected']), '', 'Flips: ' + json.dumps(R['flips'])]
open(f'{D}/stage2_fixed_config.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
log('[stage2] done')
