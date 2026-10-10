"""Stage 3: notebook 22's report-dependence comparison (LightGBM only) on the corrected feature table.

The union components, large-component treatment, folds, matched-tree removal, upsampling, validation-only
threshold and early stopping are notebook 22's own code (its cells 4 and 5 are executed verbatim). Fold order is
fixed: sp.SPLIT_SEEDS[:3] x folds 0-4. Only the feature table differs between "original" and "corrected"; the row
identities are asserted identical for every arm. The original arm is the PR #13 result stored in
results/22_report_dependence.json (same environment, same code); on the first pair it is also refitted here to
show that it reproduces. Each fitted condition is checkpointed; aggregate tests are computed only at 15 pairs.
Usage: python diagnostics/root_tree_fix/stage3_report_dependence.py [max_pairs]
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.getcwd())
import sybil_pipeline as sp

D = 'diagnostics/root_tree_fix'; CK = f'{D}/stage3'; os.makedirs(CK, exist_ok=True)
MAX_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
def log(*a):
    msg = ' '.join(str(x) for x in a); print(msg, flush=True)
    with open('/home/user/fix_diag/progress.log', 'a') as f: f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {msg}\n")
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()[:16]

old = pd.read_parquet(f'{D}/cache/master_old.parquet'); new = pd.read_parquet(f'{D}/cache/master_new.parquet')
meta = {t: json.load(open(f'{D}/cache/master_{t}.json')) for t in ('old', 'new')}
for c in ('addr', 'sybil', 'category', 'provision_tree'):
    assert (old[c].to_numpy() == new[c].to_numpy()).all(), c
df = new                                   # conditions use only addr, sybil, category, provision_tree (identical in both)
nbk = json.load(open('22_report_dependence.ipynb'))
code = lambda i: ''.join(nbk['cells'][i]['source'])
fcfs = pd.read_csv(sp.data_paths('./data')['sybil']); fcfs.columns = fcfs.columns.str.lower().str.strip()
fcfs['address'] = fcfs['address'].str.lower()
REPORT = df['addr'].map(fcfs.set_index('address')['commonwealth report link'])
assert REPORT[df['sybil'] == 1].notna().all() and REPORT[df['sybil'] == 0].isna().all()
FEATS = sp.FEATS; XGB_P, LGBM_P = sp.load_selected_params()
NS = dict(np=np, pd=pd, sp=sp, stats=stats, coo_matrix=coo_matrix, connected_components=connected_components, time=time,
          df=df, fcfs=fcfs, REPORT=REPORT, FEATS=FEATS, XGB_P=XGB_P, LGBM_P=LGBM_P)
exec(compile(code(4), 'nb22_cell4', 'exec'), NS)        # union_groups, group_kway, matched_tree_removal, build_conditions, make_S
exec(compile(code(5), 'nb22_cell5', 'exec'), NS)        # components, large components (asserts the published counts)
comp, big, rest = NS['comp'], NS['big'], NS['rest']
build_conditions, group_kway, make_S = NS['build_conditions'], NS['group_kway'], NS['make_S']
def reports_of(idx):
    s = df.loc[idx, 'sybil'] == 1
    return set(REPORT.loc[idx][s])

orig = json.load(open('results/22_report_dependence.json'))
O = {(r['repeat'], r['k'], r['condition']): r for r in orig['per_fold'] if r['model'] == 'LightGBM'}
OC = {(r['repeat'], r['k'], r['condition']): r for r in orig['composition']}
assert len(O) == 30
code_key = dict(old_code=meta['old']['code_sha256'], new_code=meta['new']['code_sha256'], inputs=meta['new']['inputs_sha256'],
                lgbm=LGBM_P, model_seed=sp.SEED, n_feats=len(FEATS), nb22_cells=hashlib.sha256((code(4) + code(5)).encode()).hexdigest()[:16])
KEY = hashlib.sha256(json.dumps(code_key, sort_keys=True, default=str).encode()).hexdigest()
METRICS = ['threshold', 'precision', 'recall', 'f1', 'auroc', 'ap']

def fit_condition(tag, S, r, k, cond):
    p = f'{CK}/{tag}_r{r}_k{k}_{cond}.json'
    if os.path.exists(p):
        z = json.load(open(p))
        if z.get('key') == KEY: return z
    t0 = time.time()
    fit = sp.fit_lgbm(S, LGBM_P, seeds=(42,), verbose=False)
    e = sp.evaluate(f'LightGBM {cond}', S['y_val'], fit['val'], S['y_test'], fit['test'])
    assert e['threshold'] == sp.best_f1_threshold(S['y_val'], fit['val'])           # threshold from validation only
    z = dict(key=KEY, tag=tag, repeat=r, k=k, condition=cond, rounds=int(fit['rounds'][0]), seconds=round(time.time() - t0, 1),
             **{m: float(e[m]) for m in METRICS})
    json.dump(z, open(p, 'w'), indent=1)
    np.savez(f'{D}/cache/s3_{tag}_r{r}_k{k}_{cond}.npz', val=fit['val'], test=fit['test'])
    return z

def summarize(pairs):
    """pairs: list of dicts with orig/new held_out+seen. Descriptive only; tests only at 15."""
    L = ['| Pair | k_a Sybils | F1 orig held / seen (diff) | F1 corrected held / seen (diff) | AP orig diff | AP corr diff | AUROC orig diff | AUROC corr diff |', '|---|---|---|---|---|---|---|---|']
    for q in pairs:
        o, n = q['orig'], q['new']
        d = lambda x, m: x['seen'][m] - x['held_out'][m]
        L.append(f"| r{q['repeat']} f{q['k']} | {q['test_sybils']} | {o['held_out']['f1']:.3f} / {o['seen']['f1']:.3f} ({d(o,'f1'):+.3f}) | "
                 f"{n['held_out']['f1']:.3f} / {n['seen']['f1']:.3f} ({d(n,'f1'):+.3f}) | {d(o,'ap'):+.3f} | {d(n,'ap'):+.3f} | {d(o,'auroc'):+.3f} | {d(n,'auroc'):+.3f} |")
    S = {}
    for tag in ('orig', 'new'):
        for m in ('f1', 'ap', 'auroc'):
            diffs = np.array([q[tag]['seen'][m] - q[tag]['held_out'][m] for q in pairs])
            S[f'{tag}_{m}'] = dict(mean=float(diffs.mean()), sd=float(diffs.std(ddof=1)) if len(diffs) > 1 else None, seen_higher=int((diffs > 0).sum()))
    return L, S

pairs, t_all = [], time.time()
n_done = 0
for r, rep_seed in enumerate(sp.SPLIT_SEEDS[:3]):
    fold_of = pd.Series(-1, index=df.index)
    fold_of[rest] = group_kway(comp[rest], df['sybil'][rest], [0.2] * 5, seed=rep_seed)
    for k in range(5):
        if n_done >= MAX_PAIRS: break
        t0 = time.time()
        C = build_conditions(df, comp, big, k, fold_of, ab_seed=1000 * rep_seed + k, rm_seed=2000 * rep_seed + k)
        ka, kb, val = C['k_a'], C['k_b'], C['val']; H, Se = C['held_out_train'], C['seen_train']
        # acceptance checks of notebook 22 for this fold-repeat
        for a, b in [(ka, val), (ka, H), (ka, Se), (val, H), (val, Se)]:
            assert len(a.intersection(b)) == 0
        assert not set(df.loc[ka, 'provision_tree']) & set(df.loc[kb, 'provision_tree'])
        assert not reports_of(ka) & reports_of(H)
        hs, ss = int(df.loc[H, 'sybil'].sum()), int(df.loc[Se, 'sybil'].sum())
        assert abs(len(Se) - len(H)) <= 0.01 * len(H) and abs(ss - hs) <= 0.01 * hs
        # recorded composition of the original run equals this construction
        for cond, tr in (('held_out', H), ('seen', Se)):
            oc = OC[(r, k, cond)]
            assert oc['train_unique'] == len(tr) and oc['train_sybils'] == int(df.loc[tr, 'sybil'].sum()), (r, k, cond)
        ident = dict(held_out=sha_arr(H), seen=sha_arr(Se), val=sha_arr(val), test=sha_arr(ka))
        res = {'orig': {}, 'new': {}}
        for cond, tr in (('held_out', H), ('seen', Se)):
            S_new = make_S(new, FEATS, tr, val, ka, seed=sp.SEED)
            S_old = make_S(old, FEATS, tr, val, ka, seed=sp.SEED)
            # actual row identities (not only counts): upsampled training rows and their order, validation, test, labels
            for kk in ('X_train', 'X_val', 'X_test'):
                assert np.array_equal(S_old[kk].index.to_numpy(), S_new[kk].index.to_numpy()), (r, k, cond, kk)
            for kk in ('y_train', 'y_val', 'y_test'):
                assert np.array_equal(S_old[kk].to_numpy(), S_new[kk].to_numpy()), (r, k, cond, kk)
            res['new'][cond] = fit_condition('new', S_new, r, k, cond)
            stored = O[(r, k, cond)]
            res['orig'][cond] = {m: float(stored[m]) for m in METRICS} | {'rounds': int(stored['rounds'])}
            if n_done == 0:      # establish comparability: refit the original arm here and compare with the stored PR #13 row
                z = fit_condition('old', S_old, r, k, cond)
                dmax = max(abs(z[m] - stored[m]) for m in METRICS)
                res['orig'][cond]['refit_here_max_abs_diff'] = dmax; res['orig'][cond]['refit_rounds'] = z['rounds']
                log(f"[stage3] comparability r{r} f{k} {cond}: refit original here vs stored PR #13: max |diff| {dmax:.3g}, rounds {z['rounds']} vs {int(stored['rounds'])}")
        q = dict(repeat=r, k=k, test_rows=len(ka), test_sybils=int(df.loc[ka, 'sybil'].sum()), identities=ident, orig=res['orig'], new=res['new'])
        pairs.append(q); n_done += 1
        json.dump(pairs, open(f'{D}/stage3_pairs.json', 'w'), indent=1)
        L, S = summarize(pairs)
        dO, dN = q['orig']['seen']['f1'] - q['orig']['held_out']['f1'], q['new']['seen']['f1'] - q['new']['held_out']['f1']
        log(f"[stage3] pair {n_done}/15 (repeat {r} fold {k}): F1 held/seen original {q['orig']['held_out']['f1']:.4f}/{q['orig']['seen']['f1']:.4f} ({dO:+.4f}), "
            f"corrected {q['new']['held_out']['f1']:.4f}/{q['new']['seen']['f1']:.4f} ({dN:+.4f}); AUROC diff orig {q['orig']['seen']['auroc'] - q['orig']['held_out']['auroc']:+.4f} "
            f"corrected {q['new']['seen']['auroc'] - q['new']['held_out']['auroc']:+.4f}; seen>held F1 in {S['new_f1']['seen_higher']}/{n_done} (corrected); {time.time() - t0:.0f}s")
        open(f'{D}/stage3_progress.md', 'w').write(f"# Stage 3 (LightGBM only): {n_done} of 15 pairs complete (descriptive until 15)\n\n" + '\n'.join(L) + '\n\n'
                                                    + json.dumps(S, indent=1) + '\n')
    if n_done >= MAX_PAIRS: break

if n_done == 15:      # planned aggregate tests, exactly as notebook 22
    kk = [q for q in orig['composition'] if q['condition'] == 'k_a']
    tr = [q for q in orig['composition'] if q['condition'] == 'held_out']
    RATIO = float(np.mean([a['test_rows'] / b['train_unique'] for a, b in zip(sorted(kk, key=lambda x: (x['repeat'], x['k'])), sorted(tr, key=lambda x: (x['repeat'], x['k'])))]))
    tests = []
    for tag in ('orig', 'new'):
        for m in ('f1', 'ap', 'auroc', 'recall', 'precision'):
            d = np.array([q[tag]['seen'][m] - q[tag]['held_out'][m] for q in pairs]); v = d.var(ddof=1)
            t = d.mean() / np.sqrt((1 / 15 + RATIO) * v)
            tests.append(dict(arm=tag, metric=m, held_out_mean=float(np.mean([q[tag]['held_out'][m] for q in pairs])), seen_mean=float(np.mean([q[tag]['seen'][m] for q in pairs])),
                              diff_mean=float(d.mean()), diff_sd=float(d.std(ddof=1)), seen_higher_n=int((d > 0).sum()), t_corrected=float(t),
                              p_corrected=float(2 * stats.t.sf(abs(t), 14)), p_wilcoxon_uncorrected=float(stats.wilcoxon(d).pvalue)))
    json.dump(dict(ratio=RATIO, tests=tests), open(f'{D}/stage3_tests.json', 'w'), indent=1)
    log('[stage3] all 15 pairs complete; planned tests written to stage3_tests.json')
log(f'[stage3] finished {n_done} pairs in {time.time() - t_all:.0f}s')
