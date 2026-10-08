"""Do the 03/04/06 runs on this machine reproduce the committed results and predictions?

Compares (1) each regenerated results JSON with the committed one, field by field, ignoring only
code_commit; (2) the regenerated output/pred_*.parquet with the predictions saved by the committed
run (path given by REFERENCE_OUTPUT, from the 0d9eacc run), value for value. Exits 1 on any difference.
"""
import json, os, subprocess, sys
import numpy as np, pandas as pd
LOG = 'logs/rerun_x86/regen'
REF = os.environ.get('REFERENCE_OUTPUT', '/home/user/layerzero_xgboost/output')
ok = True
def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f'{p}[{i}]')
    else: yield p, o
for nb in ['03_xgboost_sybil', '04_lightgbm_sybil', '06_cross_ensemble_sybil']:
    new = json.load(open(f'{LOG}/{nb}.json'))
    old = json.loads(subprocess.check_output(['git', 'show', f'HEAD:results/{nb}.json']))
    a, b = dict(flat(old)), dict(flat(new))
    diff = [k for k in sorted(set(a) | set(b)) if k != '.code_commit' and a.get(k) != b.get(k)]
    print(f'{nb}: committed at {old["code_commit"]}, regenerated at {new["code_commit"]}; '
          f'{len(a)} fields, {len(diff)} differ (code_commit ignored)')
    for k in diff[:20]: print(f'   {k}: committed {a.get(k)!r} regenerated {b.get(k)!r}')
    ok &= not diff
    for part in ['val', 'test']:
        f = f'pred_{nb}_{part}.parquet'
        n = pd.read_parquet(f'output/{f}')
        if not os.path.exists(f'{REF}/{f}'):
            print(f'   {f}: no reference file at {REF}'); continue
        r = pd.read_parquet(f'{REF}/{f}')
        same_rows = list(n.columns) == list(r.columns) and len(n) == len(r) and (n['addr'].values == r['addr'].values).all()
        pcol = [c for c in n.columns if c.startswith('prob_')][0]
        exact = same_rows and np.array_equal(n[pcol].values, r[pcol].values)
        maxdiff = float(np.abs(n[pcol].values - r[pcol].values).max()) if same_rows else None
        print(f'   {f}: rows {len(n):,}; same rows and order {same_rows}; {pcol} bit-identical {exact}; max |diff| {maxdiff}')
        ok &= exact
print('REPRODUCED' if ok else 'NOT REPRODUCED')
sys.exit(0 if ok else 1)
