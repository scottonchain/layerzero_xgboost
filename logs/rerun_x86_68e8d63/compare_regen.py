"""Do 03/04/06 regenerated here reproduce the committed results?

Compares each regenerated results JSON (run/regen/) with the committed one at HEAD, field by field at
full precision, ignoring only code_commit. The committed run's prediction files (output/, gitignored) are
not available here, so the predictions are checked through every metric, threshold and early-stopping
round they produce, and again by 14, which recomputes the metrics from output/ and requires 1e-9 agreement.
Their SHA-256 hashes are recorded in run/regen/predictions.sha256. Exits 1 on any difference.
"""
import json, subprocess, sys
import pandas as pd
R = 'logs/rerun_x86_68e8d63/run/regen'
def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f'{p}[{i}]')
    else: yield p, o
ok = True
for nb in ['03_xgboost_sybil', '04_lightgbm_sybil', '06_cross_ensemble_sybil']:
    new = json.load(open(f'{R}/{nb}.json'))
    old = json.loads(subprocess.check_output(['git', 'show', f'HEAD:results/{nb}.json']))
    a, b = dict(flat(old)), dict(flat(new))
    diff = [k for k in sorted(set(a) | set(b)) if k != '.code_commit' and a.get(k) != b.get(k)]
    print(f'{nb}: committed at {old["code_commit"]}, regenerated at {new["code_commit"]}; '
          f'{len(a)} fields, {len(diff)} differ (code_commit ignored)')
    for k in diff[:20]: print(f'   {k}: committed {a.get(k)!r} regenerated {b.get(k)!r}')
    ok &= not diff
    for part in ['val', 'test']:
        n = pd.read_parquet(f'output/pred_{nb}_{part}.parquet')
        print(f'   pred_{nb}_{part}.parquet: {len(n):,} rows, columns {list(n.columns)}')
print('REPRODUCED' if ok else 'NOT REPRODUCED')
sys.exit(0 if ok else 1)
