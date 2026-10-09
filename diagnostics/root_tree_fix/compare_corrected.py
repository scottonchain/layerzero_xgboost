"""Corrected publication run versus the pre-fix reference (PR #13 head b7a9a3f): what changed, by notebook.

Reads results/*.json at the working tree and at b7a9a3f. Reports, for every notebook that has been rerun in the
corrected run, how many stored fields differ, and prints the numbers the paper cites most. Writes
diagnostics/root_tree_fix/corrected_vs_prefix.json and .md.
Usage: python diagnostics/root_tree_fix/compare_corrected.py   (from the repository root)
"""
import json, math, os, subprocess, sys
REF = 'b7a9a3f'
SKIP = ('code_commit', 'platform', 'seconds', 'runtime', 'elapsed', 'cost', 'memory')
def ref_json(path):
    r = subprocess.run(['git', 'show', f'{REF}:{path}'], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None
def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f'{p}[{i}]')
    else: yield p, o
def isnum(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def same(a, b):
    if a == b: return True
    return isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)

names = sorted(f[:-5] for f in os.listdir('results') if f.endswith('.json') and f[:2].isdigit())
out, L = {}, ['# Corrected run versus the pre-fix reference (b7a9a3f)', '']
L += ['| Notebook | Corrected run at | Stored fields compared | Differ | Max abs numeric difference |', '|---|---|---|---|---|']
for nb in names:
    cur = json.load(open(f'results/{nb}.json')); old = ref_json(f'results/{nb}.json')
    if old is None or cur.get('code_commit') == old.get('code_commit'):
        continue     # not rerun in the corrected run yet (or new)
    a, b = dict(flat(old)), dict(flat(cur))
    keys = [k for k in set(a) & set(b) if not any(s in k.lower() for s in SKIP)]
    diff = [k for k in keys if not same(a[k], b[k])]
    mx = max((abs(a[k] - b[k]) for k in diff if isnum(a[k]) and isnum(b[k]) and not (math.isnan(a[k]) or math.isnan(b[k]))), default=0.0)
    out[nb] = dict(compared=len(keys), differ=len(diff), max_abs=mx, corrected_commit=cur['code_commit'])
    L.append(f"| {nb} | {cur['code_commit']} | {len(keys):,} | {len(diff):,} | {mx:.4g} |")

def models(nb):
    cur = json.load(open(f'results/{nb}.json')); old = ref_json(f'results/{nb}.json')
    return cur, old
L += ['', '## Headline model results (test, validation-selected threshold)', '', '| Notebook | Model | F1 pre-fix | F1 corrected | AP pre-fix | AP corrected | AUROC pre-fix | AUROC corrected |', '|---|---|---|---|---|---|---|---|']
for nb in ('03_xgboost_sybil', '04_lightgbm_sybil', '05_logistic_regression_sybil', '06_cross_ensemble_sybil', '13_random_forest_sybil'):
    if nb in out:
        cur, old = models(nb)
        for mc in cur['models']:
            mo = next((x for x in old['models'] if x['name'] == mc['name']), None)
            if mo: L.append(f"| {nb} | {mc['name']} | {mo['f1']:.4f} | {mc['f1']:.4f} | {mo['ap']:.4f} | {mc['ap']:.4f} | {mo['auroc']:.4f} | {mc['auroc']:.4f} |")
if '99_tie_out_revision' in out:
    c = json.load(open('results/99_tie_out_revision.json'))
    n_ok = sum(x['passed'] for x in c['checks'])
    L += ['', f"## Notebook 99 (corrected run): {n_ok} of {len(c['checks'])} checks pass; all_passed = {c['all_passed']}; multiple_platforms_cited = {c['multiple_platforms_cited']}"]
    for x in c['checks']:
        if not x['passed']: L.append(f"- FAIL: {x['check']}: {x['detail']}")
open('diagnostics/root_tree_fix/corrected_vs_prefix.md', 'w').write('\n'.join(L) + '\n')
json.dump(out, open('diagnostics/root_tree_fix/corrected_vs_prefix.json', 'w'), indent=1)
print('\n'.join(L))
