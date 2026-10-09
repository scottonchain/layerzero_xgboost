"""Compare this rerun's results with (a) the results committed at 68e8d63 (PR #12; 13-20 and 22 from the
Apple M5, arm64) and (b) PR #8's x86 rerun (cf47a0f; 13-16 on a Xeon @ 2.10GHz host, 17-25 and 99 on a
Xeon @ 2.80GHz host). Every numeric leaf of each results JSON is compared at full precision; code_commit,
platform and timing fields are reported separately. Also compares results CSVs, tables/ and figures/ bytes.
Usage: python compare_results.py [--refs 68e8d63,paven/pr-8] > compare_results.txt
"""
import json, subprocess, sys, hashlib, math, os
REFS = {'68e8d63 (PR #12, committed)': '68e8d63', 'PR #8 x86 (cf47a0f)': 'paven/pr-8'}
NBS = ['13_random_forest_sybil', '14_entity_level_recall', '15_split_tree_report', '16_label_robustness_initial_list',
       '17_models_10_splits', '18_mimicry_stress', '19_ablation_by_category', '20_temporal_holdout',
       '22_report_dependence', '23_revision_figures', '24_mimicry_controls', '25_paper_tables', '99_tie_out_revision']
SKIP = ('code_commit', 'platform', 'seconds', 'cost', 'runtime', 'elapsed', 'previous_commit', 'committed_code_commit')
git = lambda *a: subprocess.run(['git', *a], capture_output=True)
def flat(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f'{p}.{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f'{p}[{i}]')
    else: yield p, o
def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def load(ref, path):
    r = git('show', f'{ref}:{path}')
    return r.stdout if r.returncode == 0 else None
summary = {}
for nb in NBS:
    path = f'results/{nb}.json'
    if not os.path.exists(path): print(f'{nb}: no result in this run'); continue
    new = dict(flat(json.load(open(path))))
    print(f'=== {nb}  (this run: {new.get(".code_commit")})')
    for label, ref in REFS.items():
        raw = load(ref, path)
        if raw is None: print(f'  vs {label}: no result at {ref}'); continue
        old = dict(flat(json.loads(raw)))
        keys = sorted(set(old) | set(new))
        skip = [k for k in keys if any(s in k.lower() for s in SKIP)]
        cmp_ = [k for k in keys if k not in skip]
        only = [k for k in cmp_ if (k in old) != (k in new)]
        both = [k for k in cmp_ if k in old and k in new]
        diff = [k for k in both if old[k] != new[k] and not (num(old[k]) and num(new[k]) and (old[k] == new[k] or (isinstance(old[k], float) and math.isnan(old[k]) and math.isnan(new[k]))))]
        ndiff = [k for k in diff if num(old[k]) and num(new[k])]
        mx = max((abs(new[k] - old[k]) for k in ndiff), default=0.0)
        print(f'  vs {label} [{old.get(".code_commit")}]: {len(both)} shared fields, {len(diff)} differ '
              f'({len(ndiff)} numeric, max |diff| {mx:.3g}), {len(only)} only on one side, {len(skip)} commit/platform/timing fields skipped')
        for k in diff[:8]: print(f'      {k}: ref {old[k]!r}  now {new[k]!r}')
        summary[(nb, label)] = dict(shared=len(both), differ=len(diff), max_abs=mx, one_side=len(only))
print('\n=== files (byte-identical?)')
files = sorted(set(subprocess.run(['git', 'ls-files', 'results/*.csv', 'tables', 'figures/rev_*'], capture_output=True, text=True).stdout.split()))
for f in files:
    h = hashlib.sha256(open(f, 'rb').read()).digest()
    row = []
    for label, ref in REFS.items():
        raw = load(ref, f)
        row.append('absent' if raw is None else ('same' if hashlib.sha256(raw).digest() == h else 'DIFF'))
    print(f'  {f:52s} ' + '  '.join(f'{l.split()[0]}:{r}' for l, r in zip(REFS, row)))
json.dump({f'{k[0]} | {k[1]}': v for k, v in summary.items()}, open('logs/rerun_x86_68e8d63/run/compare_summary.json', 'w'), indent=1)
