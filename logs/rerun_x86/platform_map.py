"""Platform of every revision figure and table, from the source records that 23 and 25 save.

00-12 results record no platform; they are the committed x86_64 Linux run at 0d9eacc. This rerun shows
that this machine reproduces them (03/04/06 results and predictions bit-identical; 13's LightGBM on the
10 splits of 08 equal to 08). An asset is single-platform when all its sources are x86_64/Linux or 00-12.
"""
import json, glob, os
R00_12 = 'x86_64 / Linux (committed 00-12 run, 0d9eacc; reproduced here)'
def plat(p):
    return R00_12 if p.startswith('not recorded') else p
rows = []
for nb in ['23_revision_figures', '25_paper_tables']:
    r = json.load(open(f'results/{nb}.json'))
    for asset, srcs in r['sources'].items():
        res = [s for s in srcs if 'results' in s]
        ps = sorted({plat(s['platform']) for s in res})
        files = [s['file'] for s in srcs if 'file' in s]
        x86 = all(p.startswith('x86_64') for p in ps)
        rows.append((asset, nb[:2], ', '.join(f"{os.path.basename(s['results'])[:2]}@{s['code_commit']}" for s in res)
                     + (('; ' + ', '.join(files)) if files else ''), ' + '.join(ps), 'single (x86_64)' if x86 else 'MIXED'))
# Paper figures drawn by 12 (00-12 inputs only)
r12 = json.load(open('results/12_figures.json'))
for f in sorted(glob.glob('figures/*.png')):
    b = os.path.basename(f)[:-4]
    if not b.startswith('rev_'):
        rows.append((b, '12', f"12@{r12['code_commit']}", R00_12, 'single (x86_64)'))
w = max(len(r[0]) for r in rows)
print('| Asset | Generator | Sources (results@commit) | Platform | Status |\n|---|---|---|---|---|')
for r in rows: print('| ' + ' | '.join(r) + ' |')
mixed = [r[0] for r in rows if r[4] == 'MIXED']
print(f'\n{len(rows)} assets; mixed-platform: {mixed if mixed else "none"}')
