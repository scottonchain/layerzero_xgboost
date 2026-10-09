"""Paired training counts of 22 (seen vs held out) per fold-repeat, from results/22_report_dependence.json.

train_unique and train_sybils are recorded by 22 itself (unique rows and Sybils of each training arm).
Rows fed to the model follow make_S: non-Sybils kept, Sybils resampled with replacement to the same number,
so model_rows = 2 * (train_unique - train_sybils) and model_sybil_rows = train_unique - train_sybils.
"""
import json
r = json.load(open('results/22_report_dependence.json'))
comp = [c for c in r['composition'] if c['condition'] in ('held_out', 'seen')]
pairs = {}
for c in comp:
    p = pairs.setdefault((c['repeat'], c['k']), {})
    neg = c['train_unique'] - c['train_sybils']
    p[c['condition']] = dict(train_unique=c['train_unique'], train_sybils=c['train_sybils'], train_reports=c['train_reports'],
                             model_rows=2 * neg, model_sybil_rows=neg)
ka = {(c['repeat'], c['k']): c for c in r['composition'] if c['condition'] == 'k_a'}
rows = []
for (rep, k), p in sorted(pairs.items()):
    h, s = p['held_out'], p['seen']
    rows.append(dict(repeat=rep, fold=k, test_rows=ka[(rep, k)]['test_rows'], test_sybils=ka[(rep, k)]['test_sybils'],
                     held_out=h, seen=s, rows_diff=s['train_unique'] - h['train_unique'], sybils_diff=s['train_sybils'] - h['train_sybils'],
                     rows_within_1pct=abs(s['train_unique'] - h['train_unique']) <= 0.01 * h['train_unique'],
                     sybils_within_1pct=abs(s['train_sybils'] - h['train_sybils']) <= 0.01 * h['train_sybils']))
out = dict(source='results/22_report_dependence.json', code_commit=r['code_commit'], pairs=rows,
           summary=dict(pairs=len(rows), equal_train_rows=sum(x['rows_diff'] == 0 for x in rows),
                        equal_train_sybils=sum(x['sybils_diff'] == 0 for x in rows),
                        max_abs_rows_diff=max(abs(x['rows_diff']) for x in rows), max_abs_sybils_diff=max(abs(x['sybils_diff']) for x in rows),
                        all_within_1pct=all(x['rows_within_1pct'] and x['sybils_within_1pct'] for x in rows),
                        held_out_sybils_mean=sum(x['held_out']['train_sybils'] for x in rows) / len(rows)))
json.dump(out, open('logs/rerun_x86_68e8d63/paired_training_counts_22.json', 'w'), indent=1)
print(json.dumps(out['summary']))
for x in rows: print(x['repeat'], x['fold'], 'held', x['held_out']['train_unique'], x['held_out']['train_sybils'], '| seen', x['seen']['train_unique'], x['seen']['train_sybils'], '| model rows', x['held_out']['model_rows'], x['seen']['model_rows'], '| test', x['test_rows'], x['test_sybils'])
