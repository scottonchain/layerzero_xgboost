"""Numbers for REPORT.md, read from results/ at HEAD and from the two references (68e8d63, PR #8 cf47a0f)."""
import json, subprocess, csv, io
REF = {'68e8d63': '68e8d63', 'PR #8': 'paven/pr-8'}
def show(ref, path):
    r = subprocess.run(['git', 'show', f'{ref}:{path}'], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None
cur = lambda nb: json.load(open(f'results/{nb}.json'))
print('## runtimes (timeline.tsv)')
for line in open('logs/rerun_x86_68e8d63/run/timeline.tsv'):
    f = line.rstrip('\n').split('\t')
    if f[1] in ('END', 'RESTART', 'ABSENT', 'HOSTCHECK', 'PASS', 'FAILED', 'DONE'): print('  ', ' | '.join(f))
print('\n## 14 Section 5.5: LightGBM at the validation threshold, bucket all')
def e14(d):
    if d is None: return {}
    return {r['entity']: r for r in d['table'] if r['model'] == 'LightGBM' and r['threshold_label'] == 'validation' and r['bucket'] == 'all'}
n14 = cur('14_entity_level_recall'); a = e14(n14); refs = {k: e14(show(v, 'results/14_entity_level_recall.json')) for k, v in REF.items()}
thr = n14['thresholds']['validation']['LightGBM'] if 'thresholds' in n14 else None
print('   threshold', thr, '| predictions_match_committed_results', n14['inputs']['predictions_match_committed_results'])
for ent, r in a.items():
    for m in ['entities', 'any_hit', 'majority_hit', 'full_hit', 'within_recall_unweighted', 'within_recall_weighted', 'residual_allocation_share', 'entities_majority_allocation_unflagged_share']:
        vals = [r[m]] + [refs[k].get(ent, {}).get(m) for k in REF]
        print(f'   {ent:38s} {m:46s} now {vals[0]!r:22} 68e8d63 {vals[1]!r:22} PR#8 {vals[2]!r}')
print('\n## 22 central paired comparison (seen - held out)')
n22 = cur('22_report_dependence'); r22 = {k: show(v, 'results/22_report_dependence.json') for k, v in REF.items()}
for t in n22['tests']:
    o = {k: next(x for x in d['tests'] if x['model'] == t['model'] and x['metric'] == t['metric']) for k, d in r22.items()}
    print(f"   {t['model']:8s} {t['metric']:9s} held {t['held_out_mean']:.6f} seen {t['seen_mean']:.6f} diff {t['diff_mean']:+.6f} ± {t['diff_sd']:.6f} higher {t['seen_higher_n']}/15 p {t['p_corrected']:.3g}"
          + ''.join(f" | {k}: diff {x['diff_mean']:+.6f} ± {x['diff_sd']:.6f} p {x['p_corrected']:.3g}" for k, x in o.items()))
print('\n## 99')
n99 = cur('99_tie_out_revision')
print('   all_passed', n99['all_passed'], '| passed', sum(c['passed'] for c in n99['checks']), 'of', len(n99['checks']), '| multiple_platforms_cited', n99['multiple_platforms_cited'])
for c in n99['checks']:
    if not c['passed']: print('   FAIL', c['check'], c['detail'])
