"""Correct stale notebook language metadata after all scientific executions pass."""
import pathlib,json,sys,subprocess
root=pathlib.Path(sys.argv[1]).resolve();out=root/'docs/cloud-reproduction'
state=json.loads((out/'execution.json').read_text());latest={int(x['notebook'][:2]):x for x in state['stages'] if x['status']=='passed'}
order=list(range(21))+[22,23,24,25,99];assert set(latest)==set(order)
p=json.loads((out/'platform.json').read_text())
assert subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()==''
for n in order:
 file=root/latest[n]['notebook'];nb=json.loads(file.read_text());before=[c['source'] for c in nb['cells'] if c['cell_type']=='code']
 assert all(c.get('execution_count') is not None and not any(o['output_type']=='error' for o in c.get('outputs',[])) for c in nb['cells'] if c['cell_type']=='code')
 for cell in nb['cells']:
  if cell['cell_type']=='code':cell.setdefault('metadata',{}).pop('execution',None)
 meta=nb.setdefault('metadata',{});meta.setdefault('language_info',{})['version']=p['python']
 meta['cloud_reproduction']=dict(execution_commit=latest[n]['execution_commit'],platform_manifest='docs/cloud-reproduction/platform.json',python=p['python'],cpu_model=p['processor'],architecture=p['machine'],model_threads=p['n_jobs'],execution_method=p['execution_method'])
 if n==13:
  r13=json.loads((root/'results/13_random_forest_sybil.json').read_text());r08=json.loads((root/'results/08_ablation_families.json').read_text());assert r13['platform']==r08['platform'] and r13['summary']['lgbm_reproduces_08']
  nb['cells']=[c for c in nb['cells'] if c.get('id')!='cloud-platform-note']
  nb['cells'].append(dict(cell_type='markdown',id='cloud-platform-note',metadata={},source=['### Cloud run provenance note\n','For this completed reproduction, notebook 08 and the LightGBM fits in this notebook use the same documented AMD cloud platform and pass the paired numerical tie-out. Static wording about another machine or platform differences in the inherited comparison code refers to historical revisions; the executed numbers and platform records above are preserved.']))
 if n==15:
  r15=json.loads((root/'results/15_split_tree_report.json').read_text());r08=json.loads((root/'results/08_ablation_families.json').read_text());assert r15['platform']==r08['platform']
  nb['cells']=[c for c in nb['cells'] if c.get('id')!='cloud-platform-note']
  nb['cells'].append(dict(cell_type='markdown',id='cloud-platform-note',metadata={},source=['### Cloud run provenance and feasibility note\n','The historical field name delta_vs_08_all_features_other_machine is retained for schema compatibility; this run’s notebooks 15 and 08 use the same documented AMD platform. The main union split passes its existing size/class-rate criteria with a large-component warning. The optional reporter-linked variant fails its class-rate feasibility criteria and is stopped before training, as recorded in components_reporter_variant. No tolerance is relaxed.']))
 file.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n');assert before==[c['source'] for c in json.loads(file.read_text())['cells'] if c['cell_type']=='code']
print('Corrected actual-language/platform metadata in all26 executed notebooks; scientific code/output unchanged.')
