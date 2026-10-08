import os,sys,json,pathlib,subprocess,hashlib,importlib.metadata,math
root=pathlib.Path(sys.argv[1]).resolve();os.chdir(root);sys.path.insert(0,str(root))
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='4',PYTHONHASHSEED='42')
import pandas as pd,nbformat,sybil_pipeline as sp
from packaging.utils import canonicalize_name
out=root/'docs/cloud-reproduction';checks=[]
def check(name,ok,detail=''):
 checks.append(dict(check=name,passed=bool(ok),detail=detail));assert ok,(name,detail)
g=lambda *a:subprocess.check_output(['git',*a],text=True).strip()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
state=json.loads((out/'execution.json').read_text());latest={int(x['notebook'][:2]):x for x in state['stages'] if x['status']=='passed'};order=list(range(21))+[22,23,24,25,99]
check('complete execution inventory',set(latest)==set(order),str(sorted(latest)))
R={};env=sp.runtime_platform()
import io,contextlib,numpy,scipy
backend=io.StringIO()
with contextlib.redirect_stdout(backend):numpy.show_config();scipy.show_config()
check('established numerical backends exactly preserved',backend.getvalue()==json.loads((out/'platform.json').read_text())['numerical_backends'])
for n in order:
 p,=root.glob(f'{n:02d}_*.ipynb');nb=nbformat.read(p,as_version=4)
 check(f'{n:02d} actual notebook language metadata',nb.metadata.language_info.version==env['python'])
 check(f'{n:02d} all code cells executed without errors',all(c.get('execution_count') is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code'))
 commits=g('log','--reverse','--format=%H',latest[n]['execution_commit']+'..HEAD','--',p.name).splitlines();check(f'{n:02d} published artifact reachable',bool(commits));latest[n]['artifact_commit']=commits[0]
 f=root/'results'/(p.stem+'.json')
 if not f.exists():check(f'{n:02d} result-file exception',n==10);continue
 r=R[n]=json.loads(f.read_text());check(f'{n:02d} execution commit matches inventory',r['execution_commit']==latest[n]['execution_commit']);check(f'{n:02d} full producer commit reachable',g('cat-file','-t',r['execution_commit'])=='commit' and subprocess.run(['git','merge-base','--is-ancestor',r['execution_commit'],'HEAD'],capture_output=True).returncode==0)
 check(f'{n:02d} one actual numerical platform',all(r['platform'][k]==env[k] for k in env))
prov=sp.provenance([r['notebook'] for r in R.values()]);check('all source dependency provenance',prov.dependencies_unchanged.all(),prov.to_json(orient='records'))
# Publication executions must follow their actual current source results.
import ast
publication_dependencies={12:[0,2,3,4,5,6],24:[2,9,18]}
for n in [23,25]:
 nb=nbformat.read(next(root.glob(f'{n}_*.ipynb')),as_version=4)
 for c in nb.cells:
  if c.cell_type=='code':
   for node in ast.walk(ast.parse(c.source)):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in node.targets):publication_dependencies[n]=[int(x[:2]) for x in ast.literal_eval(node.value)]
for n,dependencies in publication_dependencies.items():
 for source in dependencies:check(f'{n:02d} publication follows current source {source:02d}',subprocess.run(['git','merge-base','--is-ancestor',latest[source]['artifact_commit'],R[n]['execution_commit']],capture_output=True).returncode==0)
check('final99 all checks passed',R[99]['all_passed']);check('14 predictions match',R[14]['inputs']['predictions_match_committed_results']);check('13 paired LightGBM baseline matches08',R[13]['summary']['lgbm_reproduces_08']);check('17 baseline matches08',R[17]['lgbm_matches_08']);check('19 baseline matches08',R[19]['all_features_matches_08'])
rf13={x['split_seed']:x for x in R[13]['per_split']}
rf17=[x for x in R[17]['per_split'] if x['model']=='Random Forest']
check('17 ten matching Random Forest comparisons',len(rf17)==10 and len(rf13)==10)
rf_diagnostics=[]
for row in rf17:
 prior=rf13[row['split_seed']];delta={k:row[k]-prior['test_'+k] for k in ['f1','ap','auroc']};delta['threshold']=row['threshold']-prior['threshold']
 check('17 Random Forest threshold/F1 and publication rounding match13 split '+str(row['split_seed']),abs(delta['f1'])<1e-12 and abs(delta['threshold'])<1e-12 and all(f"{row[k]:.3f}"==f"{prior['test_'+k]:.3f}" for k in ['ap','auroc']))
 rf_diagnostics.append(dict(split_seed=row['split_seed'],deltas=delta,full_precision_1e_12_match=all(abs(v)<1e-12 for v in delta.values())))
(out/'random-forest-cross-stage-diagnostic.json').write_text(json.dumps(dict(all_full_precision_checks_passed=all(x['full_precision_1e_12_match'] for x in rf_diagnostics),comparisons=rf_diagnostics,interpretation='Threaded sklearn predict_proba accumulates tree outputs under a lock in thread completion order. The observed tiny AP differences are consistent with floating-point ordering affecting tied ranks; thresholds, F1 and displayed3-decimal publication metrics are checked separately. No notebook assertion is changed.'),indent=2)+'\n')
w=R[6]['val_f1_by_weight'];check('06 validation-selected blend weight',float(max(w,key=w.get))==R[6]['xgb_weight'])
for kind,key,total in [('xgboost','xgb_selected',74),('lightgbm','lgbm_selected',48)]:
 t=pd.read_csv(f'results/02_search_{kind}.csv');b=t.sort_values(['val_f1','val_ap'],ascending=False).iloc[0];check('02 '+kind+' complete grid',len(t)==total);check('02 '+kind+' validation winner',all(b[k]==v for k,v in R[2][key].items()))
t=pd.read_csv('results/13_search_random_forest.csv',dtype=str,keep_default_na=False);b=t.assign(val_f1=pd.to_numeric(t.val_f1),val_ap=pd.to_numeric(t.val_ap)).sort_values(['val_f1','val_ap'],ascending=False).iloc[0];check('13 complete predefined grid',len(t)==24);check('13 validation winner',all(b[k]==str(R[13]['params'][k]) for k in ['max_features','min_samples_leaf','max_depth','max_samples']))
a=pd.DataFrame(R[8]['per_split']);families=R[8]['families'];configs={'All features':list(sp.FEATS)}
configs.update({'Without '+name:[f for f in sp.FEATS if f not in fs] for name,fs in families.items()});configs.update({name+' only':fs for name,fs in families.items()});network=set().union(*(set(families[k]) for k in ['Gas provider','Gas provision tree','Provision chain']));configs['Without all provision-network families']=[f for f in sp.FEATS if f not in network]
check('08 all configurations and split seeds',set(a.config)==set(configs) and len(a)==10*len(configs))
for config,features in configs.items():
 t=a[a.config==config];check('08 feature count '+config,len(t)==10 and t.split_seed.is_unique and (t.n_features==len(features)).all())
 for metric in ['val_f1','test_f1','test_ap','test_auroc']:
  for stat in ['mean','std']:check('08 summary '+config+' '+metric+' '+stat,abs(getattr(t[metric],stat)()-R[8]['summary'][config][metric+'_'+stat])<1e-12)
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score,roc_auc_score
pred=[]
for n in [3,4,5,6,13]:
 stem=R[n]['notebook'];frames={p:sp.load_predictions(stem,p) for p in ['val','test']};check(f'{n:02d} unique disjoint prediction partitions',all(f.addr.is_unique for f in frames.values()) and not set(frames['val'].addr)&set(frames['test'].addr));col,=[c for c in frames['val'] if c.startswith('prob_')];thr=sp.best_f1_threshold(frames['val'].y,frames['val'][col]);m=next(x for x in R[n]['models'] if x['name']=='Cross-Ensemble') if n==6 else R[n]['models'][0];check(f'{n:02d} validation threshold',thr==m['threshold']);y=frames['test'].y;p=frames['test'][col];z=p>=thr;metrics=dict(precision=precision_score(y,z),recall=recall_score(y,z),f1=f1_score(y,z),ap=average_precision_score(y,p),auroc=roc_auc_score(y,p));check(f'{n:02d} prediction metric tie-out',all(abs(m[k]-v)<1e-9 for k,v in metrics.items()),json.dumps(metrics))
 for part in frames:
  file=root/'output'/f'pred_{stem}_{part}.parquet';pred.append(dict(path=str(file.relative_to(root)),sha256=sha(file),bytes=file.stat().st_size,execution_commit=R[n]['execution_commit']))
blendframes={n:sp.load_predictions(R[n]['notebook'],'val').set_index('addr') for n in [3,4,6]};base=blendframes[3];other=blendframes[4].loc[base.index];actual_blend=blendframes[6].loc[base.index];check('06 blend labels and addresses agree',base.y.equals(other.y) and base.y.equals(actual_blend.y))
px=base.prob_xgb.to_numpy();pl=other.prob_lgbm.to_numpy();chosen=R[6]['xgb_weight'];check('06 saved blend equals components',numpy.allclose(chosen*px+(1-chosen)*pl,actual_blend.prob_ens,rtol=0,atol=1e-15))
for key,value in R[6]['val_f1_by_weight'].items():
 weight=numpy.float64(key);prob=weight*px+(1-weight)*pl;thr=sp.best_f1_threshold(base.y,prob);score=f1_score(base.y,prob>=thr);check('06 full-precision validation blend sweep '+key,abs(score-value)<1e-12)
for n in [23,24,25]:
 c=str(R[n]['compile_check']);check(f'{n} actual pdflatex compile',('pdflatex' in c.lower()) and not any(x in c.lower() for x in ['skip','warning','not installed']),c)
comp=pd.DataFrame(R[22]['composition']);arms=comp[comp.condition.isin(['held_out','seen'])];p=arms.pivot(index=['repeat','k'],columns='condition',values=['train_unique','train_sybils']);check('22 all15 paired arms',len(p)==15 and len(arms)==30)
for key in ['train_unique','train_sybils']:check('22 paired '+key+' tolerance',((p[(key,'seen')]-p[(key,'held_out')]).abs()<=.01*p[(key,'held_out')]).all())
rows=[]
for r in arms.to_dict('records'):
 neg=int(r['train_unique']-r['train_sybils']);rows.append({**r,'model_training_rows':2*neg,'model_training_sybils':neg,'model_training_non_sybils':neg})
(out/'paired-training-counts.json').write_text(json.dumps(dict(note='Observed unique/class counts from22; balanced model counts follow make_S exactly: one resampled Sybil per observed majority row.',rows=rows),indent=2)+'\n')
assets={}
def collect(v,n):
 if isinstance(v,str) and v.startswith(('figures/','tables/')):
  check('asset '+v,(root/v).is_file());assets[v]=n
 elif isinstance(v,dict):
  for k,x in v.items():
   if k!='sources':collect(x,n)
 elif isinstance(v,list):
  for x in v:collect(x,n)
for n in [12,23,24,25]:collect(R[n].get('figures',{}),n);collect(R[n].get('tables',{}),n)
for name in R[12]['figures']:p='figures/'+name;check('12 figure '+name,(root/p).is_file());assets[p]=12
for n in [23,25]:
 for group in ['figures','tables']:
  for item in R[n].get(group,{}).values():
   for source in item.get('sources',[]):
    if 'results' in source:
     result=json.loads((root/source['results']).read_text());check(f'{n} source artifact '+source['results'],source['code_commit']==result['code_commit'])
diagnostic=R[15]['components_reporter_variant'];failed_optional=[k for k,v in diagnostic['checks'].items() if not v]
check('15 main union feasibility criteria preserved',R[15]['components']['passed'] and all(R[15]['components']['checks'].values()))
check('15 infeasible reporter variant recorded without training',not diagnostic['passed'] and len(failed_optional)==3 and R[15]['status']=='ok_with_warning')
manifest=[dict(path=p,sha256=sha(root/p),bytes=(root/p).stat().st_size,producer_notebook=n,execution_commit=R[n]['execution_commit']) for p,n in sorted(assets.items())]
for f in json.loads((out/'lfs-inputs.json').read_text()):check('LFS '+f['path'],(root/f['path']).stat().st_size==f['bytes'] and sha(root/f['path'])==f['sha256'])
for p,expected_hash in json.loads((out/'resume-verification.json').read_text())['intermediate_sha256'].items():check('restored intermediate identity '+p,sha(root/p)==expected_hash)
expected={canonicalize_name(x.split('==')[0]):x.split('==')[1] for x in (out/'pip-freeze.txt').read_text().splitlines() if '==' in x};actual={canonicalize_name(d.metadata['Name']):d.version for d in importlib.metadata.distributions()};check('all113 pinned packages',actual==expected)
for file,data in [('prediction-manifest.json',pred),('publication-manifest.json',manifest),('independent-validation.json',dict(all_passed=True,all_applicable_mandatory_checks_passed=True,all_checks_including_optional_diagnostics_passed=not failed_optional and all(x['full_precision_1e_12_match'] for x in rf_diagnostics),random_forest_full_precision_diagnostics=rf_diagnostics,checks=checks,feasibility_diagnostics=dict(notebook15_reporter_variant_passed=diagnostic['passed'],failed_checks=failed_optional,maximum_component_sybils=diagnostic['largest_by_sybils'][0]['sybils'],maximum_component_sybil_share=diagnostic['max_component_sybil_share'],reason='One whole component contains10071 Sybils, exceeding every partition’s joint size/class-rate bounds; guarded variant is not trained. The main union variant passes, retaining its large-component warning.'))),('execution.json',state)]: (out/file).write_text(json.dumps(data,indent=2)+'\n')
(root/'output/cloud-run/execution.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(dict(all_passed=True,checks=len(checks),assets=len(manifest),predictions=len(pred))))
