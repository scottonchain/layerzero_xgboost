"""Atomic checkpoints for completed metric-only fits; scientific arguments unchanged."""
import functools,hashlib,json,os,subprocess
from pathlib import Path

def install(root, notebook):
    import numpy as np
    import sybil_pipeline as sp
    nb=json.loads((root/notebook).read_text())
    cells=[''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='code']
    sha=lambda b:hashlib.sha256(b).hexdigest()
    git=lambda *a:subprocess.check_output(['git',*a],text=True).strip()
    identity=dict(schema=1,notebook=notebook,code_cells=sha(json.dumps(cells).encode()),pipeline=sha((root/'sybil_pipeline.py').read_bytes()),requirements=sha((root/'requirements.txt').read_bytes()),checkpoint_code=sha(Path(__file__).read_bytes()),data_tree=git('rev-parse','HEAD:data'),platform=sp.runtime_platform())
    producer=git('rev-parse','HEAD')
    directory=root/'output/cloud-run/metric-cache'/Path(notebook).stem
    directory.mkdir(parents=True,exist_ok=True)
    original=sp.lgbm_fit_eval
    def add(h,v):
        a=np.asarray(v);h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode())
        h.update(json.dumps(a.tolist()).encode() if a.dtype.hasobject else memoryview(np.ascontiguousarray(a)).cast('B'))
    @functools.wraps(original)
    def fit(S,feats,params,model_seeds=(42,)):
        h=hashlib.sha256(json.dumps(dict(identity=identity,feats=list(feats),params=params,model_seeds=list(model_seeds)),sort_keys=True).encode())
        for part in ['train','val','test']:
            frame=S['X_'+part][feats];add(h,frame.to_numpy());add(h,frame.index.to_numpy());add(h,S['y_'+part])
        key=h.hexdigest();target=directory/(key+'.json')
        if target.exists():
            saved=json.loads(target.read_text())
            assert saved['key']==key and saved['identity']==identity,'Fit checkpoint fingerprint mismatch'
            assert sha(json.dumps(saved['result'],sort_keys=True).encode())==saved['result_sha256'],'Fit checkpoint contents changed'
            assert git('cat-file','-t',saved['execution_commit'])=='commit','Fit producer commit unavailable'
            print(f'CACHE HIT {Path(notebook).stem} {key[:12]} ({len(feats)} features)',flush=True)
            return saved['result']
        result=original(S,feats,params,model_seeds=model_seeds)
        saved=dict(key=key,identity=identity,execution_commit=producer,result=result,result_sha256=sha(json.dumps(result,sort_keys=True).encode()))
        tmp=target.with_suffix('.tmp')
        with tmp.open('w') as f:json.dump(saved,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,target)
        print(f'CACHE SAVE {Path(notebook).stem} {key[:12]} ({len(feats)} features)',flush=True)
        return result
    sp.lgbm_fit_eval=fit
