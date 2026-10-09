"""Build the original (b7a9a3f) and corrected master feature tables from identical inputs and cache both.

The original module is the verbatim b7a9a3f sybil_pipeline.py (sybil_pipeline_b7a9a3f.py, same directory).
Each cached table carries a JSON sidecar with the SHA-256 of the code and of the inputs it was built from.
Usage: python diagnostics/root_tree_fix/build_tables.py   (from the repository root)
"""
import hashlib, importlib.util, json, os, platform, subprocess, sys, time
import pandas as pd
sys.path.insert(0, os.getcwd())
import sybil_pipeline as sp_new

D = 'diagnostics/root_tree_fix'
C = f'{D}/cache'
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open('/home/user/fix_diag/progress.log', 'a') as f: f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {msg}\n")

spec = importlib.util.spec_from_file_location('sybil_pipeline_b7a9a3f', f'{D}/sybil_pipeline_b7a9a3f.py')
sp_old = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp_old)

# input hashes: the Git LFS pointer files (oid = SHA-256 of each data file) plus the other data files
inputs = {}
for root in ('data', 'review_support'):
    for dp, _, fs in os.walk(root):
        for fn in sorted(fs):
            p = os.path.join(dp, fn)
            inputs[p] = sha(p)
inputs_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
env = dict(python=platform.python_version(), pandas=pd.__version__, machine=platform.machine(),
           cpu=[l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0])
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
for tag, mod, codefile in (('old', sp_old, f'{D}/sybil_pipeline_b7a9a3f.py'), ('new', sp_new, 'sybil_pipeline.py')):
    parq, meta = f'{C}/master_{tag}.parquet', f'{C}/master_{tag}.json'
    want = dict(code_sha256=sha(codefile), inputs_sha256=inputs_hash, env=env)
    if os.path.exists(parq) and os.path.exists(meta):
        have = json.load(open(meta))
        if all(have.get(k) == v for k, v in want.items()):
            log(f'[{tag}] cache hit {parq}'); continue
    t0 = time.time(); log(f'[{tag}] building master table (code {want["code_sha256"][:12]})')
    df, tfm, anchors = mod.build_master_df('./data', verbose=False)
    df.to_parquet(parq, index=False)
    json.dump(dict(**want, git_head=head, rows=len(df), cols=len(df.columns), seconds=round(time.time() - t0, 1),
                   n_anchors=len(anchors), n_edges=len(tfm)), open(meta, 'w'), indent=1)
    log(f'[{tag}] built {len(df):,} rows x {len(df.columns)} cols in {time.time() - t0:.0f}s')
json.dump(inputs, open(f'{D}/input_hashes.json', 'w'), indent=0)
log('tables ready')
