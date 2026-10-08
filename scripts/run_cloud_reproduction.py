"""Sequential clean-tree notebook execution with durable stage checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='4', PYTHONHASHSEED='42', MPLBACKEND='Agg')
import nbformat

ORDER = list(range(21)) + [22, 24, 23, 99]
OUT = ROOT / 'output/cloud-run'
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / 'execution.json'

def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()

def run(number, state):
    path, = ROOT.glob(f'{number:02d}_*.ipynb')
    assert not git('status', '--porcelain'), 'Execution requires a clean tree'
    commit = git('rev-parse', 'HEAD')
    nb = nbformat.read(path, as_version=4)
    logpath = OUT / (path.stem + '.log')
    started = time.time()
    record = dict(notebook=path.name, execution_commit=commit, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), status='running')
    state['stages'].append(record)
    MANIFEST.write_text(json.dumps(state, indent=2) + '\n')
    with logpath.open('w', buffering=1) as log:
        def emit(text):
            log.write(text + '\n')
            print(text, flush=True)
        emit(f'START {path.name} at {commit}')
        def cell_done(cell, cell_index, **kwargs):
            emit(f'CELL {cell_index} completed')
            for out in cell.get('outputs', []):
                if out.output_type == 'stream': emit(out.text)
                elif out.output_type in ('execute_result', 'display_data') and 'text/plain' in out.data:
                    emit(out.data['text/plain'])
                elif out.output_type == 'error': emit('\n'.join(out.traceback))
            nbformat.write(nb, OUT / (path.stem + '.checkpoint.ipynb'))
        try:
            subprocess.run([sys.executable, __file__, '--execute-one', path.name],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            nb = nbformat.read(OUT / (path.stem + '.checkpoint.ipynb'), as_version=4)
            if number == 1:
                r = json.loads((ROOT / 'results/01_ablation_gini.json').read_text())
                assert r['decision'] == 'remove', 'Gini decision changed: align FEATS with the predefined decision before proceeding'
            if number == 2:
                r = json.loads((ROOT / 'results/02_hyperparameter_search.json').read_text())
                assert r['n_configs'] == {'xgboost': 74, 'lightgbm': 48}, r['n_configs']
        except Exception as exc:
            nbformat.write(nb, OUT / (path.stem + '.failed.ipynb'))
            record.update(status='failed', seconds=time.time()-started, error=str(exc))
            MANIFEST.write_text(json.dumps(state, indent=2)+'\n')
            emit(f'FAILED: {exc}')
            raise
        nbformat.write(nb, path)
        record.update(status='passed', seconds=time.time()-started,
                      completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        emit(f'PASS {path.name}: {record["seconds"]:.1f}s')
    dest = ROOT / 'docs/cloud-reproduction/logs'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(logpath, dest / logpath.name)
    (ROOT / 'docs/cloud-reproduction/execution.json').write_text(json.dumps(state, indent=2)+'\n')
    git('add', path.name, 'results', 'figures', 'tables', 'docs/cloud-reproduction')
    git('commit', '-m', f'Cloud reproduction: execute {path.stem}')
    record['artifact_commit'] = git('rev-parse', 'HEAD')
    MANIFEST.write_text(json.dumps(state, indent=2)+'\n')

def execute_one(filename):
    """Run unmodified notebook cells in a fresh process without a Jupyter socket."""
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    nb = nbformat.read(ROOT / filename, as_version=4)
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic('matplotlib', 'inline')
    count = 0
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code': continue
        count += 1
        started = time.time()
        print(f'CELL {index} started', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = []
        for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if value:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=value))
                print(value, flush=True)
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        error = result.error_before_exec or result.error_in_exec
        cell.metadata['cloud_execution'] = dict(seconds=time.time()-started, execution_method='IPython in isolated process')
        if error:
            cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__, evalue=str(error), traceback=traceback.format_exception(error)))
        nbformat.write(nb, OUT / (Path(filename).stem + '.checkpoint.ipynb'))
        print(f'CELL {index} completed in {time.time()-started:.1f}s', flush=True)
        if error: raise error

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--execute-one':
    execute_one(sys.argv[2])
elif __name__ == '__main__':
    state = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else dict(baseline_commit=git('rev-parse', 'HEAD'), stages=[])
    completed = {int(x['notebook'][:2]) for x in state['stages'] if x['status']=='passed'}
    for number in ([int(sys.argv[2])] if len(sys.argv) > 1 and sys.argv[1] == '--stage' else ORDER):
        if number not in completed: run(number, state)
    print('COMPLETE: all notebooks executed', flush=True)
