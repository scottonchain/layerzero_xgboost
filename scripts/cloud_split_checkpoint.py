"""Atomic, fingerprinted full-precision model-comparison split checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


class SplitCheckpoint:
    def __init__(self, notebook, parameters, models):
        import sybil_pipeline as sp
        self.root = Path.cwd()
        self.directory = self.root / 'output/cloud-run/metric-cache' / Path(notebook).stem
        self.directory.mkdir(parents=True, exist_ok=True)
        nb = json.loads((self.root / notebook).read_text())
        cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.identity = dict(schema=1, notebook=notebook, parameters=parameters, models=models,
            code_cells=self.digest(json.dumps(cells).encode()),
            pipeline=self.digest((self.root / 'sybil_pipeline.py').read_bytes()),
            requirements=self.digest((self.root / 'requirements.txt').read_bytes()),
            checkpoint_code=self.digest(Path(__file__).read_bytes()),
            data_tree=self.git('rev-parse', 'HEAD:data'), platform=sp.runtime_platform())
        self.producer = self.git('rev-parse', 'HEAD')

    @staticmethod
    def digest(value):
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()

    def key(self, S, features, seed, categories):
        import numpy as np
        h = hashlib.sha256(json.dumps(dict(identity=self.identity, features=list(features), seed=seed), sort_keys=True).encode())
        def add(value):
            a = np.asarray(value)
            h.update(str(a.dtype).encode()); h.update(json.dumps(a.shape).encode())
            h.update(json.dumps(a.tolist()).encode() if a.dtype.hasobject else memoryview(np.ascontiguousarray(a)).cast('B'))
        for part in ['train', 'val', 'test']:
            frame = S['X_' + part][features]
            add(frame.to_numpy()); add(frame.index.to_numpy()); add(S['y_' + part])
        add(categories)
        return h.hexdigest()

    def load(self, S, features, seed, categories):
        key = self.key(S, features, seed, categories)
        target = self.directory / (key + '.json')
        if not target.exists():
            return None
        saved = json.loads(target.read_text())
        assert saved['key'] == key and saved['identity'] == self.identity, 'Split fingerprint mismatch'
        assert saved['result_sha256'] == self.digest(json.dumps(saved['result'], sort_keys=True).encode()), 'Split contents changed'
        assert self.git('cat-file', '-t', saved['execution_commit']) == 'commit', 'Producer unavailable'
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', saved['execution_commit'], 'HEAD'], capture_output=True).returncode == 0, 'Producer outside current history'
        return saved['result']

    def save(self, S, features, seed, categories, result):
        key = self.key(S, features, seed, categories)
        target = self.directory / (key + '.json')
        saved = dict(key=key, identity=self.identity, execution_commit=self.producer,
            result=result, result_sha256=self.digest(json.dumps(result, sort_keys=True).encode()))
        temporary = target.with_suffix('.tmp')
        with temporary.open('w') as f:
            json.dump(saved, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, target)
