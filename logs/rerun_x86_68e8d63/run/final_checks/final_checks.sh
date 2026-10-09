#!/bin/bash
# After the run: (1) 03 again, its results must equal the first regeneration (numeric host check deferred at restart 3);
# (2) 10_tie_out, the tie-out of 00-09, executed into this folder; any result it writes is restored afterwards.
cd "$(git rev-parse --show-toplevel)"; export PATH=/home/user/paven86/venv311/bin:$PATH; H=logs/rerun_x86_68e8d63/run/final_checks
for nb in 03_xgboost_sybil 10_tie_out; do
  t0=$(date +%s)
  jupyter nbconvert --to notebook --execute $nb.ipynb --output-dir $H --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $H/$nb.log 2>&1
  echo "$nb rc=$? seconds=$(( $(date +%s) - t0 )) changed: $(git status --porcelain --untracked-files=no | tr '\n' ' ')" >> $H/final_checks.txt
  for f in $(git status --porcelain --untracked-files=no | awk '{print $2}'); do cp $f $H/; git checkout -- $f; done
done
python - >> $H/final_checks.txt <<'PY'
import json
a = json.load(open('logs/rerun_x86_68e8d63/run/regen/03_xgboost_sybil.json')); b = json.load(open('logs/rerun_x86_68e8d63/run/final_checks/03_xgboost_sybil.json'))
a.pop('code_commit'); b.pop('code_commit'); print('03 results identical to the first regeneration:', a == b)
PY
sha256sum -c logs/rerun_x86_68e8d63/run/regen/predictions.sha256 >> $H/final_checks.txt 2>&1
echo FINAL_DONE >> $H/final_checks.txt
