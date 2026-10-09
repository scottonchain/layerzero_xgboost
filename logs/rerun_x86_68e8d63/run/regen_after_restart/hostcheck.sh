#!/bin/bash
# Host check after the container restart: 03/04/06 must give the same results and byte-identical predictions
# as before the restart, and one Random Forest configuration must reproduce its pre-restart validation scores.
cd "$(git rev-parse --show-toplevel)"; export PATH=/home/user/paven86/venv311/bin:$PATH
H=logs/rerun_x86_68e8d63/run/regen_after_restart; ok=1
for nb in 03_xgboost_sybil 04_lightgbm_sybil 06_cross_ensemble_sybil; do
  t0=$(date +%s)
  jupyter nbconvert --to notebook --execute $nb.ipynb --output-dir $H --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $H/$nb.log 2>&1 || ok=0
  echo "$nb rc=$? seconds=$(( $(date +%s) - t0 ))" >> $H/hostcheck.txt
  cp results/$nb.json $H/$nb.json; git checkout -- results/$nb.json
done
python - >> $H/hostcheck.txt <<'PY'
import json
for nb in ['03_xgboost_sybil', '04_lightgbm_sybil', '06_cross_ensemble_sybil']:
    a = json.load(open(f'logs/rerun_x86_68e8d63/run/regen/{nb}.json')); b = json.load(open(f'logs/rerun_x86_68e8d63/run/regen_after_restart/{nb}.json'))
    a.pop('code_commit'); b.pop('code_commit'); print(nb, 'results identical to pre-restart regen:', a == b)
PY
sha256sum -c logs/rerun_x86_68e8d63/run/regen/predictions.sha256 >> $H/hostcheck.txt 2>&1 || ok=0
python $H/rf_check.py >> $H/hostcheck.txt 2>&1 || ok=0
echo "HOSTCHECK_DONE ok=$ok" >> $H/hostcheck.txt
