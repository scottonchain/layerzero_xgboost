#!/bin/bash
# 10_tie_out (00-09) needs every model's test predictions in output/; 05's were never regenerated in this run.
# Regenerate 05 (its result must equal the committed one), then run 10. Results written are copied here and restored.
cd "$(git rev-parse --show-toplevel)"; export PATH=/home/user/paven86/venv311/bin:$PATH; H=logs/rerun_x86_68e8d63/run/final_checks/tieout_10
for nb in 05_logistic_regression_sybil 10_tie_out; do
  test -z "$(git status --porcelain --untracked-files=no)" || { echo "tree not clean before $nb" >> $H/tieout_10.txt; exit 1; }
  t0=$(date +%s)
  jupyter nbconvert --to notebook --execute $nb.ipynb --output-dir $H --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $H/$nb.rerun.log 2>&1
  echo "$nb rc=$? seconds=$(( $(date +%s) - t0 )) at $(git rev-parse --short HEAD) changed: $(git status --porcelain --untracked-files=no | tr '\n' ' ')" >> $H/tieout_10.txt
  for f in $(git status --porcelain --untracked-files=no | awk '{print $2}'); do cp $f $H/; git checkout -- $f; done
done
python - >> $H/tieout_10.txt <<'PY'
import json, subprocess
new = json.load(open('logs/rerun_x86_68e8d63/run/final_checks/tieout_10/05_logistic_regression_sybil.json'))
old = json.loads(subprocess.check_output(['git', 'show', 'HEAD:results/05_logistic_regression_sybil.json']))
new.pop('code_commit'); old.pop('code_commit'); print('05 results identical to the committed 0d9eacc result:', new == old)
PY
echo TIEOUT_DONE >> $H/tieout_10.txt
