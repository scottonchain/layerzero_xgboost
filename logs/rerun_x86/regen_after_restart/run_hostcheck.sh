#!/bin/bash
# Host check after the container restart: 03/04/06 regenerated and compared to the committed run's
# predictions; 13's Random Forest refitted and compared to the 2.10 GHz host's predictions.
set -u
SP=/tmp/claude-0/-home-user-layerzero-xgboost/60465b48-3e92-51f6-b127-5b66dd881cdc/scratchpad
cd /home/user/lz_rerun; L=logs/rerun_x86/regen_after_restart; mkdir -p $L
test -z "$(git status --porcelain --untracked-files=no)" || { echo "DIRTY"; exit 1; }
python3 $SP/hostcheck/rf_check.py > $L/rf_check.txt 2>&1; echo "rf_check rc=$?" >> $L/rf_check.txt
for nb in 03_xgboost_sybil 04_lightgbm_sybil 06_cross_ensemble_sybil; do
  jupyter nbconvert --to notebook --execute $nb.ipynb --output-dir $L --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $L/$nb.log 2>&1 || { echo "$nb failed"; exit 1; }
  cp results/$nb.json $L/$nb.json; git checkout -- results/$nb.json
done
REGEN_LOG=$L python3 $SP/hostcheck/compare_regen_any.py > $L/compare_regen.txt 2>&1; echo "compare rc=$?" >> $L/compare_regen.txt
test -z "$(git status --porcelain --untracked-files=no)" && echo "tree clean" || echo "TREE DIRTY"
echo HOSTCHECK_DONE
