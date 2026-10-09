#!/bin/bash
# Corrected publication rerun: 00-25 (21 does not exist) and 99, in numerical dependency order, on the branch
# fix/root-tree-metrics-2026-10-09. Fresh caches: results/02_search_*.csv and results/13_search_random_forest.csv
# are removed first (they are kept at b7a9a3f), and the stale output/ files are moved away so that every prediction
# file is regenerated before it is read. Each notebook starts from a clean tree (sybil_pipeline.CODE_COMMIT must not
# end in -dirty) and its outputs are committed before the next one starts. Logs stay untracked in run/ until the end.
# Stops after 01 if the Gini keep-or-drop decision differs from the declared feature set (remove).
set -u
cd "$(git rev-parse --show-toplevel)"
export PATH=/home/user/paven86/venv311/bin:$PATH
L=logs/rerun_fix_20261009; R=$L/run; TL=$R/timeline.tsv; mkdir -p $R/inputs
exec 9>$R/.lock; flock -n 9 || { echo "another runner holds $R/.lock"; exit 1; }
BRANCH=fix/root-tree-metrics-2026-10-09
clean() { test -z "$(git status --porcelain --untracked-files=no)"; }
die() { echo -e "$(date -u +%FT%TZ)\tFAILED\t$1" >> $TL; echo "FAILED $1"; exit 1; }
push() { for i in 1 2 3 4; do git push -q origin HEAD:refs/heads/$BRANCH 2>>$R/push.log && return 0; sleep $((2**i)); done; echo "push failed after $1" >> $R/push.log; }
inputs() { { git rev-parse HEAD; sha256sum results/* output/* tables/* figures/* 2>/dev/null; } > $R/inputs/$1.sha256; }
nbrun() {
  local nb=$1; local t0=$(date +%s) head=$(git rev-parse --short HEAD)
  clean || die "$nb: tree not clean before start"
  inputs $nb
  echo -e "$(date -u +%FT%TZ)\tSTART\t$nb\t$head" >> $TL
  jupyter nbconvert --to notebook --execute $nb.ipynb --inplace \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $R/$nb.log 2>&1
  local rc=$?
  echo -e "$(date -u +%FT%TZ)\tEND\t$nb\t$head\trc=$rc\tseconds=$(( $(date +%s) - t0 ))" >> $TL
  return $rc
}
FROM=${1:-00}
if [ "$FROM" = 00 ]; then
  clean || die "tree not clean at start"
  if [ -d output ]; then mv output /home/user/fix_diag/output_before_corrected_run_$(date -u +%H%M%S); fi
  mkdir -p output
  git rm -q --ignore-unmatch results/02_search_xgboost.csv results/02_search_lightgbm.csv results/13_search_random_forest.csv
  git commit -q -F - <<'MSG' || die "could not commit the cache removal"
Start fresh search caches for the corrected publication run

results/02_search_xgboost.csv, results/02_search_lightgbm.csv and
results/13_search_random_forest.csv were scored on features that gave root
wallets zero tree features. They stay available at b7a9a3f; the corrected run
searches from scratch.

Co-Authored-By: Claude <noreply@anthropic.com>
MSG
  push cache-removal
fi
ALL=$(ls [0-9][0-9]_*.ipynb | sed 's/.ipynb$//' | grep -v '^99_' | awk -v f=$FROM -F_ '$1>=f && $1<=25')
echo -e "$(date -u +%FT%TZ)\tPLAN\t$(echo $ALL | tr '\n' ' ') 99_tie_out_revision" >> $TL
for n in $(seq 0 25); do nn=$(printf '%02d' $n); ls ${nn}_*.ipynb >/dev/null 2>&1 || echo -e "$(date -u +%FT%TZ)\tABSENT\t$nn" >> $TL; done
for nb in $ALL 99_tie_out_revision; do
  nbrun $nb || die "$nb rc!=0 (see $R/$nb.log)"
  head=$(git rev-parse --short HEAD)
  git add -A results figures tables $nb.ipynb
  printf 'Corrected run: %s at %s\n\nExecuted from a clean tree at %s with the root-tree feature fix (663e026).\nLog: logs/rerun_fix_20261009/run/%s.log (untracked until the end of the run)\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n' \
    "$nb" "$head" "$head" "$nb" | git commit -q -F - || die "$nb: commit failed"
  push $nb
  if [ "$nb" = 01_ablation_gini ]; then
    python -c "import json,sys; d=json.load(open('results/01_ablation_gini.json')); print(d['decision'], d['splits_lowered'])" > $R/01_decision.txt
    grep -q '^remove ' $R/01_decision.txt || { echo -e "$(date -u +%FT%TZ)\tSTOP\t01 decision changed: $(cat $R/01_decision.txt); reconcile the declared feature set before continuing" >> $TL; echo "STOP: 01 decision changed"; exit 3; }
  fi
done
echo -e "$(date -u +%FT%TZ)\tDONE\tall" >> $TL
