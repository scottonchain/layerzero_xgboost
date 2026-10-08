#!/bin/bash
# Rerun of 13-25 and 99 at 68e8d63 (PR #12) on x86_64 Linux, Intel Xeon @ 2.80GHz, 4 cores,
# Python 3.11.15 with the pinned packages (env_kernel.json, pip_freeze.txt).
# Stage regen: 03, 04, 06 are executed into run/regen/ to regenerate output/pred_*.parquet for 14;
#   compare_regen.py checks them against the committed results, then the committed JSONs are restored.
# Stage main: every existing notebook numbered 13-25, then 99, in numerical order. Each starts from a
#   clean tree (sybil_pipeline.CODE_COMMIT must not end in -dirty) and its outputs are committed before
#   the next one starts, so each result records the commit it ran at. Code, data and requirements are
#   identical across those commits. Logs stay untracked (run/) until the end.
set -u
cd "$(git rev-parse --show-toplevel)"
export PATH=/home/user/paven86/venv311/bin:$PATH
L=logs/rerun_x86_68e8d63; R=$L/run; TL=$R/timeline.tsv; mkdir -p $R/regen $R/inputs
exec 9>$R/.lock; flock -n 9 || { echo "another runner holds $R/.lock"; exit 1; }
BRANCH=claude/original-x86-rerun-2026-10-08
clean() { test -z "$(git status --porcelain --untracked-files=no)"; }
die() { echo -e "$(date -u +%FT%TZ)\tFAILED\t$1" >> $TL; echo "FAILED $1"; exit 1; }
inputs() {  # sha256 of every result, prediction, table and figure present before the notebook starts
  { git rev-parse HEAD; sha256sum results/* output/* tables/* figures/* 2>/dev/null; } > $R/inputs/$1.sha256
}
push() { for i in 1 2 3 4; do git push -q origin HEAD:refs/heads/$BRANCH 2>>$R/push.log && return 0; sleep $((2**i)); done; echo "push failed after $1" >> $R/push.log; }
nbrun() {  # notebook, extra nbconvert args...
  local nb=$1; shift; local t0=$(date +%s) head=$(git rev-parse --short HEAD)
  clean || die "$nb: tree not clean before start"
  inputs $nb
  echo -e "$(date -u +%FT%TZ)\tSTART\t$nb\t$head" >> $TL
  jupyter nbconvert --to notebook --execute $nb.ipynb "$@" \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $R/$nb.log 2>&1
  local rc=$?
  echo -e "$(date -u +%FT%TZ)\tEND\t$nb\t$head\trc=$rc\tseconds=$(( $(date +%s) - t0 ))" >> $TL
  return $rc
}
stage=${1:-all}
if [ "$stage" = all ] || [ "$stage" = regen ]; then
  for nb in 03_xgboost_sybil 04_lightgbm_sybil 06_cross_ensemble_sybil; do
    nbrun $nb --output-dir $R/regen || die "$nb rc!=0 (see $R/$nb.log)"
    cp results/$nb.json $R/regen/$nb.json
    git status --porcelain --untracked-files=no > $R/regen/$nb.changed_files
    git checkout -- results/$nb.json && clean || die "$nb: could not restore the committed result"
  done
  sha256sum output/pred_0[346]_*.parquet > $R/regen/predictions.sha256
  python $L/compare_regen.py > $R/regen/compare_regen.txt 2>&1 || die "03/04/06 do not reproduce the committed results (see $R/regen/compare_regen.txt)"
  echo -e "$(date -u +%FT%TZ)\tPASS\tcompare_regen" >> $TL
fi
if [ "$stage" = all ] || [ "$stage" = main ]; then
  FROM=${2:-13}   # resume point: first notebook number to run (completed notebooks are not rerun)
  for n in $(seq 13 25); do ls ${n}_*.ipynb >/dev/null 2>&1 || echo -e "$(date -u +%FT%TZ)\tABSENT\t$n" >> $TL; done
  for nb in $(ls [12][0-9]_*.ipynb | sed 's/.ipynb$//' | awk -F_ -v f=$FROM '$1>=f && $1>=13 && $1<=25') 99_tie_out_revision; do
    nbrun $nb --inplace || die "$nb rc!=0 (see $R/$nb.log)"
    head=$(git rev-parse --short HEAD)
    git add -A results figures tables $nb.ipynb
    printf 'x86 rerun at 68e8d63: %s at %s\n\nExecuted from a clean tree at %s on x86_64 Linux, Intel Xeon @ 2.80GHz, 4 cores,\nPython 3.11.15, pinned packages (logs/rerun_x86_68e8d63/env_kernel.json).\nLog: logs/rerun_x86_68e8d63/run/%s.log\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n' \
      "$nb" "$head" "$head" "$nb" | git commit -q -F - || die "$nb: commit failed"
    push $nb
  done
fi
echo -e "$(date -u +%FT%TZ)\tDONE\t$stage" >> $TL
