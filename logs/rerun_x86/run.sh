#!/bin/bash
# Single-platform rerun of 13-25 and 99 (x86_64), with 03/04/06 predictions regenerated first.
# Every notebook starts from a clean tree (sybil_pipeline.CODE_COMMIT must not end in -dirty);
# each notebook's outputs are committed before the next one starts, so each result records the
# commit it ran at. Code, data and requirements are identical across those commits.
# 03/04/06 are executed into logs/rerun_x86/regen/ and their committed results are restored
# afterwards: they only regenerate output/pred_*.parquet for 14, after compare_regen.py has
# checked that they reproduce the committed results.
set -u
cd "$(git rev-parse --show-toplevel)"
LOG=logs/rerun_x86
TL=$LOG/timeline.tsv
clean() { test -z "$(git status --porcelain --untracked-files=no)"; }
die() { echo -e "$(date -u +%FT%TZ)\tFAILED\t$1" >> $TL; echo "FAILED $1"; exit 1; }
nbrun() {  # notebook, extra nbconvert args...
  local nb=$1; shift; local t0=$(date +%s) head=$(git rev-parse --short HEAD)
  clean || die "$nb: tree not clean before start"
  echo -e "$(date -u +%FT%TZ)\tSTART\t$nb\t$head" >> $TL
  jupyter nbconvert --to notebook --execute $nb.ipynb "$@" \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $LOG/$nb.log 2>&1
  local rc=$?
  echo -e "$(date -u +%FT%TZ)\tEND\t$nb\t$head\trc=$rc\tseconds=$(( $(date +%s) - t0 ))" >> $TL
  return $rc
}
stage=${1:-all}
if [ "$stage" = all ] || [ "$stage" = regen ]; then
  mkdir -p $LOG/regen
  for nb in 03_xgboost_sybil 04_lightgbm_sybil 06_cross_ensemble_sybil; do
    nbrun $nb --output-dir $LOG/regen || die "$nb rc!=0"
    cp results/$nb.json $LOG/regen/$nb.json
    git status --porcelain --untracked-files=no > $LOG/regen/$nb.changed_files
    git checkout -- results/$nb.json && clean || die "$nb: could not restore the committed result"
  done
  python3 $LOG/compare_regen.py > $LOG/compare_regen.txt 2>&1 || die "03/04/06 do not reproduce the committed results (see compare_regen.txt)"
  echo -e "$(date -u +%FT%TZ)\tPASS\tcompare_regen" >> $TL
fi
if [ "$stage" = all ] || [ "$stage" = main ]; then
  FROM=${2:-13}   # resume point: first notebook number to run (completed notebooks are not rerun)
  for nb in $(ls [12][0-9]_*.ipynb | sed 's/.ipynb$//' | awk -F_ -v f=$FROM '$1>=f && $1>=13 && $1<=25') 99_tie_out_revision; do
    nbrun $nb --inplace || die "$nb rc!=0 (see $LOG/$nb.log)"
    head=$(git rev-parse --short HEAD)
    # Stage only what the notebook writes; logs stay untracked until the end of the run, because a
    # tracked log appended to before the next kernel starts would mark the tree dirty.
    git add -A results figures tables $nb.ipynb
    printf 'x86 rerun: %s at %s\n\nExecuted from a clean tree at %s on this platform (logs/rerun_x86/env.txt).\nLog: logs/rerun_x86/%s.log\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Cofo5QgMp6ieGLC9NEdHip\n' \
      "$nb" "$head" "$head" "$nb" | git commit -q -F - || die "$nb: commit failed"
  done
fi
echo -e "$(date -u +%FT%TZ)\tDONE\t$stage" >> $TL
