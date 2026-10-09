#!/bin/bash
# Remainder of the corrected publication run after 09_shap_importance, reordered (Scott, 2026-10-09) so that the
# manuscript-critical results (16, 19, then 15 and 22) arrive first. The experiments are unchanged: the order was
# checked against each notebook's code-level dependencies (sp.provenance, sp.load_results, sp.load_selected_params,
# reads from results/ and output/, assertions on prior notebooks) and against the DEPS mapping of 99.
#
# run.sh was stopped (SIGSTOP) while 09 was executing, so that it would not start 10; 09 itself was left to finish.
# This script waits for 09's nbconvert to exit, takes its exit status, ends the stopped run.sh, and commits and pushes
# 09 exactly as run.sh would have. It then runs the queue in run/queue.txt with the same steps as run.sh: clean
# tracked tree, input hashes, nbconvert --execute --inplace, commit, push, one notebook at a time. The queue is read
# at each notebook boundary only, so it can be changed between notebooks without touching a running one; a line HOLD
# stops the runner cleanly at that point. A push that fails four times stops the runner, so no notebook starts
# before the previous one is pushed.
set -u
cd "$(git rev-parse --show-toplevel)"
export PATH=/home/user/paven86/venv311/bin:$PATH
L=logs/rerun_fix_20261009; R=$L/run; TL=$R/timeline.tsv; Q=$R/queue.txt
BRANCH=fix/root-tree-metrics-2026-10-09
OLD_RUNNER=${OLD_RUNNER:-}; NB09_PID=${NB09_PID:-}; NB09_START=${NB09_START:-}; NB09_HEAD=${NB09_HEAD:-}
ts() { date -u +%FT%TZ; }
clean() { test -z "$(git status --porcelain --untracked-files=no)"; }
die() { echo -e "$(ts)\tFAILED\t$1" >> $TL; echo "FAILED $1"; exit 1; }
push() { for i in 1 2 3 4; do git push -q origin HEAD:refs/heads/$BRANCH 2>>$R/push.log && return 0; sleep $((2**i)); done
         die "push failed after $1"; }
commit_nb() {   # $1 notebook, $2 commit it was executed at
  git add -A results figures tables $1.ipynb
  printf 'Corrected run: %s at %s\n\nExecuted from a clean tree at %s with the root-tree feature fix (663e026).\nLog: logs/rerun_fix_20261009/run/%s.log (untracked until the end of the run)\n%s\nCo-Authored-By: Claude <noreply@anthropic.com>\n' \
    "$1" "$2" "$2" "$1" "${3:-}" | git commit -q -F - || die "$1: commit failed"
  push $1
}
inputs() { { git rev-parse HEAD; sha256sum results/* output/* tables/* figures/* 2>/dev/null; } > $R/inputs/$1.sha256; }
LOCKED=
lock() { [ -n "$LOCKED" ] && return; exec 9>$R/.lock; flock -w 600 9 || die "another runner holds $R/.lock"; LOCKED=1; }

# 1. Finish 09 (only when the stopped run.sh is still waiting on it).
if [ -n "$NB09_PID" ]; then
  while :; do
    st=$(awk '{sub(/^.*\) /, ""); print $1}' /proc/$NB09_PID/stat 2>/dev/null)
    [ "$st" = Z ] && break
    [ -z "$st" ] && die "09: nbconvert $NB09_PID vanished without an exit status"
    sleep 20
  done
  code=$(awk '{sub(/^.*\) /, ""); print $50}' /proc/$NB09_PID/stat)
  rc=$(( (code >> 8) & 255 )); sig=$(( code & 127 ))
  echo -e "$(ts)\tEND\t09_shap_importance\t$NB09_HEAD\trc=$rc\tsignal=$sig\tseconds=$(( $(date +%s) - NB09_START ))\t(exit status read from /proc; run.sh stopped)" >> $TL
  kill -9 $OLD_RUNNER 2>/dev/null; sleep 2
  echo -e "$(ts)\tNOTE\trun.sh ($OLD_RUNNER) ended at the 09/10 boundary; the remaining order is taken from run/queue.txt" >> $TL
  lock
  [ "$rc" = 0 ] && [ "$sig" = 0 ] || die "09_shap_importance rc=$rc signal=$sig (see $R/09_shap_importance.log)"
  python - "$NB09_HEAD" <<'EOF' || die "09: result or notebook check failed"
import json, sys
r = json.load(open('results/09_shap_importance.json'))
assert r['code_commit'] == sys.argv[1], (r['code_commit'], sys.argv[1])
nb = json.load(open('09_shap_importance.ipynb'))
errs = [o for c in nb['cells'] if c['cell_type'] == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error']
assert not errs, 'error output in the executed notebook'
EOF
  commit_nb 09_shap_importance "$NB09_HEAD" "
The notebook ran to completion under run.sh; run.sh was stopped at the 09/10 boundary
and this commit was made by run_reordered.sh with the same paths and message.
"
fi

# 2. The queue.
lock
echo -e "$(ts)\tPLAN\t$(grep -v '^#' $Q | tr '\n' ' ')" >> $TL
while :; do
  nb=$(grep -v -e '^#' -e '^[[:space:]]*$' $Q | head -1)
  [ -z "$nb" ] && break
  if [ "$nb" = HOLD ]; then echo -e "$(ts)\tHOLD\tremaining: $(grep -v -e '^#' -e '^[[:space:]]*$' $Q | tail -n +2 | tr '\n' ' ')" >> $TL; exit 4; fi
  [ -f $nb.ipynb ] || die "$nb.ipynb not found"
  head=$(git rev-parse --short HEAD); t0=$(date +%s)
  clean || die "$nb: tree not clean before start"
  inputs $nb
  echo -e "$(ts)\tSTART\t$nb\t$head" >> $TL
  jupyter nbconvert --to notebook --execute $nb.ipynb --inplace \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 > $R/$nb.log 2>&1
  rc=$?
  echo -e "$(ts)\tEND\t$nb\t$head\trc=$rc\tseconds=$(( $(date +%s) - t0 ))" >> $TL
  [ $rc = 0 ] || die "$nb rc!=0 (see $R/$nb.log)"
  commit_nb $nb $head
  awk -v nb=$nb 'done || $0 != nb {print; next} {done = 1}' $Q > $Q.tmp && mv $Q.tmp $Q
done
echo -e "$(ts)\tDONE\tall" >> $TL
