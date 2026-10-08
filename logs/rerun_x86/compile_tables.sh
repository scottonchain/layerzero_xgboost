#!/bin/bash
# Compile every LaTeX table fragment in tables/ inside the manuscript's document class (elsarticle,
# with the manuscript's table packages), one document per table with -halt-on-error, then one
# combined document. Reports errors, undefined control sequences and overfull boxes per table.
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=logs/rerun_x86/latex; rm -rf $OUT; mkdir -p $OUT
PRE='\documentclass[preprint,12pt]{elsarticle}
\usepackage{amssymb,amsmath,booktabs,multirow,makecell,array,adjustbox,xcolor,graphicx}
\begin{document}'
status=0
for f in tables/*.tex; do
  b=$(basename $f .tex)
  printf '%s\n\\begin{table}[p]\\centering\\caption{%s}\n\\begin{adjustbox}{max width=\\textwidth}\n\\input{%s}\n\\end{adjustbox}\n\\end{table}\n\\end{document}\n' \
    "$PRE" "${b//_/\\_}" "$(pwd)/$f" > $OUT/$b.tex
  (cd $OUT && pdflatex -interaction=nonstopmode -halt-on-error $b.tex > /dev/null 2>&1); rc=$?
  err=$(grep -c '^!' $OUT/$b.log); undef=$(grep -c 'Undefined control sequence' $OUT/$b.log)
  over=$(grep -c 'Overfull' $OUT/$b.log); pages=$(grep -o 'Output written on [^ ]* ([0-9]* page' $OUT/$b.log | grep -o '[0-9]* page')
  printf '%-40s rc=%s errors=%s undefined=%s overfull=%s %s\n' "$b" "$rc" "$err" "$undef" "$over" "$pages"
  [ $rc -eq 0 ] && [ $err -eq 0 ] || { status=1; grep -A3 '^!' $OUT/$b.log | head -12; }
done
{ echo "$PRE"; for f in tables/*.tex; do b=$(basename $f .tex); printf '\\begin{table}[p]\\centering\\caption{%s}\n\\begin{adjustbox}{max width=\\textwidth}\n\\input{%s}\n\\end{adjustbox}\n\\end{table}\n\\clearpage\n' "${b//_/\\_}" "$(pwd)/$f"; done; echo '\end{document}'; } > $OUT/all_tables.tex
(cd $OUT && pdflatex -interaction=nonstopmode -halt-on-error all_tables.tex > /dev/null 2>&1); rc=$?
printf 'all_tables: rc=%s errors=%s %s\n' $rc "$(grep -c '^!' $OUT/all_tables.log)" "$(grep -o 'Output written on [^(]*([0-9]* pages' $OUT/all_tables.log)"
[ $rc -eq 0 ] || status=1
find $OUT -type f ! -name '*.pdf' ! -name '*.log' ! -name '*.tex' -delete
exit $status
