# ChatGPT cloud reproduction — 2026-10-08

Status: integrated execution baseline; analyses pending. Historical JSON results and publication assets are retained until each stage replaces them and must not be described as this run's results.

Branch: `chatgpt/cloud-reproduction-2026-10-08`; target: Paven's `main` (`5b2bfc3`). Integrated PR #11 through `790f80d` (includes #10 and notebook 24), plus PR #8 `cfe3627`. README/tracker conflicts preserve the newer entries and adopt #8's historical x86 entity row. No AGENTS.md exists in the repository or workspace ancestors; CLAUDE.md instructions were read.

Inventory/order: 00–20, 22, 24, 23, 99. Notebook 21 was renamed to 99 by #10. Notebook 25 is absent from current main and all integrated revisions; no result for 25 can be reproduced or cited. 24 generates its own controls assets. 23 then regenerates the remaining publication assets; 99 runs last.

Environment: Debian Linux, kernel 6.18.44, AMD EPYC 9V74, x86_64, 9 visible CPUs / 8-core cgroup quota, 8 GiB cgroup memory, Python 3.12.14. Exact package versions, backends and thread settings are in platform.json and pip-freeze.txt. About 29 GiB disk was free before setup. All 37 LFS inputs were checked against their committed SHA-256 and size; none is a pointer. LaTeX is available.

This is a fresh numerical platform, distinct from the historical Intel Xeon / Python 3.11.15 run and Apple M5 / Python 3.13.7 revisions. Package pins are unchanged. Models retain 4 threads, LightGBM deterministic col-wise histograms, predefined splits, selection rules, search spaces and seeds. BLAS threads are fixed at 1; OMP and NumExpr at 4; PYTHONHASHSEED=42.

Necessary repairs before execution:
- Every saved result records the full numerical platform and full execution SHA; historical records only identified some platforms.
- 99 verifies platforms across both original and revision results rather than inferring them from matching metrics. Existing acceptance checks are retained.
- Blend sweep and operating-point threshold serialization retain full precision; displayed publication values are rounded by generators. Model selection and metrics are unchanged by this serialization repair.
- Search CSVs were removed from the committed baseline, forcing fresh 74-configuration XGBoost, 48-configuration LightGBM, and 24-configuration Random Forest searches. Their historical versions remain in Git history.
- Historical notebook outputs were cleared. The runner saves notebook checkpoints after each cell, text logs, and stage commits, and stops on failed assertions. Predictions remain in the prescribed gitignored output directory.

Final validations, result comparisons (particularly Sections 5.2 and 5.5), generated asset inventory and outstanding author actions will be recorded after execution. No manuscript or reviewer text is committed.

Execution transport repair: native Jupyter TCP and IPC socket creation both fail with Operation not permitted before notebook code runs. The runner therefore uses a fresh Python process and IPython shell per notebook, capturing stream and inline rich outputs; cell code is unmodified. Failed transport attempts are retained in the local run manifest. No numerical result was produced by those attempts.

Publication transport: local Git has no authenticated HTTPS push credentials. The connected GitHub API has write access to scottonchain/layerzero_xgboost, and read access to Paven’s fork. Baseline and each stage are published through Git data APIs, fetched back, and become the clean HEAD before subsequent execution. This preserves reachable execution provenance. The PR will target Paven from Scott’s branch.
