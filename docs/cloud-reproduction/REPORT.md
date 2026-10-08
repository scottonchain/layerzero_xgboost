# ChatGPT cloud reproduction — 2026-10-08

Status: integrated execution baseline; 00 and 01 passed; remaining analyses pending. Historical JSON results and publication assets are retained until each stage replaces them and must not be described as this run's results.

Branch: `chatgpt/cloud-reproduction-2026-10-08`; target: Paven's `main` (`5b2bfc3`). Integrated PR #11 through `790f80d` (includes #10 and notebook 24), plus PR #8 `cfe3627`. README/tracker conflicts preserve the newer entries and adopt #8's historical x86 entity row. No AGENTS.md exists in the repository or workspace ancestors; CLAUDE.md instructions were read.

Inventory/order: 00–20, 22, 23, 24, 25, 99. Notebook 21 was renamed to 99 by #10. PR #12 (`2c2c05a`) supplies notebook 25 and generator refinements; it was found in the broader PR search after the initial main/#11 inventory. Its code and documentation are incorporated while its computed results and assets are excluded. All three publication generators run after analyses, and 99 runs last.

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

Isolated process initialization also requires adding the repository root to sys.path; this replaces the working-directory import behavior of a notebook kernel. The terminal IPython shell supplies inline Matplotlib display hooks without a GUI or socket.

Completed: 00 (61.5 seconds), 01 (652.2 seconds). The predefined Gini rule again removes the feature: 3/10 splits lowered validation F1 when removed (8 required). The first scientific execution baseline is `a2f2dd79ecdc833da38de7c2ff30fc88b3a8c40d`. Integrating #12 does not change completed notebook code, sybil_pipeline.py, data or requirements, so completed work is retained.

24 now serializes DataFrame values at full Python float precision with missing cells as null, replacing pandas to_json’s default ten-decimal truncation. This affects saved precision, not fits or selection. The runner streams fit logs while capturing notebook outputs.

The separately requested original-hardware rerun is not launched: the available host is AMD EPYC 9V74/Python 3.12.14, unlike Intel Xeon @ 2.80 GHz/Python 3.11.15. No tool exposes the original machine, whose committed provenance lacks a full CPU model. This run makes no exact-hardware claim and does not modify the other job.

## Interruption checkpoint

The execution was interrupted during notebook 02 after 25 initial XGBoost grid configurations completed. `results/02_search_xgboost.csv` and the live log preserve these validation-only fits; no LightGBM grid fits or downstream analyses have completed. Notebooks 00 and 01 remain complete. The committed legacy 02+ JSON results and publication assets have not been replaced and are historical, not cloud results. No completed PR, tie-out outcome, Section 5.5 rerun or paired 22 comparison is claimed. Resume notebook 02 on the same documented platform with the isolated runner; it skips cached configurations and retains the fixed search spaces and validation-only rule.

## Resume verification

The cloud reproduction was resumed on the established AMD EPYC 9V74 platform. The original Xeon requirement was superseded by the user. No active worker was present. All 37 LFS input hashes, numerical package versions, thread settings, prerequisite provenance, and all 25 cached XGBoost configurations and their logged metrics matched the published checkpoint. No checkpoints were lost. See resume-verification.json.

## Workspace replacement and checkpoint repair

Notebooks00–07 completed and were published. The execution server disconnected during08, then the entire original scratch directory was replaced. Ignored master/split/prediction files and notebook08 in-memory fit metrics are lost. At least12 fits (split42) completed; additional completed-fit count and exact interruption runtime are unrecoverable. No scientific assertion failed. The replacement CPU, OS, kernel, Python, CPU quota, memory limit, all113 packages and all37 actual LFS input hashes match the established numerical platform. Completed scientific results pass provenance. The original25 cached XGBoost fits and the completed74/48 search grids remain published and valid. Only lost intermediate outputs (00 and03/04/05/06) will be regenerated;01/02/07 remain valid. Atomic code/input/platform-keyed checkpoints now preserve completed metric-only fits without changing their function or arguments. Isolated fixture tests passed cache reuse, changed-input invalidation and corruption rejection. The per-stage completion message is clarified. Unpublished scratch report-preparation scripts also require recovery.
