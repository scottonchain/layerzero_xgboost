Completes the revised LayerZero pipeline with freshly executed analyses and publication assets from one documented ChatGPT cloud numerical platform. Integrates PRs #8, #10, #11 and #12 without adopting another platform’s computed outputs. Scientific assumptions, features, splits, search spaces and validation-only selection rules are preserved.

Branch: `scottonchain:chatgpt/cloud-reproduction-2026-10-08`
Results and validation commit: `a029120fed8418341d6d1e2413ceae4ef95ab317`
Initial scientific execution baseline: `a2f2dd79ecdc833da38de7c2ff30fc88b3a8c40d`
Final notebook99 execution commit: `67a14e035f72876d8c40bfff1432faaa3ff9683f`

[Completed reproduction report](https://github.com/scottonchain/layerzero_xgboost/blob/a029120fed8418341d6d1e2413ceae4ef95ab317/docs/cloud-reproduction/REPORT.md) includes every full execution/artifact SHA, input/prediction/publication manifests, full-precision historical differences and recovery history.

## Actual platform

Ubuntu24.04.3LTS; Linux6.18.44; **AMD EPYC9V74 80-Core Processor**, x86_64, family25/model17/stepping1. Nine visible CPUs, an eight-core cgroup quota, 8GiB memory limit; **four model threads**. Python3.12.14, XGBoost3.2.0, LightGBM4.6.0, scikit-learn1.8.0, NumPy2.4.6. All113 established package versions and numerical backends were preserved. OMP4/OpenBLAS1/MKL1/NumExpr4, PYTHONHASHSEED42, LightGBM deterministic=True/force_col_wise=True. Fresh isolated Python/IPython process per notebook with persistent stream/rich-output logs and checkpoints.

This is the available AMD cloud platform; it is not described as Scott’s original Xeon. All37 LFS inputs contain actual data and pass size/SHA-256 checks. The newer x86 PR8 checkpoint `cf47a0f` was inspected read-only: scientific code13–16,18–20,22 is identical; current PR12 publication refinements and cloud repairs are retained. The other reproduction branch was never changed.

## Complete execution inventory

All26 required notebooks completed. Notebook21 was renamed to99 by PR10;24 and25 were discovered and executed. Analysis order is00–20,22; then refreshed10/12,23,24,25,99 last. Recovery reruns and interrupted attempts are separately retained.

<details>
<summary>Executed notebooks and latest runtimes</summary>

| Notebook | Latest runtime (s) |
|---|---:|
| 00_data_pipeline.ipynb | 62.4 |
| 01_ablation_gini.ipynb | 652.2 |
| 02_hyperparameter_search.ipynb | 4068.0 |
| 03_xgboost_sybil.ipynb | 235.9 |
| 04_lightgbm_sybil.ipynb | 171.4 |
| 05_logistic_regression_sybil.ipynb | 54.2 |
| 06_cross_ensemble_sybil.ipynb | 5.4 |
| 07_sensitivity_label_vintage.ipynb | 1078.2 |
| 08_ablation_families.ipynb | 4556.8 |
| 09_shap_importance.ipynb | 2522.5 |
| 10_tie_out.ipynb | 4.6 |
| 11_split_comparison.ipynb | 432.4 |
| 12_figures.ipynb | 202.8 |
| 13_random_forest_sybil.ipynb | 7841.3 |
| 14_entity_level_recall.ipynb | 34.8 |
| 15_split_tree_report.ipynb | 934.4 |
| 16_label_robustness_initial_list.ipynb | 1392.3 |
| 17_models_10_splits.ipynb | 4497.5 |
| 18_mimicry_stress.ipynb | 317.4 |
| 19_ablation_by_category.ipynb | 2400.0 |
| 20_temporal_holdout.ipynb | 651.8 |
| 22_report_dependence.ipynb | 3705.0 |
| 23_revision_figures.ipynb | 6.3 |
| 24_mimicry_controls.ipynb | 934.8 |
| 25_paper_tables.ipynb | 33.0 |
| 99_tie_out_revision.ipynb | 4.2 |

</details>

Notebook02’s4068.0s is the resumed portion, excluding25 verified cached XGBoost fits. All74 XGBoost and48 LightGBM validation configurations were recomputed for this cloud reproduction. Notebook13’s7841.3s resumed portion excludes ten verified cached Random Forest search fits; all24 predefined configurations completed and original per-fit times remain in its CSV.

## Validation and publication outputs

- Final99: **85/85 checks passed**, `multiple_platforms_cited=false`. Original10 passed.
- **508 independent mandatory consistency checks passed**: complete/error-free notebook execution, reachable producer commits, unchanged scientific dependencies, one actual platform, search winners/thresholds/blend weights selected on validation, feature counts/ablation summaries and publication-source chronology.
- Notebook14 consumes matching regenerated03/04/06 predictions. All10 saved validation/test prediction files have unique/disjoint addresses and independently reproduced metrics and thresholds. Large predictions remain in prescribed ignored `output/` storage; their hashes/producers are committed.
- Notebook22 completed all15 paired comparisons/30 training arms with partition/report/tree assertions intact. Actual unique training sizes266590–266592 and Sybil counts14611–14613 satisfy the predefined1% matching tolerance; they are not claimed to be exactly equal. Observed unique/class counts and exact post-upsampling counts derived from unchanged make_S are in `paired-training-counts.json`.
- **41 current publication outputs refreshed** by12/23/24/25, including the original/revision/control figures,18 LaTeX table fragments and207 cited values with none missing. Sources/producers/hashes are in `publication-manifest.json`.
- All18 current tables compiled together with pdfTeX/TeXLive; the combined PDF, source and compiler logs are committed under `docs/cloud-reproduction/latex-proof/`. Generator23/24/25 compile checks also passed.

**Checks including optional diagnostics do not all pass.** Notebook15’s main union split passes size/class criteria but retains its large-component warning (45.36% of Sybils in one component, above the5% warning level). The stricter reporter-linked variant fails all three class-rate feasibility checks and correctly stops before training: one whole component contains10071 Sybils (55.30%); even the largest allowed partition could hold only about9761 under the joint bounds. No assertion or tolerance was weakened. Notebook22 is a separate paired design and does not establish reporter-level independence.

The independent strict full-precision Random Forest cross-stage diagnostic also records an AP difference of9.74147e-9 on split4. All ten thresholds/F1/AUROC agree exactly; publication rounding is unchanged. This is consistent with scikit-learn’s threaded probability-summation ordering affecting tied ranks. A bit-identical RF AP claim is not made.

## Numerical comparisons

Cloud minus committed main, Section5.2 mean F1: XGBoost+0.000276, LightGBM−0.000181, RandomForest−0.000363, Ensemble−0.000384; logistic regression unchanged. Full AP/precision/recall/AUROC comparisons and corrected tests are in the report.

Section5.5, E1 validation operating point: LightGBM weighted recall0.711148 (main0.718653), majority hit0.755044, residual allocation0.379626; ensemble weighted recall0.738788 (main0.718653), majority hit0.784797, residual allocation0.354931. The entity values match the historical PR8 x86 results, using this run’s own regenerated inputs.

Notebook22: LightGBM held/seen mean F1 **0.369207/0.598032**, paired difference **+0.228825**, corrected p1.85484e-5; paired AP difference+0.293917. XGBoost paired F1 difference+0.218234. The report compares these against PR10’s earlier ARM results and the latest PR8 x86 checkpoint; the central cloud values match the latter at displayed precision. Code/revision/platform effects are not attributed solely to hardware.

## Necessary repairs and recovery

Actual CPU/library/thread/full producer metadata, full-precision threshold/blend/generator serialization, complete-platform99 guards, cloud module paths and socket-free execution were repaired. Notebook17’s namespace PID could not be found in the host procfs view; its memory monitor now obtains the same process’s actual PID from /proc/self/stat. Atomic metric and whole-comparison-split caches validate code/input/parameter/feature/platform fingerprints and checksums; reuse/invalidation/corruption tests passed.

Five workspace losses and an exited notebook17 worker caused unrecoverable intermediate work; exact losses and extra runtimes are reported. Valid published stages and verified caches were reused; lost master/split/prediction prerequisites were regenerated on the same restored environment. The failed17 attempt’s ten unsaved full-precision comparison groups required one repeat; all current groups are now checkpointed. No duplicate worker was launched. Historical logs/status are retained and not presented as completed stages.

## Before submission

Paven must apply the refreshed assets/cited values to the private manuscript and reviewer response, confirm tentative feature-cost classes, preserve the entity/paired/label/temporal and infeasible-reporter caveats, and resolve the existing DATA.md licensing/citation/archival questions. Private manuscript tasks remain open. No merge was performed.
