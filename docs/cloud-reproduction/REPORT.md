# Completed ChatGPT cloud reproduction

All 26 required notebooks completed; mandatory notebook 10/final 99 checks, independent consistency checks and publication LaTeX compilation passed. The optional reporter-linked split fails its existing feasibility checks and is stopped before training; this limitation is retained and explained below. This is one documented cloud numerical platform, distinct from the historical Xeon and Apple M5 runs.

Branch: `chatgpt/cloud-reproduction-2026-10-08`, published to Scott’s fork for a cross-fork PR to Paven main. First scientific execution baseline: `a2f2dd79ecdc833da38de7c2ff30fc88b3a8c40d`. Latest PR12 follow-ups at `68e8d63fa14c2efb8f906f3885a1650bd0669154` are incorporated; no foreign computed outputs were adopted as cloud results. A later revision recheck found PR8 at cf47a0fb7122d1ea64c81e0abbb5ae5d44ee0ed0: analyses13–16,18–20,22 have identical scientific code. Its publication code predates the latest PR12 refinements retained here, and its extra f-string escaping fix targets Python3.11 rather than this Python3.12 environment. Its executed outputs are comparison baselines only; the other reproduction branch was never changed.

## Actual platform

Ubuntu 24.04.3 LTS, Linux 6.18.44, AMD EPYC 9V74 80-Core Processor, x86_64; 9 visible CPUs, 8-core cgroup quota, 8 GiB memory limit; Python 3.12.14. Model threads 4; OMP 4, OpenBLAS 1, MKL 1, NumExpr 4, PYTHONHASHSEED 42. LightGBM deterministic=True, force_col_wise=True. Isolated pinned Python/IPython process per notebook with stream/rich-output capture. NumPy/SciPy backend configuration and all 113 package versions match the established run. All 37 actual LFS input sizes/hashes verified. See platform.json, pip-freeze.txt and lfs-inputs.json.

| Library | Version |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| shap | 0.51.0 |
| pyarrow | 25.0.1 |
| matplotlib | 3.11.2 |

## Execution inventory

| Notebook | Latest runtime(s) | Execution commit | Published artifact |
|---|---:|---|---|
| 00_data_pipeline.ipynb | 62.4 | `a8f8d8eabd6941dcb7d3a45b9d77746041c997f6` | `f6caa9c068cba43910b8ffcaf469d5919d70b927` |
| 01_ablation_gini.ipynb | 652.2 | `666e2477787da82259491149db3884c983e33725` | `643f31681f9b11c704c9d328a00e119936eb070e` |
| 02_hyperparameter_search.ipynb | 4068.0 | `60838d89cfce084b89a4e031ad1403befd35c6cc` | `f9e3f5cca26fe5a4f0c48418ed5af275fbc01ee1` |
| 03_xgboost_sybil.ipynb | 235.9 | `f6caa9c068cba43910b8ffcaf469d5919d70b927` | `e3cc058915d33b813923f166fbd98386976a5ab1` |
| 04_lightgbm_sybil.ipynb | 171.4 | `e3cc058915d33b813923f166fbd98386976a5ab1` | `a2a4a69fd0f01078d5409f767106910647da1be3` |
| 05_logistic_regression_sybil.ipynb | 54.2 | `a2a4a69fd0f01078d5409f767106910647da1be3` | `cb17cb9942c4f0a0ede69d3c81f493e95a146d3c` |
| 06_cross_ensemble_sybil.ipynb | 5.4 | `cb17cb9942c4f0a0ede69d3c81f493e95a146d3c` | `8cf6323c89efdc1b1ee1a2522ed5ba0fbf015e8c` |
| 07_sensitivity_label_vintage.ipynb | 1078.2 | `5f70994a51ea2ff5889a4ceea7f436537fd5ea65` | `ad601232025cdf3e8f0b54808f99204cde2ecd83` |
| 08_ablation_families.ipynb | 4556.8 | `1c6f5ff3b6a1379cbaa6f702966ea2c90d864257` | `633b71c9d6fb644df9b077ff0d1b91c6dbd355b5` |
| 09_shap_importance.ipynb | 2522.5 | `e5885b4a341bc2fd6ef9ce8881cc7a5911141563` | `033b1a437f73876ee49a547500f98301109bd96e` |
| 10_tie_out.ipynb | 4.6 | `b8628e73cb75a7ed5bfd4238e81417c9899c3f04` | `7f50b5b509b9215124c0085a90b058fcc8473195` |
| 11_split_comparison.ipynb | 432.4 | `7f9e8fee3e2e4bdd52a3be63f9a8b6726a06ef1d` | `a4eac5419b48acf4f4b532b625050b079b16b647` |
| 12_figures.ipynb | 202.8 | `7f50b5b509b9215124c0085a90b058fcc8473195` | `fed2febf5d060383c575b0b56c84cf8dc6a6d3e3` |
| 13_random_forest_sybil.ipynb | 7841.3 | `3d1774c8975c9b471d4c7fdec823cf087cdc98fe` | `81edf1564aeb98f6c4c10201c9c9e291c54f8e5c` |
| 14_entity_level_recall.ipynb | 34.8 | `81edf1564aeb98f6c4c10201c9c9e291c54f8e5c` | `8e6a32e0ac4f66d71b7d885e770e0b4ef97ee207` |
| 15_split_tree_report.ipynb | 934.4 | `8e6a32e0ac4f66d71b7d885e770e0b4ef97ee207` | `0e923a7883236329af2055f8ad2028fe059f17b2` |
| 16_label_robustness_initial_list.ipynb | 1392.3 | `0e923a7883236329af2055f8ad2028fe059f17b2` | `2ef61811ab3228dd69393616f61ca037d8f1a69e` |
| 17_models_10_splits.ipynb | 4497.5 | `a578dd33d81257ce6f439b28f6cb3f33a3093eb9` | `5da7f51a0ccd06a84f6f23874b5fe0bd15c09f78` |
| 18_mimicry_stress.ipynb | 317.4 | `5da7f51a0ccd06a84f6f23874b5fe0bd15c09f78` | `4fbf48766c7305fc9a4ad7d7654151c864be872f` |
| 19_ablation_by_category.ipynb | 2400.0 | `4fbf48766c7305fc9a4ad7d7654151c864be872f` | `e0c0016d4c5f68ebf009709cf130de590b8c9556` |
| 20_temporal_holdout.ipynb | 651.8 | `e0c0016d4c5f68ebf009709cf130de590b8c9556` | `1f362bb82e322ac1c031199b2b58d33d1baf0253` |
| 22_report_dependence.ipynb | 3705.0 | `1f362bb82e322ac1c031199b2b58d33d1baf0253` | `b8628e73cb75a7ed5bfd4238e81417c9899c3f04` |
| 23_revision_figures.ipynb | 6.3 | `fed2febf5d060383c575b0b56c84cf8dc6a6d3e3` | `6feb6942602ab7cb79edf579d063bb339547a4ac` |
| 24_mimicry_controls.ipynb | 934.8 | `6feb6942602ab7cb79edf579d063bb339547a4ac` | `443f11b0bec2468fa086e57a6f8e7609e92b262f` |
| 25_paper_tables.ipynb | 33.0 | `443f11b0bec2468fa086e57a6f8e7609e92b262f` | `67a14e035f72876d8c40bfff1432faaa3ff9683f` |
| 99_tie_out_revision.ipynb | 4.2 | `67a14e035f72876d8c40bfff1432faaa3ff9683f` | `65cce3a0ba06ab8f7aebeb8f9d4ec931d02d86ef` |

Notebook 21 was renamed to99 by PR10. Notebooks24 and25 were discovered and both ran. Order:00–20,22,23,24,25,99. Original tie-out 10 was repeated against the latest restored outputs. Generator 12 was refreshed again after the latest prediction restoration and revision analyses, followed by 23, 24 and 25; 99 runs last. Notebook 02 runtime is the resumed portion, excluding 25 earlier cached XGBoost fits. All 74 XGBoost and 48 LightGBM validation configurations belong to this cloud reproduction. Notebook 13 also reports the resumed portion: ten full-precision cached Random Forest search fits were reused after code/input/parameter/platform verification; their original per-fit runtimes remain in the CSV. Random Forest’s predefined 24-configuration validation grid was regenerated.

## Recovery and repairs

Two scratch workspace replacements interrupted notebook 08: at least 12 fits in the first and 79 in the second were lost. Completed00–07 results survived in Git. Ignored master/split/prediction files required regeneration of00 and03–06;01/02/07 passed provenance and were reused. The archive service accepted intermediate snapshots but their download returned HTTP403, so those files could not be verified/reused. A third checkout loss occurred after notebook 08 was completed and published: all 120 completed fits and its results were recovered from Git, while ignored master/split/prediction files again required regeneration of 00 and 03–06. Valid 01/02/07/08 stages were reused. A fourth checkout loss occurred during the first notebook 13 search fit, before a partial checkpoint was published. Its unpublished log/cache could not be recovered; the completed fit count is unknown. Completed stages through 12 remained valid and were retained; 00 and 03–06 were regenerated to restore ignored prerequisites. A fifth workspace loss removed that checkout and the virtual environment during notebook 13. Its final published partial checkpoint preserved all ten completed validation fits. The former worker exited, the exact 113-package environment was restored under /tmp, and code/input/parameter/platform hashes were checked before reusing the ten fits. The current checkout is /tmp/layerzero-cloud-reproduction with ten-minute GitHub partial snapshots and a directory liveness heartbeat. Unpublished intermediate files again required 00 and 03–06 restoration. No duplicate analysis worker was launched during recovery. Notebook17 later completed ten model-comparison groups but its memory monitor failed because the namespace PID was absent from the host procfs view. Full-precision in-memory results were not serialized and could not be recovered after worker exit; those ten groups required one repeat. The failed 3340.1-second attempt is retained in failed-attempts/. The monitor now obtains this same process’s actual procfs PID from /proc/self/stat. Complete split rows/categories/weights are now atomically cached with code/input/parameter/platform hashes; fixture tests verify reuse, changed-input invalidation and tampering rejection. No scientific fit or selection rule changed. Logs, cell checkpoints and metric-only fits remain traceable. Historical attempts are retained in execution.json/RUN_HISTORY.md; no interrupted stage is claimed complete.

Necessary repairs record actual platform/full producer SHA, correct stale notebook language/platform metadata without changing code or executed outputs, verify platforms across original and revision results, retain full-precision thresholds/blend-sweep/generator values, support socket-free IPython execution and atomically checkpoint metric-only fits by code, feature order, parameters, arrays, labels and platform. Cache fixture tests passed reuse, changed-input invalidation and corruption rejection. Latest PR12 display names/table refinements preserve internal feature definitions. No selection rule, split, search space or assertion was weakened.

## Validation and refreshed assets

508 independent checks passed. Every producer commit is reachable, every result matches its recorded execution and actual numerical platform, all applicable provenance checks passed.14 matches regenerated03/04/06 probabilities. Prediction metrics and validation-selected thresholds, search winners and blend weights recompute correctly.13/17/19 LightGBM baselines match 08. Random Forest cross-stage threshold/F1 and publication rounding are checked independently; tiny full-precision AP differences are retained in random-forest-cross-stage-diagnostic.json. Threaded scikit-learn probability summation can affect tied ranks at floating-point precision, so a bit-identical Random Forest AP comparison is not claimed.22’s partition assertions executed successfully and all 15 paired comparisons (30 arms) meet the predefined 1% matching tolerance. Its observed unique/class counts and exact post-upsampling model counts derived from make_S are in paired-training-counts.json.

41 current publication outputs are in publication-manifest.json with hashes and producer commits. All 18 current LaTeX table fragments compiled; notebook25 produced 207 named values with none missing; combined proof and compiler logs are in latex-proof/. Prediction hashes/producers are in prediction-manifest.json. Large predictions remain in prescribed ignored output storage. Unlisted root figures/tables are historical or unreferenced and are not claimed as refreshed cloud outputs.

## Numerical comparisons

Full-precision differences against main, PR8 entity results and PR10 paired results are in numerical-differences.json. Older-revision comparisons include code/platform differences and should not be attributed solely to CPU hardware.

### Section 5.2

| Model | F1 mean ± SD | AP mean ± SD | AUROC mean ± SD |
|---|---:|---:|---:|
| Logistic regression | 0.2336 ± 0.0041 | 0.1605 ± 0.0088 | 0.8323 ± 0.0030 |
| XGBoost | 0.7066 ± 0.0076 | 0.7529 ± 0.0085 | 0.9641 ± 0.0020 |
| LightGBM | 0.7131 ± 0.0060 | 0.7607 ± 0.0060 | 0.9682 ± 0.0018 |
| Random Forest | 0.7022 ± 0.0056 | 0.7535 ± 0.0073 | 0.9685 ± 0.0015 |
| Ensemble | 0.7128 ± 0.0082 | 0.7610 ± 0.0070 | 0.9680 ± 0.0018 |

### Section 5.5

| Entity/model, validation threshold, all bucket | Weighted recall | Majority hit | Residual allocation |
|---|---:|---:|---:|
| E1 gas provision tree / XGBoost | 0.720483 | 0.767067 | 0.368790 |
| E1 gas provision tree / LightGBM | 0.711148 | 0.755044 | 0.379626 |
| E1 gas provision tree / Ensemble | 0.738788 | 0.784797 | 0.354931 |
| E2 bounty report / XGBoost | 0.720483 | 0.552301 | 0.368790 |
| E2 bounty report / LightGBM | 0.711148 | 0.539749 | 0.379626 |
| E2 bounty report / Ensemble | 0.738788 | 0.577406 | 0.354931 |
| E2 bounty report (all Sybils in test) / XGBoost | 0.050000 | 0.055556 | 0.981827 |
| E2 bounty report (all Sybils in test) / LightGBM | 0.050000 | 0.055556 | 0.981827 |
| E2 bounty report (all Sybils in test) / Ensemble | 0.050000 | 0.055556 | 0.981827 |
| E3 reporter / XGBoost | 0.720483 | 0.704545 | 0.368790 |
| E3 reporter / LightGBM | 0.711148 | 0.674242 | 0.379626 |
| E3 reporter / Ensemble | 0.738788 | 0.727273 | 0.354931 |

### Central paired comparison 22

| Model/metric | Previous held/seen | Cloud held/seen | Cloud paired difference | Corrected p |
|---|---:|---:|---:|---:|
| LightGBM/f1 | 0.366702/0.598080 | 0.369207/0.598032 | +0.228825 | 1.85484e-05 |
| LightGBM/ap | 0.304796/0.596586 | 0.305047/0.598964 | +0.293917 | 1.35349e-06 |
| XGBoost/f1 | 0.363252/0.580279 | 0.362983/0.581217 | +0.218234 | 5.50912e-05 |
| XGBoost/ap | 0.305494/0.596929 | 0.305010/0.595040 | +0.290030 | 1.10872e-06 |

### Comparison with latest x86 PR8 checkpoint

| Model/metric | Latest x86 held/seen | Cloud held/seen | Change in paired difference |
|---|---:|---:|---:|
| LightGBM/f1 | 0.369207/0.598032 | 0.369207/0.598032 | +0.000000 |
| LightGBM/ap | 0.305047/0.598964 | 0.305047/0.598964 | +0.000000 |
| XGBoost/f1 | 0.362983/0.581217 | 0.362983/0.581217 | +0.000000 |
| XGBoost/ap | 0.305010/0.595040 | 0.305010/0.595040 | +0.000000 |

Arms satisfy 1% size/class matching, not exact equality. Seen training intentionally includes test-report mates; held-out excludes them. Validation/test rows remain separate.22’s per-fold composition is recorded rather than inferred from nominal sizes.

## Selected historical differences

Values below use full-precision source results; displayed differences are cloud minus historical. The main comparison is against the committed revision platform; the PR8 comparison is against its historical x86 entity results.

| Section 5.2 model | Main F1 mean | Cloud F1 mean | Difference | Main AP mean | Cloud AP mean | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.233605 | 0.233605 | +0.000000 | 0.160486 | 0.160486 | +0.000000 |
| XGBoost | 0.706294 | 0.706570 | +0.000276 | 0.753430 | 0.752918 | -0.000512 |
| LightGBM | 0.713273 | 0.713093 | -0.000181 | 0.761092 | 0.760687 | -0.000406 |
| Random Forest | 0.702564 | 0.702201 | -0.000363 | 0.753676 | 0.753537 | -0.000139 |
| Ensemble | 0.713136 | 0.712752 | -0.000384 | 0.761229 | 0.760985 | -0.000245 |

| Section 5.5 baseline/entity/model | Weighted recall old/cloud/delta | Majority hit old/cloud/delta | Residual allocation old/cloud/delta |
|---|---:|---:|---:|
| main / E1 gas provision tree / XGBoost | 0.730734/0.720483/-0.010251 | 0.777461/0.767067/-0.010393 | 0.359631/0.368790/+0.009159 |
| main / E1 gas provision tree / LightGBM | 0.718653/0.711148/-0.007505 | 0.764214/0.755044/-0.009171 | 0.372188/0.379626/+0.007438 |
| main / E1 gas provision tree / Ensemble | 0.718653/0.738788/+0.020135 | 0.764214/0.784797/+0.020583 | 0.372188/0.354931/-0.017257 |
| main / E2 bounty report / XGBoost | 0.730734/0.720483/-0.010251 | 0.556485/0.552301/-0.004184 | 0.359631/0.368790/+0.009159 |
| main / E2 bounty report / LightGBM | 0.718653/0.711148/-0.007505 | 0.552301/0.539749/-0.012552 | 0.372188/0.379626/+0.007438 |
| main / E2 bounty report / Ensemble | 0.718653/0.738788/+0.020135 | 0.552301/0.577406/+0.025105 | 0.372188/0.354931/-0.017257 |
| main / E2 bounty report (all Sybils in test) / XGBoost | 0.050000/0.050000/+0.000000 | 0.055556/0.055556/+0.000000 | 0.981827/0.981827/+0.000000 |
| main / E2 bounty report (all Sybils in test) / LightGBM | 0.050000/0.050000/+0.000000 | 0.055556/0.055556/+0.000000 | 0.981827/0.981827/+0.000000 |
| main / E2 bounty report (all Sybils in test) / Ensemble | 0.050000/0.050000/+0.000000 | 0.055556/0.055556/+0.000000 | 0.981827/0.981827/+0.000000 |
| main / E3 reporter / XGBoost | 0.730734/0.720483/-0.010251 | 0.704545/0.704545/+0.000000 | 0.359631/0.368790/+0.009159 |
| main / E3 reporter / LightGBM | 0.718653/0.711148/-0.007505 | 0.689394/0.674242/-0.015152 | 0.372188/0.379626/+0.007438 |
| main / E3 reporter / Ensemble | 0.718653/0.738788/+0.020135 | 0.689394/0.727273/+0.037879 | 0.372188/0.354931/-0.017257 |
| pr8 / E1 gas provision tree / XGBoost | 0.720483/0.720483/+0.000000 | 0.767067/0.767067/+0.000000 | 0.368790/0.368790/+0.000000 |
| pr8 / E1 gas provision tree / LightGBM | 0.711148/0.711148/+0.000000 | 0.755044/0.755044/+0.000000 | 0.379626/0.379626/+0.000000 |
| pr8 / E1 gas provision tree / Ensemble | 0.738788/0.738788/+0.000000 | 0.784797/0.784797/+0.000000 | 0.354931/0.354931/+0.000000 |
| pr8 / E2 bounty report / XGBoost | 0.720483/0.720483/+0.000000 | 0.552301/0.552301/+0.000000 | 0.368790/0.368790/+0.000000 |
| pr8 / E2 bounty report / LightGBM | 0.711148/0.711148/+0.000000 | 0.539749/0.539749/+0.000000 | 0.379626/0.379626/+0.000000 |
| pr8 / E2 bounty report / Ensemble | 0.738788/0.738788/+0.000000 | 0.577406/0.577406/+0.000000 | 0.354931/0.354931/+0.000000 |
| pr8 / E2 bounty report (all Sybils in test) / XGBoost | 0.050000/0.050000/+0.000000 | 0.055556/0.055556/+0.000000 | 0.981827/0.981827/+0.000000 |
| pr8 / E2 bounty report (all Sybils in test) / LightGBM | 0.050000/0.050000/+0.000000 | 0.055556/0.055556/+0.000000 | 0.981827/0.981827/+0.000000 |
| pr8 / E2 bounty report (all Sybils in test) / Ensemble | 0.050000/0.050000/+0.000000 | 0.055556/0.055556/+0.000000 | 0.981827/0.981827/+0.000000 |
| pr8 / E3 reporter / XGBoost | 0.720483/0.720483/+0.000000 | 0.704545/0.704545/+0.000000 | 0.368790/0.368790/+0.000000 |
| pr8 / E3 reporter / LightGBM | 0.711148/0.711148/+0.000000 | 0.674242/0.674242/+0.000000 | 0.379626/0.379626/+0.000000 |
| pr8 / E3 reporter / Ensemble | 0.738788/0.738788/+0.000000 | 0.727273/0.727273/+0.000000 | 0.354931/0.354931/+0.000000 |

## Author actions before submission

Paven must apply refreshed assets/named values to the private manuscript/reviewer response; confirm tentative feature cost classes and preserve entity-proxy/allocation, paired-design, label and temporal-cohort caveats; resolve existing DATA.md license/citation/archival questions. The main union split 15 retains its large-component warning: its largest component holds 45.36% of all Sybils, above the existing 5% warning level. Historical field names referring to another machine are retained for schema compatibility, but all current sources use the recorded AMD platform. Its stricter reporter-linked variant fails all three class-rate checks: train 4.70%, validation/test 3.69%, versus target 4.19% ±0.30 percentage points. One component contains 10,071 Sybils (55.30% of all Sybils). Even the largest permitted partition, 50% of 434,786 rows at at most 4.49% Sybils, could contain only about 9,761 Sybils; a whole-component split therefore cannot satisfy the joint constraints. The code stops that diagnostic variant before training, and no assertion or tolerance is weakened. This is a structural limitation, not an execution defect; the paired report-dependence design 22 is evaluated separately and does not establish reporter-level independence. Mandatory checks pass, while checks including this optional feasibility diagnostic do not all pass. Private manuscript tasks remain open; no merge was performed.

## PR publication status

All execution, result and publication work is complete and pushed to Scott’s existing cloud branch. Cross-fork PR creation targeting Paven main was attempted with the requested title, but GitHub returned **403: Resource not accessible by integration**. No PR or merge is claimed. The complete reviewable PR body is in [PULL_REQUEST.md](PULL_REQUEST.md); the target integration needs permission to create pull requests, or an approved authenticated browser fallback is needed.
