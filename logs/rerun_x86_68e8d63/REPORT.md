# Rerun of 13–25 and 99 at 68e8d63 on x86_64 (2026-10-08 to 2026-10-09)

Source: `revision/a25-tables` at 68e8d63 (PR #12, stacked on #11 and #10). Execution commit 412db11.
Every notebook numbered 13–25 that exists was run in numerical order, then `99`. Notebook 21 does not exist (renamed to `99` in PR #10).
Each notebook started from a clean tree and was committed before the next one started.

## Result

- `99_tie_out_revision`: **83 of 83 checks pass**, `multiple_platforms_cited` false (`results/99_tie_out_revision.json`).
- `14` regenerated its entity-level results from `03`/`04`/`06` predictions regenerated here: `predictions_match_committed_results` true for XGBoost, LightGBM and the ensemble.
- `13`–`22` and `25` equal PR #8's x86 rerun (cf47a0f), which ran the same code on the same data. The only differences are Random Forest average precision at most 6.7e-7 (`13`, `17`) and timing or memory measurements (`compare_results.txt`).
- Against the results committed at 68e8d63 (13–20 and 22 from an Apple M5, arm64), most numbers change (below).

This is a rerun on hardware that matches the recorded specification. It is not claimed to be the identical physical host: the CPU model string, family, model, stepping, microcode and flags are identical to the original 2.80 GHz host's record, which cannot prove physical identity. Random Forest scores are not bit-reproducible run to run: scikit-learn sums tree probabilities across threads in varying order (13's search rows differ from PR #8's by up to 1.1e-7 in validation AP, and a configuration re-scored on this host after a restart differed by 1e-16; F1, thresholds and the selection are identical).

## Environment (read in the notebook kernel)

| Item | Requested | Actual |
|---|---|---|
| System | x86_64 Linux | x86_64 Linux, kernel 6.18.44-fc-v80 (KVM guest) |
| CPU | Intel Xeon @ 2.80 GHz | Intel(R) Xeon(R) Processor @ 2.80GHz, family 6, model 85, stepping 7, microcode 0x1; L2 4 MiB, L3 33 MiB; flags SHA-256 e28bebf6… (identical to `logs/rerun_x86/env_after_restart.txt` of PR #8) |
| Cores | 4 | 4 (affinity 4, no cgroup CPU quota, 1 thread per core) |
| Memory | | 15.7 GiB, no cgroup limit |
| Python | 3.11.15 | 3.11.15 (uv standalone build) |
| xgboost / lightgbm / scikit-learn / numpy | 3.2.0 / 4.6.0 / 1.8.0 / 2.4.6 | 3.2.0 / 4.6.0 / 1.8.0 / 2.4.6 |
| Other pins | pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2, pyarrow 25.0.1, shap 0.51.0 | identical |
| Remaining packages | | the original run's freeze, 99 of 102 identical; the 3 Ubuntu system bindings are absent (`pip_freeze.txt`) |
| Threads | 4, deterministic | `N_JOBS` 4, LightGBM `force_col_wise` and `deterministic`; OpenBLAS and OpenMP pools 4; no thread variables set |

Files: `env_kernel.json` (before the run), `run/env_after_restart*.json` (after each restart, all identical to it), `lscpu.txt`, `meminfo.txt`, `pip_freeze.txt`.

## Inputs and retained results

- Git LFS: all 37 objects fetched and checked, SHA-256 and size equal to their pointers (`lfs_inputs.sha256`). No pointer file was read.
- `00`–`12` results retained: `sp.provenance` passes for every one (code, `sybil_pipeline.py`, `requirements.txt`, `data/` unchanged since 0d9eacc; `12` since cae1b67) (`run/prereq_provenance.txt`). Their host CPU is not recorded.
- `03`, `04`, `06` regenerated before `14`: every field equals the committed 0d9eacc results (85, 84, 96 fields) (`run/regen/compare_regen.txt`); prediction hashes in `run/regen/predictions.sha256`. On that basis the committed `01` feature selection and `02` search were kept.
- `results/13_search_random_forest.csv` at 68e8d63 was scored on arm64, so 412db11 removes it and `13` searched all 24 configurations here.
- Every notebook's inputs (results, predictions, tables, figures) are hashed at its start: `run/inputs/<notebook>.sha256`.

## Code changes in the execution commit

- `24_mimicry_controls`: the table cell's `replace('_', '\\_')` moved out of the f-string (a SyntaxError before Python 3.12), as in 693757c of PR #8; the produced text is unchanged.
- `results/13_search_random_forest.csv` removed (above). No other code, data or requirement changes.

## Notebooks

| Notebook | Executed at | Committed | Seconds | Note |
|---|---|---|---|---|
| 13_random_forest_sybil | 412db11 | 9c634b7 | 4,517 | resumed after restart 1 from 21 of 24 search rows scored here (host check passed) |
| 14_entity_level_recall | 9c634b7 | 1e6aa04 | 49 | |
| 15_split_tree_report | 1e6aa04 | 3a0e8ee | 1,046 | |
| 16_label_robustness_initial_list | 3a0e8ee | 17dd144 | 1,655 | rerun from the start after restart 2 |
| 17_models_10_splits | 17dd144 | daa8986 | 4,946 | |
| 18_mimicry_stress | daa8986 | cdb7aed | 416 | |
| 19_ablation_by_category | cdb7aed | cb1243e | 2,702 | |
| 20_temporal_holdout | cb1243e | 1588c7c | 731 | |
| 21 | | | | absent |
| 22_report_dependence | 1588c7c | bb458c6 | 6,261 | rerun from the start after restart 3 |
| 23_revision_figures | bb458c6 | 37bdba6 | 19 | |
| 24_mimicry_controls | 37bdba6 | 2bfbee1 | 1,713 | |
| 25_paper_tables | 2bfbee1 | 8eed3f4 | 76 | |
| 99_tie_out_revision | 8eed3f4 | b050c3e | 7 | 83 of 83 |

Regeneration before `14`: `03` 258 s, `04` 212 s, `06` 7 s. Timeline: `run/timeline.tsv`; logs: `run/<notebook>.log`.

Restarts. The container restarted three times (during `13`, `16` and `22`); no result was written by an interrupted notebook. Interrupted logs are kept as `*.interrupted_by_restart.log`, and `13`'s 21 rows as `run/13_search_random_forest.partial_before_restart.csv`. After restarts 1 and 2 the environment was re-read and `03` (and after restart 1 also `04`, `06` and one Random Forest configuration) reproduced the pre-restart results exactly (`run/regen_after_restart*/`). After restart 3 only the environment was re-read before `22` resumed; `22` then equalled PR #8 in every field, and after the run `03` again reproduced its first regeneration exactly, with all six prediction files unchanged (`run/final_checks/`; that `03` run started while the tracker edit was uncommitted, so it records a `-dirty` commit; only `docs/REVISION_LEAKAGE.md` differed).

## Checks requested

- **14 consumes matching predictions.** It asserts the three prediction files have identical rows and labels, and records `matches_committed` true for every model (thresholds, precision, recall, F1, AUROC, AP within 1e-9 of the committed `03`/`04`/`06`).
- **22 partition separation.** For each of 15 fold-repeats it asserted fold sizes within 1 % of 20 %, no tree and no report spanning two folds, k_a disjoint from validation and both training sets, no shared tree between k_a and k_b, no k_a report-mate in held-out training, and training sets within 1 % in rows and Sybils. All held.
- **Paired training counts** (`paired_training_counts_22.json`, from 22's recorded composition): unique Sybils equal in all 15 pairs (mean 14,612.2); unique rows equal in 14 of 15, differing by 2 in the other. Rows fed to the model after 1:1 upsampling are 2 × non-Sybils: 503,958 or 503,960 per arm.
- **Selection on validation only.** Thresholds come from `sp.evaluate` (`best_f1_threshold` on validation), early stopping uses the validation set, blend weights `select_blend_weight` on validation, and `13`'s search asserts the test partition is absent. No notebook computes a threshold or selection on test labels; `22` asserts each threshold equals the validation optimum.
- **Consistency.** `99` checks provenance of every result and dependency, stored hyperparameters against `02` and `13`, `24` against `18`, `14`'s inputs, platforms and 207 named numbers (none missing). `23` and `25` compile every table with pdflatex.

## Differences from the committed results (68e8d63, arm64)

`14`, Section 5.5 (LightGBM at its validation threshold, 0.6001 here, 0.5748 committed):

| Entity | Metric | Committed | Rerun |
|---|---|---|---|
| E1 provision tree | any-hit | 0.7646 | 0.7555 |
| E1 provision tree | majority-hit | 0.7642 | 0.7550 |
| E1 provision tree | Sybil-weighted recall | 0.7187 | 0.7111 |
| E2 bounty report | any-hit | 0.7029 | 0.6946 |
| E2 bounty report | majority-hit | 0.5523 | 0.5397 |
| E2 bounty report | reports with half their allocation unflagged | 0.4895 | 0.5063 |
| E3 reporter | any-hit | 0.8712 | 0.8636 |
| E3 reporter | majority-hit | 0.6894 | 0.6742 |
| All | unflagged allocation share | 0.3722 | 0.3796 |

`22`, central paired comparison (seen minus held out, 15 fold-repeats, all 15 higher in both runs):

| Model | Metric | Committed | Rerun |
|---|---|---|---|
| LightGBM | F1 | +0.2314 ± 0.0843, p 4.7e-5 | +0.2288 ± 0.0763, p 1.9e-5 |
| LightGBM | AP | +0.2918 ± 0.0748, p 9.8e-7 | +0.2939 ± 0.0775, p 1.4e-6 |
| LightGBM | AUROC | +0.0785 ± 0.0411, p 0.0012 | +0.0793 ± 0.0424, p 0.0015 |
| XGBoost | F1 | +0.2170 ± 0.0810, p 5.9e-5 | +0.2182 ± 0.0809, p 5.5e-5 |

Field counts per notebook against both references, and table and figure comparisons: `compare_results.txt`. Tables carry the same numbers as PR #8's; `rev_levels`, `rev_taxonomy`, `rev_temporal_holdout` and `rev_union_split_appendix` differ in layout from PR #12's own changes. In `tables/rev_text_numbers.json`, 2 of the 199 names shared with PR #8 differ in value (Random Forest AP mean and SD, at most 1e-8), plus 21 timing and memory measurements; 8 names are new in PR #12.

## Not done here

- `README.md` still describes `13`–`24` with the Apple M5 numbers and runtimes of PR #12; it was not edited.
- Physical identity with the original host cannot be established from inside the guest.
