# Root-wallet tree-feature fix: interim report (2026-10-09 16:41 UTC)

Interim evidence from one environment (x86_64 Linux, Xeon 2.80 GHz, 4 cores, Python 3.11.15, pinned packages). The corrected publication rerun of notebooks 00 to 25 and 99 has not been done; results, figures and tables committed in the repository are still those from before the fix.

## 1. Branch, fix commit, draft PR
- Branch `fix/root-tree-metrics-2026-10-09` in `scottonchain/layerzero_xgboost`, started from b7a9a3f (PR #13's head, preserved untouched).
- Fix commit 663e026; regression checks included. Later commits add the Section 5.3 reporting change (7c97305) and the evidence under `diagnostics/root_tree_fix/`.
- Draft PR to `paven86/layerzero_xgboost`: the Claude GitHub App is refused (403) on that repository, so it must be opened from https://github.com/paven86/layerzero_xgboost/compare/main...scottonchain:layerzero_xgboost:fix/root-tree-metrics-2026-10-09?expand=1 with "Create draft pull request".

## 2. Cause and inherited history
`provision_features` skipped every wallet absent from the funding forest's parent map (`if a not in parent: rows.append({})`), so roots got all-zero tree features through the later `fillna(0)`, although their descendants carried the whole tree's summaries. Interactors with no recorded incoming funding edge were missing from the provider table and were zero-filled too. The behavior is in the original featurization notebook from its earliest committed version (efc1c50, May 2025). Reproduction: a labeled exchange funds A and A funds B; both have `provision_tree` = A, but A got `tree_size` 0 and B got 2.

## 3. Regression checks
`tests/test_root_tree_features.py`: 11 checks pass on the fixed code (service-funded root with a child, singleton root, root with no incoming record but a child, isolated wallet, zero-total entropy, hand-computed descendant and non-tree values, deeper depths, tree split by a labeled edge, input-order independence, one row per interactor, merge zero-fill). 10 of the 11 fail on the original code (8 assertion failures, 2 errors including the entropy division by zero).

## 4. Full-dataset feature changes
- Selected model inputs (62 features) change for **307,622 of 434,786 wallets (70.75 %)**. All of them had all-zero tree features before.
- By category: IxL 306,658 of 306,658 (100.0 %); IxI 37 of 34,391 (0.108 %); IxE 251 of 93,061 (0.27 %); No provider 676 of 676 (100.0 %).
- Sybils changed 15,947 of 18,211; non-Sybils 291,675 of 416,575.
- Roots with descendants 12,942 (255 Sybils); singleton roots 294,680 (15,692 Sybils); non-roots 0.
- The tracker's historical count is confirmed: 12,886 IxL wallets are roots of multi-wallet groups.
- Changed model features (17): breadth_factor, tree_size, sparsity, leaf_to_internal_ratio, breadth_to_depth_ratio, longest_chain_ratio, branching_factor, total_gas, leaf_provision_proportion, max_depth, avg_depth, gas_distribution_entropy, gas_distribution_skewness, star_like_ratio, leaf_gas_distribution_entropy, leaf_gas_distribution_skewness, balance_factor. `depth` is unchanged. Changed computed columns outside the model: avg_leaf_gas, depth_weighted_avg_gas, depth_variance, gini_coefficient.
- Unexpected changes: none (non-root wallets: none; non-tree columns: none).
- Identical between original and corrected: address order True, labels True, categories True, `provision_tree` ids True, anchors (4,308) True, train/validation/test row ids True.

## 5. Fixed-configuration sensitivity analysis (not the retuned paper result)
LightGBM, committed hyperparameters, 62 features, seed-42 tree split, one model seed (42), identical rows; early stopping and threshold from validation for each fit.

| Test metric | Original | Corrected | Difference |
|---|---|---|---|
| F1 | 0.7105 | 0.7169 | +0.0064 |
| AP | 0.7650 | 0.7668 | +0.0018 |
| AUROC | 0.9672 | 0.9672 | +0.0000 |
| Precision | 0.6769 | 0.7158 | +0.0389 |
| Recall | 0.7476 | 0.7179 | -0.0297 |
| Validation threshold | 0.4355 | 0.5841 | +0.1486 |

| Category | Test n | Sybils | Recall original | corrected | F1 original | corrected |
|---|---|---|---|---|---|---|
| IxE | 28,171 | 258 | 0.198 | 0.178 | 0.315 | 0.292 |
| IxI | 9,967 | 426 | 0.397 | 0.373 | 0.477 | 0.467 |
| IxL | 92,079 | 4,770 | 0.809 | 0.778 | 0.738 | 0.747 |
| No provider | 218 | 9 | 0.778 | 0.778 | 0.875 | 0.875 |

Affected Sybil roots in test: 4,779; recall 0.809 to 0.778 (mean score 0.773 to 0.777); with descendants 70: 0.471 to 0.386. The recall drop comes mostly from the higher validation-selected threshold.

Binary predictions that change: 702 of 130,435 (0.54 %): 74 to flagged, 628 to unflagged; Sybils 222 (30 missed to caught, 192 caught to missed), non-Sybils 480 (44 false positives added, 436 removed).

## 6. Central evidence: notebook 22 report dependence, LightGBM only, **10 of 15 pairs complete**
Same components, folds (repeats 42, 1, 2; folds 0 to 4), matched-tree removal, upsampling and validation-only selection; row identities asserted equal for every arm. The original arm is PR #13's stored result (the first pair was refitted here and is bit-identical). Descriptive until all 15 pairs are done.

| Pair | Test Sybils | F1 held / seen: original | corrected | F1 diff original | corrected | AP diff original | corrected | AUROC diff original | corrected |
|---|---|---|---|---|---|---|---|---|---|
| r0 f0 | 901 | 0.462 / 0.626 | 0.472 / 0.615 | +0.164 | +0.144 | +0.213 | +0.212 | +0.057 | +0.051 |
| r0 f1 | 899 | 0.404 / 0.598 | 0.389 / 0.604 | +0.194 | +0.215 | +0.246 | +0.257 | +0.060 | +0.060 |
| r0 f2 | 899 | 0.328 / 0.580 | 0.323 / 0.598 | +0.253 | +0.274 | +0.321 | +0.306 | +0.068 | +0.071 |
| r0 f3 | 900 | 0.221 / 0.598 | 0.214 / 0.600 | +0.377 | +0.386 | +0.418 | +0.417 | +0.155 | +0.150 |
| r0 f4 | 900 | 0.451 / 0.628 | 0.460 / 0.616 | +0.177 | +0.156 | +0.263 | +0.264 | +0.053 | +0.055 |
| r1 f0 | 901 | 0.419 / 0.633 | 0.424 / 0.624 | +0.214 | +0.200 | +0.251 | +0.242 | +0.051 | +0.049 |
| r1 f1 | 899 | 0.392 / 0.569 | 0.388 / 0.565 | +0.178 | +0.178 | +0.203 | +0.208 | +0.056 | +0.058 |
| r1 f2 | 900 | 0.347 / 0.583 | 0.343 / 0.603 | +0.236 | +0.260 | +0.351 | +0.357 | +0.080 | +0.076 |
| r1 f3 | 899 | 0.247 / 0.604 | 0.241 / 0.616 | +0.356 | +0.375 | +0.428 | +0.427 | +0.160 | +0.148 |
| r1 f4 | 899 | 0.429 / 0.616 | 0.429 / 0.621 | +0.187 | +0.192 | +0.241 | +0.235 | +0.048 | +0.046 |

| Seen minus held out, mean ± SD over completed pairs | Original | Corrected |
|---|---|---|
| F1 | +0.2335 ± 0.0755 (10/10 higher) | +0.2380 ± 0.0855 (10/10 higher) |
| AP | +0.2934 ± 0.0817 (10/10 higher) | +0.2927 ± 0.0814 (10/10 higher) |
| AUROC | +0.0789 ± 0.0425 (10/10 higher) | +0.0764 ± 0.0394 (10/10 higher) |

## 7. What can be concluded now, what still needs the full rerun, and the ETA
- Now (descriptive, 10 of 15 pairs): the feature fix is verified as intended and touches only roots; with a fixed configuration it changes test F1 by +0.0064 and leaves AUROC unchanged; in every completed report-dependence pair the seen arm beats the held-out arm for both feature versions, and the mean effect moves from +0.234 to +0.238 F1. No change in the interpretation is visible so far. No significance test is quoted until all 15 pairs are complete.
- Still needs the full rerun: retuned hyperparameters (02 and 13 from scratch), the Gini keep-or-drop rule (01), all models and predictions, entity-level recall (14), the ten-split comparisons (13, 17, 19), the revised analyses (15, 16, 18, 20, 22, 24), the figures, tables and named numbers (23, 25), and the tie-out (99). The 83 of 83 checks of the earlier run are evidence for the original implementation only.
- Observed ETA: 10 pairs done; about 3.1 minutes per pair, so the remaining 5 should finish near 16:56 UTC (10:56 Denver).
- The corrected publication rerun (00 to 25, 99; fresh search caches) takes roughly 12 hours of compute on this four-core host, based on the earlier run's notebook timings (13 to 99: about 7.7 hours; 00 to 12 with the 80-minute search: about 5 hours), plus any container restarts; it starts when Stage 3 ends.

## 8. Required manuscript corrections
- **"Single-wallet tree".** The tree ids (`provision_tree`) are unchanged, so counts of single-wallet groups stand. What changes is what the tree *features* said: before the fix every one of the 306,658 IxL wallets had all-zero tree features. 293,772 IxL wallets (95.8 %) are single-wallet trees (15,669 of the 15,921 IxL Sybils, 98.4 %), but 12,886 (4.2 %) are roots of multi-wallet trees of up to 610 wallets (252 of them Sybils). Text saying IxL wallets are singletons, or that their tree features were computed from a tree, must be corrected.
- **"Uninformative by construction".** The old statement cannot stand as written: the tree features were zero for all IxL wallets because of the bug, not because of the data. After the fix, the 18 tree-derived model features take one identical value vector for all 293,772 IxL singleton wallets (tree_size 1, breadth_factor 1, sparsity 1, the rest 0), so within singletons they cannot discriminate, but that is a property of singleton trees; for the 12,886 IxL roots with descendants they now carry the tree's summaries. Whether they help for IxL must be shown by the corrected ablation (notebook 19), not asserted by construction.
- **Numbers.** Every model result, table, figure and named number that depends on the tree features (notebooks 00 to 25 and 99) is to be replaced by the corrected rerun; the values above are fixed-configuration diagnostics only.
- **Section 5.3 false-positive rate** (reporting change, notebooks 25 and 99; values below are from the current, pre-fix results/16 and will regenerate): N = 124,972 original negative test addresses, I = 10,315 of them on the initial list; F = 1,461 false positives at the validation threshold, J = 556 of them on the list. Original-label FPR = F/N = 1,461/124,972 = 1.169 %. With every initial-list address counted as positive: (F-J)/(N-I) = 905/114,657 = 0.789 %. The denominator removes all 10,315 list addresses among the test negatives, flagged or not.
