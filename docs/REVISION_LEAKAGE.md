# Revision tracker: data leakage

Manuscript BCRA-D-26-00643, resubmission due Oct 12, 2026. Repo baseline: paven86/layerzero_xgboost @ 11366f4.

Check an item only when the code, the manuscript text, and the response letter are all updated.

**Notebook numbering (2026-10-02).** Notebooks were renumbered to run order; findings below use the old numbers.
Old → new: `00` → `00`; `08_ablation_gini` → `01`; `05_hyperparameter_search` → `02`; `01_xgboost` → `03`;
`02_lightgbm` → `04`; `03_logistic_regression` → `05`; `04_cross_ensemble` → `06`; `07` → `07`;
`09_ablation_families` → `08`; `10_shap_importance` → `09`; `06_split_comparison` → split into `10_tie_out`
(checks and reported numbers) and `11_split_comparison` (the original random split, trained only there).

## Status

Code status per leakage item. "Code done" means the repository change is committed; the checkbox
in sections A to C stays open until the manuscript and response letter are updated too. Numbers
for the paper come from `10_tie_out.ipynb` (and `11_split_comparison.ipynb` for the random-split comparison) in the tagged final run (section E). Notebook numbers in this table, C and D are the current ones.

| Item | Reviewer point | Code status | Commit | Evidence |
|---|---|---|---|---|
| A1 group split | R2 relational leakage | Code done | fd6411f | `sybil_pipeline.make_splits` asserts no gas provision tree spans two partitions |
| A2 tree-overlap check | R2 relational leakage | Code done | fd6411f | `00` leakage cell; `11` leakage table |
| A3 blend weight on test | R2 test isolation | Code done | fd6411f | `06` section 2 selects on validation F1 (`sp.select_blend_weight`) |
| A4 search not in repo; tables on test | R1 search/validation; R2 test isolation | Done: validation-only search, 74 XGBoost and 48 LightGBM configurations | ea602c6 | `02_hyperparameter_search` (validation only; test deleted before fitting) |
| A5 feature selection | R1 | Code done. Authors: the 63 were chosen manually; `gini_coefficient` later removed by a pre-specified rule (C). Label-based EDA moved to the training partition | fd6411f | Methods text still needed |
| A6 post-snapshot reference data | R1 temporal cutoff | Audited (findings E). Labels: fixed rule, pre-snapshot vintage as sensitivity (`07`): test F1 −0.005 ± 0.007 | ea602c6 | `README` data table gives each source's cutoff; `07` |
| A7 provision graph date filter | R1 temporal cutoff | Code done: cutoff enforced; tree features recomputed from filtered edges | b2904f4 | `00` steps 2 and 4 |
| A8 both splits reported | R2 relational leakage | Done: final run, both splits | ea602c6, 06dbd82 | `11_split_comparison` |
| A9 descriptive figure on val+test | R1 | Done: `12_figures` redraws it on all IxI and IxE addresses (34,391 and 93,061) | this PR | `figures/non_interacting_dist_from_root.png` |
| A10 duplicate wallet row | New (row-level leakage) | Code done | b2904f4 | `00` quality check: 0 duplicates; build asserts uniqueness |
| A11 model nondeterminism | New (R1 reproducibility) | Code done: XGBoost `n_jobs` pinned to 4 (b2904f4); LightGBM `force_col_wise` and `deterministic` (`sp.LGBM_REPRO`) | b2904f4, ea602c6 | Findings E; `10` checks that `06`'s single models equal `03` and `04` |
| A12 44 hand-added labeled addresses | New (possible label leakage) | Code done: hand step replaced by a two-step labeling rule (public lists, then an Etherscan check of every other funder of ≥ 50 interactors; 36 checked, 10 labeled). Pre-snapshot label vintage as sensitivity | 25ef316, b1594ce, 91cfd02 | `data/20260930_etherscan_service_labels/etherscan_lookups.csv`; `07`; findings E |
| C `gini_coefficient` identically zero | R1 graph-feature formulas | Code done: formula corrected, then removed by the pre-specified rule (3 of 10 splits; 62 features) | b1594ce, 91cfd02 | `01_ablation_gini`; findings E |
| B10 repo text | R2 test isolation | Done: notebook headers, README benchmark and robustness tables from the final run | d84263a, 9558973 | `README.md` |
| A13 Random Forest baseline | R2 baselines | Code done: validation-only search (24 configurations after the extension), 3-seed model, 10 splits against a same-machine LightGBM | ce0a3c0, 3bbec2c | `13_random_forest_sybil`; findings E (2026-10-05) |
| A14 entity-level recall | R1 entity-level evaluation | Code done; run on arm64 predictions, flagged (do not match committed `03`/`04`/`06`) | e156d9d | `14_entity_level_recall`; findings E (2026-10-05) |
| A15 split by tree and bounty report | New (internal review) | Done with a warning: the 5 % component rule became a warning; union-split models and 10 splits run (union − tree test F1 −0.457) | e156d9d, 3bbec2c | `15_split_tree_report`; findings E (2026-10-05) |
| A16 label robustness: initial list | R1 label methodology; R2 positive-unlabeled | Code done: list verified against the 24 May 2024 Wayback capture (byte-identical) and 5 copies; `16` run | 5888aa3 | `review_support/initial_list/`; `16`; findings E (2026-10-06) |
| A17 all models over 10 splits; significance; cost | R1 significance and cost | Code done | 5888aa3 | `17`; README model table |
| A18 mimicry stress test | R1 evasion | Code done; feature cost classes tentative (authors to confirm) | 5888aa3 | `18` |
| A19 provision-network ablation by category | R2 novelty and ablation | Code done | 5888aa3 | `19` |
| A20 temporal holdout | R1 drift | Code done | 5888aa3 | `20` |
| A15b report-mates in training | R2 relational leakage (report level) | Code done: seen minus held out, test F1 +0.231 ± 0.084 (15 of 15) | ef82c78 | `22`; findings E (2026-10-06) |
| A23 revision figures and tables | R1 verifiability; manuscript | Code done: 5 figures (PDF and PNG), 6 booktabs tables and blend-weight stats; follow-up fixes (T1 without the weighted average, T6, F2B AUROC) from `results/` only | 4a84790, 7648749 | `23`; `figures/rev_*`, `tables/` |
| A24 mimicry controls | R1 adversarial evasion | Code done: single-feature, strict-cheap, structural-only and random-k controls; reproduces `18` | 9b87db5 | `24`; `figures/rev_mimicry_controls`, `tables/rev_mimicry_controls.tex` |
| A25 paper tables and cited numbers | R1 verifiability; manuscript | Code done: 10 tables and 199 named numbers from `results/`; T6 four decimals; controls table k = 7 | b8747c0 | `25`; `tables/rev_*.tex`, `tables/rev_text_numbers.json` |
| A21 tie-out for 13–25 | R1 verifiability | Code done; renamed `99_tie_out_revision` (runs last); 82 of 83 checks pass (14's inputs fail until it is rerun on the machine of the committed results) | c3d7217, 3c2afe5, 2c4b8e5, 42d4cf5, 201c9b4 | `99` |
| A22 data-release documents | R1 release | Drafts: `DATA.md`, `CITATION.cff` (validated); no LICENSE; Zenodo steps below | this PR | `DATA.md`, `CITATION.cff` |
| A26 single-platform rerun of 13–25 and 99 at 68e8d63 | R1 verifiability | Done: x86_64 Linux, Xeon @ 2.80 GHz, 4 cores, Python 3.11.15, pinned packages; `03`/`04`/`06` reproduce the committed results; `14` on regenerated predictions (`predictions_match_committed_results: true`); `99` 83 of 83 | 412db11 to b050c3e | `logs/rerun_x86_68e8d63/REPORT.md`; findings E (2026-10-09) |

## A. Code and experiments

- [ ] **A1. Relational leakage from the random split.**
  - Where: 00_data_pipeline (`train_test_split(..., stratify=y)`), duplicated in 01, 02, 04.
  - Problem: Wallets from the same gas provision tree land in both train and test. Provider and tree features (`provider_*`, `gini_coefficient`, `branching_factor`, `tree_size`, etc.) are identical within a tree, so test performance partly measures recognition of trees seen in training.
  - Fix: Use a stratified group split (`StratifiedGroupKFold`).
    - The group is the gas provision tree, identified by the root of the unlabeled part of the provision chain. Walk `tfm` until the next hop is a labeled anchor or there is no provider; the last unlabeled node is the root.
    - IxL and No-provider wallets are roots of their own trees (shared only with wallets they fund). A CEX is not an operator.
    - Do not key on the immediate provider. A CEX hot wallet would become one giant tree.
  - Done when: The split code asserts that no tree spans two partitions.
- [ ] **A2. Leakage check is too narrow.**
  - Where: 00, "No leakage ✓" cell (checks row-index overlap only).
  - Fix: Add a tree-overlap check. Report the count of test Sybils sharing a gas provision tree with a train Sybil, under both the old and new splits.
  - Done when: Both counts appear in the notebook output and in the paper.
- [ ] **A3. Blend weight selected on test.**
  - Where: 04, weight-sensitivity cell. It scores `f1_score(y_test, ...)`, plots "Test F1", and outputs a best weight of 0.16.
  - Fix: Score the sweep on validation, relabel the axis, and re-derive the weight.
  - Done when: The weight chosen on validation matches the paper text and the code comment.
- [ ] **A4. Hyperparameter search not in repo; tuning tables show test F1.**
  - Where: Parameters are hard-coded as "found via grid search"; no search code exists.
  - Fix: Commit a tuning notebook that selects on validation only. Evaluate test once, for the final configs.
  - Done when: The notebook is committed, and Tables 7 and 8 are regenerated from it.
- [ ] **A5. Feature selection procedure undocumented.**
  - Where: `FEATS` (63 features) in 00. EDA cells compute label correlations on the full df, including test.
  - Risk: If the 63 were chosen by label correlation or importance on the full data, that is leakage.
  - Fix: Document how the 63 were chosen. If the choice was data-driven, redo it on train only. Move label-based EDA to the train split.
  - Done when: The selection method is stated in Methods.
- [ ] **A6. Post-snapshot reference data.**
  - Where: labeled addresses (2024-12-14), CEX list (2024-10-13), cex_dex_indegree (2025-02-08, date filter unknown).
  - Fix:
    - Verify that the cex_dex query filters `<= 2024-05-01`.
    - For the labeled list, use a May 2024 snapshot if one is available. Otherwise state the date and argue the effect is limited to labels, not behavior.
  - Done when: Methods gives a date for every data source.
- [ ] **A7. Provision graph query has no date filter.**
  - Where: `data/20241114_gas_provision/readme.txt` SQL (first-funding tx from all traces).
  - Fix: Add `block_timestamp <= '2024-05-01'` and rerun, or report the max provision timestamp in the network to show it predates the snapshot.
  - Done when: The filter or the max timestamp is reported.
- [ ] **A8. Report both splits.**
  - Fix: Report random-split and group-split results side by side. A temporal split (train on wallets funded earlier, test on later) is optional but strong.
  - Done when: A new table exists, with discussion of the gap between splits.
- [x] **A9. Descriptive figure drawn on val+test.** Redrawn on all addresses by `12_figures`.
  - Where: `non_interacting_dist_from_root` figure (caption says combined val+test).
  - Fix: Redraw on the full dataset (it is descriptive and uses no labels), or on train only.

## B. Manuscript language

- [ ] **B1. Sec 3.3 wrong rationale for upsampling.**
  - Current text: "This approach avoids the label leakage that would arise from cross-validation when minority-class examples are replicated many times in training."
  - Problem: Cross-validation is fine if upsampling is done inside each training fold.
  - Rewrite: Upsampling is applied after splitting, to the training partition only, so duplicated Sybil rows never appear in validation or test.
- [ ] **B2. Sec 4.1 test isolation claim.**
  - Current text: "the test set is accessed only for final reporting."
  - Fix: Keep this sentence only after A3 and A4 make it true.
- [ ] **B3. Sec 4.1 and Sec 12 circular overfitting argument.**
  - Current text: "The validation-to-test gap ... below 0.01, confirming that validation-based decisions did not over-fit."
  - Problem: Under a random split, validation and test share gas provision trees, so both are inflated equally and the gap proves nothing.
  - Fix: Remove, or restate under the group split as "consistent with."
- [ ] **B4. Sec 4.4 blend weight description.**
  - Current text: "selected by optimizing F1 on the validation set ... optimum falls near 0.34."
  - Problem: Contradicts the code, which sweeps on test with an optimum of 0.16.
  - Fix: Replace with rerun numbers from A3.
- [ ] **B5. Sec 4.6 overclaims no leakage.**
  - Current text: "Consequently, no Sybil label leakage occurs across the split boundary."
  - Rewrite: Labels do not enter feature computation, but features are relational. Wallets sharing a gas provision tree share feature values, so the split is grouped by gas provision tree (A1). Name this relational or tree-level leakage explicitly.
- [ ] **B6. Tables 7 and 8 captions.**
  - Current text: "Metrics on held-out test set."
  - Fix: Selection column is validation. Test is either dropped or labeled "for reference, not used in selection."
- [ ] **B7. Sec 3.2 snapshot statement.**
  - Current text: "prior to the snapshot date of May 1, 2024."
  - Fix: This holds for the Flipside features. Add per-source dates, including the post-snapshot reference lists (A6).
- [ ] **B8. Sec 4.1 split description.**
  - Current text: "stratified by class."
  - Fix: Change to "stratified group split," and define the group key.
- [ ] **B9. Abstract and Sec 7 "held-out test set."**
  - Fix: Say "group-held-out test set" so the claim is precise.
- [ ] **B10. Repo text.** Fix all of these:
  - 04 comment "tuned on validation set".
  - README benchmark table claims ensemble F1 0.741, but the notebook output is 0.738.
  - The "No leakage ✓" wording.
  - Add the group split to README "Key design decisions".

## C. Knock-on: the rerun changes every number

- [ ] Regenerate all results from one tagged commit, and cite the commit hash in Data availability. This covers:
  - Abstract.
  - Contributions list.
  - Tables 1 through 12 and all figures.
  - Sec 4.4 disagreement count (649).
  - Conclusion (0.720 → 0.735, 0.723 → 0.739).
- [ ] Table 5: IxI and IxE counts are swapped relative to the code. Add the "No provider" category (676 addresses).
- [ ] Table 12: Sybil counts per category from the final run (labeling rule): IxL 15,921; IxI 1,566; IxE 698; No provider 26 (paper: 16,241 / 1,534 / 436).
- [ ] Table 2 says `min_child_weight = 1`, while Sec 9 says mcw = 10. The validation search selects 1 (mcw 10 is fourth); use the `02` configuration throughout.
- [ ] Feature importance: fixed in 06dbd82. The XGBoost notebook (old `01`) reported average gain per split and the LightGBM notebook (old `02`) total gain, both from the last seed and labeled "Gini gain"; both now report total gain averaged over the three seeds. Proposed for the paper: SHAP on the test partition (`09`) in place of Tables 10 and 11 (ledger N13).
- [ ] "Share only 2 of top-15": with the same measure for both models (total gain, mean of 3 seeds, group split), XGBoost and LightGBM share all 15 of their top-15 features. The low overlap in the submission came from comparing two different gain measures.
- [ ] Dataset size: 434,786 addresses (18,211 Sybil; 416,575 non-Sybil) after removing the duplicated wallet (A10), not 434,787.
- [ ] Table 5 from the final run (labeling rule): IxL 306,658 (70.53 %), IxE 93,061 (21.40 %), IxI 34,391 (7.91 %), No provider 676 (0.16 %). The IxI/IxE swap must be fixed.
- [ ] `gini_coefficient` was identically 0 up to rounding (max |value| 9.9e-16 over all wallets): the original formula `(n-1)/n * (1 - sum(sorted(x)/sum(x)))` always gives (n-1)/n × 0. Table 11 ranked it with importance 125e-4, which was splitting on floating-point noise. **Fixed** (authors chose to fix, 2026-09-30): `sybil_pipeline._gini` computes the standard Gini coefficient of the non-root provision amounts in the tree, G = 2·Σ i·x₍ᵢ₎ / (n·Σx) − (n+1)/n, 0 for fewer than two amounts. In the final run (0d9eacc) it is nonzero for 19.4 % of wallets (IxI mean 0.345, IxE 0.332, IxL 0). Over all wallets the Sybil mean (0.049) is below the non-Sybil mean (0.101), but outside IxL, where the feature is nonzero, the order reverses (0.387 vs 0.333), so the data do not support the manuscript's claim that a high Gini coefficient characterizes star-like Sybil funding, nor its opposite. (An earlier count, made before the final run, gave 20.4 %, 0.051 and 0.108.) **Then removed** by the rule fixed in advance (findings E, 2026-10-01): removing it lowered validation F1 on 3 of 10 group splits, short of the 8 required. Manuscript: drop it from the feature list, Table 11 and Sec 5.2's sentence; the response letter explains the zero values, the correction and the test.
- [ ] `total_gas` is the total ETH provisioned within the wallet's gas provision tree (from the tree featurization), not "cumulative ETH gas consumed by an address" as Sec 5.3.1 and Appendix A say. Fix the definitions; R1 asks for formulas of graph features, and `sybil_pipeline.provision_features` is the reference implementation.
- [ ] The committed notebook outputs at 11366f4 gave LightGBM F1 0.7371 and LR AP 0.159, not the paper's 0.739 and 0.094. Superseded by the rerun, but the response letter should not quote the old values.
- [ ] Tables 7 and 8 "baseline" rows (default parameters) and the "directed grid search over 15 configurations" have no code in the repo. Replace with the `02_hyperparameter_search` results (validation only, full grids).
- [ ] Contribution 5 (cross-model ensemble): confirmed in the final run. Validation selects XGBoost weight 0.06 under both splits; on the group split the blend (F1 0.717) does not beat LightGBM alone (0.719). Restate (proposed text in ledger N12); the title names "Cross-Model Ensemble" (author decision).

## D. Response letter mapping

| Reviewer point | Items |
|---|---|
| R1: temporal cutoff, future-information leakage | A6, A7, A12, B7 |
| R1: search space, optimization method, validation scheme | A4, B6 |
| R1: code and data release, verifiability | A7 (tree features now computed from repo data), A11, `10` provenance checks |
| R1: formulas for graph-based features | C (`gini_coefficient`, `total_gas`); `sybil_pipeline.provision_features` |
| R1: precision/recall per category; FPR at operating threshold | `10` per-category and FPR tables |
| R1: ablations of tuning choices | `02` marginal-effect table |
| R1/R2: which features matter; graph features | `01` (Gini), `08` family ablation, `09` SHAP by category |
| R2: relational leakage from random split | A1, A2, A8, A10, B5, B8, B9 |
| R2: test isolation contradicted by manuscript and repo | A3, A4, B2, B3, B4, B6, B10 |
| R2: baselines | A13 (`13`) |
| R1: entity-level evaluation | A14 (`14`) |
| New (internal review): report-level dependence | A15 (`15`) |
| R1: label methodology; R2: positive-unlabeled labels | A16 (`16`, `review_support/initial_list/`) |
| R1: significance of model differences; cost | A17 (`17`) |
| R1: evasion once features are published | A18 (`18`) |
| R2: novelty of the graph features; ablation | A19 (`19`) |
| R1: concept drift | A20 (`20`) |
| R1: release and verifiability | A21 (`99`, was `21`), A22 (`DATA.md`, `CITATION.cff`, Zenodo) |
| R2: relational leakage, report level | A15 (`15`), A15b (`22`) |
| R1: adversarial evasion (strict-cheap, random-k and structural-only attacks) | A18 (`18`), A24 (`24`) |

## E. Findings log

Measured facts, with the commit they were measured at. Append; don't rewrite.

**2026-09-30, @ 11366f4 (baseline, before any fix).** Environment: `requirements.txt`.

- Baseline reproduces: 434,787 addresses, 18,211 Sybil (4.19%). Split sizes match Table 1 (213,045 / 91,305 / 130,437 before upsampling).
- Taxonomy from code (confirms C, Table 5 swap): IxL 302,796 (15,886 Sybil), IxE 96,277 (707), IxI 35,038 (1,592), No provider 676 (26).
- A1 gas provision trees as defined above: 367,923 trees; 22,912 with more than one wallet; largest tree 2,103 wallets. Only 2,359 of 18,211 Sybils sit in multi-wallet trees, because 87% of Sybils are IxL singletons.
- A2 under the current random split: 20,891 of 130,437 test addresses share a gas provision tree with a train address; **608 of 5,463 test Sybils share a tree with a train Sybil.**
- A7: the latest `first_gas_provision_time` for any L0 interactor is 2024-05-01 23:56:47 UTC, which is on the snapshot date. 163 interactor rows fall on May 1 itself. The full network file (which includes upstream non-interactor providers) runs to 2024-10-31 and has 460 rows after 2024-05-01 00:00. Those upstream rows can affect `chain_length`, `interactors_in_chain`, and the A1 group root. The tree features (`20241117_tree_features`) were built from this same network, so check their inputs too.
- A6: `data/20250208_cex_dex_indegree/readme.txt` SQL filters `block_timestamp <= '2024-05-01'`. That sub-item is verified.
- A5: `legacy/20250519 XGBoost Sybil Detection.ipynb` (the 2025 paper's code) also hard-codes the 63-feature list. The repo holds no selection code. The procedure must come from the authors.
- Caveat for the discussion section: an IxL wallet starts its own gas provision tree (it joins no group upstream of the labeled funder), so an operator who funds wallets through separate CEX withdrawals is invisible to the provision-graph grouping. The group split removes the leakage the provision graph can see, and no more.

**2026-09-30, @ fd6411f (group split added; checkpoint before the temporal fix).** Runs in `results/` were not committed; they are superseded by the final run.

- Random split through `sybil_pipeline` reproduces the original `splits.npz` bit for bit.
- LightGBM on the random split reproduces the committed 11366f4 output exactly (F1 0.7371). XGBoost does not (F1 0.7329 vs 0.7356; best iterations 465/465/516 vs 493/449/480): XGBoost `hist` depends on the thread count, and the original ran on a different machine. Fixed from b2904f4 by pinning `n_jobs` (A11).
- Group split, test F1 / AUROC / AP: XGBoost 0.707 / 0.968 / 0.767; LightGBM 0.723 / 0.971 / 0.778; LR 0.234 / 0.833 / 0.157. Random split: XGBoost 0.733 / 0.974 / 0.796; LightGBM 0.737 / 0.976 / 0.801; LR unchanged. The linear model does not move, consistent with the tree models isolating feature values shared within a gas provision tree.
- Blend weight chosen on validation: 0.00 on the group split (the ensemble is LightGBM alone), 0.26 on the random split.
- 04's individual models equal 01 and 02 exactly under both splits.

**2026-09-30, temporal and data audit (R1), @ b2904f4.**

- Snapshot boundary: the latest `latest_l0_tx_time` is 2024-05-01 23:59:58 UTC, so the L0 snapshot table covers all of May 1. Ethereum transaction features cut at 2024-05-01 00:00 UTC (`<=` for outgoing, `<` for incoming; see the L0 query). CEX/DEX in-degree cuts at `<= 2024-05-01`.
- Provision network: 296 edges dated at or after 2024-05-02 00:00 UTC, all between 2024-08-17 and 2024-10-31, overlapping the Sybil list process (list snapshot 2024-09-15). None enters an interactor and none creates an `is_provider` flag; 295 join two unlabeled addresses. Effect before the fix: one IxE non-Sybil wallet's `chain_length`, gas provision tree, and tree features. Fixed by the cutoff in `load_provision`.
- Tree features: the original precomputed file came from `data/20241117_tree_features/20241117 Gas Provision Featurization.ipynb`, run on the unfiltered network with inputs outside the repo. Ported to `sybil_pipeline.provision_features`. Against the original file on 434,110 wallets: provider fan-out, max and min amounts match 100 %; tree features match on all but 13 or 14 wallets, which sit in one 14-node tree that the pipeline's labeled list splits, plus the one cutoff wallet; `provider_is_star_like_attack` differs on 9 wallets for the same labeled-list reason; provider total and average amounts differ on the 16,101 wallets of one provider because the original double-counted the duplicated wallet. Skewness uses the scipy < 1.9 rule (0 for near-constant data), which the original file reflects.
- Duplicate wallet (A10): `0x3aecba06e531a982cfdba16f0589ece5dc200fa9` appeared twice in the precomputed file, so it appeared twice in the master table (non-Sybil, IxL). 00's own quality check reported "Duplicate addresses : 1 … Issues found". Under the random split the two copies could land in train and test.
- Labeled addresses (A6): compiled Oct–Dec 2024 from Flipside L0 address labels, hildobby CEX list, BigQuery and Dune contract lists, dawsbot and brianleect label sets, plus 44 hand-added addresses (`data/20241214_labeled_addresses/readme.txt`). None of the 44 is on the Sybil list. One labeled address in the network is on the Sybil list (`0x3df1…9159`); it funds no other address, so it affects no other wallet's features.
- A12 (open): the 44 hand-added addresses directly fund 3,153 interactors, of which 10 are Sybil (0.32 % vs 4.19 % overall). Labeling them changes those wallets' `provider_is_labeled`, `provider_is_star_like_attack`, chain, and tree features. If they were identified from public identity sources (explorer tags, exchange documentation), this is not leakage; if Sybil rates influenced the choice, it is. Ask the authors; a sensitivity run without the 44 would settle it empirically.
- `total_gas` does not exist in the L0 query; it comes from the tree featurization (see C).

**2026-09-30, A12 resolved (hand-added labeled addresses).** Evidence in `review_support/`.

- 20 of the 44 hand-added addresses are also in the hildobby CEX list (`data/20241013_hildobby_cex_evms`, loaded by the same build script; verified by exact address match): Shakepay, MoonPay, Voyager, 13 ShapeShift. Their labels never depended on the hand step. The remaining 24 are "hand-only"; 15 affect any feature, funding 653 interactors in total.
- Etherscan name tags for the 24, read from each page on 2026-09-30 (`etherscan_lookups.csv`, `lookup_reason = hand_added_2024`): 20 carry a public entity tag, mostly bridges and relayers (ZigZag, Owlto, Umbria, Chaineye, Synapse, Hyphen, Multichain, Allbridge, Argent), plus Newton (exchange), an OKX deposit address, Union Chain, Kuailian wallet contract, Layer Zero Executor, Seaport 1.6, EtherDelta 2. Four have no entity tag (`0x6b8f…`, `0x0000…055a` with a personal ENS name only, `0xffff…61ef`, `0x0000…29d8`); together they fund 8 interactors.
- Author's account: the addresses were checked on Etherscan after anomalies in exploratory, label-free analytics and found to be entities the hildobby CEX list does not cover. The tags are consistent with that.
- Sensitivity, 5 group-split seeds (`hand_only_seed_band.csv`; LightGBM, 05 configuration from 82148f1, pre-Gini-fix features): removing the 24 hand-only labels changes test F1 by −0.002 ± 0.014 (mean ± SD across seeds; per seed −0.023, +0.006, +0.008, −0.009, +0.008), AUROC by −0.002 ± 0.002, AP by −0.004 ± 0.011. Indistinguishable from zero. The earlier single-seed −0.017 removed all 44, including the 20 hildobby addresses, and is superseded.
- Side finding for R1 (significance) and the ensemble claim: with the labels, LightGBM test F1 ranges 0.702 to 0.726 across the 5 group-split seeds (SD ≈ 0.009). Model differences smaller than that are within split noise.
- Completeness: resolved by the labeling rule below (2026-10-01).

**2026-10-01, pre-specified decision rule for `gini_coefficient` (recorded before the ablation is run).**

The corrected `gini_coefficient` is the only feature whose definition changed in the revision, so it alone gets a keep-or-drop test. Train LightGBM (configuration from the last completed search) with and without it on 10 stratified group splits (seeds 42, 1 to 9), on the final feature table (corrected Gini, new labeling rule). Compare validation F1 per split; the test set is not used. Retain the feature only if removing it lowers validation F1 on at least 8 of the 10 splits (one-sided sign test, p ≈ 0.055); otherwise remove it and use 62 features. Agreed with the first author (Paven) and Scott on 2026-10-01.

**2026-10-01, labeling rule complete (A12), @ 25ef316 and b1594ce; tables reformatted in the next commit.**

- Rule, applied identically to every address and without Sybil labels: (1) public label datasets: the consolidated 2024 label file minus its 44 hand-added addresses, plus Dune Spellbook's CEX, DEX and bridge lists at pinned commits (`data/20260128_dune_spellbook_labels`); (2) an Etherscan check of every other address that funded at least 50 LayerZero interactors, labeled if its name tag identifies a shared service. Exchange deposit addresses, personal ENS names and untagged addresses are not labeled.
- Primary labels use the current list versions; labels known before the snapshot (2024-05-01) are the `presnapshot` sensitivity in `07`. Reason (authors, 2026-10-01): an address does not become a service after the snapshot; it is identified as one later, and the pre-snapshot lists miss services that were active then.
- Etherscan check: 65 addresses read on 2026-09-30 and 2026-10-01 (`review_support/etherscan_lookups.csv`: the 24 hand-only additions, the top 30 unlabeled funders by network fan-out, and the remaining 11 funders of 50+). 36 are in the rule's scope; 10 are labeled: Relay Solver (2,046 interactors), Layerswap 1 (656), ZigZag Bridge (190), Shakepay 7 (152), Shakepay 6 (150), Union Chain (130), Owlto Finance Bridge 2 (81), Umbria Narni Bridge 2 (73), Owlto Finance Bridge (56), DeGate Hot Wallet (52). 25 have no entity tag and one has a personal ENS name only; none is labeled. Four tagged exchanges found in the top 30 (BingX 42, Newton, Newton 5, Shakepay 9) are already in the current Spellbook CEX list. `build_etherscan_lookups.py` asserts that every address in scope was looked up.
- Each tag was matched to its address through the address printed on the pasted Etherscan page, not the link opened. A reviewer can re-check any row at `etherscan_url`.
- The 24 hand-only additions under the rule: 5 stay labeled (Newton via Spellbook; ZigZag, Union Chain, Owlto 2 and Umbria 2 via Etherscan; together 589 funded interactors); 19 are below the threshold and unlabeled (64 funded interactors).
- The 2024 build code also labeled one address directly (`0x9241…3e64`, removed in 25ef316). It funds 9 interactors, is itself an interactor and is in no public list, so it is unlabeled under the rule.

**2026-10-01, `gini_coefficient` decision (pre-specified rule above), @ b1594ce.**

- `08_ablation_gini`: removing the corrected feature lowered validation F1 on 3 of 10 group splits (one-sided sign test p = 0.945); mean change when removed +0.0011 F1, +0.0010 AP. Decision: remove. `sp.FEATS` is now 62 features; `sp.FEATS_SUBMITTED` keeps the submitted 63 in their order for `08`.
- The labeled set at b1594ce is identical to the reformatted tables (same 36 rows in scope, same 10 labeled), so the decision applies to the final feature table. The final batch reruns `08` from the final commit.

**2026-10-01, full batch @ 91cfd02 rejected by `06`; LightGBM made deterministic.**

- `06` stopped at its check that 04's individual models equal 01 and 02: under the random split, 04's LightGBM seed 123 stopped at 1,347 rounds and 02's at 1,324 (test F1 0.74101 vs 0.74105). Cause: LightGBM picks col-wise or row-wise histogram construction by a timing test at startup, which depends on machine load; forcing col-wise reproduces 02's 1,324 rounds and forcing row-wise reproduces 04's 1,347. Within one mode, predictions still differed in the last bits between runs (thread-order summation).
- Fix: `sp.LGBM_REPRO = dict(force_col_wise=True, deterministic=True)` in every LightGBM model (02, 04, 05, 10, and `sp.lgbm_fit_eval` for 07 to 09). Two runs then give bit-identical test predictions, also with 2 threads instead of 4, at the same speed (about 50 s per fit).
- `08` at 91cfd02 reproduced the b1594ce decision run exactly (all 10 splits). The whole batch is rerun from the fix commit; the 91cfd02 numbers are superseded.
- `09` gains one report-only row, all three provision-network families removed together, because single-family removals of correlated families can understate what the network adds.

**2026-10-01, final run.** Code commits: ea602c6 for 00, 03, 05, 07 to 10; 06dbd82 for 01, 02, 04 (operating-point and importance fixes, which change no metric); `06` passes every provenance and consistency check at 06dbd82.

- Data (labeling rule): 434,786 addresses, 18,211 Sybil; 4,308 labeled addresses in the network; 371,764 gas provision trees (22,952 multi-wallet, largest 1,627). Taxonomy: IxL 306,658, IxE 93,061, IxI 34,391, No provider 676.
- A2: random split, 599 of 5,463 test Sybils share a gas provision tree with a training Sybil (19,884 of 130,436 test rows share a tree with a training row); group split, 0.
- Search (validation only): XGBoost lr 0.05, depth 8, subsample 0.7, colsample 1.0, mcw 1 (val F1 0.703); LightGBM 127 leaves, lr 0.05, subsample 0.7, λ 1 (val F1 0.707).
- Test, group / random: XGBoost F1 0.720 / 0.741, AUROC 0.966 / 0.974, AP 0.764 / 0.798; LightGBM 0.719 / 0.742, 0.968 / 0.976, 0.770 / 0.802; ensemble 0.717 / 0.743; LR 0.241 / 0.234. The tree models lose 0.021 to 0.025 F1 and 0.033 AP under the group split; LR does not.
- Per category (LightGBM F1, group / random): IxI 0.47 / 0.85, IxE 0.29 / 0.54, IxL 0.75 / 0.74. The leakage sits in the categories with multi-wallet trees.
- Blend weight 0.06 (XGBoost) under both splits; disagreement at 0.5: 703 (group), 579 (random). In the nondeterministic 91cfd02 run the random-split weight was 0.58: the validation F1 curve is flat, so the weight is not a stable quantity.
- Split noise: LightGBM test F1 over 10 group splits 0.713 ± 0.006 (`09`).
- `07`: pre-snapshot labels (4,115 vs 4,308) change 10,575 wallets' features; test F1 −0.005 ± 0.007, validation F1 +0.009 ± 0.012 (opposite signs, both within split noise). Sybil share of wallets funded by CEX addresses by date added to the list: before the snapshot 5.30 % (264,646 wallets), between snapshot and Sybil list 2.40 % (1,082), after the Sybil list 0.92 % (1,086). Later labels do not concentrate Sybils.
- `09` (test F1 change vs all 62 features): without LayerZero transactions −0.066, Ethereum transactions −0.012, gas provider −0.005 (lower on 9 of 10 splits), gas provision tree −0.001, provision chain −0.002; all three provision-network families together −0.008 (lower on 7 of 10; validation F1 lower on 3 of 10). Alone: LayerZero 0.672, Ethereum 0.625, gas provider 0.438, gas provision tree 0.101, provision chain 0.101 (AUROC 0.61).
- `10` (mean |SHAP| summed by family, test partition): LayerZero transactions 4.88, Ethereum transactions 2.44, gas provider 0.84, gas provision tree 0.49, provision chain 0.04. Funding-tree features weigh more within IxI (1.35) and IxE (0.98) than IxL (0.25).

**2026-10-01, Blockscout cross-check of the Etherscan tags (A12).** `review_support/blockscout_crosscheck.py`, read 2026-10-01; a check only, the labeling rule is unchanged.

- All 65 addresses looked up. In the rule's scope (36), the two explorers agree on 34: 9 named on both with the same name tag (Relay Solver, Layerswap 1, ZigZag Bridge, Shakepay 6 and 7, Union Chain, Owlto Finance Bridge and Bridge 2, Umbria Narni Bridge 2) and 25 named on neither.
- Two differ. DeGate: Hot Wallet (52 interactors, labeled) has an Etherscan tag but none on Blockscout. `0x08b0…8b06` (66 interactors, not labeled) has no entity tag on either explorer, but Blockscout shows its verified contract name, WethUnwrapper. A contract's own name is not a third-party entity tag, so the rule leaves it unlabeled; the authors may want to note it.
- Outside the scope (29 addresses), the two agree on all: 19 named on both, 10 on neither.
- Caveat: some Blockscout tags come from shared public sources (for example the Open Labels Initiative), so agreement shows consistency between explorers, not fully independent confirmation.

**2026-10-01, automated labeling rule as a robustness check (A12), @ cf244f3.** `review_support/build_blockscout_rule.py` and `sensitivity_blockscout_rule.ipynb`.

- Alternative rule, fixed before the run: replace the Etherscan step with Blockscout's tag service for every funder of at least 2 addresses (13,992; 1,726 tagged), labeled when it carries a service category tag (Exchange, Hot Wallet, Bridge, DEX, Router, Fiat Gateway, Layer 2, Derivatives, Payments, OTC) and no Deposit Address tag. No interactor threshold, no human step. Public lists unchanged.
- Why it is not the primary rule: Blockscout's category tags miss 6 of the 10 services the Etherscan step confirmed (Layerswap 1, Shakepay 6 and 7, Union Chain, Umbria Narni Bridge 2, DeGate), and its free-text name tags mix services, individuals and scam addresses, so a complete automated mapping would need hand classification of about 1,250 name tags.
- Result: the labeled sets differ by 15 addresses (6 only primary, the ones above; 9 only automated: small exchange hot wallets and bridges funding 2 to 32 addresses, e.g. Chaineye Mini Bridge, changehero, ACE). 3,476 wallets' features change (54 Sybil). Automated minus primary over 10 group splits: test F1 −0.003 ± 0.010 (lower on 6 of 10), AP −0.005 ± 0.012, AUROC −0.002 ± 0.003, validation F1 +0.007 ± 0.012. Within split noise, the same pattern as the label-vintage check.

**2026-10-02, restructure (Scott's PR review).** Notebooks renumbered to run order (map above). Model training, evaluation, the blend weight, test metrics, operating points and gain importance are defined once in `sybil_pipeline.py`; the search and robustness notebooks use the same model definitions (`sp.xgb_model`, `sp.lgbm_model`). The ensemble notebook reads the XGBoost and LightGBM predictions instead of retraining. The model notebooks run on the group split only; the random split is trained only in `11_split_comparison`. Check before any rerun: the shared functions reproduce the ea602c6/06dbd82 group-split results exactly (all metrics, thresholds, rounds, blend weight 0.06, importances; maximum difference 0).
- Correction: earlier text said IxL wallets are singletons. They are not: 12,886 of 306,658 IxL wallets (4 %) are in multi-wallet gas provision trees, as roots of the wallets they fund; IxI ≈100 % (34,390 of 34,391), IxE 42 % (38,642 of 93,061). Earlier text used "cluster" and "funding group" for the gas provision tree; both are replaced throughout.

**2026-10-02, evidence for labels resting on the 2024 file alone (A12), @ e75cad4.** `review_support/build_2024_only_evidence.py`, Blockscout read 2026-10-02.

- Of the 4,308 labeled addresses in the provision network, 3,466 come only from the 2024 labeled-address file (not Spellbook, not the Etherscan step); they fund 33,103 addresses. The file survives only as the merged list of its six sources.
- Scope of the table: those that funded at least 50 interactors (the Etherscan step's threshold, same count): 32 addresses, 22,532 interactors. Earlier text said 49; that counted all addresses funded, not interactors.
- 27 of 32 carry a Blockscout entity tag (Stargate, Orbiter, Across, Aztec, Synapse, Rhino.fi, deBridge, THORChain, Ethermine, Nanopool, Tornado Cash pools, batch-send tools and others). Of the other 5, two are verified contracts whose code names a service (WooCrossChainRouterV2, Disperse), one is a Gnosis Safe, and two have no name or tag.
- For the paper: 6 of the 32 are batch-send tools (Disperse twice, Bulksender, CoinTool MultiSender, Multisender.app, GasliteDrop), funding 2,650 interactors. Labeling them ends the provision chain there, so wallets funded through the same tool are not linked into one tree. Consistent with the rule (shared services), but a reviewer may ask.

**2026-10-02, how much the split seed varies the group split, @ 08cc963.** `sp._group_partition` over split seeds 42 and 1 to 9.

- Share of wallets in the same partition as under seed 42 (mean over the 9 other seeds): single-wallet trees 0.374 (348,812 wallets; chance for 49/21/30 is 0.374), trees of 2 to 10 wallets 0.37, 11 to 100 wallets 0.43, over 100 wallets 1.000.
- The 34 trees over 100 wallets (9,020 wallets, 2.1 %; 97 Sybils, 0.5 %) land in the same partition for every seed: largest first into the partition with the most room, and `argmax` takes the first partition on ties (the largest tree, 1,627 wallets, is always in train). So the 10-split robustness checks do not vary their placement, and the split-to-split SD slightly understates the variation.
- Decision (Scott): keep the code; state this in the paper (ledger B13).

**2026-10-02, `review_support/` restructured (Scott's PR review).** `build_etherscan_lookups.py` and `etherscan_lookups.csv` moved to `data/20260930_etherscan_service_labels/`, next to the `service_labels.csv` they generate (the builder reproduces both files byte for byte). The rest is grouped into `blockscout_crosscheck/`, `blockscout_rule_sensitivity/` and `labels_2024_only/`, each with a README; nothing in the pipeline reads `review_support/`. Removed as intermediate (still in git history before this commit): `hand_labeled_addresses.ipynb` and its two sheets, `top30_unlabeled_funders.csv`, `hand_only_seed_band.csv`. `code_review/` removed: its copies matched the notebooks exactly. Paths in earlier findings refer to the old layout.

**2026-10-02, final run @ 21adb3c.** All notebooks `00` to `11` executed from a clean tree at 21adb3c (after the code review pass); `10_tie_out` confirms every result traces to that commit. Every reported number equals the previous run (ea602c6 / 06dbd82): same selected hyperparameters (74 and 48 configurations), test metrics, blend weight 0.06, 703 disagreements, boosting rounds; leakage 599 of 5,463 under the random split, 0 under the group split; Gini 3 of 10; label vintage −0.0047 ± 0.0070; families SD 0.0060, all network families −0.0079; per-category F1 identical. The supplementary `review_support/blockscout_rule_sensitivity` notebook also ran at 21adb3c: automated minus primary labels −0.0033 ± 0.0098 test F1, 3,476 wallets changed (same as before). Later changes are checked against 21adb3c (`sybil_pipeline.py`, `requirements.txt`, `data/`, notebook code cells); a change that passes leaves every result valid.

**2026-10-02, review edits to `sybil_pipeline.py` after the final run.** c5244f5 (docstring note on label vintage) and the rename of `provision_features`' parameter `labeled` to `anchors`. Checked against 21adb3c: the syntax trees are identical once docstrings are dropped and the parameter is renamed back, and the master table, labeled set and seed-42 group and random splits are identical cell for cell. Because the file's bytes changed, `10_tie_out` will accept the results only after a rerun at the final commit.

**2026-10-02, rerun @ 0d9eacc after the review edits, with `02` searched from scratch.** All notebooks `00` to `11` and `review_support/blockscout_rule_sensitivity` executed from a clean tree at 0d9eacc. Every result equals the 21adb3c run apart from the recorded commit. `02` was then rerun with no cached search scores (122 configurations, 7,168 s): every validation F1, AP, log loss, threshold and round count equals the committed search, and the selection is unchanged. `10_tie_out` at 6eeb76c accepts every result. Runtimes are in the README.

**2026-10-02, numbers in the manuscript draft that no notebook printed, @ 6eeb76c.** Every number in the current draft was checked against this run. Two were not printed by any notebook, and `10_tie_out` now prints both: chain length by class and category (Sybil: IxI 1,566 wallets, 8.663 hops; IxL 15,921, 1.000; IxE 698, 9.090; No provider 26, 0; all 1.968. Non-Sybil: IxI 32,825, 4.000; IxL 290,737, 1.000; IxE 92,363, 3.908; No provider 650, 0; all 1.880), and the precision of the top 8% of the cross-ensemble's flagged addresses (463 of 5,790; 0.981). Measured during the same check and not printed by a notebook:
- Validation vs test F1 of the 3-seed models at their validation thresholds: XGBoost 0.699 vs 0.720, LightGBM 0.708 vs 0.719, ensemble 0.708 vs 0.717. Under the group split test F1 is the higher one, by 0.009 to 0.021.
- Disagreement zone (703 addresses at threshold 0.5): correct at 0.5, ensemble 366, LightGBM 362, XGBoost 341 (as printed by `06`); at each model's own validation threshold, ensemble 371, LightGBM 423, XGBoost 452.
- Gain importance (mean of 3 seeds): `provider_is_null` ranks 57 of 62 for XGBoost and 62 for LightGBM; the two models' top 15 features are the same set.
- From `results/02_search_*.csv`: XGBoost defaults (lr 0.3, depth 6, no subsampling) validation F1 0.6907, selected 0.7032; LightGBM defaults (31 leaves, lr 0.1, no subsampling, no L2) 0.6871, selected 0.7067, stopping early at 2,059 of 5,000 rounds; LightGBM with L2 = 1 beats L2 = 0 in 22 of 24 paired configurations (mean +0.0036).
- The corrected `gini_coefficient` figures in section C now come from this run.

**2026-10-02, figures @ cae1b67.** `12_figures` draws every figure in the paper from this run into `figures/` under the manuscript's file names. Learning curve (one XGBoost, seed 42, selected hyperparameters): early stopping at 1,680 trees, as in `03`; validation F1 0.7032 at its F1-maximizing threshold, equal to the `02` search; training F1 0.9993 on the upsampled 1:1 training set. PR and ROC curves reproduce every AP and AUROC in `results/`. Depth distribution on all addresses (IxI 34,391, IxE 93,061; A9). The three provision-graph examples match the manuscript's address tables exactly: 0xf70768a8… (IxI), 0x92bb85ed… (IxE), 0x47127232… (IxE, Sybil). All 15 of the XGBoost and LightGBM top-15 features are shared, so the comparison figure shows every bar as shared.

**2026-10-05, Apple M5 (arm64) does not reproduce `03` and `04`.** Both rerun from a clean tree at d51af2c (pinned `requirements.txt`, Python 3.13.7, macOS arm64), to regenerate the `output/` predictions `14` reads; the committed `results/` were left unchanged.

- Data and split agree: 434,786 addresses, 18,211 Sybil, 371,764 trees; test 130,435 rows and 5,463 Sybils, 0 tree overlap.
- The models do not. XGBoost: test F1 0.7183 (committed 0.7200), validation threshold 0.5414 (0.5824), AP 0.7647 (0.7645). LightGBM: F1 0.7168 (0.7188), threshold 0.5748 (0.6001), AP 0.7697 (0.7696), rounds 1,224 / 1,267 / 1,225 (1,277 / 1,248 / 1,143). `06` on these predictions (ce0a3c0) selects XGBoost weight 0.00 (committed 0.06), so the ensemble equals LightGBM there.
- So `sp.N_JOBS` and `sp.LGBM_REPRO` make runs reproducible on one machine, not across machines. The differences are below the split-to-split SD (0.006). The README's "reproduce across runs and machines" is corrected. `10_tie_out` cannot detect this, because it checks code and data, not the platform.

**2026-10-05, A14 entity-level recall, `14` @ e156d9d.** The authors chose to run `14` on the arm64 predictions above, flagged: `results/14_entity_level_recall.json` records `predictions_match_committed_results: false`, the platform, and the metrics of the predictions used next to the committed ones. Numbers are LightGBM at its validation threshold (0.5748) on those predictions; XGBoost differs by at most 0.015 in any rate below. The acceptance checks pass against the predictions used: E1 Sybil-weighted recall equals address-level recall (0.7187) to 1e-9; every entity type sums to 5,463 test Sybils and 32,696.7 ZRO.

- E1, gas provision tree: 4,907 trees hold the test Sybils; 97.1 % are single-wallet trees, and 98.5 % of IxL test Sybils sit in single-wallet trees, so E1 is close to address level. Any-hit 0.765, majority-hit 0.764, full-hit 0.762; within-tree recall 0.764 unweighted, 0.719 weighted. 1,160 trees (23.6 %) keep at least half their allocation unflagged.
- E2, bounty report: 239 of 311 reports have test Sybils (median report: 31 % of its Sybils in test). Any-hit 0.703, majority-hit 0.552, full-hit 0.159; within-report recall 0.512 unweighted. 117 reports (49.0 %) keep at least half their allocation unflagged. Only 18 reports (20 Sybils) have every Sybil in test. On those, any-hit is 0.056, because they are small (16 of 18 have one Sybil).
- E3, reporter (descriptive): 132 reporters; any-hit 0.871, majority-hit 0.689, full-hit 0.121; 50 (37.9 %) keep at least half their allocation unflagged.
- Allocation: 37.2 % of the test Sybils' ZRO is on unflagged addresses, against an address miss rate of 28.1 %. The missed Sybils carry larger allocations than the detected ones.
- Caveat: a tree or a report is a proxy for an operator, not the operator. One operator can fund many wallets directly from an exchange, which yields many single-wallet trees. `allocation` is intended payout, not profit (it excludes costs).

**2026-10-05, A15 report-level dependence, `15` @ e156d9d: stopped for review.** No model was trained.

- Report overlap: under the random split (seed 42), 5,434 of 5,463 test Sybils share a Commonwealth report with a training Sybil. Under the tree split, 5,406 (seed 42), and 5,402.5 ± 22.3 over the 10 seeds. GitHub issue: 1,935 / 1,798 / 1,824.4 ± 21.8. Reporter: 5,461 / 5,445 / 5,444.5 ± 16.6. Under the tree split, 212 of the 239 reports with test Sybils also have training Sybils. The tree split removes tree overlap (599 → 0) but almost none of the report overlap.
- Union groups (tree ∪ report ∪ GitHub issue; GitHub issues never span two reports, so they add no links): 355,619 components (22,649 with more than one row). The largest has 13,346 rows and 8,260 Sybils (**45.4 % of all Sybils**), joining 6,651 trees through 57 reports. The second is a single report of 954 Sybils in 954 trees (5.2 %), so even a report-only grouping breaks the 5 % rule. Partition Sybil rates (4.19 % each) and sizes (49.0 / 21.0 / 30.0) pass. The union split has 0 tree and 0 report overlap (asserted).
- Stop condition failed: a component holds more than 5 % of Sybils. Status `stopped_for_review`. Parts C and D (models on the union split, 10-split comparison with `08`) were not run. The partitioner was not modified and no edges were dropped.
- Stricter variant (also group by reporter): the largest component has 16,840 rows and 10,071 Sybils (55.3 %). The partition Sybil rates then fail too (train 4.70 %, validation 3.69 %, test 3.69 %). Not run.
- For the authors: report-level dependence cannot be removed by a group split with this partitioner. Options include a leave-reports-out evaluation (for example, holding out whole reports below a size cap), restricting the test set to reports with no training Sybils, or a relaxed component rule. Each would be a new pre-specified design, not a change to this notebook.

**2026-10-05, A13 Random Forest baseline, `13` @ ce0a3c0 (Apple M5, arm64; 32 min).** Selected like `02`: validation only, test deleted before any fit, 12 configurations, 300 trees, no `class_weight`.

- Memory guard: one fully grown tree pickles to 1.13 MiB (`'sqrt'`) or 0.90 MiB (0.3), so 300 trees project to 0.33 GiB, well under 50 % of available RAM (5.1 GiB). `max_samples` stays unset.
- Search (validation F1): `max_features` 0.3, `min_samples_leaf` 1, `max_depth` None is selected (0.7044, AP 0.7508). That is above the selected XGBoost (0.7032) and below LightGBM (0.7067). Unlimited depth beats depth 24 in every pair (by 0.02 to 0.08); at unlimited depth, smaller leaves are better.
- Test (3 seeds, threshold 0.3567 from validation): precision 0.7266, recall 0.6982, F1 0.7121, AUROC 0.9710, AP 0.7686, Brier 0.0176, FPR 0.0115. Against the committed models: XGBoost F1 0.7200 / AP 0.7645, LightGBM 0.7188 / 0.7696, LR 0.2412 / 0.1656. Random Forest has the highest AUROC of the four and an AP between XGBoost and LightGBM. Its F1 gap to the boosted models (0.007 to 0.008) is about one split-noise SD. Per category, F1: IxL 0.751, IxE 0.277, IxI 0.186 (LightGBM, group split: 0.75, 0.29, 0.47); recall is low in IxI (0.103) at high precision (0.936).
- 10 group splits (model seed 42): test F1 0.7026 ± 0.0063, AP 0.7537 ± 0.0073, AUROC 0.9687 ± 0.0015, against `08` "All features" (LightGBM) 0.7131 ± 0.0060, 0.7607 ± 0.0060, 0.9682 ± 0.0018. Paired by split, Random Forest minus LightGBM: F1 −0.0105 ± 0.0071 (lower on 9 of 10), AP −0.0070 ± 0.0034, AUROC +0.0005 ± 0.0013. Caveat: `08` ran on the machine of the committed results and `13` on arm64; that alone moved LightGBM's seed-42 test F1 by 0.002 (above).
- For the manuscript: replace the prose dismissal with these results. A tuned Random Forest is a strong baseline, close to the boosted models on AUROC and AP, and behind LightGBM on F1 by about 0.01 over 10 splits. Boosting's advantage is real but small.

**2026-10-06, step 0 follow-up on A13 and A15, `13` and `15` @ 3bbec2c (Apple M5, arm64; 60 min and 13.5 min).** Both results now record the platform.

- A13, grid extended to `max_features` ∈ {`'sqrt'`, 0.3, 0.5, 0.7} (24 configurations; the 12 earlier ones resumed from `results/13_search_random_forest.csv`, same machine). The selection is unchanged: 0.3, leaf 1, depth None (validation F1 0.7044), ahead of 0.5 (0.7039) and 0.7 (0.7009). Test metrics are unchanged (F1 0.7121, AUROC 0.9710, AP 0.7686).
- A13, same-machine baseline: LightGBM (model seed 42, `sp.lgbm_fit_eval`) on the same 10 splits, 0.7133 ± 0.0086 test F1 (`08`, other machine: 0.7131 ± 0.0060; per split the two differ by +0.0002 ± 0.0044). Random Forest minus LightGBM, paired, same machine: F1 −0.0107 ± 0.0069 (RF lower on 9 of 10), AP −0.0074 ± 0.0028, AUROC +0.0005 ± 0.0012. This replaces the cross-machine −0.0105 ± 0.0071.
- A15, the 5 % component rule is now a warning (status `ok_with_warning`); the partition rate and size checks still pass. Composition: the two components over 5 % of Sybils (`BIG`: 14,300 rows, 9,214 Sybils, 50.6 %) land in the same place for all 10 seeds: the 13,346-row component (8,260 Sybils) in train and the 954-Sybil single-report component in validation. Under the union split (seed 42), 8,260 of 8,924 training Sybils come from one component spanning 57 reports; validation's Sybils are 25 % one report; test holds 99 reports and only 67 IxI and 106 IxE Sybils (tree split: 426 and 258).
- A15 Part C (seed 42, 3 seeds, parameters from `02`): XGBoost test F1 0.255, AP 0.220, AUROC 0.818; LightGBM 0.283, 0.224, 0.809 (tree split, committed: 0.720 / 0.764 / 0.966 and 0.719 / 0.770 / 0.968).
- A15 Part D (LightGBM, model seed 42, 10 splits, both splits trained here): union 0.256 ± 0.022 vs tree 0.713 ± 0.009 test F1. Union minus tree: F1 −0.457 ± 0.023, AP −0.564 ± 0.016, AUROC −0.199 ± 0.022; IxI −0.463 ± 0.093, IxL −0.476 ± 0.024, IxE −0.181 ± 0.062; lower on 10 of 10 for every metric. Same composition (each model scored on its test rows outside `BIG`): F1 −0.375 ± 0.022, AP −0.449 ± 0.015.
- Reading: a model trained on Sybils from a few reports does not find Sybils from other reports nearly as well, so the tree-split test score depends heavily on report-level similarity between train and test. The size of the drop is not a clean estimate of that dependence: the union split forces 93 % of training Sybils into one component of 57 reports, which no feasible report-grouped split avoids (one report alone holds 5.2 % of Sybils). The AUROC drop rules out a thresholding artefact. For the authors: this is the strongest robustness result so far and needs a sentence in the limitations, with a design that holds out reports at a controlled composition if the paper is to quantify it.

**2026-10-05 to 2026-10-06, A16 search for LayerZero's initial list (phase 1).** The original `initialList.txt` was in `github.com/LayerZero-Labs/sybil-report`, now deleted.

- GitHub: `gh api repos/LayerZero-Labs/sybil-report` and its forks endpoint return 404. `gh search repos "sybil-report"` and `"layerzero sybil"` (100 results each) and `gh search code "LayerZero-Labs/sybil-report"` were read. Four unrelated repos hold a 34,560,871-byte file with the same git blob (`0fc2e788…`): `psrvere/layerzero-sybil-checker` (`initialList.csv`, first committed 2024-05-18 13:44 UTC), `ibuyshite/LayerZero-Sybil` (`LO Report`, 2024-05-18 00:31), `tomkysar/sybil.rip` (`addys.csv`, 2024-05-18 02:10) and `shuail0/layerzero-sybil-query` (`initialList.csv`, 2024-05-20 08:35; its README calls it "the 800K Sybil data provided by LayerZero").
- Wayback Machine: a capture of the original page exists (`web.archive.org/web/20240524040829/https://github.com/LayerZero-Labs/sybil-report/blob/main/initialList.txt`, cited by `davidgasquez/sybil-detection-pond-challenge-tainted`), but the CDX API and the capture returned 429 (rate limit) to every request for over 25 minutes; the archived bytes were not compared.
- From the authors: a Google Drive zip whose `initialList.csv` (dated 2024-05-18 07:08) is byte-identical to the GitHub copies, and a Telegram `initialList.txt` with the same 803,093 addresses in the same order, 297 of them written in scientific notation by a spreadsheet. This repo's history also held `fcfs_provisional_list.csv` (efc1c50, deleted in e2718ac): a 151,972-row superset of `fcfs_list.csv` in the bounty format, not the initial list (none of its 188 extra addresses is an interactor).
- Verification: 803,093 unique lowercase addresses (801,932 20-byte, 1,161 32-byte, consistent with Aptos); 0 overlap with `fcfs_list.csv`; SHA-256 `1a643f70056fe0c0b38d4a13468e968a2de0bd18de6bc25ed7139d9fa691a7cc`. The authors accepted it as the original (2026-10-06); stored in `review_support/initial_list/` (Git LFS) with its provenance.

**2026-10-06, A16–A20 @ 5888aa3 (Apple M5, arm64; every comparison within a notebook is same-machine).**

- A16 (`16`, 21 min). 34,545 of 434,786 interactors are on the initial list, all labeled non-Sybil: 8.29 % of the 416,575 non-Sybils (IxL 29,646, 9.7 % of IxL; IxI 2,206, 6.4 %; IxE 2,669, 2.9 %; No provider 24); none is a positive. Seed-42 partitions: 17,145 / 7,085 / 10,315 of the train / validation / test negatives (8.4 / 8.1 / 8.3 %). Baseline LightGBM (3 seeds, trained here; equals the arm64 rerun of `04`): test F1 0.7168. At the validation threshold (0.5748), 625 of the 1,566 test false positives (39.9 %) are on the initial list; 941 legitimate-looking addresses are flagged; precision 0.715 becomes 0.829 if initial-list addresses count as Sybil (at 0.95: 65 of 373 false positives, precision 0.875 → 0.897). Variant A (initial-list negatives removed from training and validation; test unchanged): on the test set without initial-list rows, F1 0.760 → 0.787, AP 0.807 → 0.832, recall 0.719 → 0.786; on the full test set F1 drops to 0.648 (AP 0.457) because the model now flags initial-list addresses (7,784 flagged). Over 10 splits (model seed 42), Variant A minus baseline on the test set without initial-list rows: F1 +0.034 ± 0.010 (higher on 10 of 10), AP +0.028 ± 0.002, recall +0.075 ± 0.021; on the full test set F1 −0.069 ± 0.006, AP −0.310 ± 0.005. Variant B (exploratory; `y = Sybil or on the initial list`, prevalence 12.1 %): test F1 0.744, AUROC 0.922, AP 0.798; recall 0.814 on bounty Sybils and 0.649 on initial-list addresses. Reading: a large share of the "false positives" are addresses LayerZero itself flagged, so the reported precision is a lower bound under the bounty-only labels; the model trained on cleaner negatives finds more bounty Sybils.
- A17 (`17`, 40 min). One model seed per fit, 10 splits, mean ± SD test F1 / AP / AUROC: LightGBM 0.7133 ± 0.0086 / 0.7611 ± 0.0065 / 0.9682 ± 0.0017; ensemble 0.7131 ± 0.0082 / 0.7612 ± 0.0075 / 0.9681 ± 0.0018 (validation weight per split 0.00 to 0.64); XGBoost 0.7063 ± 0.0068 / 0.7534 ± 0.0083 / 0.9643 ± 0.0019; Random Forest 0.7026 ± 0.0063 / 0.7537 ± 0.0073 / 0.9687 ± 0.0015; LR 0.2336 ± 0.0041 / 0.1605 ± 0.0088 / 0.8323 ± 0.0030. Corrected resampled t-test vs LightGBM (n_test/n_train 0.612, df 9, Holm per metric): F1 XGBoost −0.0070 ± 0.0048 (p 0.30), RF −0.0107 ± 0.0069 (p 0.30), ensemble −0.0001 (p 0.96); AP XGBoost p 0.020, RF p 0.022; AUROC XGBoost −0.0039 (p 0.003), RF +0.0005 (p 1). XGBoost vs ensemble F1 p 0.22. The uncorrected paired t-test and Wilcoxon call the XGBoost and Random Forest F1 differences significant (p ≤ 0.004); the correction removes that. LightGBM here differs from `08` (other machine) per split. Cost (seed-42 split, 408,244 upsampled training rows): single-seed fit XGBoost 27 s, LightGBM 33 s, RF 95 s, LR 11 s (3 seeds: 82, 100, 286, 32 s); scoring the 130,435 test rows XGBoost 0.44 s, LightGBM 3.4 s, RF 0.63 s, LR 0.01 s; peak RSS increase 47–94 MiB; pickled XGBoost 14 MiB, LightGBM 16 MiB, RF 270 MiB; `build_master_df` 12 s. The feature-extraction SQL is not timed.
- A18 (`18`, 4 min). Fixed LightGBM (3 seeds, threshold 0.5748, test recall 0.719). Top features by mean |SHAP|: `l0_tx_time_span`, `n_l0_source_contracts`, `l0_avg_stargate_swap`, `earliest_l0_tx_time`, `latest_l0_tx_time`. Recall of test Sybils after replacing the top-k features (5 repetitions; SD ≤ 0.006): all features, M1 marginal k = 1 / 2 / 3 / 5 / 10 / 20: 0.420 / 0.184 / 0.079 / 0.013 / 0.000 / 0.000; M2 profile copy 0.420 / 0.240 / 0.120 / 0.039 / 0.005 / 0.004. Cheap features only (25 of 62): M1 k = 5 0.027, k = 10 0.004; M2 k = 5 0.045, k = 10 0.011. Adding the structural features (30) changes nothing up to k = 10, because they rank below the cheap ones. Cost classes (25 cheap, 7 time-bound, 30 structural) are tentative; `gas_provision_block_number` is time-bound although it is in the gas-provider family. Reading: the published model is fragile to wallets that copy a handful of ordinary users' cheap transaction statistics; it measures fragility, not the cost of an attack (caveats in the results).
- A19 (`19`, 28 min). LightGBM, model seed 42, 10 splits; All features minus Without all provision-network families: test F1 overall +0.0081 ± 0.0079 (All higher on 9 of 10, corrected p 0.26); IxL +0.0038 ± 0.0033 (9 of 10, p 0.20); IxI +0.105 ± 0.124 (9 of 10, p 0.34); IxE −0.046 ± 0.043 (All higher on 1 of 10, p 0.23). Recall: IxI +0.095, IxE −0.036. AUROC differences ≤ 0.008. Single families: gas provider overall +0.003, IxI +0.011; gas provision tree +0.001, IxI +0.027; provision chain +0.002, IxI +0.015. Reading: the network features matter mainly for IxI (F1 0.483 with them, 0.378 without), consistent with the taxonomy argument, but the split-to-split variation in IxI (426 test Sybils per split) is large and none of the differences survives the correction; in IxE they hurt.
- A20 (`20`, 6 min). Cohort date = earliest `earliest_l0_tx_time` in the wallet's gas provision tree. Contrary to the expectation in the task spec, Sybils are concentrated in the early cohorts: late 30 % (from 2023-08-20) Sybil rate 1.02 % against 5.55 % early; late 20 % (from 2023-12-17) 0.29 %; late 40 % (from 2023-06-27) 2.80 %. Trained on the early cohort (70/30 group split), tested on the late 30 %: precision 0.80, recall 0.095, F1 0.170, AP 0.292, AUROC 0.891; the tree-split model on its test rows in the same cohort (383 Sybils): F1 0.736, AP 0.775, AUROC 0.978. Late 40 %: F1 0.205, AUROC 0.896 (reference 0.727, 0.983); late 20 %: F1 0.008, AUROC 0.747 (reference 0.494, 0.938). Reading: a model trained only on earlier cohorts transfers poorly, at the validation threshold and in AUROC; the gap also reflects the prevalence shift and the shorter feature histories of late wallets (caveats in the results).

**2026-10-06, A21 tie-out, `21` @ c3d7217.** 43 of 44 checks pass: provenance of `13`–`20` and of every notebook they depend on (`02`, `03`, `04`, `06`, `08`, `09`, `13`), stored hyperparameters equal to `02` (and `17`'s Random Forest equal to `13`), a platform record in every result. The failing check is `14`'s inputs (`predictions_match_committed_results` is false); it passes once `03`, `04`, `06` and then `14` are rerun on the machine that produced the committed results. All of `13`–`20` ran on arm64 / Darwin; `00`–`12` ran elsewhere, so the paper cites two platforms (warned; comparisons inside `13`–`20` are same-machine).

**2026-10-06, A22 data-release documents.** `DATA.md` (per `data/` folder: contents, source, retrieval date, rows and columns, LFS object ids, notebooks, known issues, terms) and a draft `CITATION.cff` (validated with `cffconvert --validate`, cffconvert 2.0.0 in a throwaway environment). Found while writing them: the 10 L0-feature and CEX/DEX chunk files each end with a stray UTF-8 byte-order mark, read by pandas as one extra row per file and dropped by the pipeline; the current and pre-snapshot Spellbook DEX files are byte-identical, as are the two bridge files; `20241214_labeled_addresses/additional_addresses.csv` holds `0x9241…3e64`, the address the 2024 code labeled directly (removed in 25ef316, above). Open questions for the authors are at the end of `DATA.md`: terms of use for every third-party source (none documented in the repo), the license for the code (no LICENSE added), author order, name split and ORCIDs for `CITATION.cff`, and the wording for `fcfs_list.csv` (README: "final Sybil list"; it is the bounty-accepted list).

Zenodo steps (for Paven; not done here): (1) sign in to zenodo.org with GitHub and enable the integration for `paven86/layerzero_xgboost` (Account → GitHub → toggle the repository); (2) after the final merge, confirm `CITATION.cff` and choose a license, then create a GitHub release (tag, e.g. `v1.0-revision`), which Zenodo archives and assigns a DOI; (3) cite the DOI in the data-availability statement, with the release's commit.

**2026-10-06, A16 provenance confirmed against the archive.** The authors retrieved the list from the Wayback Machine's 24 May 2024 capture of `github.com/LayerZero-Labs/sybil-report/raw/main/` and shared it (Google Drive id `1GJ356Vu59_188nxBAygkky4Bv5XbKvX1`, `initialList.csv`). Downloaded and checked: 34,560,871 bytes, SHA-256 `1a643f70056fe0c0b38d4a13468e968a2de0bd18de6bc25ed7139d9fa691a7cc`, byte-identical to `review_support/initial_list/initial_list.csv`. The archive still returned 429 to this machine, so the capture was retrieved by the authors, not fetched here. No result changes.

**2026-10-06, A15b report dependence, `22` @ ef82c78 (Apple M5, arm64; 53 min).** Isolates report dependence from the training-diversity collapse in `15`'s union split. Union components as in `15` (reproduced: 355,619 components, largest 13,346 rows and 8,260 Sybils; the two components over 5 % of Sybils, 14,300 rows and 9,214 Sybils, are in every training set). The other components form 5 folds (20 % ± 1 % of rows and Sybils; no report or tree spans two folds), 3 repeats (fold seeds 42, 1, 2). Per fold k: its trees split 50/50 into k_a (test) and k_b; fold k+1 is validation; held-out training = 3 folds + large components; seen training = that plus k_b minus matched whole trees. Every acceptance check holds: the two training sets match exactly on Sybils (14,612.2 on average, rows within 2) and test and validation rows are identical in both conditions; no k_a Sybil's report is among the held-out training Sybils.

- Composition: k_a 42,048–42,050 rows, 899–901 Sybils (prevalence 2.14 %); 98.7 % of k_a Sybils (97.7–99.3 %) have at least one report-mate in k_b; training Sybils from the large components 63.1 % in both conditions (tree split: 50.2 %; union split: 92.6 %); reports represented in training 209.8 (held out) vs 247.1 (seen). n_test / n_train = 0.158.
- LightGBM, mean ± SD over 15: held out F1 0.367, AP 0.305, AUROC 0.891; seen F1 0.598, AP 0.597, AUROC 0.970. Seen minus held out: F1 +0.231 ± 0.084, AP +0.292 ± 0.075, AUROC +0.079 ± 0.041, recall +0.300 ± 0.110, precision +0.180 ± 0.080; seen higher on 15 of 15 for every metric; corrected repeated k-fold t-test p < 0.001 (F1, AP), 0.0012 (AUROC); Wilcoxon (uncorrected) p = 0.0001. XGBoost replicates: F1 +0.217 ± 0.081, AP +0.291 ± 0.078, AUROC +0.078 ± 0.043, 15 of 15.
- Dose-response (LightGBM, pooled over the 15 fold-repeats; recall held → seen, mean score gain): 0 report-mates in k_b, 181 Sybils, 0.122 → 0.127, +0.015; 1–9, 1,275, 0.194 → 0.432, +0.249; 10–99, 6,111, 0.417 → 0.773, +0.363; ≥ 100, 5,927, 0.489 → 0.753, +0.275. The gain is near zero without report-mates and large with them; it peaks at 10–99, not ≥ 100 (the largest reports are already easier to detect when held out).
- By category (LightGBM, seen minus held out): IxL F1 +0.232 ± 0.082, AUROC +0.092 (15 of 15; 860 test Sybils per fold); IxE F1 +0.154 ± 0.236, AUROC +0.044 (13 of 15; 28 per fold); IxI F1 +0.030 ± 0.066, AUROC +0.024 (3 and 12 of 15; 11 per fold). IxI and IxE are too small here to say more.
- Reading: with training size and diversity held fixed, seeing a test Sybil's report-mates in training is worth about 0.23 F1 and 0.08 AUROC. Most of `15`'s drop is report dependence, not the diversity collapse. Absolute F1 and AP are at 2.1 % prevalence and not comparable with the main table; the main tree-split results (where 99 % of test Sybils share a report with a training Sybil, `15`) therefore include report-level similarity, which the paper should state.

**2026-10-06, tie-out renamed `21` → `99_tie_out_revision` (runs last under the glob), extended to `22`.** At 3c2afe5: 49 of 50 checks pass; the failing one is still `14`'s inputs. `results/21_tie_out_revision.json` removed; the result is now `results/99_tie_out_revision.json`.

**2026-10-07, A23 revision figures and tables, `23` @ 4a84790 (5 s; reads `results/` only).** Figures (`figures/rev_*.pdf` and `.png`): `rev_leakage_hierarchy` (A: `11`, address vs tree split, F1 −0.021 / −0.023, AP −0.033, AUROC −0.008 / −0.007 for XGBoost / LightGBM; B: `22`, seen vs held out per fold-repeat, with the note that test rows are 2.1 % Sybil), `rev_report_dependence` (`22`), `rev_label_effects` (`16`), `rev_mimicry` (`18`; "cheap + structural" equals "cheap only" for k ≤ 10, confirmed from the results), `rev_shap_by_category` (`09`, top 15, each column as a share of its total). F6 (10-split dot plot) not drawn: Table T1 carries it. Tables (`tables/rev_*.tex`, booktabs `tabular` fragments): `rev_models_10split` (`17`), `rev_report_dependence` (`22`), `rev_label_robustness` and `rev_operating_points` (`16`), `rev_mimicry` (`18`). Checks: the eight sources pass `sp.provenance`; every figure exists as PDF and PNG; T1 and T2 cells equal the JSON values after rounding (T1 means and SDs recomputed from `per_split` with ddof = 1 equal `17`'s summary); every table row has its tabular's column count. `pdflatex` is not installed on this machine, so the compile check was skipped with a warning. `rev_leakage_hierarchy` mixes platforms (`11` from the machine of the committed results, `22` arm64), recorded as its `platform_note`. Correction to the A15b hand-back: LightGBM held-out F1 SD over the 15 fold-repeats is 0.088 (not 0.083); all values are in `results/22_report_dependence.json` → `summary`. Tie-out `99` at 2c4b8e5: 59 of 60 checks pass (the failing one is still `14`'s inputs).

**2026-10-07, A23 follow-up, `23` @ 7648749.** T1 (`rev_models_10split`) without the weighted average; p-values exactly as stored by `17` (Holm over five comparisons per metric: four models vs LightGBM and XGBoost vs the weighted average). New T6 `rev_ensemble_appendix` (appendix tab:ens): LightGBM 0.713 ± 0.009 / 0.761 ± 0.006 / 0.968 ± 0.002 (F1 / AP / AUROC); weighted average 0.713 ± 0.008 / 0.761 ± 0.008 / 0.968 ± 0.002; average minus LightGBM 0.000 ± 0.003 / 0.000 ± 0.002 / 0.000 ± 0.000, Holm p 0.961 / 0.923 / 1.000, average higher on 4 / 7 / 3 of 10; XGBoost minus average −0.007 ± 0.004 / −0.008 ± 0.002 / −0.004 ± 0.001, Holm p 0.216 / 0.003 / 0.003. `rev_ensemble_weight_stats.json`: validation-selected XGBoost weight per split 0.00–0.64, mean 0.264 (`17`, arm64); seed-42 curve (`06`, machine of the committed results): validation F1 0.7076 at w = 0, 0.7082 at the selected 0.06, 0.7063–0.7082 over w ≤ 0.2. F2 panel B now shows AUROC by category (IxL +0.092 ± 0.047, 15/15; IxE +0.044 ± 0.036, 13/15; IxI +0.024 ± 0.046, 12/15; 860 / 28 / 11 test Sybils per fold) with the note that IxE and IxI are too small for reliable category estimates; F1 panel B held out in dark grey; F4 legend clear of the no-attack line. T1, T2 and T6 equal the JSON after rounding. `pdflatex` is still not installed here; the compile check is skipped with a warning.

**2026-10-07, A24 mimicry controls, `24` @ 9b87db5 (Apple M5, arm64; 10 min).** Setup copied from `18`; the baseline (recall 0.7187, F1, AP) and 12 recall values (all features and cheap only, M1 and M2, k = 1, 5, 10) reproduce `results/18` to 1e-9.

- A, single feature (M1, k = 1, 5 repetitions): most sensitive `l0_tx_time_span` (time-bound, SHAP rank 1) 0.4197 ± 0.0060, `n_l0_source_contracts` (cheap, 2) 0.4200 ± 0.0014, `earliest_l0_tx_time` (time-bound, 4) 0.4403, `l0_avg_stargate_swap` (cheap, 3) 0.4935, `earliest_tx_block_in` (time-bound, 10) 0.5319. The two recalls that looked identical in `18` differ at 6 decimals (0.419733 vs 0.419952): a coincidence at 3 decimals; the columns, donor values and per-Sybil scores differ (2,273 vs 2,300 Sybils flagged in repetition 0, 1,767 in both). No structural feature is among the 15 most sensitive.
- B, strict cheap (Ethereum-side counts and values, 7 features; rule: Ethereum transactions family minus time-bound): M1 / M2 recall k = 1 0.578 / 0.578, k = 3 0.261 / 0.267, k = 5 0.167 / 0.167, k = 7 0.139 / 0.146. SHAP share: top 5 19.4 %, all 7 21.5 % (top 5 overall 37.7 %, top 5 cheap 31.2 %).
- C, structural only (30 features): k = 1 0.697 / 0.697, k = 5 0.537 / 0.546, k = 10 0.493 / 0.513, all 30 0.470 / 0.484. SHAP share of all 30: 13.2 %. Replacing every funding-graph feature at once leaves recall at about two thirds of its unattacked value.
- D, random k (20 subsets; M2 mean, min–max): all 62 features k = 1 0.682 (0.437–0.719), k = 5 0.433 (0.179–0.719), k = 10 0.257 (0.027–0.648), k = 20 0.080 (0.008–0.487); cheap pool k = 1 0.663 (0.423–0.718), k = 5 0.350 (0.141–0.654), k = 10 0.185 (0.030–0.490), k = 20 0.018 (0.008–0.040).
- Reading: the collapse in `18` is specific to the top-ranked features (at k = 5, SHAP order 0.039 vs random 0.433), not general fragility to replacement. A rule-based strict-cheap attack on Ethereum-side statistics alone still cuts recall to 0.17 at k = 5; randomizing the whole gas-provision structure cuts it only to 0.48. Caveats in the results (those of `18`, plus: random k is not an attack model; strict cheap is defined by data source, not monetary cost).

**2026-10-07, tie-out `99` @ 42d4cf5.** Adds `24` (provenance; dependencies `02`, `09`, `18`; parameters; platform; the regression check against `18`). 66 of 67 checks pass; the failing one is still `14`'s inputs.

**2026-10-07, A25 paper tables and cited numbers, `25` @ b8747c0 (14 s; reads `results/`, the Sybil list and the master table; no model).** Tables: `rev_data_labels`, `rev_taxonomy`, `rev_levels`, `rev_per_category`, `rev_family_ablation`, `rev_provision_ablation_by_category`, `rev_entity_level`, `rev_temporal_holdout`, `rev_union_split_appendix`, `rev_search_top`; `tables/rev_text_numbers.json` holds 199 named scalars, each with its source and platform (none missing).

- Bounty-list statistics, computed from `fcfs_list.csv` matched to the 18,211 Ethereum Sybils: 311 reports, 145 reporters, ten largest reports 35.6 % of Sybils, ten most prolific reporters 44.6 % (the manuscript cites 311 / 145 / 36 % / 45 %). Flag times run from 2024-05-18 to 2024-05-30.
- Trees over 100 wallets: 34 trees, 9,020 wallets (2.07 %), 97 Sybils (0.53 %), computed from the master table (not stored in any result before).
- Per-category performance over the ten tree splits (`17`, LightGBM, mean ± SD): IxL F1 0.741 ± 0.006, recall 0.761; IxI F1 0.483 ± 0.097, recall 0.338, precision 0.901; IxE F1 0.234 ± 0.057, recall 0.142, precision 0.694; No provider about 7 test Sybils per split (unreliable). XGBoost: IxL F1 0.736, IxI 0.409, IxE 0.221. **Correction for the manuscript:** the seed-42 per-category values in an earlier draft (IxI F1 0.534, IxE F1 0.265) are more favorable than these 10-split means; cite the 10-split means.
- Searches: the selected configuration ranks first in each grid; the library defaults rank 60 of 74 (XGBoost) and 48 of 48 (LightGBM).
- `rev_entity_level` carries `UNVERIFIED` in its first line: `results/14` → `predictions_match_committed_results` is false (arm64 predictions) until `14` is rerun on the machine of the committed results.
- Part 1: T6 (`23`) prints the weighted-average differences to four decimals (F1 −0.0001 ± 0.0032, AP +0.0001 ± 0.0016, AUROC −0.0002 ± 0.0003); the controls table (`24`, which writes it; rerun at 61884d1 with unchanged numbers) has a k = 7 column for strict cheap. `pdflatex` is still not installed here; every table passes the structural column check.

**2026-10-07, tie-out `99` @ 201c9b4.** Adds `25` (provenance; its 13 source notebooks; no missing named numbers). 82 of 83 checks pass; the failing one is still `14`'s inputs.

**2026-10-08, display names for the feature classes; A25 table follow-ups, @ 2f39463.** Display names agreed by the authors: `cheap` → wallet-local (25 features), strict cheap → Ethereum-local (7), `structural` → funding-graph (30), `time-bound` → time-based (7). Only figure legends, tags and table labels change (`24`: `rev_mimicry_controls` figure and tables; `23`: F4 legend and T5 rows); the class keys in code and JSON are unchanged. `24` now asserts that every number equals its previous same-platform result: maximum absolute difference 0.0 against 61884d1 in every block (single feature, Ethereum-local, funding-graph, random k, SHAP order, baseline, SHAP shares). `25`: two-row header in `rev_temporal_holdout` (temporal model, same-cohort reference); `rev_levels` gains "3. Report-mates seen" (tree 0, report-mate coverage 98.7 %); `rev_taxonomy` shares to two decimals; `rev_union_split_appendix` without the empty column. New named numbers (207 in total, none missing): `union_components_total` 355,619; `level3_training_sybils` 14,612.2; `blend_weight_seed42` 0.06; `level3_pairs_equal_train_rows` 14 of 15, `level3_pairs_equal_train_sybils` 15 of 15, `level3_max_train_rows_diff` 2 (repeat 1, fold 2: 266,592 held out vs 266,590 seen); `level3_n_test_over_n_train` 0.1577. Tie-out `99`: 82 of 83 (the failing check is still `14`'s inputs).

**2026-10-09, single-platform rerun of `13`–`25` and `99` at 68e8d63 (A26), execution commit 412db11.** Platform, read in the notebook kernel: x86_64 Linux, Intel Xeon @ 2.80 GHz (family 6, model 85, stepping 7; flags identical to PR #8's 2.80 GHz host), 4 cores, Python 3.11.15, every pinned package at its pin (`logs/rerun_x86_68e8d63/env_kernel.json`). 412db11 removes `results/13_search_random_forest.csv` (scored on arm64) and moves `24`'s table escape out of the f-string for Python 3.11, as 693757c did; no other code, data or requirement change. All 37 LFS inputs match their pointers' SHA-256. Notebook 21 is absent (renamed `99`).
- `03`, `04`, `06` regenerated at 412db11: every field equals the committed 0d9eacc results (85, 84, 96 fields); `00`–`12` pass `sp.provenance`, so `01` and `02` are kept. `14` read these predictions: `predictions_match_committed_results` true.
- Each notebook ran from a clean tree and was committed before the next (9c634b7 to b050c3e). Runtimes: `13` 4,517 s (resumed after a restart from 21 of 24 search rows scored here), `14` 49 s, `15` 1,046 s, `16` 1,655 s, `17` 4,946 s, `18` 416 s, `19` 2,702 s, `20` 731 s, `22` 6,261 s, `23` 19 s, `24` 1,713 s, `25` 76 s, `99` 7 s. Three container restarts interrupted `13`, `16` and `22` before they saved anything; the environment and `03` were rechecked after each (`logs/rerun_x86_68e8d63/run/`).
- `13`–`22` and `25` equal PR #8's x86 results (same code and data) except Random Forest AP (at most 6.7e-7 in `17`; threaded summation in scikit-learn) and timings. `14` (Section 5.5), LightGBM at 0.6001: E1 any-hit 0.7555, majority 0.7550; E2 any-hit 0.6946, majority 0.5397; E3 any-hit 0.8636, majority 0.6742; unflagged allocation 0.3796 (committed arm64: 0.7646, 0.7642, 0.7029, 0.5523, 0.8712, 0.6894, 0.3722). `22`, seen minus held out, LightGBM F1 +0.2288 ± 0.0763, AP +0.2939 ± 0.0775, AUROC +0.0793 ± 0.0424, 15 of 15 (committed: +0.2314 ± 0.0843, +0.2918 ± 0.0748, +0.0785 ± 0.0411); training Sybils equal in all 15 pairs, rows in 14 (the other differs by 2).
- `99` @ 8eed3f4: 83 of 83 checks pass, `multiple_platforms_cited` false. `23` and `25` compile every table with pdflatex. Field-by-field comparisons: `logs/rerun_x86_68e8d63/compare_results.txt`. `README.md` still cites the arm64 numbers for `13`–`24`.
