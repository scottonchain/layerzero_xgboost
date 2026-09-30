# Revision tracker: data leakage

Manuscript BCRA-D-26-00643, resubmission due Oct 12, 2026. Repo baseline: paven86/layerzero_xgboost @ 11366f4.

Check an item only when the code, the manuscript text, and the response letter are all updated.

## A. Code and experiments

- [ ] **A1. Relational leakage from the random split.**
  - Where: 00_data_pipeline (`train_test_split(..., stratify=y)`), duplicated in 01, 02, 04.
  - Problem: Wallets from the same funding cluster land in both train and test. Provider and tree features (`provider_*`, `gini_coefficient`, `branching_factor`, `tree_size`, etc.) are identical within a cluster, so test performance partly measures recognition of clusters seen in training.
  - Fix: Use a stratified group split (`StratifiedGroupKFold`).
    - Group key is the root of the unlabeled part of the provision chain. Walk `tfm` until the next hop is a labeled anchor or there is no provider; the last unlabeled node is the group.
    - IxL and No-provider wallets are singletons. A CEX is not an operator.
    - Do not group by the immediate provider. A CEX hot wallet would become one giant group.
  - Done when: The split code asserts that no group spans two partitions.
- [ ] **A2. Leakage check is too narrow.**
  - Where: 00, "No leakage ✓" cell (checks row-index overlap only).
  - Fix: Add a group-overlap check. Report the count of test Sybils sharing a group with a train Sybil, under both the old and new splits.
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
- [ ] **A9. Descriptive figure drawn on val+test.**
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
  - Problem: Under a random split, validation and test share clusters, so both are inflated equally and the gap proves nothing.
  - Fix: Remove, or restate under the group split as "consistent with."
- [ ] **B4. Sec 4.4 blend weight description.**
  - Current text: "selected by optimizing F1 on the validation set ... optimum falls near 0.34."
  - Problem: Contradicts the code, which sweeps on test with an optimum of 0.16.
  - Fix: Replace with rerun numbers from A3.
- [ ] **B5. Sec 4.6 overclaims no leakage.**
  - Current text: "Consequently, no Sybil label leakage occurs across the split boundary."
  - Rewrite: Labels do not enter feature computation, but features are relational. Wallets sharing a funding cluster share feature values, so the split is grouped by cluster (A1). Name this relational or cluster-level leakage explicitly.
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
- [ ] Table 12: Sybil counts per category differ from the code (1,534/16,241/436 vs. 1,592/15,886/707+26).
- [ ] Table 2 says `min_child_weight = 1`, while Sec 9 says mcw = 10. Pick one and match the code.
- [ ] Feature importance: The code uses the last-trained seed (456), while the paper says seed 42. Average across seeds or state which one.
- [ ] "Share only 2 of top-15" becomes 3 in the current code. Recheck after the rerun.

## D. Response letter mapping

| Reviewer point | Items |
|---|---|
| R1: temporal cutoff, future-information leakage | A6, A7, B7 |
| R1: search space, optimization method, validation scheme | A4, B6 |
| R2: relational leakage from random split | A1, A2, A8, B5, B8, B9 |
| R2: test isolation contradicted by manuscript and repo | A3, A4, B2, B3, B4, B6, B10 |

## E. Findings log

Measured facts, with the commit they were measured at. Append; don't rewrite.

**2026-09-30, @ 11366f4 (baseline, before any fix).** Environment: `requirements.txt`.

- Baseline reproduces: 434,787 addresses, 18,211 Sybil (4.19%). Split sizes match Table 1 (213,045 / 91,305 / 130,437 before upsampling).
- Taxonomy from code (confirms C, Table 5 swap): IxL 302,796 (15,886 Sybil), IxE 96,277 (707), IxI 35,038 (1,592), No provider 676 (26).
- A1 group key as defined above: 367,923 groups; 22,912 with more than one wallet; largest group 2,103 wallets. Only 2,359 of 18,211 Sybils sit in multi-wallet groups, because 87% of Sybils are IxL singletons.
- A2 under the current random split: 20,891 of 130,437 test addresses share a group with a train address; **608 of 5,463 test Sybils share a group with a train Sybil.**
- A7: the latest `first_gas_provision_time` for any L0 interactor is 2024-05-01 23:56:47 UTC, which is on the snapshot date. 163 interactor rows fall on May 1 itself. The full network file (which includes upstream non-interactor providers) runs to 2024-10-31 and has 460 rows after 2024-05-01 00:00. Those upstream rows can affect `chain_length`, `interactors_in_chain`, and the A1 group root. The tree features (`20241117_tree_features`) were built from this same network, so check their inputs too.
- A6: `data/20250208_cex_dex_indegree/readme.txt` SQL filters `block_timestamp <= '2024-05-01'`. That sub-item is verified.
- A5: `legacy/20250519 XGBoost Sybil Detection.ipynb` (the 2025 paper's code) also hard-codes the 63-feature list. The repo holds no selection code. The procedure must come from the authors.
- Caveat for the discussion section: IxL wallets are singletons by design, so an operator who funds wallets through separate CEX withdrawals is invisible to the provision-graph grouping. The group split removes the leakage the provision graph can see, and no more.
