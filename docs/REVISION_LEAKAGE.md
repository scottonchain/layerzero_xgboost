# Revision tracker: data leakage

Manuscript BCRA-D-26-00643, resubmission due Oct 12, 2026. Repo baseline: paven86/layerzero_xgboost @ 11366f4.

Check an item only when the code, the manuscript text, and the response letter are all updated.

## Status

Code status per leakage item. "Code done" means the repository change is committed; the checkbox
in sections A to C stays open until the manuscript and response letter are updated too. Numbers
for the paper come from `06_split_comparison.ipynb` in the tagged final run (section E).

| Item | Reviewer point | Code status | Commit | Evidence |
|---|---|---|---|---|
| A1 group split | R2 relational leakage | Code done | fd6411f | `sybil_pipeline.make_splits` asserts no group spans two partitions |
| A2 group-overlap check | R2 relational leakage | Code done | fd6411f | `00` leakage cell; `06` leakage table |
| A3 blend weight on test | R2 test isolation | Code done | fd6411f | `04` section 6 selects on validation F1 |
| A4 search not in repo; tables on test | R1 search/validation; R2 test isolation | Code done; full search running | 82148f1 | `05_hyperparameter_search` (validation only; test deleted before fitting) |
| A5 feature selection | R1 | Code done. Authors: the 63 were chosen manually. Label-based EDA moved to the training partition | fd6411f | Methods text still needed |
| A6 post-snapshot reference data | R1 temporal cutoff | Audited (findings E). One question to the authors: A12 | — | `README` data table gives each source's cutoff |
| A7 provision graph date filter | R1 temporal cutoff | Code done: cutoff enforced; tree features recomputed from filtered edges | b2904f4 | `00` steps 2 and 4 |
| A8 both splits reported | R2 relational leakage | Code done; final run pending | d84263a | `06_split_comparison` |
| A9 descriptive figure on val+test | R1 | Not in repo: the figure is drawn outside these notebooks | — | Redraw on full data or train only |
| A10 duplicate wallet row | New (row-level leakage) | Code done | b2904f4 | `00` quality check: 0 duplicates; build asserts uniqueness |
| A11 XGBoost thread nondeterminism | New (R1 reproducibility) | Code done: `n_jobs` pinned to 4 | b2904f4 | Findings E |
| A12 44 hand-added labeled addresses | New (possible label leakage) | Resolved: 20 in hildobby; 20 of 24 hand-only publicly tagged; removing them changes F1 by −0.002 ± 0.014 over 5 seeds. Completeness check of top 30 unlabeled funders pending | ce15ded, 998fa57 | `review_support/`; findings E |
| B10 repo text | R2 test isolation | Notebook headers done (d84263a); README with final numbers | — | |

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
- [ ] Dataset size: 434,786 addresses (18,211 Sybil; 416,575 non-Sybil) after removing the duplicated wallet (A10), not 434,787.
- [ ] Table 5 percentages from the code: IxL 69.64 %, IxE 22.14 %, IxI 8.06 %, No provider 0.16 %. The paper's "IxL ≈ 71 %" and the IxI/IxE swap both change.
- [ ] `gini_coefficient` is identically 0 up to rounding (max |value| 9.9e-16 over all wallets). The original formula `(n-1)/n * (1 - sum(sorted(x)/sum(x)))` always gives (n-1)/n × 0, because the normalized amounts sum to 1. Table 11 ranks it with importance 125e-4, which is splitting on floating-point noise. Decision needed: implement the real Gini coefficient of provision amounts, or drop the feature and say so. Not leakage; not changed in code yet.
- [ ] `total_gas` is the total ETH provisioned within the wallet's funding tree (from the tree featurization), not "cumulative ETH gas consumed by an address" as Sec 5.3.1 and Appendix A say. Fix the definitions; R1 asks for formulas of graph features, and `sybil_pipeline.provision_features` is the reference implementation.
- [ ] The committed notebook outputs at 11366f4 gave LightGBM F1 0.7371 and LR AP 0.159, not the paper's 0.739 and 0.094. Superseded by the rerun, but the response letter should not quote the old values.
- [ ] Tables 7 and 8 "baseline" rows (default parameters) and the "directed grid search over 15 configurations" have no code in the repo. Replace with the `05_hyperparameter_search` results (validation only, full grids).
- [ ] Contribution 5 (cross-model ensemble): at the fd6411f checkpoint the validation-selected blend weight on the group split was 0.00, i.e. pure LightGBM. Confirm in the final run; if it holds, the ensemble claim must be dropped or restated.

## D. Response letter mapping

| Reviewer point | Items |
|---|---|
| R1: temporal cutoff, future-information leakage | A6, A7, A12, B7 |
| R1: search space, optimization method, validation scheme | A4, B6 |
| R1: code and data release, verifiability | A7 (tree features now computed from repo data), A11, `06` provenance checks |
| R1: formulas for graph-based features | C (`gini_coefficient`, `total_gas`); `sybil_pipeline.provision_features` |
| R1: precision/recall per category; FPR at operating threshold | `06` per-category and FPR tables |
| R1: ablations of tuning choices | `05` marginal-effect table |
| R2: relational leakage from random split | A1, A2, A8, A10, B5, B8, B9 |
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

**2026-09-30, @ fd6411f (group split added; checkpoint before the temporal fix).** Runs in `results/` were not committed; they are superseded by the final run.

- Random split through `sybil_pipeline` reproduces the original `splits.npz` bit for bit.
- LightGBM on the random split reproduces the committed 11366f4 output exactly (F1 0.7371). XGBoost does not (F1 0.7329 vs 0.7356; best iterations 465/465/516 vs 493/449/480): XGBoost `hist` depends on the thread count, and the original ran on a different machine. Fixed from b2904f4 by pinning `n_jobs` (A11).
- Group split, test F1 / AUROC / AP: XGBoost 0.707 / 0.968 / 0.767; LightGBM 0.723 / 0.971 / 0.778; LR 0.234 / 0.833 / 0.157. Random split: XGBoost 0.733 / 0.974 / 0.796; LightGBM 0.737 / 0.976 / 0.801; LR unchanged. The linear model does not move, consistent with cluster memorisation driving the tree models' gap.
- Blend weight chosen on validation: 0.00 on the group split (the ensemble is LightGBM alone), 0.26 on the random split.
- 04's individual models equal 01 and 02 exactly under both splits.

**2026-09-30, temporal and data audit (R1), @ b2904f4.**

- Snapshot boundary: the latest `latest_l0_tx_time` is 2024-05-01 23:59:58 UTC, so the L0 snapshot table covers all of May 1. Ethereum transaction features cut at 2024-05-01 00:00 UTC (`<=` for outgoing, `<` for incoming; see the L0 query). CEX/DEX in-degree cuts at `<= 2024-05-01`.
- Provision network: 296 edges dated at or after 2024-05-02 00:00 UTC, all between 2024-08-17 and 2024-10-31, overlapping the Sybil list process (list snapshot 2024-09-15). None enters an interactor and none creates an `is_provider` flag; 295 join two unlabeled addresses. Effect before the fix: one IxE non-Sybil wallet's `chain_length`, funding group, and tree features. Fixed by the cutoff in `load_provision`.
- Tree features: the original precomputed file came from `data/20241117_tree_features/20241117 Gas Provision Featurization.ipynb`, run on the unfiltered network with inputs outside the repo. Ported to `sybil_pipeline.provision_features`. Against the original file on 434,110 wallets: provider fan-out, max and min amounts match 100 %; tree features match on all but 13 or 14 wallets, which sit in one 14-node tree that the pipeline's labeled list splits, plus the one cutoff wallet; `provider_is_star_like_attack` differs on 9 wallets for the same labeled-list reason; provider total and average amounts differ on the 16,101 wallets of one provider because the original double-counted the duplicated wallet. Skewness uses the scipy < 1.9 rule (0 for near-constant data), which the original file reflects.
- Duplicate wallet (A10): `0x3aecba06e531a982cfdba16f0589ece5dc200fa9` appeared twice in the precomputed file, so it appeared twice in the master table (non-Sybil, IxL). 00's own quality check reported "Duplicate addresses : 1 … Issues found". Under the random split the two copies could land in train and test.
- Labeled addresses (A6): compiled Oct–Dec 2024 from Flipside L0 address labels, hildobby CEX list, BigQuery and Dune contract lists, dawsbot and brianleect label sets, plus 44 hand-added addresses (`data/20241214_labeled_addresses/readme.txt`). None of the 44 is on the Sybil list. One labeled address in the network is on the Sybil list (`0x3df1…9159`); it funds no other address, so it affects no other wallet's features.
- A12 (open): the 44 hand-added addresses directly fund 3,153 interactors, of which 10 are Sybil (0.32 % vs 4.19 % overall). Labeling them changes those wallets' `provider_is_labeled`, `provider_is_star_like_attack`, chain, and tree features. If they were identified from public identity sources (explorer tags, exchange documentation), this is not leakage; if Sybil rates influenced the choice, it is. Ask the authors; a sensitivity run without the 44 would settle it empirically.
- `total_gas` does not exist in the L0 query; it comes from the tree featurization (see C).

**2026-09-30, A12 resolved (hand-added labeled addresses).** Evidence in `review_support/`.

- 20 of the 44 hand-added addresses are also in the hildobby CEX list (`data/20241013_hildobby_cex_evms`, loaded by the same build script; verified by exact address match): Shakepay, MoonPay, Voyager, 13 ShapeShift. Their labels never depended on the hand step. The remaining 24 are "hand-only"; 15 affect any feature, funding 653 interactors in total.
- Etherscan name tags for the 24, read from each page on 2026-09-30 (`etherscan_tags_2026-09-30.csv`): 20 carry a public entity tag, mostly bridges and relayers (ZigZag, Owlto, Umbria, Chaineye, Synapse, Hyphen, Multichain, Allbridge, Argent), plus Newton (exchange), an OKX deposit address, Union Chain, Kuailian wallet contract, Layer Zero Executor, Seaport 1.6, EtherDelta 2. Four have no entity tag (`0x6b8f…`, `0x0000…055a` with a personal ENS name only, `0xffff…61ef`, `0x0000…29d8`); together they fund 8 interactors.
- Author's account: the addresses were checked on Etherscan after anomalies in exploratory, label-free analytics and found to be entities the hildobby CEX list does not cover. The tags are consistent with that.
- Sensitivity, 5 group-split seeds (`hand_only_seed_band.csv`; LightGBM, 05 configuration from 82148f1, pre-Gini-fix features): removing the 24 hand-only labels changes test F1 by −0.002 ± 0.014 (mean ± SD across seeds; per seed −0.023, +0.006, +0.008, −0.009, +0.008), AUROC by −0.002 ± 0.002, AP by −0.004 ± 0.011. Indistinguishable from zero. The earlier single-seed −0.017 removed all 44, including the 20 hildobby addresses, and is superseded.
- Side finding for R1 (significance) and the ensemble claim: with the labels, LightGBM test F1 ranges 0.702 to 0.726 across the 5 group-split seeds (SD ≈ 0.009). Model differences smaller than that are within split noise.
- Open: completeness. The top 30 unlabeled funders by addresses funded (`top30_unlabeled_funders.csv`) are being checked on Etherscan the same way.
