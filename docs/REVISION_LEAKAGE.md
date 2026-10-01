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
| A4 search not in repo; tables on test | R1 search/validation; R2 test isolation | Done: validation-only search, 74 XGBoost and 48 LightGBM configurations | ea602c6 | `05_hyperparameter_search` (validation only; test deleted before fitting) |
| A5 feature selection | R1 | Code done. Authors: the 63 were chosen manually; `gini_coefficient` later removed by a pre-specified rule (C). Label-based EDA moved to the training partition | fd6411f | Methods text still needed |
| A6 post-snapshot reference data | R1 temporal cutoff | Audited (findings E). Labels: fixed rule, pre-snapshot vintage as sensitivity (`07`): test F1 −0.005 ± 0.007 | ea602c6 | `README` data table gives each source's cutoff; `07` |
| A7 provision graph date filter | R1 temporal cutoff | Code done: cutoff enforced; tree features recomputed from filtered edges | b2904f4 | `00` steps 2 and 4 |
| A8 both splits reported | R2 relational leakage | Done: final run, both splits | ea602c6, 06dbd82 | `06_split_comparison` |
| A9 descriptive figure on val+test | R1 | Not in repo: the figure is drawn outside these notebooks | — | Redraw on full data or train only |
| A10 duplicate wallet row | New (row-level leakage) | Code done | b2904f4 | `00` quality check: 0 duplicates; build asserts uniqueness |
| A11 model nondeterminism | New (R1 reproducibility) | Code done: XGBoost `n_jobs` pinned to 4 (b2904f4); LightGBM `force_col_wise` and `deterministic` (`sp.LGBM_REPRO`) | b2904f4, ea602c6 | Findings E; `06` checks that 04's single models equal 01 and 02 |
| A12 44 hand-added labeled addresses | New (possible label leakage) | Code done: hand step replaced by a two-step labeling rule (public lists, then an Etherscan check of every other funder of ≥ 50 interactors; 36 checked, 10 labeled). Pre-snapshot label vintage as sensitivity | 25ef316, b1594ce, 91cfd02 | `review_support/etherscan_lookups.csv`; `07`; findings E |
| C `gini_coefficient` identically zero | R1 graph-feature formulas | Code done: formula corrected, then removed by the pre-specified rule (3 of 10 splits; 62 features) | b1594ce, 91cfd02 | `08_ablation_gini`; findings E |
| B10 repo text | R2 test isolation | Done: notebook headers, README benchmark and robustness tables from the final run | d84263a, this commit | `README.md` |

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
- [ ] Table 12: Sybil counts per category from the final run (labeling rule): IxL 15,921; IxI 1,566; IxE 698; No provider 26 (paper: 16,241 / 1,534 / 436).
- [ ] Table 2 says `min_child_weight = 1`, while Sec 9 says mcw = 10. The validation search selects 1 (mcw 10 is fourth); use the `05` configuration throughout.
- [ ] Feature importance: fixed in 06dbd82. 01 reported XGBoost's average gain per split and 02 LightGBM's total gain, both from the last seed and labeled "Gini gain"; both now report total gain averaged over the three seeds. Proposed for the paper: SHAP on the test partition (`10`) in place of Tables 10 and 11 (ledger N13).
- [ ] "Share only 2 of top-15": with the same measure for both models (total gain, mean of 3 seeds, group split), XGBoost and LightGBM share all 15 of their top-15 features. The low overlap in the submission came from comparing two different gain measures.
- [ ] Dataset size: 434,786 addresses (18,211 Sybil; 416,575 non-Sybil) after removing the duplicated wallet (A10), not 434,787.
- [ ] Table 5 from the final run (labeling rule): IxL 306,658 (70.53 %), IxE 93,061 (21.40 %), IxI 34,391 (7.91 %), No provider 676 (0.16 %). The IxI/IxE swap must be fixed.
- [ ] `gini_coefficient` was identically 0 up to rounding (max |value| 9.9e-16 over all wallets): the original formula `(n-1)/n * (1 - sum(sorted(x)/sum(x)))` always gives (n-1)/n × 0. Table 11 ranked it with importance 125e-4, which was splitting on floating-point noise. **Fixed** (authors chose to fix, 2026-09-30): `sybil_pipeline._gini` computes the standard Gini coefficient of the non-root provision amounts in the tree, G = 2·Σ i·x₍ᵢ₎ / (n·Σx) − (n+1)/n, 0 for fewer than two amounts. After the fix it is nonzero for 20.4 % of wallets (IxI mean 0.358, IxE 0.347, IxL 0), and the **Sybil mean (0.051) is lower than the non-Sybil mean (0.108)**, the opposite of Sec 5.2's claim that a high Gini coefficient characterizes star-like Sybil funding. **Then removed** by the rule fixed in advance (findings E, 2026-10-01): removing it lowered validation F1 on 3 of 10 group splits, short of the 8 required. Manuscript: drop it from the feature list, Table 11 and Sec 5.2's sentence; the response letter explains the zero values, the correction and the test.
- [ ] `total_gas` is the total ETH provisioned within the wallet's funding tree (from the tree featurization), not "cumulative ETH gas consumed by an address" as Sec 5.3.1 and Appendix A say. Fix the definitions; R1 asks for formulas of graph features, and `sybil_pipeline.provision_features` is the reference implementation.
- [ ] The committed notebook outputs at 11366f4 gave LightGBM F1 0.7371 and LR AP 0.159, not the paper's 0.739 and 0.094. Superseded by the rerun, but the response letter should not quote the old values.
- [ ] Tables 7 and 8 "baseline" rows (default parameters) and the "directed grid search over 15 configurations" have no code in the repo. Replace with the `05_hyperparameter_search` results (validation only, full grids).
- [ ] Contribution 5 (cross-model ensemble): confirmed in the final run. Validation selects XGBoost weight 0.06 under both splits; on the group split the blend (F1 0.717) does not beat LightGBM alone (0.719). Restate (proposed text in ledger N12); the title names "Cross-Model Ensemble" (author decision).

## D. Response letter mapping

| Reviewer point | Items |
|---|---|
| R1: temporal cutoff, future-information leakage | A6, A7, A12, B7 |
| R1: search space, optimization method, validation scheme | A4, B6 |
| R1: code and data release, verifiability | A7 (tree features now computed from repo data), A11, `06` provenance checks |
| R1: formulas for graph-based features | C (`gini_coefficient`, `total_gas`); `sybil_pipeline.provision_features` |
| R1: precision/recall per category; FPR at operating threshold | `06` per-category and FPR tables |
| R1: ablations of tuning choices | `05` marginal-effect table |
| R1/R2: which features matter; graph features | `08` (Gini), `09` family ablation, `10` SHAP by category |
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

- Data (labeling rule): 434,786 addresses, 18,211 Sybil; 4,308 labeled addresses in the network; 371,764 funding groups (22,952 multi-wallet, largest 1,627). Taxonomy: IxL 306,658, IxE 93,061, IxI 34,391, No provider 676.
- A2: random split, 599 of 5,463 test Sybils share a funding group with a training Sybil (19,884 of 130,436 test rows share a group with a training row); group split, 0.
- Search (validation only): XGBoost lr 0.05, depth 8, subsample 0.7, colsample 1.0, mcw 1 (val F1 0.703); LightGBM 127 leaves, lr 0.05, subsample 0.7, λ 1 (val F1 0.707).
- Test, group / random: XGBoost F1 0.720 / 0.741, AUROC 0.966 / 0.974, AP 0.764 / 0.798; LightGBM 0.719 / 0.742, 0.968 / 0.976, 0.770 / 0.802; ensemble 0.717 / 0.743; LR 0.241 / 0.234. The tree models lose 0.021 to 0.025 F1 and 0.033 AP under the group split; LR does not.
- Per category (LightGBM F1, group / random): IxI 0.47 / 0.85, IxE 0.29 / 0.54, IxL 0.75 / 0.74. The leakage sits in the clustered categories.
- Blend weight 0.06 (XGBoost) under both splits; disagreement at 0.5: 703 (group), 579 (random). In the nondeterministic 91cfd02 run the random-split weight was 0.58: the validation F1 curve is flat, so the weight is not a stable quantity.
- Split noise: LightGBM test F1 over 10 group splits 0.713 ± 0.006 (`09`).
- `07`: pre-snapshot labels (4,115 vs 4,308) change 10,575 wallets' features; test F1 −0.005 ± 0.007, validation F1 +0.009 ± 0.012 (opposite signs, both within split noise). Sybil share of wallets funded by CEX addresses by date added to the list: before the snapshot 5.30 % (264,646 wallets), between snapshot and Sybil list 2.40 % (1,082), after the Sybil list 0.92 % (1,086). Later labels do not concentrate Sybils.
- `09` (test F1 change vs all 62 features): without LayerZero transactions −0.066, Ethereum transactions −0.012, gas provider −0.005 (lower on 9 of 10 splits), funding tree −0.001, funding chain −0.002; all three provision-network families together −0.008 (lower on 7 of 10; validation F1 lower on 3 of 10). Alone: LayerZero 0.672, Ethereum 0.625, gas provider 0.438, funding tree 0.101, funding chain 0.101 (AUROC 0.61).
- `10` (mean |SHAP| summed by family, test partition): LayerZero transactions 4.88, Ethereum transactions 2.44, gas provider 0.84, funding tree 0.49, funding chain 0.04. Funding-tree features weigh more within IxI (1.35) and IxE (0.98) than IxL (0.25).

