# Stage 1: wallets whose inputs change when roots receive their tree metrics

Original code: `sybil_pipeline_b7a9a3f.py` (sha256 ba23ca091ee5); corrected: `sybil_pipeline.py` (7b6c935e9bed); inputs sha256 d76974a04291. Same environment, same inputs, one build each.

- Wallets: 434,786. Selected model inputs (62 features) change for **307,622** wallets (70.75 %); any computed column changes for 307,622.
- Changed model features: breadth_factor, tree_size, sparsity, leaf_to_internal_ratio, breadth_to_depth_ratio, longest_chain_ratio, branching_factor, total_gas, leaf_provision_proportion, max_depth, avg_depth, gas_distribution_entropy, gas_distribution_skewness, star_like_ratio, leaf_gas_distribution_entropy, leaf_gas_distribution_skewness, balance_factor.
- Changed columns excluded from the model: avg_leaf_gas, depth_weighted_avg_gas, depth_variance, gini_coefficient.
- Model tree features unchanged: depth.
- Unexpected non-root changes: none. Non-tree columns changed: none.

| Category | Changed | Total | % |
|---|---|---|---|
| IxL | 306,658 | 306,658 | 100.0 |
| IxI | 37 | 34,391 | 0.108 |
| IxE | 251 | 93,061 | 0.27 |
| No provider | 676 | 676 | 100.0 |

Sybil changed 15,947 of 18,211; non-Sybil changed 291,675 of 416,575.
Changed by kind: {'roots_with_descendants': 12942, 'singleton_roots': 294680, 'non_roots': 0}.

Identity checks: {"same_columns_in_same_order": true, "address_order_identical": true, "labels_identical": true, "category_identical": true, "provision_tree_identical": true, "anchors_flags_identical": true, "n_anchors_old": 4308, "n_anchors_new": 4308, "n_edges_old": 604365, "n_edges_new": 604365, "split_train_row_ids_identical": true, "split_val_row_ids_identical": true, "split_test_row_ids_identical": true, "y_train_identical": true, "upsampled_train_index_identical": true}
