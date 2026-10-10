# Sensitivity: an automated labeling rule

Replaces the Etherscan step of the labeling rule with a rule that has no interactor threshold and no human
step: every address that funded at least 2 addresses is labeled if Blockscout's tag service gives it a
service category tag (rule fixed before the run). Everything else follows the primary pipeline. Result
(LightGBM, 10 group splits, corrected run): test F1 changes by −0.001 ± 0.011 (pre-fix −0.003 ± 0.010).

| File | Contents |
|---|---|
| `build_blockscout_rule.py` | Reads Blockscout tags for every funder of 2+ addresses and applies the rule |
| `blockscout_tags.csv` | Every such funder with any Blockscout tag, with the read date |
| `blockscout_rule.csv` | The addresses the automated rule labels |
| `sensitivity_blockscout_rule.ipynb` | Primary vs automated labels on 10 group splits |
| `review_blockscout_rule.json` | Results of the corrected run (code commit b2f2ece recorded inside) |

Run from the repo root: `python review_support/blockscout_rule_sensitivity/build_blockscout_rule.py`
(network access to Blockscout), then execute the notebook. It needs `results/02_hyperparameter_search.json`.
