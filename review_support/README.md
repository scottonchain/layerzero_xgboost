# Review support: labeled addresses (tracker item A12)

Evidence for how the labeled-address set was built. The model pipeline does not read this folder,
except through the generated `data/20260930_etherscan_service_labels/service_labels.csv`.

| File | What it is |
|---|---|
| `hand_labeled_addresses.ipynb` | Reconstructs the 44 addresses added by hand in 2024: which affect any feature, and whether a label-free ranking explains them |
| `hand_labeled_addresses.csv`, `top_funders_not_added.csv` | Sheets written by that notebook |
| `hand_only_seed_band.csv` | Sensitivity: the 24 hand-only labels removed vs kept, 5 group-split seeds |
| `top30_unlabeled_funders.csv` | The 30 largest funders not covered by the 2024 labels, by addresses funded |
| `build_etherscan_lookups.py` | Records the Etherscan tags read for 65 addresses, applies the labeling rule, and writes the two tables below |
| `etherscan_lookups.csv` | Every Etherscan lookup with its date, tag and the rule's decision (columns in `data/20260930_etherscan_service_labels/readme.md`) |
| `blockscout_crosscheck.py`, `blockscout_crosscheck.csv` | The same 65 addresses on a second explorer (Blockscout API and its tag service), read 2026-10-01; a check only, no label depends on it |
| `build_blockscout_rule.py`, `blockscout_tags.csv`, `blockscout_rule.csv` | Automated alternative to the Etherscan step: Blockscout tags for every funder of 2+ addresses, labeled by a fixed list of service categories |
| `build_2024_only_evidence.py`, `labels_2024_only_evidence.csv` | The 32 addresses labeled only by the 2024 labeled-address file (not by Spellbook or the Etherscan step) that funded 50+ interactors, with their Blockscout name, entity tags and contract flag; a record only, no label depends on it |
| `sensitivity_blockscout_rule.ipynb`, `review_blockscout_rule_group.json` | Primary vs automated labels, LightGBM on 10 group splits: test F1 −0.003 ± 0.010 |

The labeling rule and its results are in `docs/REVISION_LEAKAGE.md`, findings for 2026-09-30 and 2026-10-01.
