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

The labeling rule and its results are in `docs/REVISION_LEAKAGE.md`, findings for 2026-09-30 and 2026-10-01.
