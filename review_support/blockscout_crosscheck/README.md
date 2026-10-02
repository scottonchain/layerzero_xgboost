# Blockscout cross-check of the Etherscan tags

The labeling rule's second step labels a funder of 50+ interactors when its Etherscan name tag identifies a
shared service. This check reads the same 65 addresses from a second, open explorer (Blockscout's API and its
tag service) and records whether Blockscout also names an entity. Result: Blockscout agrees on 34 of the 36
addresses in the rule's scope. A check only: no label depends on it.

| File | Contents |
|---|---|
| `blockscout_crosscheck.py` | Reads Blockscout for every address in `data/20260930_etherscan_service_labels/etherscan_lookups.csv` |
| `blockscout_crosscheck.csv` | The Etherscan and Blockscout tags side by side, with the read date |

Run from the repo root (network access to `eth.blockscout.com` and `metadata.services.blockscout.com`):

    python review_support/blockscout_crosscheck/blockscout_crosscheck.py

Tags can change after the read date, so a rerun may differ slightly.
