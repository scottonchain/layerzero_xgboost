# Review support (non-essential)

Supporting files used to respond to the reviewers' comments on the labeled addresses (tracker item A12).
Nothing in the pipeline or the notebooks `00` to `11` reads this folder, and no reported number depends
on it; every result can be reproduced without it. Each subfolder has its own README.

| Folder | What it supports |
|---|---|
| `blockscout_crosscheck/` | A second explorer (Blockscout) agrees with the Etherscan tags behind the labeling rule |
| `blockscout_rule_sensitivity/` | Replacing the Etherscan step with a fully automated Blockscout rule barely changes the results |
| `labels_2024_only/` | The most influential labels that rest on the 2024 labeled-address file alone are recognizable services |

The labeling rule itself, and the Etherscan lookups the pipeline uses, are in
`data/20260930_etherscan_service_labels/`. Files used only in intermediate steps of the revision
(the reconstruction of the 2024 hand additions and early funder rankings) were removed; they remain in
the git history, and their findings are in `docs/REVISION_LEAKAGE.md`.
