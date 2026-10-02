# Labels resting on the 2024 labeled-address file alone

Of the 4,308 labeled addresses in the provision network, 3,466 are labeled only by the 2024 labeled-address
file (not by Dune Spellbook or the Etherscan step); that file survives only as the merged list of its six
sources. This table records Blockscout's view of the 32 of them that funded at least 50 interactors (the
same threshold as the Etherscan step). 27 carry a Blockscout entity tag (bridges, mixers, mining pools,
batch-send tools). A record only: no label depends on it.

| File | Contents |
|---|---|
| `build_2024_only_evidence.py` | Selects the 32 addresses and reads Blockscout for each |
| `labels_2024_only_evidence.csv` | Address, interactors funded, Blockscout name, entity tags, contract flag, read date |

Run from the repo root (network access to Blockscout):

    python review_support/labels_2024_only/build_2024_only_evidence.py
