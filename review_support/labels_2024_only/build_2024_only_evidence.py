"""Evidence for the high-fanout addresses labeled only by the 2024 labeled-address file (tracker A12).

Scope: addresses that the 2024 file labels (its 44 hand additions excluded), that neither Dune
Spellbook nor the Etherscan step labels, and that funded at least `sp.ETHERSCAN_MIN_FANOUT` LayerZero
interactors in the filtered network. These are the labels with the most weight that rest on the
2024 file alone. For each, Blockscout's explorer API and tag service are read (the same lookup as
`review_support/blockscout_crosscheck/blockscout_crosscheck.py`). This is a record only: no label depends on it.

Writes review_support/labels_2024_only/labels_2024_only_evidence.csv. Run from the repo root:
    python review_support/labels_2024_only/build_2024_only_evidence.py
"""
import datetime
import os
import sys
import time

import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'review_support/blockscout_crosscheck')
import sybil_pipeline as sp
from blockscout_crosscheck import lookup

OUT = 'review_support/labels_2024_only/labels_2024_only_evidence.csv'


def main():
    P = sp.data_paths()
    tfm = sp.provision_map(sp.load_provision(P['gas_prov']))
    iset = set(sp.load_l0(P['l0'])['addr'])
    fo = pd.Series(tfm)[lambda s: s.index.isin(iset)].value_counts()   # as in build_etherscan_lookups.py
    funders = set(fo[fo >= sp.ETHERSCAN_MIN_FANOUT].index)

    drop = sp.hand_added_addresses(P['labeled_readme'])
    file_2024 = set()
    with open(P['labeled']) as f:
        for line in f:
            a = line.strip()
            if a in funders and a not in drop:
                file_2024.add(a)
    spellbook = set()
    for fn in sp.SPELLBOOK['current']:
        spellbook |= set(pd.read_csv(os.path.join(P['spellbook_dir'], fn), dtype=str)['address'])
    e = pd.read_csv(P['etherscan_services'])
    etherscan = set(e.loc[e['labeled'].astype(str).str.lower() == 'true', 'address'].str.lower())
    scope = sorted(file_2024 - spellbook - etherscan, key=lambda a: -fo[a])

    today = datetime.date.today().isoformat()
    rows = []
    for a in scope:
        rows.append(dict(address=a, interactors_funded=int(fo[a]), checked_on=today,
                         blockscout_url='https://eth.blockscout.com/address/' + a,
                         etherscan_url='https://etherscan.io/address/' + a, **lookup(a)))
        time.sleep(0.5)   # stay well inside the public rate limit
    t = pd.DataFrame(rows)
    # an entity tag, not a verified contract's code name (e.g. GnosisSafeProxy, ERC1967Proxy)
    t['blockscout_entity_tag'] = t['blockscout_name_tags'] != ''
    t.to_csv(OUT, index=False)
    print(f'{len(t)} addresses labeled only by the 2024 file with {sp.ETHERSCAN_MIN_FANOUT}+ interactors funded; '
          f'with a Blockscout entity tag: {int(t.blockscout_entity_tag.sum())}')


if __name__ == '__main__':
    main()
