"""Automated alternative to the Etherscan step of the labeling rule (robustness check; tracker A12).

Rule, fixed before any model was trained on it: an address that funded at least 2 addresses in the
filtered provision network is labeled if Blockscout's tag service gives it a generic tag in SERVICE and
none in EXCLUDE. No interactor threshold and no human step. The primary rule (public lists plus the
Etherscan check of funders of 50+ interactors) is unchanged; this only feeds
`sensitivity_blockscout_rule.ipynb`.

Writes
  review_support/blockscout_tags.csv    every funder of 2+ addresses that has any Blockscout tag
  review_support/blockscout_rule.csv    address, labeled (the rows the alternative rule labels)

Run from the repo root: python review_support/build_blockscout_rule.py
"""
import datetime
import json
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, '.')
import sybil_pipeline as sp

META = 'https://metadata.services.blockscout.com/api/v1/metadata?addresses={}&chainId=1'
SERVICE = {'Exchange', 'HOT WALLET', 'Bridge', 'DEX', 'Router', 'Fiat Gateway', 'Layer 2', 'Derivatives',
           'Payments', 'OTC'}
EXCLUDE = {'DEPOSIT ADDRESS'}   # one customer's account at an exchange, not a shared service
BATCH = 200                      # 500 exceeds the service's URL limit


def fetch(addrs):
    req = urllib.request.Request(META.format(','.join(addrs)),
                                 headers={'User-Agent': 'layerzero_xgboost-review/1.0'})   # default UA gets 403
    with urllib.request.urlopen(req, timeout=60) as r:
        return {k.lower(): v for k, v in (json.load(r).get('addresses') or {}).items()}


def main():
    f = sp.load_provision(sp.data_paths()['gas_prov'])
    fan = f.groupby('gas_provider')['activated_address'].nunique()
    funders = sorted(fan[fan >= 2].index)
    tags = {}
    for i in range(0, len(funders), BATCH):
        tags.update(fetch(funders[i:i + BATCH]))
    today = datetime.date.today().isoformat()
    rows = []
    for a in funders:
        ts = (tags.get(a) or {}).get('tags', [])
        if not ts:
            continue
        generic = {t['name'] for t in ts if t['tagType'] == 'generic'}
        rows.append(dict(address=a, checked_on=today, addresses_funded=int(fan[a]),
                         name_tags='; '.join(t['name'] for t in ts if t['tagType'] == 'name'),
                         protocol_tags='; '.join(t['name'] for t in ts if t['tagType'] == 'protocol'),
                         generic_tags='; '.join(sorted(generic)),
                         labeled=bool(generic & SERVICE) and not (generic & EXCLUDE)))
    t = pd.DataFrame(rows)
    t.to_csv('review_support/blockscout_tags.csv', index=False)
    t.loc[t.labeled, ['address', 'labeled']].to_csv('review_support/blockscout_rule.csv', index=False)
    print(f'{len(funders):,} funders of 2+ addresses; {len(t):,} with any tag; {int(t.labeled.sum()):,} labeled')


if __name__ == '__main__':
    main()
