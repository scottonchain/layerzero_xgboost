"""Cross-check the Etherscan tags with a second, open explorer (Blockscout; tracker A12).

For every address in `etherscan_lookups.csv`, read its public name and tags from the Blockscout
API and record whether Blockscout also identifies a named entity. This is a check only: the
labeling rule still uses the Etherscan tags in `service_labels.csv`.

Writes review_support/blockscout_crosscheck.csv. Run from the repo root:
    python review_support/blockscout_crosscheck.py
"""
import datetime
import json
import time
import urllib.request

import pandas as pd

API = 'https://eth.blockscout.com/api/v2/addresses/{}'
META = 'https://metadata.services.blockscout.com/api/v1/metadata?addresses={}&chainId=1'
OUT = 'review_support/blockscout_crosscheck.csv'


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'layerzero_xgboost-review/1.0'})   # default UA gets 403
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def lookup(address):
    """Name and contract flag from the explorer API; entity tags from Blockscout's metadata service."""
    d = get(API.format(address))
    meta = {k.lower(): v for k, v in (get(META.format(address)).get('addresses') or {}).items()}
    tags = (meta.get(address.lower()) or {}).get('tags', [])
    named = [t['name'] for t in tags if t.get('tagType') in ('name', 'protocol')]
    named += [t.get('display_name', '') for t in d.get('public_tags') or []]
    generic = [t['name'] for t in tags if t.get('tagType') not in ('name', 'protocol')]
    return dict(blockscout_name=d.get('name') or '', blockscout_name_tags='; '.join(t for t in named if t),
                blockscout_generic_tags='; '.join(generic), blockscout_is_contract=bool(d.get('is_contract')))


def main():
    t = pd.read_csv('review_support/etherscan_lookups.csv')
    today = datetime.date.today().isoformat()
    rows = []
    for a in t['address']:
        rows.append(dict(address=a, checked_on=today, **lookup(a)))
        time.sleep(0.5)   # stay well inside the public rate limit
    b = pd.DataFrame(rows)
    m = t[['address', 'in_rule_scope', 'name_tag', 'entity_type', 'labeled']].merge(b, on='address')
    m['blockscout_named'] = (m['blockscout_name'].fillna('') != '') | (m['blockscout_name_tags'].fillna('') != '')
    m['etherscan_named'] = m['entity_type'].isin(
        ['exchange_hot_wallet', 'bridge_or_relayer', 'protocol_contract', 'other_service', 'exchange_deposit'])
    m.to_csv(OUT, index=False)
    s = m[m.in_rule_scope]
    print(f'{len(m)} addresses checked; in rule scope {len(s)}')
    print(pd.crosstab(s['etherscan_named'], s['blockscout_named'],
                      rownames=['Etherscan named'], colnames=['Blockscout named']).to_string())


if __name__ == '__main__':
    main()
