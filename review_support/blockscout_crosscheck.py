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
OUT = 'review_support/blockscout_crosscheck.csv'


def lookup(address):
    with urllib.request.urlopen(API.format(address), timeout=30) as r:
        d = json.load(r)
    tags = [t.get('name', '') for t in (d.get('metadata') or {}).get('tags', [])]
    tags += [t.get('display_name', '') for t in d.get('public_tags') or []]
    return dict(blockscout_name=d.get('name') or '', blockscout_tags='; '.join(t for t in tags if t),
                blockscout_is_contract=bool(d.get('is_contract')))


def main():
    t = pd.read_csv('review_support/etherscan_lookups.csv')
    today = datetime.date.today().isoformat()
    rows = []
    for a in t['address']:
        rows.append(dict(address=a, checked_on=today, **lookup(a)))
        time.sleep(0.5)   # stay well inside the public rate limit
    b = pd.DataFrame(rows)
    m = t[['address', 'in_rule_scope', 'name_tag', 'entity_type', 'labeled']].merge(b, on='address')
    m['blockscout_named'] = (m['blockscout_name'] != '') | (m['blockscout_tags'] != '')
    m['etherscan_named'] = m['entity_type'].isin(
        ['exchange_hot_wallet', 'bridge_or_relayer', 'protocol_contract', 'other_service', 'exchange_deposit'])
    m.to_csv(OUT, index=False)
    s = m[m.in_rule_scope]
    print(f'{len(m)} addresses checked; in rule scope {len(s)}')
    print(pd.crosstab(s['etherscan_named'], s['blockscout_named'],
                      rownames=['Etherscan named'], colnames=['Blockscout named']).to_string())


if __name__ == '__main__':
    main()
