"""Build data/20260930_etherscan_service_labels/service_labels.csv (labeling rule step 2).

Candidates: every address that funded >= sp.ETHERSCAN_MIN_FANOUT LayerZero interactors and is not
covered by the public label datasets. Tags come from the review_support/etherscan_tags_*.csv files.
Run from the repo root: python review_support/build_etherscan_service_labels.py
"""
import sys; sys.path.insert(0, ".")
import os, pandas as pd, sybil_pipeline as sp
P = sp.data_paths()
f = sp.load_provision(P['gas_prov']); tfm = sp.provision_map(f)
iset = set(sp.load_l0(P['l0']).addr)
fo = pd.Series(tfm)[lambda s: s.index.isin(iset)].value_counts()
# public labels only (no Etherscan step): stream with a temporary empty services file
out = P['etherscan_services']
# public labels only: the Etherscan step itself must not shrink its own candidate list
empty = '/tmp/_no_etherscan_services.csv'
pd.DataFrame(columns=['address', 'label']).to_csv(empty, index=False)
pub = sp.stream_labeled_anchors({**P, 'etherscan_services': empty}, set(fo.index), 'current')
cand = fo[(fo >= sp.ETHERSCAN_MIN_FANOUT) & ~fo.index.isin(pub)]
tags = pd.concat([pd.read_csv('review_support/etherscan_tags_2026-09-30.csv')[['address', 'etherscan_name_tag', 'etherscan_category']],
                  pd.read_csv('review_support/etherscan_tags_top30_2026-09-30.csv')[['address', 'etherscan_name_tag', 'etherscan_category']]])
extra = 'review_support/etherscan_tags_threshold50_2026-10-01.csv'
if os.path.exists(extra):
    tags = pd.concat([tags, pd.read_csv(extra)[['address', 'etherscan_name_tag', 'etherscan_category']]])
tags = tags.drop_duplicates('address', keep='last').set_index('address').fillna('')
rows = []
for a, n in cand.items():
    if a not in tags.index:
        rows.append(dict(address=a, interactors_funded=int(n), etherscan_name_tag='', etherscan_category='',
                         label=False, reason='lookup pending')); continue
    t, c = tags.loc[a, 'etherscan_name_tag'], tags.loc[a, 'etherscan_category']
    if not t:
        lab, why = False, 'no entity tag' + (f' ({c})' if c else '')
    elif 'Deposit' in c or t.startswith(('OKX Dep', 'Binance Dep')) or ' Dep:' in t:
        lab, why = False, 'exchange deposit address (one customer account)'
    else:
        lab, why = True, 'shared service'
    rows.append(dict(address=a, interactors_funded=int(n), etherscan_name_tag=t, etherscan_category=c, label=lab, reason=why))
df = pd.DataFrame(rows).sort_values('interactors_funded', ascending=False)
df.to_csv(out, index=False)
print(df.to_string(index=False)); print(df.reason.value_counts().to_dict())
