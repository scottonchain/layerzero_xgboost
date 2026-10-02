# %% [markdown]
# # Review support — hand-added labeled addresses (tracker A12)
#
# `data/20241214_labeled_addresses/readme.txt` adds 44 addresses to the labeled set by hand. Labeling a
# funder changes the features of every wallet it funded, so reviewers can ask how they were chosen.
# The authors' recollection: high-volume addresses (for example Coinbase hot wallets) that rose to the top
# of features like provider fan-out, confirmed on Etherscan.
#
# 20 of the 44 are also in the hildobby CEX list (`data/20241013_hildobby_cex_evms`, loaded by the same build
# script), so their labels did not depend on the hand step. Rankings below treat only the other 24 ("hand-only")
# as hand-labeled.
#
# This notebook reconstructs the selection from repository data only:
# 1. Which of the 44 affect the dataset at all.
# 2. Whether a label-free ranking (interactor fan-out, network fan-out, tree size) explains the selection,
#    on the current and the archived provision network.
# 3. Which high-ranked funders were *not* added (exceptions a label-free rule would have to explain).
# 4. Two sheets for the Etherscan check (`hand_labeled_addresses.csv`, `top_funders_not_added.csv`). The tags
#    read on Etherscan are recorded in `etherscan_lookups.csv` (built by `build_etherscan_lookups.py`).
#
# Sybil rates of funded wallets are reported only as an audit diagnostic; they are not a selection criterion.

# %%
import os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, '..')
import sybil_pipeline as sp
pd.set_option('display.width', 250)
DATA = '../data'
P = sp.data_paths(DATA)
ARCHIVE = os.path.join(DATA, '20241114_gas_provision', 'archive', '20241114_layerzero_provisions_000000000000.csv')
readme = open(os.path.join(DATA, '20241214_labeled_addresses', 'readme.txt')).read()
HAND = list(dict.fromkeys(a.lower() for a in re.findall(r"labeled_addresses\.add\('(0x[0-9a-fA-F]{40})'", readme)))
labfile = set(l.strip() for l in open(P['labeled']))
HILDOBBY = set(pd.read_csv(os.path.join(DATA, '20241013_hildobby_cex_evms',
                                        'All_Known_EVM_CEX_Addresses.2024-10-13.csv'))['address'].str.lower())
HAND_ONLY = [a for a in HAND if a not in HILDOBBY]   # hand-added and not in the hildobby CEX list
CODE_ADDED = '0x9241f27daffd0bb1df4f2a022584dd6c77843e64'   # one more addition in the 2024 build code (removed in the revision)
OTHER_LABELED = (labfile - set(HAND_ONLY)) | {CODE_ADDED}   # the 2024 labeled set without the 24 hand-only additions
df = sp.load_l0(P['l0'])
syb = set(pd.read_csv(P['sybil'])['Address'].str.lower())
INTER = set(df['addr'])
print(f'Hand-added: {len(HAND)} ({len(set(HAND) & labfile)} present in the labeled file); '
      f'also in hildobby: {len(HAND) - len(HAND_ONLY)}; hand-only: {len(HAND_ONLY)}')


# %%
def profile(path, tag):
    f = pd.read_csv(path, usecols=['activated_address', 'gas_provider'], na_values=['', 'null'])
    f = f[f.gas_provider.notna()].drop_duplicates(['activated_address', 'gas_provider'])
    f['inter'] = f.activated_address.isin(INTER)
    f['sybil'] = f.activated_address.isin(syb) & f.inter
    g = f.groupby('gas_provider').agg(net_fanout=('activated_address', 'size'), int_fanout=('inter', 'sum'),
                                      sybil_funded=('sybil', 'sum'))
    g = g[~g.index.isin(OTHER_LABELED)].copy()            # candidates: not labeled by a public source
    # unlabeled forest (public labels only): subtree size under each candidate
    ok = f[~f.gas_provider.isin(OTHER_LABELED) & ~f.activated_address.isin(OTHER_LABELED)
           & (f.gas_provider != f.activated_address)]
    ch = {}
    for a, p in zip(ok.activated_address, ok.gas_provider):
        ch.setdefault(p, []).append(a)
    def subtree(r):
        seen, st = set(), [r]
        while st:
            x = st.pop()
            if x not in seen:
                seen.add(x); st += ch.get(x, [])
        return len(seen)
    top = g.sort_values('net_fanout', ascending=False).head(3000).index.union(pd.Index(HAND).intersection(g.index))
    g['subtree'] = np.nan
    g.loc[top, 'subtree'] = [subtree(a) for a in top]
    for c in ['int_fanout', 'net_fanout', 'subtree']:
        g[f'rank_{c}'] = g[c].rank(ascending=False, method='min')
    g['hand'] = g.index.isin(HAND_ONLY)
    g['in_network'] = True
    acts = set(f.activated_address)
    return g.add_prefix(f'{tag}_'), acts

cur, acts_cur = profile(P['gas_prov'], 'cur')
arc, acts_arc = profile(ARCHIVE, 'arc')
t = pd.DataFrame(index=pd.Index(HAND, name='addr'))
t['list_position'] = range(1, len(HAND) + 1)
t['is_interactor'] = t.index.isin(INTER)
t['funds_anyone_cur'] = t.index.isin(cur.index)
t['funds_anyone_arc'] = t.index.isin(arc.index)
t['activated_in_network'] = t.index.isin(acts_cur | acts_arc)
t = t.join(cur.drop(columns=['cur_hand', 'cur_in_network'])).join(arc.drop(columns=['arc_hand', 'arc_in_network']))
effective = t.funds_anyone_cur | t.funds_anyone_arc | t.is_interactor
print(f'Hand-added addresses that fund anyone in either network, or are interactors: {effective.sum()} of {len(t)}')
print(f'No role in either network (labeling them changes nothing): {(~effective).sum()}')
t[['list_position', 'is_interactor', 'funds_anyone_cur', 'cur_int_fanout', 'cur_rank_int_fanout',
   'cur_net_fanout', 'cur_rank_net_fanout', 'arc_subtree', 'arc_rank_subtree']]

# %% [markdown]
# ## Does a label-free ranking explain the selection?
#
# For each ranking: how many of the effective hand-added addresses sit in its top N, and how many top-N funders were *not* added.

# %%
rows = []
for tag, g in [('cur', cur), ('arc', arc)]:
    for c in ['int_fanout', 'net_fanout', 'subtree']:
        r = g[f'{tag}_rank_{c}']
        for n in [10, 25, 50, 100]:
            topn = r[r <= n].index
            rows.append(dict(network=tag, ranking=c, top_n=n, hand_in_top=int(pd.Index(HAND_ONLY).isin(topn).sum()),
                             top_not_added=int((~topn.isin(HAND)).sum())))
cov = pd.DataFrame(rows)
cov.pivot_table(index=['network', 'ranking'], columns='top_n', values=['hand_in_top', 'top_not_added'])

# %% [markdown]
# ## Exceptions: top funders that were not added
#
# Top 30 candidates by network fan-out on the archived network (the likely working version at the time), with the Sybil share of the interactors they funded. If a label-free rule was used, these need a reason other than their funded wallets' labels.

# %%
ex = arc.sort_values('arc_net_fanout', ascending=False).head(30).copy()
ex['sybil_rate_funded'] = (ex.arc_sybil_funded / ex.arc_int_fanout.replace(0, np.nan)).round(3)
ex[['arc_net_fanout', 'arc_int_fanout', 'arc_subtree', 'arc_hand', 'sybil_rate_funded']]

# %%
# Audit diagnostic: Sybil share among interactors funded by added vs not-added top funders
top100 = arc[arc.arc_rank_net_fanout <= 100]
for flag, grp in top100.groupby('arc_hand'):
    print(f"{'added' if flag else 'not added'}: {len(grp)} funders, {int(grp.arc_int_fanout.sum()):,} funded interactors, "
          f"Sybil share {grp.arc_sybil_funded.sum() / max(grp.arc_int_fanout.sum(), 1) * 100:.2f}%")

# %% [markdown]
# ## Sheets for the Etherscan check
#
# Tags read on Etherscan are recorded in `etherscan_lookups.csv`, not in these sheets.

# %%
sheet = t[['list_position', 'is_interactor', 'funds_anyone_cur', 'funds_anyone_arc',
           'cur_int_fanout', 'cur_net_fanout', 'arc_subtree']].copy()
sheet['affects_features'] = effective
hname = pd.read_csv(os.path.join(DATA, '20241013_hildobby_cex_evms', 'All_Known_EVM_CEX_Addresses.2024-10-13.csv'))
hname['address'] = hname['address'].str.lower()
sheet['hildobby_name'] = sheet.index.map(hname.drop_duplicates('address').set_index('address')['cex_name'])
sheet['etherscan_url'] = 'https://etherscan.io/address/' + sheet.index
sheet.to_csv('hand_labeled_addresses.csv')
ex.assign(etherscan_url='https://etherscan.io/address/' + ex.index) \
  .to_csv('top_funders_not_added.csv')
print('Wrote hand_labeled_addresses.csv and top_funders_not_added.csv')
