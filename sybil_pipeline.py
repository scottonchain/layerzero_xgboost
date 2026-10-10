"""Shared data pipeline for the LayerZero Sybil notebooks.

Builds the master feature table from the raw files, assigns gas provision trees,
and produces train / validation / test partitions. Every notebook imports
from here so that the split is defined in one place.

Split methods
-------------
'random' : address-level stratified split: the original paper's procedure and seed
           (the corrected table has one row fewer, so the partitions are not identical).
'group'  : stratified group split. No gas provision tree spans two partitions.

Gas provision tree (the key for the 'group' split)
--------------------------------------------------
The gas provision network links each address to its first funder. Removing every
edge that touches a labeled address (CEX, DEX, named entity) leaves a forest; each
of its trees is a gas provision tree. An address's tree is identified by its root:
walk toward the funders until the next hop is labeled, there is no further
provider, or a cycle; the last unlabeled address reached is the root. An address
funded directly by a labeled entity, or with no provider, is a root itself, so a
CEX hot wallet never joins its customers into one tree.

Temporal cutoff
---------------
The LayerZero snapshot table ends at 2024-05-01 23:59:59 UTC (the last L0
transaction in the data). Provision edges dated at or after SNAPSHOT_END are
dropped before any provider, chain or tree feature is computed.
"""
import gc
import json
import os
import re
import subprocess
import time
from statistics import variance

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

SEED = 42
TEST_FRAC = 0.30
VAL_FRAC = 0.30          # fraction of the non-test remainder -> 21 % of total
SPLIT_METHODS = ('random', 'group')
SNAPSHOT_END = '2024-05-02 00:00:00'   # UTC; exclusive upper bound for provision edges
# XGBoost 'hist' results depend on the thread count, so model notebooks fix it.
N_JOBS = 4
# LightGBM otherwise chooses col-wise or row-wise histograms by a timing test at startup (which
# depends on machine load) and sums in thread order; either changes the trees between runs.
LGBM_REPRO = dict(force_col_wise=True, deterministic=True)

# The 63 features of the submitted paper (order matters for NumPy splits). Chosen manually by
# the authors; no data-driven selection step.
FEATS_SUBMITTED = [
    'min_tx_value_out','gini_coefficient','cex_in_count',
    'leaf_gas_distribution_entropy','star_like_ratio','provider_is_star_like_attack',
    'leaf_gas_distribution_skewness','interactors_in_chain','provider_is_labeled',
    'provider_is_interactor','l0_to_eth_avg_native_drop_usd','l0_to_eth_max_native_drop_usd',
    'balance_factor','l0_tx_time_span','latest_l0_tx_time','time_span_in',
    'provider_total_gas_provision_amount','l0_avg_stargate_swap','avg_depth','breadth_factor',
    'indegree_per_block_in','gas_distribution_skewness','tree_size','l0_min_stargate_swap',
    'provider_max_gas_provision_amount','total_gas','num_transactions_in','branching_factor',
    'l0_to_eth_tx_time_span','n_l0_to_eth_source_contracts','n_l0_to_eth_projects',
    'provider_is_null','max_depth','leaf_provision_proportion',
    'n_l0_to_eth_project_per_source_chain','n_l0_to_eth_txs','earliest_l0_tx_time',
    'n_l0_projects','n_l0_to_eth_dest_contracts','provider_fan_out',
    'l0_to_eth_min_stargate_swap','n_l0_source_chains','longest_chain_ratio',
    'n_eth_interactions','provider_avg_gas_provision_amount','gas_distribution_entropy',
    'sparsity','max_tx_value_out','n_l0_source_contracts','gas_provision_block_number',
    'min_tx_value_in','tx_value_per_block_out','chain_length',
    'provider_min_gas_provision_amount','n_l0_to_eth_source_chains','depth',
    'l0_to_eth_max_stargate_swap','breadth_to_depth_ratio','leaf_to_internal_ratio',
    'earliest_tx_block_in','n_l0_project_per_source_chain','l0_to_eth_avg_stargate_swap',
    'is_provider'
]
# gini_coefficient was identically zero in the submission. After the formula was corrected, it was to be
# kept only if removing it lowered validation F1 on >= 8 of 10 group splits (rule fixed in advance,
# docs/REVISION_LEAKAGE.md). Removing it lowered F1 on 3 of 10 (01_ablation_gini), so it was removed and the
# reported results do not include the feature.
FEATS = [f for f in FEATS_SUBMITTED if f != 'gini_coefficient']

BURN_ADDRESS = '0x0000000000000000000000000000000000000000'

# Labeling rule (docs/REVISION_LEAKAGE.md, A12):
#   1. Public label datasets: the 2024 labeled-address file without its hand-added
#      addresses, plus Dune Spellbook's CEX, DEX and bridge lists at pinned commits.
#   2. Every other address that funded at least ETHERSCAN_MIN_FANOUT LayerZero interactors
#      was looked up on Etherscan and labeled if tagged as a shared service.
# 'presnapshot' uses only list entries known before the snapshot (sensitivity check).
LABEL_VINTAGES = ('current', 'presnapshot')
ETHERSCAN_MIN_FANOUT = 50
SPELLBOOK = {
    'current': ['cex_evms_addresses_current_9f61b0dd2.csv', 'dex_ethereum_addresses_current_d2ae74352.csv',
                'bridges_ethereum_current_9f61b0dd2.csv'],
    'presnapshot': ['cex_evms_addresses_presnapshot_839264841.csv', 'dex_ethereum_addresses_presnapshot_4eef8dc7e.csv',
                    'bridges_ethereum_presnapshot_ad08fa0cc.csv'],
}
SNAPSHOT_DATE = '2024-05-01'


def data_paths(data_dir='./data'):
    j = os.path.join
    return dict(
        l0=[j(data_dir, '20241104_layer0_sybil_features', f'l0_features_{s}.csv')
            for s in ['0_100000', '100000_200000', '200000_300000',
                      '300000_400000', '400000_500000']],
        gas_prov=j(data_dir, '20241114_gas_provision',
                   '20241114_1633_layer0_provision_network_000000000000.csv'),
        labeled=j(data_dir, '20241214_labeled_addresses', '20241214_labeled_addresses.csv'),
        labeled_readme=j(data_dir, '20241214_labeled_addresses', 'readme.txt'),
        hildobby_2024=j(data_dir, '20241013_hildobby_cex_evms', 'All_Known_EVM_CEX_Addresses.2024-10-13.csv'),
        spellbook_dir=j(data_dir, '20260128_dune_spellbook_labels'),
        etherscan_services=j(data_dir, '20260930_etherscan_service_labels', 'service_labels.csv'),
        # Original precomputed provider/tree features; used only as a regression check.
        tree_precomputed=j(data_dir, '20241117_tree_features', '20241117_graph_and_tree_features.csv'),
        cex=[j(data_dir, '20250208_cex_dex_indegree', f'cex_dex_features_in_{s}.csv')
             for s in [0, 100000, 200000, 300000, 400000]],
        sybil=j(data_dir, '20240915_final_sybil_list', 'fcfs_list.csv'),
    )


# ── Pipeline steps (00_data_pipeline.ipynb calls these one per cell) ──────────

def load_l0(paths):
    """Step 1: L0 base features, cleaned and deduplicated on addr."""
    chunks = []
    for path in paths:
        c = pd.read_csv(path, na_values=['null', 'NULL'])
        c.columns = c.columns.str.lower()
        chunks.append(c)
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    df = df[df['addr'] != BURN_ADDRESS]
    df = df.drop(columns=['in_degree', 'out_degree', 'rank'], errors='ignore')
    keep = ~df.drop(columns='addr').isnull().all(axis=1)
    return df[keep].drop_duplicates(subset='addr', keep='first').reset_index(drop=True)


def load_provision(path, cutoff=SNAPSHOT_END):
    """Step 2: provision edges (first ETH received) dated before the cutoff.

    One row per activated address with a known provider; amounts in ETH.
    The number of edges dropped by the cutoff is kept in .attrs.
    """
    f = pd.read_csv(path, na_values=['', 'null'])
    f.columns = f.columns.str.lower()
    t = pd.to_datetime(f['first_gas_provision_time'], utc=True)
    late = t >= pd.Timestamp(cutoff, tz='UTC')
    f = f[~late & f['gas_provider'].notna() & (f['activated_address'] != BURN_ADDRESS)]
    f = f.drop_duplicates(subset=['activated_address', 'gas_provider'], keep='first').copy()
    assert f['activated_address'].is_unique
    f['gas_provision_amount'] = pd.to_numeric(f['gas_provision_amount'], errors='coerce') / 1e18
    f.attrs['n_after_cutoff'] = int(late.sum())
    f.attrs['max_time'] = t[~late].max()
    return f


def provision_map(funding):
    """tfm = {activated_address: gas_provider}."""
    return dict(zip(funding['activated_address'], funding['gas_provider']))


def hand_added_addresses(readme_path):
    """Addresses the 2024 build script added by hand (labeled_addresses.add(...))."""
    text = open(readme_path).read()
    return {a.lower() for a in re.findall(r"labeled_addresses\.add\('(0x[0-9a-fA-F]{40})'", text)}


def stream_labeled_anchors(paths, candidates, vintage='current'):
    """Step 3a: labeled addresses among the candidates, by the labeling rule above.

    Pass every address in the provision network so the same set serves the
    provider flags, the chain walk, the gas provision trees, and the tree features.
    The 9M-line 2024 labeled-address file is streamed, never fully loaded.

    Note:  The "vintage" of a label does not affect its usability.  The identity 
    of "current" vintage addresses which are not in the "presnapshot" file 
    did not change over time.  The identity was *identified* after the presnapshot
    file, but the identity is consistent from the creation of the address. Therefore 
    all labels may be used.
    """
    assert vintage in LABEL_VINTAGES, vintage
    candidates = frozenset(candidates)
    drop = hand_added_addresses(paths['labeled_readme'])
    if vintage == 'presnapshot':
        h = pd.read_csv(paths['hildobby_2024'])
        drop |= set(h.loc[h['added_date'] > SNAPSHOT_DATE, 'address'].str.lower())
    anchors = set()
    with open(paths['labeled']) as f:
        for line in f:
            a = line.strip()
            if a and a in candidates and a not in drop:
                anchors.add(a)
    for fn in SPELLBOOK[vintage]:
        s = pd.read_csv(os.path.join(paths['spellbook_dir'], fn), dtype=str).fillna('')
        if vintage == 'presnapshot':
            s = s[(s['added_date'] == '') | (s['added_date'] <= SNAPSHOT_DATE)]
        anchors |= set(s['address']) & candidates
    if vintage == 'current':
        e = pd.read_csv(paths['etherscan_services'])
        anchors |= set(e.loc[e['labeled'].astype(str).str.lower() == 'true', 'address'].str.lower()) & candidates
    return anchors


def add_provider_flags(df, tfm, anchors):
    """Step 3b: provider_is_labeled / _interactor / _null."""
    iset = set(df['addr'])
    gp = df['addr'].map(tfm)
    df['provider_is_labeled'] = gp.isin(anchors)
    df['provider_is_interactor'] = gp.apply(lambda g: (g in iset) if pd.notna(g) else False)
    df['provider_is_null'] = gp.isna()
    return df


PROVIDER_COLS = ['provider_fan_out', 'provider_total_gas_provision_amount',
                 'provider_avg_gas_provision_amount', 'provider_max_gas_provision_amount',
                 'provider_min_gas_provision_amount', 'provider_is_star_like_attack']
TREE_COLS = ['tree_size', 'total_gas', 'max_depth', 'branching_factor', 'balance_factor',
             'leaf_provision_proportion', 'avg_depth', 'depth_variance', 'leaf_to_internal_ratio',
             'avg_leaf_gas', 'breadth_factor', 'breadth_to_depth_ratio', 'gini_coefficient',
             'gas_distribution_entropy', 'gas_distribution_skewness',
             'leaf_gas_distribution_entropy', 'leaf_gas_distribution_skewness', 'sparsity',
             'depth', 'longest_chain_ratio', 'star_like_ratio', 'depth_weighted_avg_gas']


def _entropy(x):
    if not x:
        return 0
    s = sum(x)
    if s <= 0:    # all-zero amounts (an unfunded singleton's leaf list): no distribution to measure
        return 0
    return -sum((v / s) * np.log2(v / s) for v in x if v / s > 0)


def _skew(x):
    """Biased sample skewness; 0 for (near-)constant data, as in scipy < 1.9."""
    if len(x) < 2:
        return 0
    a = np.asarray(x, dtype=float)
    m = a.mean()
    d = a - m
    m2, m3 = (d ** 2).mean(), (d ** 3).mean()
    if m2 <= (np.finfo(float).resolution * m) ** 2:
        return 0.0
    return m3 / m2 ** 1.5


def _gini(x):
    """Gini coefficient of non-negative amounts: 0 = equal, (n-1)/n = one holds all."""
    n = len(x)
    if n < 2:
        return 0.0
    a = np.sort(np.asarray(x, dtype=float))
    total = a.sum()
    if total <= 0:
        return 0.0
    return max(0.0, float(2 * np.sum(np.arange(1, n + 1) * a) / (n * total) - (n + 1) / n))


def provision_features(funding, interactors, anchors):
    """Step 4: provider_* and provision-tree features for each interactor.

    Ported from data/20241117_tree_features/20241117 Gas Provision
    Featurization.ipynb, which produced the precomputed file; same definitions.
    The provision forest excludes every edge that touches a labeled address; an
    interactor's tree features describe the whole tree of that forest containing it.
    That holds for roots too: a wallet funded by a labeled entity, or with no recorded
    funding at all, is the root of its own tree (a singleton if it funds nobody) and
    gets that tree's metrics with depth 0. Every interactor therefore has a row.
    """
    gas = funding.groupby('activated_address')['gas_provision_amount'].sum().to_dict()

    # provider_* over interactor edges only
    fi = funding[funding['activated_address'].isin(interactors)]
    g = fi.groupby('gas_provider')
    prov = pd.DataFrame({
        'provider_fan_out': g['activated_address'].nunique(),
        'provider_total_gas_provision_amount': g['gas_provision_amount'].sum(),
        'provider_max_gas_provision_amount': g['gas_provision_amount'].max(),
        'provider_min_gas_provision_amount': g['gas_provision_amount'].min(),
    })
    prov['provider_avg_gas_provision_amount'] = (prov['provider_total_gas_provision_amount']
                                                 / prov['provider_fan_out'])
    prov['provider_is_star_like_attack'] = ((prov['provider_fan_out'] > 1)
                                            & ~prov.index.isin(anchors)).astype(int)
    out = fi[['activated_address', 'gas_provider', 'gas_provision_amount', 'block_number']].merge(
        prov, left_on='gas_provider', right_index=True)
    out = out.rename(columns={'activated_address': 'addr',
                              'block_number': 'gas_provision_block_number'})

    # unlabeled provision forest
    parent, children = {}, {}
    for a, p in zip(funding['activated_address'], funding['gas_provider']):
        if p in anchors or a in anchors or a == p:
            continue
        parent[a] = p
        children.setdefault(p, []).append(a)

    cache = {}

    def tree_metrics(root):
        if root in cache:
            return cache[root]
        stack, seen, nodes = [(root, 0)], set(), []
        total_gas = weighted = depth_sum = 0
        max_depth, min_leaf = 0, float('inf')
        total_children = star_nodes = internal = leaves = 0
        leaf_gas, leaf_amts, provs, depths, breadths = 0, [], [], [], []
        while stack:
            cur, d = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            nodes.append(cur)
            depths.append(d)
            max_depth = max(max_depth, d)
            depth_sum += d
            while len(breadths) <= d:
                breadths.append(0)
            breadths[d] += 1
            ga = gas.get(cur, 0)
            if ga > 0 and cur != root:
                total_gas += ga
                provs.append(ga)
            weighted += ga * d
            if cur not in children:
                leaves += 1
                leaf_gas += ga
                leaf_amts.append(ga)
                min_leaf = min(min_leaf, d)
            else:
                internal += 1
                k = len(children[cur])
                total_children += k
                star_nodes += k > 1
                stack.extend((c, d + 1) for c in children[cur])
        n = len(nodes)
        m = dict(
            tree_size=n, total_gas=total_gas, max_depth=max_depth,
            branching_factor=total_children / n,
            balance_factor=max_depth - min_leaf if min_leaf != float('inf') else 0,
            leaf_provision_proportion=leaf_gas / total_gas if total_gas > 0 else 0,
            avg_depth=sum(depths) / n,
            depth_variance=variance(depths) if n > 1 else 0,
            leaf_to_internal_ratio=leaves / internal if internal > 0 else 0,
            avg_leaf_gas=sum(leaf_amts) / len(leaf_amts) if leaf_amts else 0,
            breadth_factor=sum(breadths) / len(breadths),
            # Gini coefficient of the provision amounts. The original featurization's
            # expression was identically 0 (see docs/REVISION_LEAKAGE.md, section C).
            gini_coefficient=_gini(provs),
            gas_distribution_entropy=_entropy(provs),
            gas_distribution_skewness=_skew(provs),
            leaf_gas_distribution_entropy=_entropy(leaf_amts),
            leaf_gas_distribution_skewness=_skew(leaf_amts),
            sparsity=sum(gas.get(x, 0) > 0 for x in nodes) / n,
            longest_chain_ratio=max_depth / n,
            star_like_ratio=star_nodes / n,
            depth_weighted_avg_gas=weighted / depth_sum if depth_sum > 0 else 0,
        )
        m['breadth_to_depth_ratio'] = m['breadth_factor'] / max_depth if max_depth > 0 else 0
        for x in nodes:
            cache[x] = m
        return m

    # Interactors with no recorded incoming edge have no provider_* values (zero after the merge)
    # but can still be the root of a tree; they get a row so that their tree features are computed.
    no_edge = sorted(set(interactors) - set(out['addr']))
    if no_edge:
        out = pd.concat([out, pd.DataFrame({'addr': no_edge})], ignore_index=True)

    rows = []
    for a in out['addr']:
        cur, depth, seen = a, 0, {a}
        while cur in parent and parent[cur] not in seen:
            cur = parent[cur]
            seen.add(cur)
            depth += 1
        m = dict(tree_metrics(cur))
        m['depth'] = depth
        rows.append(m)
    tf = pd.DataFrame(rows, columns=TREE_COLS, index=out.index).fillna(0).astype(float)
    out = pd.concat([out.drop(columns='gas_provider'), tf], axis=1).reset_index(drop=True)
    assert out['addr'].is_unique
    return out


def merge_provision_features(df, feats):
    """Step 4 (merge): interactors without a recorded provider get zero provider_* values."""
    return df.merge(feats, on='addr', how='left').fillna(0)


def compare_to_precomputed(feats, path):
    """Match rate of recomputed provider/tree features against the original file."""
    old = pd.read_csv(path, na_values=['', 'null']).drop_duplicates('addr').set_index('addr')
    new = feats.set_index('addr')
    com = old.index.intersection(new.index)
    rows = []
    for c in PROVIDER_COLS + TREE_COLS:
        if c == 'gini_coefficient':
            continue   # formula corrected; the original file's values are identically 0
        a = old.loc[com, c].fillna(0).astype(float).to_numpy()
        b = new.loc[com, c].astype(float).to_numpy()
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-9)
        rows.append(dict(feature=c, match_rate=ok.mean(), n_differ=int((~ok).sum())))
    return pd.DataFrame(rows), len(com), len(old), len(new)


def merge_cex_dex(df, paths):
    """Step 5: CEX / DEX in-degree counts."""
    cex = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    cex.columns = cex.columns.str.lower()
    cex = cex.drop_duplicates(subset='to_address', keep='first')
    df = df.merge(cex, left_on='addr', right_on='to_address', how='left')
    df = df.drop(columns='to_address', errors='ignore')
    df.columns = df.columns.str.lower()
    return df.fillna(0)


def add_labels(df, path, tfm):
    """Step 6: Sybil label and is_provider."""
    s = pd.read_csv(path)
    s.columns = s.columns.str.lower().str.strip()
    sybil_set = set(s['address'].str.lower())
    df['sybil'] = df['addr'].isin(sybil_set).astype(int)
    df['is_provider'] = df['addr'].isin(set(tfm.values())).astype(int)
    return df


def add_chain_features(df, tfm, anchors):
    """Step 7: chain_length and interactors_in_chain."""
    iset = set(df['addr'])
    clen, icnt = [], []
    for addr in df['addr']:
        length, interactors, cur, visited = 0, 0, addr, set()
        while True:
            if cur in visited: break          # cycle guard
            visited.add(cur)
            if cur in iset: interactors += 1
            if cur not in tfm: break          # no further provider
            prv = tfm[cur]
            if prv == cur: break              # self-loop
            length += 1
            if prv in anchors: break          # reached a known entity
            cur = prv
        clen.append(length)
        icnt.append(interactors)
    df['chain_length'] = clen
    df['interactors_in_chain'] = icnt
    return df


def add_category(df):
    """Taxonomy: IxL > IxI > IxE, plus 'No provider'."""
    df['category'] = np.select(
        [df['provider_is_null'], df['provider_is_labeled'], df['provider_is_interactor']],
        ['No provider', 'IxL', 'IxI'], default='IxE')
    return df


def provision_tree_root(addr, tfm, anchors):
    """Last unlabeled node on the provision chain above addr (see module doc)."""
    cur, seen = addr, set()
    while True:
        seen.add(cur)
        prv = tfm.get(cur)
        if prv is None or prv == cur or prv in anchors or prv in seen:
            return cur
        cur = prv


def add_provision_tree(df, tfm, anchors):
    df['provision_tree'] = [provision_tree_root(a, tfm, anchors) for a in df['addr']]
    return df


def build_master_df(data_dir='./data', verbose=True, label_vintage='current'):
    """Run steps 1-7 plus taxonomy and gas provision trees. Returns (df, tfm, anchors)."""
    p = data_paths(data_dir)
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    df = load_l0(p['l0'])
    log(f'[1] L0 features: {len(df):,} addresses, {len(df.columns)} cols')
    funding = load_provision(p['gas_prov'])
    tfm = provision_map(funding)
    log(f'[2] Gas provision: {len(tfm):,} edges before {SNAPSHOT_END} UTC '
        f'({funding.attrs["n_after_cutoff"]} later edges dropped)')
    anchors = stream_labeled_anchors(p, set(tfm) | set(tfm.values()), label_vintage)
    df = add_provider_flags(df, tfm, anchors)
    log(f'[3] Labeled addresses in provision network: {len(anchors):,}')
    df = merge_provision_features(df, provision_features(funding, set(df['addr']), anchors))
    log(f'[4] Provider and tree features: {len(df.columns)} cols')
    df = merge_cex_dex(df, p['cex'])
    log(f'[5] CEX/DEX merged: {len(df.columns)} cols')
    df = add_labels(df, p['sybil'], tfm)
    log(f'[6] Labels: Sybil={df.sybil.sum():,} ({df.sybil.mean()*100:.2f}%)')
    df = add_chain_features(df, tfm, anchors)
    df = add_category(df)
    df = add_provision_tree(df, tfm, anchors)
    log(f'[7] Chain features, taxonomy, gas provision trees '
        f'({df.provision_tree.nunique():,} trees)')
    assert df['addr'].is_unique, 'Duplicate addresses in master table'
    missing = [f for f in FEATS if f not in df.columns]
    assert not missing, f'Missing features: {missing}'
    log(f'All {len(FEATS)} features present ✓  ({time.time()-t0:.1f}s)')
    return df, tfm, anchors


# ── Splitting ────────────────────────────────────────────────────────────────

def _random_partition(df, seed):
    """Original paper split: two stratified train_test_split calls."""
    y = df['sybil']
    tr0, te = train_test_split(df.index, test_size=TEST_FRAC, random_state=seed, stratify=y)
    tr, va = train_test_split(tr0, test_size=VAL_FRAC, random_state=seed, stratify=y.loc[tr0])
    return np.asarray(tr), np.asarray(va), np.asarray(te)


# Group assignment similar to sklearn StratifiedGroupKFold.
def _group_partition(df, seed):
    """Stratified group split with the same 49 / 21 / 30 targets.

    The groups are gas provision trees. Multi-wallet trees are placed first,
    largest first, each into the partition whose per-class quota is least filled
    (relative to its target), weighted by the tree's class mix. Singletons then fill each partition's
    remaining per-class quota at random.
    """
    fracs = np.array([(1 - TEST_FRAC) * (1 - VAL_FRAC), (1 - TEST_FRAC) * VAL_FRAC, TEST_FRAC])
    rng = np.random.default_rng(seed)
    y = df['sybil'].to_numpy()
    n_cls = np.array([(y == 0).sum(), (y == 1).sum()])

    # [partition, class]: target non-Sybil and Sybil counts for train, validation, test
    target = np.outer(fracs, n_cls).astype(float)
    quota = target.copy()                                 # room left; shrinks as trees are placed

    g = df.groupby('provision_tree', sort=False)['sybil'].agg(['size', 'sum'])
    multi = g[g['size'] > 1]
    order = rng.permutation(len(multi))
    multi = multi.iloc[order].sort_values('size', ascending=False, kind='stable')
    part_of_tree = {}
    for grp, (size, n_syb) in zip(multi.index, multi[['size', 'sum']].to_numpy()):
        mix = np.array([size - n_syb, n_syb]) / n_cls

        # the partition with the most remaining room, relative to target, for this tree's classes
        p = int(np.argmax((quota / target) @ mix))
        quota[p] -= (size - n_syb, n_syb)
        part_of_tree[grp] = p

    part = np.array(df['provision_tree'].map(part_of_tree), dtype=float)

    # single-wallet trees fill each partition's remaining per-class quota at random
    single = np.isnan(part)
    for c in (0, 1):
        idx = np.flatnonzero(single & (y == c))
        idx = rng.permutation(idx)
        take = np.maximum(np.round(quota[:, c]), 0).astype(int)
        start = 0
        for p in (2, 1):                                  # test, then val; rest -> train
            part[idx[start:start + take[p]]] = p
            start += take[p]
        part[idx[start:]] = 0
    part = part.astype(int)
    ix = df.index.to_numpy()
    return ix[part == 0], ix[part == 1], ix[part == 2]


def make_splits(df, feats=FEATS, method='group', seed=SEED):
    """Partition df and upsample the Sybil class in train to 1:1.

    Returns a dict with X_/y_ train (balanced), val and test, the unbalanced
    train index (idx_train) plus idx_val / idx_test, and the method name.
    """
    assert method in SPLIT_METHODS, method
    if method == 'random':
        tr, va, te = _random_partition(df, seed)
    else:
        tr, va, te = _group_partition(df, seed)
        grp = df['provision_tree']
        s_tr, s_va, s_te = set(grp.loc[tr]), set(grp.loc[va]), set(grp.loc[te])
        assert not (s_tr & s_va or s_tr & s_te or s_va & s_te), \
            'A gas provision tree spans two partitions'

    X = df[feats].astype(np.float32)
    y = df['sybil'].astype(int)

    # Upsample Sybil (minority) in the training partition only, to 1:1
    trn = pd.concat([X.loc[tr], y.loc[tr].rename('sybil')], axis=1)
    maj = trn[trn['sybil'] == 0]
    mn = trn[trn['sybil'] == 1]
    mn_up = resample(mn, replace=True, n_samples=len(maj), random_state=seed)
    bal = pd.concat([maj, mn_up]).sample(frac=1, random_state=seed)

    return dict(
        method=method,
        X_train=bal.drop(columns='sybil').astype(np.float32), y_train=bal['sybil'],
        X_val=X.loc[va], y_val=y.loc[va],
        X_test=X.loc[te], y_test=y.loc[te],
        idx_train=tr, idx_val=va, idx_test=te,
    )


# Called by 00_data_pipeline, 03-05 (each records the leakage of the split it trained on) and
# 11_split_comparison. Under the group split the tree overlap is 0 by construction (make_splits
# asserts it); the random split is measured only in 11, to answer the reviewers' leakage question.
def leakage_report(df, s, verbose=True):
    """Row overlap and gas-provision-tree overlap between partitions.

    s: the split dict returned by make_splits; only s['method'] and the row indices
       s['idx_train'], s['idx_val'], s['idx_test'] (rows of df, before upsampling) are used.
    Asserts that no row is in two partitions. Returns a dict of counts: method, test_rows,
    test_sybils, test_rows_sharing_tree_with_train (test rows whose gas provision tree also has a
    training row) and test_sybils_sharing_tree_with_train_sybil (test Sybils whose tree also has a
    training Sybil).
    """
    tr, va, te = (set(s[k]) for k in ('idx_train', 'idx_val', 'idx_test'))
    row = dict(train_val=len(tr & va), train_test=len(tr & te), val_test=len(va & te))
    assert not any(row.values()), f'Row overlap between partitions: {row}'

    grp, syb = df['provision_tree'], df['sybil'] == 1
    tr_i, te_i = s['idx_train'], s['idx_test']
    g_tr = set(grp.loc[tr_i])
    g_tr_syb = set(grp.loc[tr_i][syb.loc[tr_i]])
    te_grp = grp.loc[te_i]
    te_syb = te_grp[syb.loc[te_i]]
    rep = dict(
        method=s['method'],
        test_rows=len(te_i),
        test_sybils=int(syb.loc[te_i].sum()),
        test_rows_sharing_tree_with_train=int(te_grp.isin(g_tr).sum()),
        test_sybils_sharing_tree_with_train_sybil=int(te_syb.isin(g_tr_syb).sum()),
    )
    if verbose:
        print(f"Split method: {rep['method']}")
        print(f"Row overlap (train/val, train/test, val/test): "
              f"{row['train_val']}, {row['train_test']}, {row['val_test']}")
        print(f"Test addresses sharing a gas provision tree with a train address: "
              f"{rep['test_rows_sharing_tree_with_train']:,} / {rep['test_rows']:,}")
        print(f"Test Sybils sharing a gas provision tree with a train Sybil:     "
              f"{rep['test_sybils_sharing_tree_with_train_sybil']:,} / {rep['test_sybils']:,}")
    return rep


def split_summary(df, s):
    """Per-partition counts before upsampling."""
    rows = []
    for name, key in [('Train', 'idx_train'), ('Validation', 'idx_val'), ('Test', 'idx_test')]:
        y = df.loc[s[key], 'sybil']
        rows.append(dict(Split=name, Total=len(y), Sybil=int(y.sum()),
                         NonSybil=int(len(y) - y.sum()), SybilPct=round(y.mean() * 100, 2)))
    return pd.DataFrame(rows)


def _git_commit():
    try:
        return subprocess.run(['git', 'describe', '--always', '--dirty'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


# Recorded when the kernel imports this module, i.e. the code that actually ran.
# Ends in '-dirty' if tracked files differed from that commit at import time.
CODE_COMMIT = _git_commit()


def best_f1_threshold(y_true, probs):
    """Threshold that maximises F1 on the given labels (used on validation only)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    return float(thr[np.argmax(f1[:-1])])


def load_selected_params(path='results/02_hyperparameter_search.json'):
    """Hyperparameters chosen on validation by 02_hyperparameter_search.ipynb."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: run 02_hyperparameter_search.ipynb first')
    with open(path) as f:
        hp = json.load(f)
    return hp['xgb_selected'], hp['lgbm_selected']


def save_results(models, notebook, leakage=None, extra=None, out_dir='results'):
    """Write scalar metrics for one notebook run to results/<notebook>.json, with the code commit."""
    os.makedirs(out_dir, exist_ok=True)
    scalars = lambda r: {k: (v.item() if isinstance(v, np.generic) else v)
                         for k, v in r.items() if np.isscalar(v)}
    record = dict(notebook=notebook, code_commit=CODE_COMMIT,
                  leakage=leakage, models=[scalars(r) for r in models], **(extra or {}))
    path = os.path.join(out_dir, f'{notebook}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'Saved {path}')


def load_results(notebook, results_dir='results'):
    path = os.path.join(results_dir, f'{notebook}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: run {notebook}.ipynb first')
    with open(path) as f:
        return json.load(f)


# Notebooks whose results depend on the hyperparameter search (and so on its code).
USES_SEARCH = ('03_xgboost_sybil', '04_lightgbm_sybil', '06_cross_ensemble_sybil',
               '07_sensitivity_label_vintage', '08_ablation_families', '09_shap_importance',
               '11_split_comparison')
# Notebooks whose results depend on saved predictions of the XGBoost and LightGBM notebooks.
USES_PREDICTIONS = {'06_cross_ensemble_sybil': ('03_xgboost_sybil', '04_lightgbm_sybil')}


# Called by 10_tie_out (results of 00-09) and 11_split_comparison (results of 03-06) before any
# number is shown; each asserts that every row's dependencies_unchanged is True, so a result produced
# by different code stops the notebook.
def provenance(notebooks, results_dir='results'):
    """For each notebook's saved result: was it produced by the code in the current commit?

    Compares, between the commit recorded in the result and HEAD: sybil_pipeline.py,
    requirements.txt and data/ byte for byte, and the code cells of the notebook itself (plus the
    search notebook, and the notebooks whose predictions it reads, where it depends on them).
    Saved outputs and markdown may differ. Results from a dirty working tree are rejected.
    """
    import nbformat
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True)

    def code_cells(ref, path):
        text = git('show', f'{ref}:{path}').stdout if ref else open(path).read()
        if not text:
            return None
        return [c.source for c in nbformat.reads(text, as_version=4).cells if c.cell_type == 'code']

    rows = []
    for nb in notebooks:
        commit = load_results(nb, results_dir)['code_commit']
        files_same = (not commit.endswith('-dirty')) and git(
            'diff', '--quiet', commit, 'HEAD', '--', 'sybil_pipeline.py', 'requirements.txt', 'data').returncode == 0
        deps = [nb] + (['02_hyperparameter_search'] if nb in USES_SEARCH else []) + list(USES_PREDICTIONS.get(nb, ()))
        # The code cells of the notebook and of the notebooks it depends on, as committed at the result's
        # commit and at HEAD; markdown and saved outputs are ignored.
        code_same = all(code_cells(commit, f'{d}.ipynb') == code_cells('HEAD', f'{d}.ipynb') for d in deps)
        rows.append(dict(notebook=nb, commit=commit, dependencies_unchanged=files_same and code_same))
    return pd.DataFrame(rows)


# ── Models (one definition, used by every notebook that trains a model) ──────

MODEL_SEEDS = (42, 123, 456)   # 3-seed ensembles in the model notebooks
LR_PARAMS = dict(C=0.01, penalty='l1', solver='liblinear', max_iter=1000, random_state=SEED)
BLEND_WEIGHTS = np.round(np.linspace(0, 1, 51), 2)   # XGBoost weight in the cross-ensemble


def xgb_model(params, seed):
    """XGBoost with the fixed settings; `params` are the tuned hyperparameters."""
    from xgboost import XGBClassifier
    return XGBClassifier(objective='binary:logistic', n_estimators=5000, eval_metric='logloss',
                         early_stopping_rounds=50, verbosity=0, tree_method='hist',
                         n_jobs=N_JOBS, random_state=seed, **params)


def lgbm_model(params, seed):
    """LightGBM with the fixed settings (bagging on, deterministic); `params` are tuned."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=5000, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                          n_jobs=N_JOBS, random_state=seed, **LGBM_REPRO, **params)


def fit_xgb(S, params, seeds=MODEL_SEEDS, feats=None, verbose=True):
    """Train one XGBoost per seed with early stopping on validation; average the probabilities."""
    Xtr, Xva, Xte = (S[k] if feats is None else S[k][feats] for k in ('X_train', 'X_val', 'X_test'))
    out = dict(val=[], test=[], rounds=[], models=[])
    for seed in seeds:
        t0 = time.time()
        m = xgb_model(params, seed)
        m.fit(Xtr, S['y_train'], eval_set=[(Xva, S['y_val'])], verbose=False)
        out['rounds'].append(int(m.best_iteration)); out['models'].append(m)
        out['val'].append(m.predict_proba(Xva)[:, 1]); out['test'].append(m.predict_proba(Xte)[:, 1])
        if verbose:
            print(f'  XGBoost seed={seed}  rounds={m.best_iteration}  {time.time() - t0:.1f}s', flush=True)
    out['val'], out['test'] = np.mean(out['val'], axis=0), np.mean(out['test'], axis=0)
    return out


def fit_lgbm(S, params, seeds=MODEL_SEEDS, feats=None, verbose=True):
    """Train one LightGBM per seed with early stopping on validation; average the probabilities."""
    import lightgbm as lgb
    Xtr, Xva, Xte = (S[k] if feats is None else S[k][feats] for k in ('X_train', 'X_val', 'X_test'))
    out = dict(val=[], test=[], rounds=[], models=[])
    for seed in seeds:
        t0 = time.time()
        m = lgbm_model(params, seed)
        m.fit(Xtr, S['y_train'], eval_set=[(Xva, S['y_val'])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        out['rounds'].append(int(m.best_iteration_)); out['models'].append(m)
        out['val'].append(m.predict_proba(Xva)[:, 1]); out['test'].append(m.predict_proba(Xte)[:, 1])
        if verbose:
            print(f'  LightGBM seed={seed}  rounds={m.best_iteration_}  {time.time() - t0:.1f}s', flush=True)
    out['val'], out['test'] = np.mean(out['val'], axis=0), np.mean(out['test'], axis=0)
    return out


def fit_lr(S):
    """L1 logistic regression on standardized features (the paper's untuned baseline)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    m = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(**LR_PARAMS))])
    m.fit(S['X_train'], S['y_train'])
    return dict(val=m.predict_proba(S['X_val'])[:, 1], test=m.predict_proba(S['X_test'])[:, 1], model=m)


def select_blend_weight(y_val, xgb_val, lgbm_val):
    """XGBoost weight w maximizing validation F1 of w*XGB + (1-w)*LGBM (threshold re-chosen per w)."""
    from sklearn.metrics import f1_score
    f1 = []
    for w in BLEND_WEIGHTS:
        p = w * xgb_val + (1 - w) * lgbm_val
        f1.append(f1_score(y_val, (p >= best_f1_threshold(y_val, p)).astype(int), zero_division=0))
    f1 = np.array(f1)
    return float(BLEND_WEIGHTS[np.argmax(f1)]), f1


def evaluate(name, y_val, val_probs, y_test, test_probs):
    """Test metrics at the F1-maximizing threshold chosen on validation (plus ROC/PR curves)."""
    from sklearn.metrics import (auc, average_precision_score, brier_score_loss, f1_score,
                                 precision_recall_curve, precision_score, recall_score, roc_curve)
    thr = best_f1_threshold(y_val, val_probs)
    y_pred = (test_probs >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    prc, rec, _ = precision_recall_curve(y_test, test_probs)
    return dict(name=name, threshold=thr,
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1=f1_score(y_test, y_pred, zero_division=0),
                auroc=auc(fpr, tpr), ap=average_precision_score(y_test, test_probs),
                brier=brier_score_loss(y_test, test_probs),
                fpr=fpr, tpr=tpr, prc=prc, rec_c=rec)


def print_metrics(r):
    for k, label in [('threshold', 'Threshold'), ('precision', 'Precision'), ('recall', 'Recall'),
                     ('f1', 'F1'), ('auroc', 'AUROC'), ('ap', 'Avg Prec'), ('brier', 'Brier')]:
        print(f'  {label:<10}: {r[k]:.4f}')


def plot_curves(results):
    """ROC and precision-recall curves for one or more evaluate() results."""
    import matplotlib.pyplot as plt
    colors = ['#1565C0', '#2E7D32', '#B71C1C', '#6A1B9A', '#E65100']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r, c in zip(results, colors):
        axes[0].plot(r['fpr'], r['tpr'], color=c, lw=2, label=f"{r['name']}  AUC={r['auroc']:.4f}")
        axes[1].plot(r['rec_c'], r['prc'], color=c, lw=2, label=f"{r['name']}  AP={r['ap']:.4f}")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    axes[1].legend(fontsize=9, loc='upper right'); axes[1].grid(alpha=0.2)
    plt.tight_layout(); plt.show()


def operating_points(y_test, test_probs, threshold, levels=(0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.50)):
    """Precision, recall, F1 and counts at fixed thresholds and at the validation threshold."""
    y = np.asarray(y_test)
    rows = []
    for t in sorted(set(levels) | {threshold}, reverse=True):
        pred = test_probs >= t
        if not pred.any():
            continue
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        p, r = tp / (tp + fp), tp / int(y.sum())
        rows.append(dict(threshold=round(t, 4), precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
                         flagged=int(pred.sum()), false_pos=fp,
                         note='validation threshold' if t == threshold else ''))
    return pd.DataFrame(rows)


def gain_importance(models, feats):
    """Total gain per feature, normalized to sum to 1 per model, averaged over models (seeds)."""
    imp = []
    for m in models:
        if hasattr(m, 'get_booster'):   # XGBoost; features never split on are absent
            g = m.get_booster().get_score(importance_type='total_gain')
            v = np.array([g.get(f, g.get(f'f{i}', 0.0)) for i, f in enumerate(feats)])
        else:                           # LightGBM: 'gain' is total gain
            v = m.booster_.feature_importance(importance_type='gain')
        imp.append(v / v.sum())
    return (pd.DataFrame({'Feature': feats, 'NormGain': np.mean(imp, axis=0)})
            .sort_values('NormGain', ascending=False).reset_index(drop=True))


def plot_importance(imp, title, color, n=20):
    """Horizontal bar chart of the top-n features from gain_importance()."""
    import matplotlib.pyplot as plt
    top = imp.head(n)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top['Feature'][::-1], top['NormGain'][::-1], color=color)
    ax.set(xlabel='Normalized total gain (mean over seeds)', title=title)
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout(); plt.show()


def metrics_by_category(y, probs, category, threshold):
    """Test metrics within each taxonomy category at a fixed threshold."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    y, probs, category = np.asarray(y), np.asarray(probs), np.asarray(category)
    rows = []
    for c in sorted(set(category)):
        k = category == c
        yc, pc = y[k], probs[k]
        yhat = (pc >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yc, yhat, labels=[0, 1]).ravel()
        rows.append(dict(category=c, n=int(k.sum()), sybil=int(yc.sum()),
                         precision=precision_score(yc, yhat, zero_division=0),
                         recall=recall_score(yc, yhat, zero_division=0),
                         f1=f1_score(yc, yhat, zero_division=0),
                         auroc=roc_auc_score(yc, pc) if len(set(yc)) == 2 else np.nan,
                         fpr=fp / (fp + tn) if (fp + tn) else np.nan))
    return pd.DataFrame(rows)


def save_predictions(df, S, notebook, **probs):
    """Validation and test probabilities, with address, category and label, to output/."""
    os.makedirs('output', exist_ok=True)
    for part, idx, y in [('val', S['idx_val'], S['y_val']), ('test', S['idx_test'], S['y_test'])]:
        pred = pd.DataFrame({'addr': df.loc[idx, 'addr'].to_numpy(),
                             'category': df.loc[idx, 'category'].to_numpy(), 'y': np.asarray(y),
                             **{k: v[part] for k, v in probs.items()}})
        path = f'output/pred_{notebook}_{part}.parquet'
        pred.to_parquet(path, index=False)
        print(f'Saved {path} ({len(pred):,} rows)')


def load_predictions(notebook, part):
    return pd.read_parquet(f'output/pred_{notebook}_{part}.parquet')


# ── Ablation and sensitivity helpers (notebooks 01, 07-09) ───────────────────

SPLIT_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 10 group splits

# Used to evaluate the impact of features by family (08, 09)
FEATURE_FAMILIES = {
    'LayerZero transactions': [
        'l0_tx_time_span', 'latest_l0_tx_time', 'earliest_l0_tx_time', 'l0_avg_stargate_swap',
        'l0_min_stargate_swap', 'n_l0_source_chains', 'n_l0_source_contracts', 'n_l0_projects',
        'n_l0_project_per_source_chain', 'n_eth_interactions', 'l0_to_eth_avg_native_drop_usd',
        'l0_to_eth_max_native_drop_usd', 'l0_to_eth_tx_time_span', 'n_l0_to_eth_source_contracts',
        'n_l0_to_eth_projects', 'n_l0_to_eth_project_per_source_chain', 'n_l0_to_eth_txs',
        'n_l0_to_eth_dest_contracts', 'l0_to_eth_min_stargate_swap', 'n_l0_to_eth_source_chains',
        'l0_to_eth_max_stargate_swap', 'l0_to_eth_avg_stargate_swap'],
    'Ethereum transactions': [
        'min_tx_value_out', 'max_tx_value_out', 'min_tx_value_in', 'time_span_in',
        'indegree_per_block_in', 'num_transactions_in', 'tx_value_per_block_out',
        'earliest_tx_block_in', 'cex_in_count'],
    'Gas provider': [
        'provider_is_labeled', 'provider_is_interactor', 'provider_is_null',
        'provider_is_star_like_attack', 'provider_fan_out', 'provider_total_gas_provision_amount',
        'provider_max_gas_provision_amount', 'provider_min_gas_provision_amount',
        'provider_avg_gas_provision_amount', 'is_provider', 'gas_provision_block_number'],
    'Gas provision tree': [
        'leaf_gas_distribution_entropy', 'leaf_gas_distribution_skewness',
        'star_like_ratio', 'balance_factor', 'avg_depth', 'breadth_factor', 'gas_distribution_skewness',
        'gas_distribution_entropy', 'tree_size', 'total_gas', 'branching_factor', 'max_depth',
        'leaf_provision_proportion', 'longest_chain_ratio', 'sparsity', 'breadth_to_depth_ratio',
        'leaf_to_internal_ratio'],
    'Provision chain': ['chain_length', 'interactors_in_chain', 'depth'],
}
_fam = [f for v in FEATURE_FAMILIES.values() for f in v]
assert sorted(_fam) == sorted(FEATS), 'FEATURE_FAMILIES must partition FEATS'


def lgbm_fit_eval(S, feats, params, model_seeds=(42,)):
    """Train LightGBM on the features `feats`; threshold on validation; return val and test metrics."""
    from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
    r = fit_lgbm(S, params, seeds=model_seeds, feats=feats, verbose=False)
    pv, pt = r['val'], r['test']
    thr = best_f1_threshold(S['y_val'], pv)
    out = dict(threshold=thr)
    for part, y, p in [('val', S['y_val'], pv), ('test', S['y_test'], pt)]:
        out[f'{part}_f1'] = f1_score(y, (p >= thr).astype(int))
        out[f'{part}_ap'] = average_precision_score(y, p)
        out[f'{part}_auroc'] = roc_auc_score(y, p)
    return out
