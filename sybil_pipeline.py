"""Shared data pipeline for the LayerZero Sybil notebooks.

Builds the master feature table from the raw files, assigns funding groups,
and produces train / validation / test partitions. Every notebook imports
from here so that the split is defined in one place.

Split methods
-------------
'random' : address-level stratified split. Reproduces the original paper.
'group'  : stratified group split. No funding group spans two partitions.

Funding group (the key for the 'group' split)
---------------------------------------------
Walk the gas provision map from an address toward its funders. Stop when the
next hop is a labeled anchor (CEX, DEX, named entity), when there is no
further provider, or on a cycle. The last unlabeled node reached is the group
root. An address funded directly by a labeled entity, or with no provider, is
its own root, so a CEX hot wallet never merges its customers into one group.

Temporal cutoff
---------------
The LayerZero snapshot table ends at 2024-05-01 23:59:59 UTC (the last L0
transaction in the data). Provision edges dated at or after SNAPSHOT_END are
dropped before any provider, chain, tree, or group feature is computed.
"""
import gc
import json
import os
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

# 63 selected features (order matters for NumPy splits). Chosen manually by
# the authors; no data-driven selection step.
FEATS = [
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

# Labeled entity missing from the labeled-address file.
EXTRA_LABELED = '0x9241f27daffd0bb1df4f2a022584dd6c77843e64'
BURN_ADDRESS = '0x0000000000000000000000000000000000000000'


def data_paths(data_dir='./data'):
    j = os.path.join
    return dict(
        l0=[j(data_dir, '20241104_layer0_sybil_features', f'l0_features_{s}.csv')
            for s in ['0_100000', '100000_200000', '200000_300000',
                      '300000_400000', '400000_500000']],
        gas_prov=j(data_dir, '20241114_gas_provision',
                   '20241114_1633_layer0_provision_network_000000000000.csv'),
        labeled=j(data_dir, '20241214_labeled_addresses', '20241214_labeled_addresses.csv'),
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


def stream_labeled_anchors(path, candidates):
    """Step 3a: labeled addresses among the candidates (streamed, never fully loaded).

    Pass every address in the provision network so the same set serves the
    provider flags, the chain walk, the funding groups, and the tree features.
    """
    candidates = frozenset(candidates)
    anchors = set()
    with open(path) as f:
        for line in f:
            a = line.strip()
            if a and a in candidates:
                anchors.add(a)
    anchors.add(EXTRA_LABELED)
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


def provision_features(funding, interactors, labeled):
    """Step 4: provider_* and provision-tree features for each interactor.

    Ported from data/20241117_tree_features/20241117 Gas Provision
    Featurization.ipynb, which produced the precomputed file; same definitions.
    The provision forest excludes every edge that touches a labeled address, so
    a tree is the unlabeled funding cluster above and below an interactor.
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
                                            & ~prov.index.isin(labeled)).astype(int)
    out = fi[['activated_address', 'gas_provider', 'gas_provision_amount', 'block_number']].merge(
        prov, left_on='gas_provider', right_index=True)
    out = out.rename(columns={'activated_address': 'addr',
                              'block_number': 'gas_provision_block_number'})

    # unlabeled provision forest
    parent, children = {}, {}
    for a, p in zip(funding['activated_address'], funding['gas_provider']):
        if p in labeled or a in labeled or a == p:
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
            # As in the original featurization. Note: this expression is ~0 for
            # every tree (normalized amounts sum to 1); see docs/REVISION_LEAKAGE.md.
            gini_coefficient=(len(provs) - 1) / len(provs) * (1 - sum(sorted(provs) / np.sum(provs))),
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

    rows = []
    for a in out['addr']:
        if a not in parent:
            rows.append({})
            continue
        cur, depth, seen = a, 0, {a}
        while cur in parent and parent[cur] not in seen:
            cur = parent[cur]
            seen.add(cur)
            depth += 1
        m = dict(tree_metrics(cur))
        m['depth'] = depth
        rows.append(m)
    tf = pd.DataFrame(rows, columns=TREE_COLS, index=out.index).fillna(0)
    out = pd.concat([out.drop(columns='gas_provider'), tf], axis=1).reset_index(drop=True)
    assert out['addr'].is_unique
    return out


def merge_provision_features(df, feats):
    """Step 4 (merge): interactors without a provider get zeros."""
    return df.merge(feats, on='addr', how='left').fillna(0)


def compare_to_precomputed(feats, path):
    """Match rate of recomputed provider/tree features against the original file."""
    old = pd.read_csv(path, na_values=['', 'null']).drop_duplicates('addr').set_index('addr')
    new = feats.set_index('addr')
    com = old.index.intersection(new.index)
    rows = []
    for c in PROVIDER_COLS + TREE_COLS:
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


def funding_root(addr, tfm, anchors):
    """Last unlabeled node on the provision chain above addr (see module doc)."""
    cur, seen = addr, set()
    while True:
        seen.add(cur)
        prv = tfm.get(cur)
        if prv is None or prv == cur or prv in anchors or prv in seen:
            return cur
        cur = prv


def add_funding_group(df, tfm, anchors):
    df['funding_group'] = [funding_root(a, tfm, anchors) for a in df['addr']]
    return df


def build_master_df(data_dir='./data', verbose=True):
    """Run steps 1-7 plus taxonomy and funding groups. Returns (df, tfm, anchors)."""
    p = data_paths(data_dir)
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    df = load_l0(p['l0'])
    log(f'[1] L0 features: {len(df):,} addresses, {len(df.columns)} cols')
    funding = load_provision(p['gas_prov'])
    tfm = provision_map(funding)
    log(f'[2] Gas provision: {len(tfm):,} edges before {SNAPSHOT_END} UTC '
        f'({funding.attrs["n_after_cutoff"]} later edges dropped)')
    anchors = stream_labeled_anchors(p['labeled'], set(tfm) | set(tfm.values()))
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
    df = add_funding_group(df, tfm, anchors)
    log(f'[7] Chain features, taxonomy, funding groups '
        f'({df.funding_group.nunique():,} groups)')
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


def _group_partition(df, seed):
    """Stratified group split with the same 49 / 21 / 30 targets.

    Multi-wallet groups are placed first, largest first, each into the
    partition whose per-class quota is least filled (relative to its target),
    weighted by the group's class mix. Singletons then fill each partition's
    remaining per-class quota at random.
    """
    fracs = np.array([(1 - TEST_FRAC) * (1 - VAL_FRAC), (1 - TEST_FRAC) * VAL_FRAC, TEST_FRAC])
    rng = np.random.default_rng(seed)
    y = df['sybil'].to_numpy()
    n_cls = np.array([(y == 0).sum(), (y == 1).sum()])
    target = np.outer(fracs, n_cls).astype(float)         # [partition, class]
    quota = target.copy()

    g = df.groupby('funding_group', sort=False)['sybil'].agg(['size', 'sum'])
    multi = g[g['size'] > 1]
    order = rng.permutation(len(multi))
    multi = multi.iloc[order].sort_values('size', ascending=False, kind='stable')
    part_of_group = {}
    for grp, (size, n_syb) in zip(multi.index, multi[['size', 'sum']].to_numpy()):
        mix = np.array([size - n_syb, n_syb]) / n_cls
        p = int(np.argmax((quota / target) @ mix))
        quota[p] -= (size - n_syb, n_syb)
        part_of_group[grp] = p

    part = np.array(df['funding_group'].map(part_of_group), dtype=float)
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
        grp = df['funding_group']
        s_tr, s_va, s_te = set(grp.loc[tr]), set(grp.loc[va]), set(grp.loc[te])
        assert not (s_tr & s_va or s_tr & s_te or s_va & s_te), \
            'A funding group spans two partitions'

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


def leakage_report(df, s, verbose=True):
    """Row overlap and funding-group overlap between partitions."""
    tr, va, te = (set(s[k]) for k in ('idx_train', 'idx_val', 'idx_test'))
    row = dict(train_val=len(tr & va), train_test=len(tr & te), val_test=len(va & te))
    assert not any(row.values()), f'Row overlap between partitions: {row}'

    grp, syb = df['funding_group'], df['sybil'] == 1
    tr_i, te_i = s['idx_train'], s['idx_test']
    g_tr = set(grp.loc[tr_i])
    g_tr_syb = set(grp.loc[tr_i][syb.loc[tr_i]])
    te_grp = grp.loc[te_i]
    te_syb = te_grp[syb.loc[te_i]]
    rep = dict(
        method=s['method'],
        test_rows=len(te_i),
        test_sybils=int(syb.loc[te_i].sum()),
        test_rows_sharing_group_with_train=int(te_grp.isin(g_tr).sum()),
        test_sybils_sharing_group_with_train_sybil=int(te_syb.isin(g_tr_syb).sum()),
    )
    if verbose:
        print(f"Split method: {rep['method']}")
        print(f"Row overlap (train/val, train/test, val/test): "
              f"{row['train_val']}, {row['train_test']}, {row['val_test']}")
        print(f"Test addresses sharing a funding group with a train address: "
              f"{rep['test_rows_sharing_group_with_train']:,} / {rep['test_rows']:,}")
        print(f"Test Sybils sharing a funding group with a train Sybil:     "
              f"{rep['test_sybils_sharing_group_with_train_sybil']:,} / {rep['test_sybils']:,}")
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


def save_results(models, notebook, method, leakage=None, extra=None, out_dir='results'):
    """Write scalar metrics for one notebook run to results/<notebook>_<method>.json.

    code_commit ends in '-dirty' if tracked files differed from that commit.
    """
    os.makedirs(out_dir, exist_ok=True)
    scalars = lambda r: {k: (v.item() if isinstance(v, np.generic) else v)
                         for k, v in r.items() if np.isscalar(v)}
    record = dict(notebook=notebook, split_method=method, code_commit=_git_commit(),
                  leakage=leakage, models=[scalars(r) for r in models], **(extra or {}))
    path = os.path.join(out_dir, f'{notebook}_{method}.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'Saved {path}')
