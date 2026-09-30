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
"""
import gc
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

SEED = 42
TEST_FRAC = 0.30
VAL_FRAC = 0.30          # fraction of the non-test remainder -> 21 % of total
SPLIT_METHODS = ('random', 'group')

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
        tree=j(data_dir, '20241117_tree_features', '20241117_graph_and_tree_features.csv'),
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


def load_provision_map(path):
    """Step 2: tfm = {activated_address: gas_provider}."""
    gp = pd.read_csv(path, usecols=['activated_address', 'gas_provider'],
                     na_values=['', 'null'])
    gp.columns = gp.columns.str.lower()
    vm = gp['gas_provider'].notna()
    return dict(zip(gp.loc[vm, 'activated_address'], gp.loc[vm, 'gas_provider']))


def stream_labeled_anchors(path, providers):
    """Step 3a: labeled addresses that appear as gas providers (streamed)."""
    providers = frozenset(providers)
    anchors = set()
    with open(path) as f:
        for line in f:
            a = line.strip()
            if a and a in providers:
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


def merge_tree_features(df, path):
    """Step 4: gas provision tree / topology features."""
    tree = pd.read_csv(path, na_values=['', 'null'])
    tree.columns = tree.columns.str.lower()
    tree = tree.drop(columns='provider_is_labeled')   # recomputed in step 3
    return df.merge(tree, on='addr', how='left').fillna(0)


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
    tfm = load_provision_map(p['gas_prov'])
    log(f'[2] Gas provision: {len(tfm):,} mappings')
    anchors = stream_labeled_anchors(p['labeled'], tfm.values())
    df = add_provider_flags(df, tfm, anchors)
    log(f'[3] Labeled anchors: {len(anchors):,}')
    df = merge_tree_features(df, p['tree'])
    log(f'[4] Tree features merged: {len(df.columns)} cols')
    df = merge_cex_dex(df, p['cex'])
    log(f'[5] CEX/DEX merged: {len(df.columns)} cols')
    df = add_labels(df, p['sybil'], tfm)
    log(f'[6] Labels: Sybil={df.sybil.sum():,} ({df.sybil.mean()*100:.2f}%)')
    df = add_chain_features(df, tfm, anchors)
    df = add_category(df)
    df = add_funding_group(df, tfm, anchors)
    log(f'[7] Chain features, taxonomy, funding groups '
        f'({df.funding_group.nunique():,} groups)')
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
