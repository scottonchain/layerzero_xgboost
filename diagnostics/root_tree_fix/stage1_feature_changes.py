"""Stage 1: which wallets and columns change when root wallets receive their tree's metrics.

Reads the cached original and corrected master tables (build_tables.py) and the repository's own
split construction. Writes stage1_feature_changes.json and stage1_feature_changes.md next to this file.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
import sybil_pipeline as sp

D = 'diagnostics/root_tree_fix'
old = pd.read_parquet(f'{D}/cache/master_old.parquet')
new = pd.read_parquet(f'{D}/cache/master_new.parquet')
meta = {t: json.load(open(f'{D}/cache/master_{t}.json')) for t in ('old', 'new')}
R = dict(old_code_sha256=meta['old']['code_sha256'], new_code_sha256=meta['new']['code_sha256'],
         inputs_sha256=meta['new']['inputs_sha256'], rows=len(new), n_features=len(sp.FEATS))

# ---- identity checks -------------------------------------------------------------------------
same_cols = list(old.columns) == list(new.columns)
ident = dict(
    same_columns_in_same_order=same_cols,
    address_order_identical=bool((old['addr'].to_numpy() == new['addr'].to_numpy()).all()),
    labels_identical=bool((old['sybil'].to_numpy() == new['sybil'].to_numpy()).all()),
    category_identical=bool((old['category'].to_numpy() == new['category'].to_numpy()).all()),
    provision_tree_identical=bool((old['provision_tree'].to_numpy() == new['provision_tree'].to_numpy()).all()),
    anchors_flags_identical=bool(all((old[c].to_numpy() == new[c].to_numpy()).all()
                                      for c in ('provider_is_labeled', 'provider_is_interactor', 'provider_is_null'))),
    n_anchors_old=meta['old']['n_anchors'], n_anchors_new=meta['new']['n_anchors'],
    n_edges_old=meta['old']['n_edges'], n_edges_new=meta['new']['n_edges'],
)
S_old = sp.make_splits(old, sp.FEATS, method='group', seed=sp.SEED)
S_new = sp.make_splits(new, sp.FEATS, method='group', seed=sp.SEED)
for part in ('train', 'val', 'test'):
    ident[f'split_{part}_row_ids_identical'] = bool(np.array_equal(np.asarray(S_old[f'idx_{part}']), np.asarray(S_new[f'idx_{part}'])))
ident['y_train_identical'] = bool(np.array_equal(S_old['y_train'].to_numpy(), S_new['y_train'].to_numpy()))
ident['upsampled_train_index_identical'] = bool(np.array_equal(S_old['X_train'].index.to_numpy(), S_new['X_train'].index.to_numpy()))
R['identity'] = ident

# ---- which columns / wallets change ----------------------------------------------------------
cols = [c for c in new.columns if c != 'addr']
def _differs(c):
    if pd.api.types.is_numeric_dtype(new[c]) or pd.api.types.is_bool_dtype(new[c]):
        return old[c].astype(float).to_numpy() != new[c].astype(float).to_numpy()
    return (old[c].astype(str).to_numpy() != new[c].astype(str).to_numpy())
chg = pd.DataFrame({c: _differs(c) for c in cols})
col_counts = chg.sum().sort_values(ascending=False)
changed_cols = [c for c in col_counts.index if col_counts[c] > 0]
feats = set(sp.FEATS)
R['changed_columns'] = {c: int(col_counts[c]) for c in changed_cols}
R['changed_model_features'] = [c for c in changed_cols if c in feats]
R['changed_columns_excluded_from_model'] = [c for c in changed_cols if c not in feats]
R['unchanged_tree_columns_in_model'] = [c for c in sp.TREE_COLS if c in feats and c not in changed_cols]

any_feat = chg[[c for c in sp.FEATS]].any(axis=1)
any_col = chg.any(axis=1)
R['wallets_any_computed_column_changed'] = int(any_col.sum())
R['wallets_selected_model_input_changed'] = int(any_feat.sum())
R['wallets_only_excluded_columns_changed'] = int((any_col & ~any_feat).sum())

def by_cat(mask):
    out = {}
    for cat in ['IxL', 'IxI', 'IxE', 'No provider']:
        tot = int((new['category'] == cat).sum()); n = int((mask & (new['category'] == cat)).sum())
        out[cat] = dict(changed=n, total=tot, pct=round(100 * n / tot, 3) if tot else None)
    return out
R['by_category_selected_inputs'] = by_cat(any_feat)
syb = new['sybil'].to_numpy() == 1
R['sybil_nonsybil_selected_inputs'] = dict(changed_sybil=int((any_feat & syb).sum()), changed_non_sybil=int((any_feat & ~syb).sum()),
                                          sybil_total=int(syb.sum()), non_sybil_total=int((~syb).sum()))
R['by_category_and_label'] = {cat: dict(sybil=int((any_feat & syb & (new['category'] == cat)).sum()),
                                        non_sybil=int((any_feat & ~syb & (new['category'] == cat)).sum())) for cat in ['IxL', 'IxI', 'IxE', 'No provider']}

# roots with descendants vs singleton roots (corrected table: depth 0 = root; tree_size 1 = singleton)
root = new['depth'].to_numpy() == 0
sing = new['tree_size'].to_numpy() == 1
R['new_table_roots'] = dict(roots_with_descendants=int((root & ~sing).sum()), singleton_roots=int((root & sing).sum()),
                            non_roots=int((~root).sum()))
R['changed_selected_inputs_by_kind'] = dict(
    roots_with_descendants=int((any_feat & root & ~sing).sum()), singleton_roots=int((any_feat & root & sing).sum()),
    non_roots=int((any_feat & ~root).sum()))
R['changed_any_column_by_kind'] = dict(
    roots_with_descendants=int((any_col & root & ~sing).sum()), singleton_roots=int((any_col & root & sing).sum()),
    non_roots=int((any_col & ~root).sum()))
R['changed_selected_inputs_kind_by_category'] = {
    cat: dict(roots_with_descendants=int((any_feat & root & ~sing & (new['category'] == cat)).sum()),
              singleton_roots=int((any_feat & root & sing & (new['category'] == cat)).sum()))
    for cat in ['IxL', 'IxI', 'IxE', 'No provider']}
R['changed_sybil_by_kind'] = dict(
    roots_with_descendants=int((any_feat & root & ~sing & syb).sum()), singleton_roots=int((any_feat & root & sing & syb).sum()),
    roots_with_descendants_nonsybil=int((any_feat & root & ~sing & ~syb).sum()), singleton_roots_nonsybil=int((any_feat & root & sing & ~syb).sum()))

# unexpected non-root changes: any column on a wallet whose corrected depth > 0, or any provider/non-tree column anywhere
nonroot_chg = chg.loc[~root].sum()
R['unexpected_nonroot_changes'] = {c: int(v) for c, v in nonroot_chg.items() if v > 0}
tree_set = set(sp.TREE_COLS)
non_tree_changed = [c for c in changed_cols if c not in tree_set]
R['non_tree_columns_changed'] = {c: int(col_counts[c]) for c in non_tree_changed}
# zero-fill check: every wallet whose old tree features were all zero but new are not
old_zero = (old[sp.TREE_COLS].astype(float).abs().sum(axis=1) == 0).to_numpy()
new_zero = (new[sp.TREE_COLS].astype(float).abs().sum(axis=1) == 0).to_numpy()
R['old_all_zero_tree_rows'] = int(old_zero.sum()); R['new_all_zero_tree_rows'] = int(new_zero.sum())
R['old_zero_now_nonzero'] = int((old_zero & ~new_zero).sum())

# historical count in the tracker: IxL wallets that are roots of multi-wallet gas provision trees
grp = new.groupby('provision_tree')['addr'].transform('size')
ixl_root_multi = (new['category'] == 'IxL') & (new['provision_tree'] == new['addr']) & (grp > 1)
R['ixl_roots_of_multi_wallet_provision_groups'] = int(ixl_root_multi.sum())
R['ixl_roots_with_descendants_in_retained_forest'] = int(((new['category'] == 'IxL') & root & ~sing).sum())
R['interactors_that_are_anchors'] = int(len(set(new['addr']) & set(
    sp.stream_labeled_anchors(sp.data_paths('./data'), set(new['addr']), 'current'))))

# ---- representative before/after values ------------------------------------------------------
show = ['category', 'sybil', 'provision_tree', 'tree_size', 'total_gas', 'max_depth', 'branching_factor', 'avg_depth',
        'breadth_factor', 'sparsity', 'depth', 'star_like_ratio', 'gas_distribution_entropy', 'avg_leaf_gas']
def example(mask, n=3):
    idx = np.flatnonzero(mask)[:n]
    ex = []
    for i in idx:
        ex.append(dict(addr=new.loc[i, 'addr'], before={c: (old.loc[i, c].item() if hasattr(old.loc[i, c], 'item') else old.loc[i, c]) for c in show},
                       after={c: (new.loc[i, c].item() if hasattr(new.loc[i, c], 'item') else new.loc[i, c]) for c in show}))
    return ex
rng_order = np.random.default_rng(0)
def pick(mask, n=3):
    idx = np.flatnonzero(mask); idx = np.sort(rng_order.choice(idx, size=min(n, len(idx)), replace=False))
    m = np.zeros(len(new), bool); m[idx] = True
    return example(m, n)
R['examples'] = dict(ixl_sybil_root_with_descendants=pick(any_feat & syb & root & ~sing & (new['category'] == 'IxL')),
                     ixl_non_sybil_root_with_descendants=pick(any_feat & ~syb & root & ~sing & (new['category'] == 'IxL')),
                     singleton_root=pick(any_feat & root & sing, 2),
                     no_provider_root_with_descendants=pick(any_feat & root & ~sing & (new['category'] == 'No provider'), 2))
json.dump(R, open(f'{D}/stage1_feature_changes.json', 'w'), indent=1, default=str)

# ---- short markdown --------------------------------------------------------------------------
L = ['# Stage 1: wallets whose inputs change when roots receive their tree metrics', '',
     f"Original code: `sybil_pipeline_b7a9a3f.py` (sha256 {R['old_code_sha256'][:12]}); corrected: `sybil_pipeline.py` ({R['new_code_sha256'][:12]}); "
     f"inputs sha256 {R['inputs_sha256'][:12]}. Same environment, same inputs, one build each.", '',
     f"- Wallets: {R['rows']:,}. Selected model inputs ({R['n_features']} features) change for **{R['wallets_selected_model_input_changed']:,}** wallets "
     f"({100 * R['wallets_selected_model_input_changed'] / R['rows']:.2f} %); any computed column changes for {R['wallets_any_computed_column_changed']:,}.",
     f"- Changed model features: {', '.join(R['changed_model_features'])}.",
     f"- Changed columns excluded from the model: {', '.join(R['changed_columns_excluded_from_model']) or 'none'}.",
     f"- Model tree features unchanged: {', '.join(R['unchanged_tree_columns_in_model']) or 'none'}.",
     f"- Unexpected non-root changes: {R['unexpected_nonroot_changes'] or 'none'}. Non-tree columns changed: {R['non_tree_columns_changed'] or 'none'}.", '',
     '| Category | Changed | Total | % |', '|---|---|---|---|']
for cat, v in R['by_category_selected_inputs'].items():
    L.append(f"| {cat} | {v['changed']:,} | {v['total']:,} | {v['pct']} |")
L += ['', f"Sybil changed {R['sybil_nonsybil_selected_inputs']['changed_sybil']:,} of {R['sybil_nonsybil_selected_inputs']['sybil_total']:,}; "
      f"non-Sybil changed {R['sybil_nonsybil_selected_inputs']['changed_non_sybil']:,} of {R['sybil_nonsybil_selected_inputs']['non_sybil_total']:,}.",
      f"Changed by kind: {R['changed_selected_inputs_by_kind']}.", '', 'Identity checks: ' + json.dumps(R['identity'])]
open(f'{D}/stage1_feature_changes.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
