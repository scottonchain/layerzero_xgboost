"""Build the Etherscan lookup tables (labeling rule, step 2; tracker A12).

Writes
  data/20260930_etherscan_service_labels/etherscan_lookups.csv     all 65 addresses looked up
  data/20260930_etherscan_service_labels/service_labels.csv  the rows the rule requires (in_rule_scope)

Tags were read from each address's etherscan.io page by the authors on the dates in `checked_on`.
Each tag was matched to its address through the address printed on the page itself. A reviewer can
re-check any row at `etherscan_url`; tags may change after `checked_on`.

Run from the repo root: python data/20260930_etherscan_service_labels/build_etherscan_lookups.py [out_dir]
"""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, '.')
import sybil_pipeline as sp

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'

# address, checked_on, lookup_reason, address_kind, name_tag, tag_badges, entity_type
L = [
    # 24 hand-added addresses not in the hildobby CEX list (read 2026-09-30)
    ('0xcc9557f04633d82fb6a1741dcec96986cd8689ae', '2026-09-30', 'hand_added_2024', 'account', 'ZigZag: Bridge', 'Bridge', 'bridge_or_relayer'),
    ('0x963737c550e70ffe4d59464542a28604edb2ef9a', '2026-09-30', 'hand_added_2024', 'account', 'Union Chain', '', 'other_service'),
    ('0x2db1d8cdf1abe8c70b531a790cdf2ff38aecf652', '2026-09-30', 'hand_added_2024', 'account', 'Newton', 'Exchange', 'exchange_hot_wallet'),
    ('0x5e809a85aa182a9921edd10a4163745bb3e36284', '2026-09-30', 'hand_added_2024', 'account', 'Owlto Finance: Bridge 2', 'Bridge', 'bridge_or_relayer'),
    ('0xa88902d6e93922893ee77234ed1c3ba4bec90224', '2026-09-30', 'hand_added_2024', 'account', 'Umbria: Narni Bridge 2', 'Bridge', 'bridge_or_relayer'),
    ('0x158353a7601c7ba9bfdea6c77e571557267cd03b', '2026-09-30', 'hand_added_2024', 'account', 'Umbria: Narni Bridge 4', 'Bridge', 'bridge_or_relayer'),
    ('0x00000000000007736e2f9aa5630b8c812e1f3fc9', '2026-09-30', 'hand_added_2024', 'account', 'Chaineye: Mini Bridge', 'Bridge', 'bridge_or_relayer'),
    ('0x84a518a35c7c361c64301b8f209c2f0edb6f608d', '2026-09-30', 'hand_added_2024', 'account', 'OKX Dep: 0x84A518...6F608d', 'OKX; Deposit Address', 'exchange_deposit'),
    ('0x6b8fac654d072d8a799f03626db7e4f4679b0e1d', '2026-09-30', 'hand_added_2024', 'account', '', '', 'none'),
    ('0x469503159ddf6bfd0a9ec8eba8e97a84fd3eae5b', '2026-09-30', 'hand_added_2024', 'contract', 'Kuailian APP: Wallet', '', 'protocol_contract'),
    ('0x000005a271a610964bb42658c7ff50fee2aa055a', '2026-09-30', 'hand_added_2024', 'account', '', '', 'personal_name'),
    ('0xe93685f3bba03016f02bd1828badd6195988d950', '2026-09-30', 'hand_added_2024', 'account', 'Layer Zero: Executor', 'LayerZero; Contract Deployer', 'bridge_or_relayer'),
    ('0xffff3dcb664c3f69b049d121fba7b7d7273961ef', '2026-09-30', 'hand_added_2024', 'account', '', '', 'none'),
    ('0x0000000000000068f116a894984e2db1123eb395', '2026-09-30', 'hand_added_2024', 'contract', 'Seaport 1.6', 'OpenSea', 'protocol_contract'),
    ('0x0000000000a8fb09af944ab3baf7a9b3e1ab29d8', '2026-09-30', 'hand_added_2024', 'contract', '', '', 'none'),
    ('0xd64791e747188b0e5061fc65b56bf20fee2e3321', '2026-09-30', 'hand_added_2024', 'account', 'Aztec: Sequencer', '', 'bridge_or_relayer'),
    ('0x230a1ac45690b9ae1176389434610b9526d2f21b', '2026-09-30', 'hand_added_2024', 'account', 'Synapse: Executor', '', 'bridge_or_relayer'),
    ('0x0385b3f162a0e001b60ecb84d3cb06199d78f666', '2026-09-30', 'hand_added_2024', 'account', 'Argent: Relayer 2', '', 'bridge_or_relayer'),
    ('0x4fb5df81b644e3bd5ad0ba07dce2b67559c764e0', '2026-09-30', 'hand_added_2024', 'account', 'Hyphen: Executor 3', '', 'bridge_or_relayer'),
    ('0x2a038e100f8b85df21e4d44121bdbfe0c288a869', '2026-09-30', 'hand_added_2024', 'account', 'Multichain: Executor', '', 'bridge_or_relayer'),
    ('0x1439eda7f9a911b9120e9a0dafb60eae317f7685', '2026-09-30', 'hand_added_2024', 'account', 'Hyphen: Executor 2', '', 'bridge_or_relayer'),
    ('0x1eaca1277bcdfa83e60658d8938b3d63cd3e63c1', '2026-09-30', 'hand_added_2024', 'account', 'Allbridge: Executor', '', 'bridge_or_relayer'),
    ('0x70217e7de3a68187905269462506f81cf344bbad', '2026-09-30', 'hand_added_2024', 'account', 'ShapeShift: Deployer', 'Contract Deployer', 'other_service'),
    ('0x8d12a197cb00d4747a1fe03395095ce2a5cc6819', '2026-09-30', 'hand_added_2024', 'contract', 'EtherDelta 2', 'DEX', 'protocol_contract'),
    # 30 largest unlabeled funders by addresses funded (read 2026-09-30)
    ('0xf70da97812cb96acdf810712aa562db8dfa3dbef', '2026-09-30', 'top30_unlabeled_funder', 'account', 'Relay: Solver', 'Relay; Bridge', 'bridge_or_relayer'),
    ('0x2fc617e933a52713247ce25730f6695920b3befe', '2026-09-30', 'top30_unlabeled_funder', 'account', 'Layerswap 1', '', 'bridge_or_relayer'),
    ('0x4c9af439b1a6761b8e549d8d226a468a6b2803a8', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0xa60efe89e07cfcf83eb30f3b063834f0f8b30466', '2026-09-30', 'top30_unlabeled_funder', 'account', 'BingX 42', 'Exchange', 'exchange_hot_wallet'),
    ('0x389044f3ac7472060a0618116e3624a5f0f20f28', '2026-09-30', 'top30_unlabeled_funder', 'account', 'Shakepay 7', 'Shakepay', 'exchange_hot_wallet'),
    ('0xe4e81a38ea6d480e4f21abdbe34df076621b9113', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0xbbd0d4d067d5af2065b1b6fd936d93237ae1c56c', '2026-09-30', 'top30_unlabeled_funder', 'account', 'Shakepay 6', 'Shakepay', 'exchange_hot_wallet'),
    ('0x63a0a1c4b8606ddeb58ec9564e83ee1e9fb20ffb', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x434ca7f0da6c47d3eb3515cbd97711aec9d4ae6e', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0xee73323912a4e3772b74ed0ca1595a152b0ef282', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x9cdc2a4e228b3ff679bb2487c609112affbdd3db', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x75e7f640bf6968b6f32c47a3cd82c3c2c9dcae68', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x4f9dd9b7e014a6a6838d16a81144ca92d0de793f', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0xc5e8ceb141fa13d0c34101de3c5e53b445c5d015', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0xa0bb4ba19f578a63fa3f67adaf7bbca15ccadc45', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x657720113dea643354e494e0de955a101d2466a7', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x7086ee078ff457f5f2c59cd140e79b2ebbc0e89c', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x4ac92d60cb6415232e62db519657c33abdcd102f', '2026-09-30', 'top30_unlabeled_funder', 'account', 'Newton 5', 'Exchange', 'exchange_hot_wallet'),
    ('0x76c49d0e2b00ded0611862f0713c624a9bd0a432', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x4c20d6c790dcd8474df3ec22dd199ff03e0fcd7d', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0xaa364c1a348f9517009207a1601e0a73c1cd530b', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x230a9e0a0eecb17c9e2cf770b15f7669e336d4f0', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x16dae7d42ad9ae86c885762caaa247fcc65ba516', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x88dcdd4a0a58b7e2208805d547043c37dca2b6dc', '2026-09-30', 'top30_unlabeled_funder', 'account', 'Shakepay 9', 'Shakepay; Exchange', 'exchange_hot_wallet'),
    ('0x556b5712c2a7cc5506fbfa785b52536fb8243765', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x7f04c1ea7e60f2c620948d5f201ca23d3d54d10d', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'personal_name'),
    ('0x797dbfab26308010199f0b18c97c1c554dd119f9', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x554ccc18f88cf3159eedfe161cd46611ef820359', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x07f0eb0c571b6cfd90d17b5de2cc51112fb95915', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    ('0x063ed6f59bd44d8bc99c3b170a3d52b49dcbcfff', '2026-09-30', 'top30_unlabeled_funder', 'account', '', '', 'none'),
    # remaining funders of 50+ interactors not covered by public lists (read 2026-10-01)
    ('0x08b067ad41e45babe5bbb52fc2fe7f692f628b06', '2026-10-01', 'funder_50_plus', 'contract', '', '', 'none'),
    ('0x24759faa9b57d2b3c215f2fca6d898a53e7799bf', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0xdb3d858dcc295be2309e69539b2de07b41d658f7', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0x81f91aca8c05b3eefebc00171139afefac17c9a6', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0xe6b3c2fd41566c8dc5354da55cfe20eeaeaa299f', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0x45a318273749d6eb00f5f6ca3bc7cd3de26d642a', '2026-10-01', 'funder_50_plus', 'account', 'Owlto Finance: Bridge', 'Bridge', 'bridge_or_relayer'),
    ('0x60fe41469307da95b78ebaa100d70afb3c7805d9', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0xbbba1983859ac05cacc453a6ae9a12ce83735b6d', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0x78c49a9ff270562f0aaea3e54c12d5841acdf599', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
    ('0xa3af00a4ca8a11b840b1cb190d1c1d66da1546fc', '2026-10-01', 'funder_50_plus', 'account', 'DeGate: Hot Wallet', '', 'exchange_hot_wallet'),
    ('0xdcd6968c5e40a6b26cabca51e818b0404082c156', '2026-10-01', 'funder_50_plus', 'account', '', '', 'none'),
]
SHARED_SERVICE = {'exchange_hot_wallet', 'bridge_or_relayer', 'protocol_contract', 'other_service'}
cols = ['address', 'checked_on', 'lookup_reason', 'address_kind', 'name_tag', 'tag_badges', 'entity_type']
t = pd.DataFrame(L, columns=cols)
assert t['address'].is_unique and len(t) == 65
assert t['address'].str.fullmatch(r'0x[0-9a-f]{40}').all()

P = sp.data_paths()
f = sp.load_provision(P['gas_prov'])
tfm = sp.provision_map(f)
iset = set(sp.load_l0(P['l0'])['addr'])
fo = pd.Series(tfm)[lambda s: s.index.isin(iset)].value_counts()
# public labels only: the Etherscan step must not count toward its own scope
with tempfile.TemporaryDirectory() as tmp:
    empty = os.path.join(tmp, 'no_etherscan_services.csv')
    pd.DataFrame(columns=['address', 'labeled']).to_csv(empty, index=False)
    public = sp.stream_labeled_anchors({**P, 'etherscan_services': empty}, set(fo.index) | set(t['address']), 'current')

t['etherscan_url'] = 'https://etherscan.io/address/' + t['address']
t['interactors_funded'] = t['address'].map(fo).fillna(0).astype(int)
t['covered_by_public_list'] = t['address'].isin(public)
t['in_rule_scope'] = (t['interactors_funded'] >= sp.ETHERSCAN_MIN_FANOUT) & ~t['covered_by_public_list']

def decide(r):
    if r.covered_by_public_list:
        return True, 'public_list'
    if not r.in_rule_scope:
        return False, 'below_threshold'
    if r.entity_type in SHARED_SERVICE:
        return True, 'shared_service_tag'
    if r.entity_type == 'exchange_deposit':
        return False, 'exchange_deposit'
    if r.entity_type == 'personal_name':
        return False, 'personal_name'
    return False, 'no_entity_tag'

t[['labeled', 'decision_reason']] = t.apply(lambda r: pd.Series(decide(r)), axis=1)
order = ['address', 'etherscan_url', 'checked_on', 'lookup_reason', 'address_kind', 'name_tag', 'tag_badges',
         'entity_type', 'interactors_funded', 'covered_by_public_list', 'in_rule_scope', 'labeled', 'decision_reason']
t = t[order].sort_values(['in_rule_scope', 'interactors_funded'], ascending=False)

# every funder the rule requires must have been looked up
scope_all = set(fo[(fo >= sp.ETHERSCAN_MIN_FANOUT) & ~fo.index.isin(public)].index)
missing = scope_all - set(t['address'])
assert not missing, f'In rule scope but not looked up: {sorted(missing)}'
assert set(t.loc[t.in_rule_scope, 'address']) == scope_all

os.makedirs(os.path.join(OUT, 'data', '20260930_etherscan_service_labels'), exist_ok=True)
t.to_csv(os.path.join(OUT, 'data', '20260930_etherscan_service_labels', 'etherscan_lookups.csv'), index=False)
t[t.in_rule_scope].to_csv(os.path.join(OUT, 'data', '20260930_etherscan_service_labels', 'service_labels.csv'), index=False)
print(f'{len(t)} lookups; {len(scope_all)} in rule scope; labeled by tag: {(t.decision_reason == "shared_service_tag").sum()}')
print(t.decision_reason.value_counts().to_string())
