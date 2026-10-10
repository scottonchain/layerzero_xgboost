"""Regression checks for the root-wallet tree features of sybil_pipeline.provision_features.

Run from the repository root:  python -m unittest tests.test_root_tree_features -v

Before the fix, a wallet funded by a labeled entity (a root of the unlabeled provision forest)
and every wallet with no recorded incoming funding got all-zero tree features, although the
descendants of a root carried the whole tree's summaries.
"""
import math
import os
import random
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sybil_pipeline as sp  # noqa: E402

SERVICE = '0xservice'
TREE_ONLY = [c for c in sp.TREE_COLS if c != 'depth']


def funding(edges):
    """edges: (activated_address, gas_provider, amount in ETH) -> the frame load_provision returns."""
    f = pd.DataFrame(edges, columns=['activated_address', 'gas_provider', 'gas_provision_amount'])
    f['block_number'] = np.arange(len(f)) + 100
    return f


def features(edges, interactors, anchors=(SERVICE,)):
    out = sp.provision_features(funding(edges), set(interactors), set(anchors))
    return out.set_index('addr')


class RootTreeFeatures(unittest.TestCase):
    def test_service_funded_root_with_child_gets_the_whole_tree(self):
        # the reproduction from the review: a labeled exchange funds A, A funds B
        f = features([('A', SERVICE, 1.0), ('B', 'A', 2.0)], ['A', 'B'])
        for col in TREE_ONLY:
            self.assertEqual(f.loc['A', col], f.loc['B', col], col)
        self.assertEqual(f.loc['A', 'tree_size'], 2)
        self.assertEqual(f.loc['B', 'tree_size'], 2)
        self.assertEqual(f.loc['A', 'depth'], 0)
        self.assertEqual(f.loc['B', 'depth'], 1)

    def test_singleton_root(self):
        f = features([('A', SERVICE, 1.0)], ['A'])
        self.assertEqual(f.loc['A', 'tree_size'], 1)
        self.assertEqual(f.loc['A', 'depth'], 0)
        self.assertEqual(f.loc['A', 'max_depth'], 0)
        self.assertEqual(f.loc['A', 'breadth_factor'], 1)
        self.assertTrue(np.isfinite(f.loc['A'].to_numpy(dtype=float)).all())

    def test_root_without_incoming_record_but_with_a_child(self):
        f = features([('B', 'A', 2.0)], ['A', 'B'])
        self.assertIn('A', f.index)
        self.assertEqual(f.loc['A', 'tree_size'], 2)
        self.assertEqual(f.loc['A', 'depth'], 0)
        self.assertEqual(f.loc['B', 'depth'], 1)
        for col in TREE_ONLY:
            self.assertEqual(f.loc['A', col], f.loc['B', col], col)
        # no recorded provider: provider_* values are absent here and become 0 in the master table
        self.assertTrue(pd.isna(f.loc['A', 'provider_fan_out']))
        self.assertEqual(f.loc['B', 'provider_fan_out'], 1)

    def test_isolated_wallet_without_any_record(self):
        f = features([('B', 'A', 2.0)], ['A', 'B', 'Z'])   # raised ZeroDivisionError in _entropy before
        self.assertEqual(f.loc['Z', 'tree_size'], 1)
        self.assertEqual(f.loc['Z', 'depth'], 0)
        self.assertEqual(f.loc['Z', 'leaf_gas_distribution_entropy'], 0)
        self.assertEqual(f.loc['Z', 'avg_leaf_gas'], 0)
        self.assertTrue(np.isfinite(f.loc['Z', sp.TREE_COLS].to_numpy(dtype=float)).all())

    def test_zero_total_entropy(self):
        self.assertEqual(sp._entropy([]), 0)
        self.assertEqual(sp._entropy([0.0]), 0)
        self.assertEqual(sp._entropy([0, 0, 0]), 0)
        self.assertAlmostEqual(sp._entropy([1.0, 1.0]), 1.0)
        self.assertAlmostEqual(sp._entropy([2.0, 3.0]), -(0.4 * math.log2(0.4) + 0.6 * math.log2(0.6)))

    def test_descendant_and_non_tree_values(self):
        # root A (funded by the service, 1.0 ETH) with children B (2.0) and C (3.0)
        f = features([('A', SERVICE, 1.0), ('B', 'A', 2.0), ('C', 'A', 3.0)], ['A', 'B', 'C'])
        want = dict(tree_size=3, total_gas=5.0, max_depth=1, branching_factor=2 / 3, balance_factor=0,
                    leaf_provision_proportion=1.0, avg_depth=2 / 3, depth_variance=1 / 3,
                    leaf_to_internal_ratio=2.0, avg_leaf_gas=2.5, breadth_factor=1.5,
                    gini_coefficient=0.1, gas_distribution_skewness=0.0, leaf_gas_distribution_skewness=0.0,
                    sparsity=1.0, longest_chain_ratio=1 / 3, star_like_ratio=1 / 3,
                    depth_weighted_avg_gas=2.5, breadth_to_depth_ratio=1.5)
        h = -(0.4 * math.log2(0.4) + 0.6 * math.log2(0.6))
        want.update(gas_distribution_entropy=h, leaf_gas_distribution_entropy=h)
        for who in 'ABC':
            for col, v in want.items():
                self.assertAlmostEqual(f.loc[who, col], v, places=12, msg=f'{who} {col}')
        self.assertEqual([f.loc[x, 'depth'] for x in 'ABC'], [0, 1, 1])
        # non-tree values: provider features of B and C describe their provider A (2 interactor children)
        for who, amt in (('B', 2.0), ('C', 3.0)):
            self.assertEqual(f.loc[who, 'provider_fan_out'], 2)
            self.assertAlmostEqual(f.loc[who, 'provider_total_gas_provision_amount'], 5.0)
            self.assertAlmostEqual(f.loc[who, 'provider_max_gas_provision_amount'], 3.0)
            self.assertAlmostEqual(f.loc[who, 'provider_min_gas_provision_amount'], 2.0)
            self.assertEqual(f.loc[who, 'provider_is_star_like_attack'], 1)
            self.assertAlmostEqual(f.loc[who, 'gas_provision_amount'], amt)
        # A's provider is the labeled service: one interactor edge, not star-like
        self.assertEqual(f.loc['A', 'provider_fan_out'], 1)
        self.assertEqual(f.loc['A', 'provider_is_star_like_attack'], 0)

    def test_deeper_descendants_keep_their_depth(self):
        f = features([('A', SERVICE, 1.0), ('B', 'A', 2.0), ('C', 'B', 3.0), ('D', 'C', 4.0)], list('ABCD'))
        self.assertEqual([f.loc[x, 'depth'] for x in 'ABCD'], [0, 1, 2, 3])
        self.assertEqual(set(f['tree_size']), {4})
        self.assertEqual(set(f['max_depth']), {3})

    def test_a_labeled_edge_still_splits_trees(self):
        # B is funded by the service, not by A: two trees, each a root
        f = features([('A', SERVICE, 1.0), ('B', SERVICE, 2.0), ('C', 'B', 1.0)], list('ABC'))
        self.assertEqual(f.loc['A', 'tree_size'], 1)
        self.assertEqual(f.loc['B', 'tree_size'], 2)
        self.assertEqual(f.loc['C', 'tree_size'], 2)

    def test_input_order_independence(self):
        edges = [('A', SERVICE, 1.0), ('B', 'A', 2.0), ('C', 'A', 3.0), ('D', 'C', 0.5),
                 ('E', SERVICE, 4.0), ('G', 'F', 1.0)]
        ids = list('ABCDEFGZ')
        base = features(edges, ids).sort_index()
        rng = random.Random(7)
        for _ in range(5):
            e, i = edges[:], ids[:]
            rng.shuffle(e)
            rng.shuffle(i)
            got = features(e, i).sort_index()
            pd.testing.assert_frame_equal(base.drop(columns=['gas_provision_block_number']),
                                          got.drop(columns=['gas_provision_block_number']))

    def test_every_interactor_has_exactly_one_row(self):
        ids = list('ABCZ')
        f = features([('A', SERVICE, 1.0), ('B', 'A', 2.0)], ids)
        self.assertEqual(sorted(f.index), sorted(ids))

    def test_master_merge_zero_fills_only_provider_values(self):
        out = sp.provision_features(funding([('B', 'A', 2.0)]), {'A', 'B'}, {SERVICE})
        df = sp.merge_provision_features(pd.DataFrame({'addr': ['A', 'B', 'Y']}), out)
        a = df.set_index('addr')
        self.assertEqual(a.loc['A', 'tree_size'], 2)
        self.assertEqual(a.loc['A', 'provider_fan_out'], 0)
        self.assertEqual(a.loc['Y', 'tree_size'], 0)    # not an interactor of this fixture: zeros, as before


if __name__ == '__main__':
    unittest.main()
