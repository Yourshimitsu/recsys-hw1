import unittest

import numpy as np
from scipy import sparse
import torch

try:
    from models.fm import FactorizationMachine, _FMParameters
except ModuleNotFoundError:
    from hw1.models.fm import FactorizationMachine, _FMParameters


class FactorizationMachineTests(unittest.TestCase):
    def setUp(self):
        self.train = sparse.csr_matrix(
            np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=np.float32)
        )
        self.metadata = np.array([[0, 1], [1, 1], [2, 2], [1, 0], [0, 2]], dtype=np.int64)

    def fit_small(self, **kwargs):
        params = dict(n_factors=3, epochs=2, batch_size=8, negative_ratio=2, random_state=7)
        params.update(kwargs)
        return FactorizationMachine(**params).fit(self.train, self.metadata, score_chunk_size=2)

    def test_vectorized_formula_matches_explicit_distinct_pairs_and_gradients(self):
        torch.manual_seed(3)
        module = _FMParameters(11, 4)
        with torch.no_grad():
            module.linear.weight.copy_(torch.linspace(-0.2, 0.3, 11)[:, None])
            module.bias.fill_(0.17)
        ids = torch.tensor([[0, 3, 6, 9], [1, 4, 7, 10]], dtype=torch.long)
        vectors = module.factors(ids)
        direct = module.bias + module.linear(ids).sum(dim=(1, 2))
        for left in range(ids.shape[1]):
            for right in range(left + 1, ids.shape[1]):
                direct = direct + (vectors[:, left] * vectors[:, right]).sum(dim=1)
        efficient = module(ids)
        torch.testing.assert_close(efficient, direct, rtol=1e-5, atol=1e-7)
        efficient.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
        self.assertGreater(float(module.factors.weight.grad.abs().sum()), 0)

    def test_cached_all_item_scores_equal_direct_fm(self):
        model = self.fit_small()
        users = np.array([1, 0, 1], dtype=np.int64)
        actual = model.score(users, self.train)
        pairs_u = np.repeat(users, model.n_items_)
        pairs_i = np.tile(np.arange(model.n_items_), users.size)
        ids = np.column_stack((pairs_u, model.item_feature_ids_[pairs_i]))
        with torch.no_grad():
            direct = model.module_(torch.as_tensor(ids)).numpy().reshape(users.size, -1)
        np.testing.assert_allclose(actual, direct, rtol=2e-5, atol=1e-6)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(model.score([], self.train).shape, (0, 5))

    def test_negative_samples_exclude_all_train_positives_and_full_rows(self):
        model = self.fit_small()
        requested = np.repeat(np.arange(3), 1000)
        users, items = model._sample_negative_pairs(requested, np.random.default_rng(19))
        self.assertEqual(users.size, 2000)
        self.assertFalse(np.any(users == 2))
        self.assertTrue(np.all(np.asarray(self.train[users, items]).ravel() == 0))
        for user, allowed in ((0, {1, 3, 4}), (1, {0, 2, 3})):
            self.assertEqual(set(items[users == user]), allowed)
        self.assertEqual(model.epoch_samples_[0], {"positives": 9, "negatives": 8})

    def test_dense_row_sampler_uses_only_its_one_unseen_item(self):
        train = sparse.csr_matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.float32)
        model = FactorizationMachine(epochs=1, use_metadata=False).fit(train)
        users, items = model._sample_negative_pairs(np.zeros(100, dtype=np.int64))
        self.assertEqual(users.size, 100)
        np.testing.assert_array_equal(items, np.full(100, 9))

    def test_metadata_shares_categories_only_within_fields_and_reserves_zero(self):
        model = self.fit_small()
        self.assertEqual(model.item_feature_ids_[1, 1], model.item_feature_ids_[3, 1])
        self.assertEqual(model.item_feature_ids_[0, 2], model.item_feature_ids_[1, 2])
        self.assertNotEqual(model.item_feature_ids_[0, 1], model.item_feature_ids_[3, 2])
        np.testing.assert_array_equal(model.metadata_cardinalities_, [3, 3])
        self.assertEqual(model.item_feature_ids_[0, 1], model.metadata_offsets_[0])

    def test_id_only_ablation_and_seed_are_reproducible(self):
        first = self.fit_small(use_metadata=False)
        second = self.fit_small(use_metadata=False)
        self.assertEqual(first.item_feature_ids_.shape, (5, 1))
        self.assertEqual(first.n_features_, 8)
        np.testing.assert_array_equal(first.score(np.arange(3)), second.score(np.arange(3)))
        self.assertEqual(first.loss_history_, second.loss_history_)
        self.assertTrue(np.isfinite(first.loss_history_).all())

    def test_metadata_validation_and_pre_fit_error(self):
        with self.assertRaises(RuntimeError):
            FactorizationMachine().score([0])
        with self.assertRaises(ValueError):
            FactorizationMachine(epochs=1).fit(self.train)
        with self.assertRaises(ValueError):
            FactorizationMachine(epochs=1).fit(self.train, self.metadata.astype(np.float32))
        invalid = self.metadata.copy()
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            FactorizationMachine(epochs=1).fit(self.train, invalid)


if __name__ == "__main__":
    unittest.main()
