from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

try:
    from models.ease import EASE
    from models.item_knn import ItemKNN
except ModuleNotFoundError:
    from hw1.models.ease import EASE
    from hw1.models.item_knn import ItemKNN


class TestEASE(unittest.TestCase):
    def test_matches_independent_leave_one_item_out_ridge(self):
        X = np.array([
            [1, 2, 0, 1], [0, 1, 3, 0], [4, 0, 1, 1],
            [1, 1, 2, 0], [0, 2, 1, 3],
        ], dtype=np.float64)
        l2 = 1.7
        reference = np.zeros((4, 4))
        for target in range(4):
            others = np.arange(4) != target
            design = X[:, others]
            reference[others, target] = np.linalg.solve(
                design.T @ design + l2 * np.eye(3), design.T @ X[:, target],
            )
        model = EASE(l2=l2)
        self.assertIs(model.fit(sparse.csr_matrix(X)), model)
        np.testing.assert_allclose(model.coefficients_, reference, rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(np.diag(model.coefficients_), np.zeros(4))
        self.assertFalse(np.allclose(reference, reference.T))

    def test_scores_use_supplied_full_history_and_requested_row_order(self):
        train = sparse.csr_matrix([[1, 1, 0], [1, 0, 2], [0, 1, 3]])
        history = sparse.csr_matrix([[0, 2, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
        model = EASE(l2=2).fit(train)
        ids = np.array([3, 0, 3, 1])
        scores = model.score(ids, history)
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(scores.shape, (4, 3))
        np.testing.assert_allclose(scores, history[ids].toarray() @ model.coefficients_)
        np.testing.assert_array_equal(scores[3], np.zeros(3))
        self.assertEqual(model.score(np.array([], dtype=int), history).shape, (0, 3))

    def test_cold_item_and_single_item_have_no_self_prediction(self):
        X = sparse.csr_matrix([[1, 1, 0], [1, 0, 0]])
        coefficients = EASE(l2=1).fit(X).coefficients_
        np.testing.assert_array_equal(coefficients[:, 2], np.zeros(3))
        np.testing.assert_array_equal(coefficients[2, :], np.zeros(3))
        single = EASE().fit(sparse.csr_matrix([[1], [2]]))
        np.testing.assert_array_equal(single.score(np.array([0, 1]), sparse.csr_matrix([[1], [2]])), [[0], [0]])


class TestItemKNN(unittest.TestCase):
    def test_cosine_shrinkage_and_target_column_topk(self):
        X = np.array([[1, 2, 0, 0], [1, 0, 3, 0], [0, 1, 1, 0]], dtype=np.float64)
        shrinkage = 0.5
        gram = X.T @ X
        norms = np.linalg.norm(X, axis=0)
        denominator = norms[:, None] * norms[None, :] + shrinkage
        similarities = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
        np.fill_diagonal(similarities, 0)
        reference = np.zeros_like(similarities)
        for target in range(4):
            neighbor = np.argmax(similarities[:, target])
            reference[neighbor, target] = similarities[neighbor, target]
        model = ItemKNN(k=1, shrinkage=shrinkage).fit(sparse.csr_matrix(X))
        learned = model.coefficients_.toarray()
        np.testing.assert_allclose(learned, reference, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(np.diag(learned), np.zeros(4))
        self.assertTrue(np.all(model.coefficients_.tocsc().getnnz(axis=0) <= 1))
        self.assertTrue(sparse.isspmatrix_csr(model.coefficients_))

    def test_no_shrinkage_scores_and_cold_history(self):
        X = sparse.csr_matrix([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
        model = ItemKNN(k=20, shrinkage=0).fit(X)
        expected_weight = 1 / np.sqrt(2)
        np.testing.assert_allclose(model.coefficients_.toarray(), [
            [0, expected_weight, 0], [expected_weight, 0, 0], [0, 0, 0],
        ], atol=1e-7)
        history = sparse.csr_matrix([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        scores = model.score(np.array([2, 0, 1, 2]), history)
        self.assertEqual(scores.dtype, np.float32)
        np.testing.assert_allclose(scores, [
            [expected_weight, expected_weight, 0], [0, 0, 0],
            [0, expected_weight, 0], [expected_weight, expected_weight, 0],
        ], atol=1e-7)
        self.assertGreater(scores[0, 0], 0)
        self.assertEqual(model.score(np.array([], dtype=int), history).shape, (0, 3))

    def test_single_item(self):
        X = sparse.csr_matrix([[1], [2]])
        model = ItemKNN().fit(X)
        self.assertEqual(model.coefficients_.nnz, 0)
        np.testing.assert_array_equal(model.score(np.array([1, 0]), X), [[0], [0]])


class TestLinearModelValidation(unittest.TestCase):
    def test_invalid_parameters_and_inputs(self):
        for kwargs in ({"k": 0}, {"k": 1.5}, {"shrinkage": -1}, {"shrinkage": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ItemKNN(**kwargs)
        for l2 in (0, -1, np.inf):
            with self.subTest(l2=l2), self.assertRaises(ValueError):
                EASE(l2=l2)
        for cls in (ItemKNN, EASE):
            model = cls()
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(RuntimeError):
                    model.score(np.array([0]), sparse.eye(2, format="csr"))
                with self.assertRaises(ValueError):
                    model.fit(sparse.csr_matrix([[np.nan]]))
                model.fit(sparse.eye(2, format="csr"))
                with self.assertRaises(ValueError):
                    model.score(np.array([0]), sparse.csr_matrix((2, 3)))
                with self.assertRaises(ValueError):
                    model.score(np.array([0.5]), sparse.eye(2, format="csr"))
                with self.assertRaises(IndexError):
                    model.score(np.array([-1]), sparse.eye(2, format="csr"))


if __name__ == "__main__":
    unittest.main()
