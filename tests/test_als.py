import unittest
import numpy as np
from scipy import sparse
try:
    from models.implicit_als import ImplicitALS, _als_update
    from models.recency_als import recency_targets, RecencyALS
except ModuleNotFoundError:
    from hw1.models.implicit_als import ImplicitALS, _als_update
    from hw1.models.recency_als import recency_targets, RecencyALS


class ALSTests(unittest.TestCase):
    def test_normal_equation_matches_dense_weighted_ridge(self):
        interactions = sparse.csr_matrix([[1, 0, 1], [0, 1, 0]], dtype=float)
        targets = np.array([[0.3, 0, 1.0], [0, 0.8, 0]])
        Y = np.array([[1., 2.], [.2, -.3], [1.1, .4]])
        observed_targets = sparse.csr_matrix(targets)
        actual = _als_update(interactions.indptr, interactions.indices, interactions.data,
            observed_targets.data, Y, 4., 0.5)
        for u in range(2):
            c = 1 + 4 * interactions.toarray()[u]
            expected = np.linalg.solve(Y.T @ np.diag(c) @ Y + .5 * np.eye(2),
                                       Y.T @ (c * targets[u]))
            np.testing.assert_allclose(actual[u], expected, atol=1e-10)

    def test_custom_changes_preferences_and_same_normal_matrix(self):
        X = sparse.csr_matrix([[1., 1.]])
        ts = sparse.csr_matrix([[86400., 172800.]])
        p = recency_targets(X, ts, 172800., half_life_days=1., floor=.25)
        np.testing.assert_allclose(p.data, [.625, 1.])
        Y = np.array([[1., .3], [.2, 1.]])
        baseline = _als_update(X.indptr, X.indices, X.data, X.data, Y, 10., 1.)
        custom = _als_update(X.indptr, X.indices, X.data, p.data, Y, 10., 1.)
        self.assertGreater(float(np.linalg.norm(baseline - custom)), .01)
        np.testing.assert_allclose(recency_targets(X, ts, 172800., floor=1).data, X.data)

    def test_future_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            recency_targets(sparse.csr_matrix([[1.]]), sparse.csr_matrix([[20.]]), 10.)

    def test_als_objective_decreases_and_seed_repeats(self):
        X = sparse.csr_matrix([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=float)
        def objective(model):
            error = X.toarray() - model.user_factors_ @ model.item_factors_.T
            return np.sum((1 + 10 * X.toarray()) * error**2) + .5 * (
                np.sum(model.user_factors_**2) + np.sum(model.item_factors_**2))
        a = ImplicitALS(2, 10, .5, 2).fit(X)
        b = ImplicitALS(2, 10, .5, 8).fit(X)
        self.assertLessEqual(objective(b), objective(a) + 1e-8)
        c = ImplicitALS(2, 10, .5, 8).fit(X)
        np.testing.assert_allclose(b.score([2, 0], X), c.score([2, 0], X))


if __name__ == '__main__':
    unittest.main()
