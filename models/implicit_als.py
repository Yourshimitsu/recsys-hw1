from __future__ import annotations

import time
import numpy as np
from scipy import sparse
from numba import njit, prange


@njit(parallel=True, cache=True)
def _als_update(indptr, indices, confidence_data, target_data, opposite, alpha, l2):
    n_rows = len(indptr) - 1
    n_factors = opposite.shape[1]
    gram = opposite.T @ opposite
    result = np.zeros((n_rows, n_factors), dtype=np.float64)
    for row in prange(n_rows):
        normal = gram.copy()
        rhs = np.zeros(n_factors, dtype=np.float64)
        for f in range(n_factors):
            normal[f, f] += l2
        for pos in range(indptr[row], indptr[row + 1]):
            item = indices[pos]
            excess = alpha * confidence_data[pos]
            target = target_data[pos]
            for a in range(n_factors):
                ya = opposite[item, a]
                rhs[a] += (1.0 + excess) * target * ya
                for b in range(n_factors):
                    normal[a, b] += excess * ya * opposite[item, b]
        result[row] = np.linalg.solve(normal, rhs)
    return result


class ImplicitALS:

    def __init__(self, n_factors=32, alpha=20.0, l2=1.0, n_iterations=12,
                 random_state=42):
        self.n_factors = int(n_factors)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.n_iterations = int(n_iterations)
        self.random_state = int(random_state)
        if self.n_factors < 1 or self.alpha < 0 or self.l2 <= 0 or self.n_iterations < 1:
            raise ValueError("Invalid ALS hyperparameters")

    def fit(self, X, targets=None):
        started = time.perf_counter()
        X = sparse.csr_matrix(X, dtype=np.float64).copy()
        X.sum_duplicates()
        X.eliminate_zeros()
        X.sort_indices()
        if np.any(X.data < 0) or not np.isfinite(X.data).all():
            raise ValueError("Interactions must be finite and nonnegative")
        X.data[:] = 1.0
        if targets is None:
            targets = X.copy()
        else:
            targets = sparse.csr_matrix(targets, dtype=np.float64).copy()
            targets.sum_duplicates()
            targets.sort_indices()
            if (targets.shape != X.shape or not np.array_equal(targets.indptr, X.indptr)
                    or not np.array_equal(targets.indices, X.indices)):
                raise ValueError("Targets must have exactly the interaction sparsity pattern")
            if not np.isfinite(targets.data).all() or np.any(targets.data < 0):
                raise ValueError("Targets must be finite and nonnegative")
        rng = np.random.default_rng(self.random_state)
        self.n_users_, self.n_items_ = X.shape
        self.item_factors_ = rng.normal(0.0, 0.05, (self.n_items_, self.n_factors))
        transposed = X.T.tocsr()
        transposed_targets = targets.T.tocsr()
        for _ in range(self.n_iterations):
            self.user_factors_ = _als_update(X.indptr, X.indices, X.data,
                targets.data, self.item_factors_, self.alpha, self.l2)
            self.item_factors_ = _als_update(transposed.indptr, transposed.indices,
                transposed.data, transposed_targets.data, self.user_factors_, self.alpha, self.l2)
        self.targets_ = targets
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def score(self, user_ids, history):
        if history.shape != (self.n_users_, self.n_items_):
            raise ValueError("History dimensions differ from fitted data")
        return (self.user_factors_[np.asarray(user_ids)] @ self.item_factors_.T).astype(np.float32)

