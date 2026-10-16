from __future__ import annotations

import numpy as np
from scipy import linalg, sparse


class EASE:

    def __init__(self, l2: float = 100.0):
        if not np.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be finite and strictly positive")
        self.l2 = float(l2)

    def fit(self, X: sparse.csr_matrix) -> "EASE":
        if not sparse.issparse(X):
            raise TypeError("X must be a scipy sparse matrix")
        interactions = sparse.csr_matrix(X, dtype=np.float64, copy=True)
        interactions.sum_duplicates()
        interactions.eliminate_zeros()
        n_items = interactions.shape[1]
        if n_items == 0:
            raise ValueError("X must contain at least one item")
        if not np.isfinite(interactions.data).all():
            raise ValueError("X must contain only finite values")

        gram = (interactions.T @ interactions).toarray()
        gram.flat[:: n_items + 1] += self.l2
        factor = linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        precision = linalg.cho_solve(
            factor, np.eye(n_items, dtype=np.float64), overwrite_b=True, check_finite=False,
        )
        coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(coefficients, 0.0)
        self.n_items_ = n_items
        self.coefficients_ = coefficients.astype(np.float32)
        return self

    def score(self, user_ids: np.ndarray, history: sparse.csr_matrix) -> np.ndarray:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Call fit before score")
        if not sparse.issparse(history):
            raise TypeError("history must be a scipy sparse matrix")
        if history.shape[1] != self.n_items_:
            raise ValueError("history has a different number of items from X")
        ids = np.asarray(user_ids)
        if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("user_ids must be a one-dimensional integer array")
        if ids.size and (ids.min() < 0 or ids.max() >= history.shape[0]):
            raise IndexError("user_ids contains an out-of-range user")
        selected = sparse.csr_matrix(history)[ids.astype(np.intp)].astype(np.float32)
        if not np.isfinite(selected.data).all():
            raise ValueError("selected history must contain only finite values")
        return np.asarray(selected @ self.coefficients_, dtype=np.float32)
