from __future__ import annotations

import numpy as np
from scipy import sparse


class ItemKNN:

    def __init__(self, k: int = 50, shrinkage: float = 10.0):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        self.k = int(k)
        self.shrinkage = float(shrinkage)

    def fit(self, X: sparse.csr_matrix) -> "ItemKNN":
        if not sparse.issparse(X):
            raise TypeError("X must be a scipy sparse matrix")
        interactions = sparse.csr_matrix(X, dtype=np.float64, copy=True)
        interactions.sum_duplicates()
        interactions.eliminate_zeros()
        if interactions.shape[1] == 0:
            raise ValueError("X must contain at least one item")
        if not np.isfinite(interactions.data).all():
            raise ValueError("X must contain only finite values")

        n_items = interactions.shape[1]
        norms = np.sqrt(np.asarray(interactions.multiply(interactions).sum(axis=0)).ravel())
        gram = (interactions.T @ interactions).tocsc()
        gram.setdiag(0)
        gram.eliminate_zeros()
        gram.sort_indices()

        neighbor_rows = []
        neighbor_values = []
        indptr = np.zeros(n_items + 1, dtype=np.int64)
        for target in range(n_items):
            start, stop = gram.indptr[target : target + 2]
            rows = gram.indices[start:stop]
            denominator = norms[rows] * norms[target] + self.shrinkage
            values = np.divide(
                gram.data[start:stop], denominator,
                out=np.zeros(stop - start, dtype=np.float64),
                where=denominator > 0,
            )
            nonzero = values != 0
            rows, values = rows[nonzero], values[nonzero]
            if values.size > self.k:
                chosen = np.argpartition(values, values.size - self.k)[-self.k :]
                rows, values = rows[chosen], values[chosen]
            order = np.argsort(rows)
            neighbor_rows.append(rows[order])
            neighbor_values.append(values[order].astype(np.float32))
            indptr[target + 1] = indptr[target] + rows.size

        coefficients = sparse.csc_matrix(
            (np.concatenate(neighbor_values), np.concatenate(neighbor_rows), indptr),
            shape=(n_items, n_items), dtype=np.float32,
        ).tocsr()
        self.n_items_ = n_items
        self.coefficients_ = coefficients
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
        return (selected @ self.coefficients_).toarray().astype(np.float32, copy=False)
