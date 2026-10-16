from __future__ import annotations
import numpy as np
from scipy import sparse
from .implicit_als import ImplicitALS


def recency_targets(interactions, timestamps, reference_time, half_life_days=3.0, floor=0.25):
    if half_life_days <= 0 or not 0 < floor <= 1:
        raise ValueError("Positive half life and floor in (0,1] required")
    interactions = sparse.csr_matrix(interactions).copy()
    interactions.eliminate_zeros()
    interactions.sort_indices()
    timestamps = sparse.csr_matrix(timestamps, dtype=np.float64).copy()
    timestamps.sort_indices()
    if (interactions.shape != timestamps.shape
        or not np.array_equal(interactions.indptr, timestamps.indptr)
        or not np.array_equal(interactions.indices, timestamps.indices)):
        raise ValueError("One historical timestamp required for every observed interaction")
    ages = (float(reference_time) - timestamps.data) / 86400.0
    if not np.isfinite(ages).all() or np.any(ages < -1e-8):
        raise ValueError("Timestamps must be finite and no later than reference_time")
    targets = timestamps.copy()
    targets.data = floor + (1.0 - floor) * np.exp2(-np.maximum(ages, 0) / half_life_days)
    return targets


class RecencyALS(ImplicitALS):
    def __init__(self, n_factors=32, alpha=20.0, l2=1.0, n_iterations=12,
                 half_life_days=3.0, floor=0.25, random_state=42):
        super().__init__(n_factors, alpha, l2, n_iterations, random_state)
        self.half_life_days = float(half_life_days)
        self.floor = float(floor)

    def fit(self, X, timestamps, reference_time):
        targets = recency_targets(X, timestamps, reference_time,
                                  self.half_life_days, self.floor)
        return super().fit(X, targets=targets)

