from __future__ import annotations

import numpy as np
from scipy import sparse


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _presence(matrix: sparse.csr_matrix, name: str) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise TypeError(f"{name} must be a scipy sparse matrix")
    result = sparse.csr_matrix(matrix, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError(f"{name} must contain finite interactions")
    return result.astype(bool)


def _top_items(scores: np.ndarray, seen_items: np.ndarray, k: int) -> np.ndarray:
    valid = np.isfinite(scores)
    valid[seen_items] = False
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return candidates
    candidate_scores = scores[candidates]
    if candidates.size > k:
        cutoff = np.partition(candidate_scores, candidate_scores.size - k)[candidate_scores.size - k]
        better = candidates[candidate_scores > cutoff]
        tied = candidates[candidate_scores == cutoff]
        candidates = np.concatenate((better, tied[: k - better.size]))
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order]


def evaluate(
    model,
    history: sparse.csr_matrix,
    truth: sparse.csr_matrix,
    k: int = 10,
    batch_size: int = 256,
) -> dict:
    k = _positive_integer(k, "k")
    batch_size = _positive_integer(batch_size, "batch_size")
    history_presence = _presence(history, "history")
    truth_presence = _presence(truth, "truth")
    if history_presence.shape != truth_presence.shape:
        raise ValueError("history and truth must have the same user and item dimensions")
    if history_presence.multiply(truth_presence).nnz:
        raise ValueError("history and truth must be disjoint for every user")

    n_items = history_presence.shape[1]
    history_counts = np.diff(history_presence.indptr)
    truth_counts = np.diff(truth_presence.indptr)
    user_ids = np.flatnonzero((history_counts > 0) & (truth_counts > 0))
    n_users = user_ids.size
    ndcg = np.zeros(n_users, dtype=np.float64)
    recall = np.zeros(n_users, dtype=np.float64)
    precision = np.zeros(n_users, dtype=np.float64)
    hit_rate = np.zeros(n_users, dtype=np.float64)
    recommended_items = np.zeros(n_items, dtype=bool)
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2)
    ideal_dcg = np.cumsum(discounts)

    for offset in range(0, n_users, batch_size):
        batch_ids = user_ids[offset : offset + batch_size]
        raw_scores = model.score(batch_ids, history)
        if sparse.issparse(raw_scores):
            raise TypeError("model.score must return a dense numeric array")
        scores = np.asarray(raw_scores)
        if scores.shape != (batch_ids.size, n_items):
            raise ValueError("model.score returned the wrong user/item dimensions")
        if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
            raise TypeError("model.score must return real numeric scores")
        if not np.issubdtype(scores.dtype, np.floating):
            scores = scores.astype(np.float64)

        for local_row, user in enumerate(batch_ids):
            seen_start, seen_stop = history_presence.indptr[user : user + 2]
            chosen = _top_items(scores[local_row], history_presence.indices[seen_start:seen_stop], k)
            relevant_start, relevant_stop = truth_presence.indptr[user : user + 2]
            relevant = truth_presence.indices[relevant_start:relevant_stop]
            hits = np.isin(chosen, relevant, assume_unique=True)
            n_hits = int(hits.sum())
            row = offset + local_row
            ndcg[row] = np.dot(hits, discounts[: chosen.size]) / ideal_dcg[min(k, relevant.size) - 1]
            recall[row] = n_hits / relevant.size
            precision[row] = n_hits / k
            hit_rate[row] = float(n_hits > 0)
            recommended_items[chosen] = True

    def user_mean(values: np.ndarray) -> float:
        return float(values.mean()) if n_users else 0.0

    return {
        f"ndcg_at_{k}": user_mean(ndcg),
        f"recall_at_{k}": user_mean(recall),
        f"precision_at_{k}": user_mean(precision),
        f"hit_rate_at_{k}": user_mean(hit_rate),
        f"catalog_coverage_at_{k}": float(recommended_items.sum() / n_items) if n_items else 0.0,
        "n_users": int(n_users),
        "per_user_ndcg": ndcg,
        "per_user_recall": recall,
        "user_ids": user_ids,
    }


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict:
    n_bootstrap = _positive_integer(n_bootstrap, "n_bootstrap")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be nonempty one-dimensional arrays of the same length")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("a and b must contain only finite values")
    delta = a - b
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    draws_per_batch = min(256, max(1, 1_000_000 // delta.size))
    for start in range(0, n_bootstrap, draws_per_batch):
        stop = min(n_bootstrap, start + draws_per_batch)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        bootstrap_means[start:stop] = delta[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {"mean_delta": float(delta.mean()), "ci_low": float(lower), "ci_high": float(upper)}
