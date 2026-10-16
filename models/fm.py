from __future__ import annotations

import time

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F


class _FMParameters(nn.Module):

    def __init__(self, n_features: int, n_factors: int):
        super().__init__()
        self.factors = nn.Embedding(n_features, n_factors)
        self.linear = nn.Embedding(n_features, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        vectors = self.factors(features)
        interactions = 0.5 * (
            vectors.sum(dim=1).square() - vectors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + self.linear(features).sum(dim=(1, 2)) + interactions


class FactorizationMachine:

    def __init__(
        self,
        n_factors=16,
        learning_rate=0.01,
        l2=1e-5,
        epochs=8,
        batch_size=2048,
        negative_ratio=3,
        random_state=42,
        use_metadata=True,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.random_state = random_state
        self.use_metadata = use_metadata

    def _validate_parameters(self):
        for name in ("n_factors", "epochs", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.negative_ratio, (int, np.integer)) or self.negative_ratio < 0:
            raise ValueError("negative_ratio must be a nonnegative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")

    def _prepare_metadata(self, item_features):
        if not self.use_metadata:
            metadata = np.empty((self.n_items_, 0), dtype=np.int64)
        else:
            if item_features is None:
                raise ValueError("item_features is required when use_metadata=True")
            metadata = np.asarray(item_features)
            if metadata.ndim != 2 or metadata.shape[0] != self.n_items_:
                raise ValueError("item_features must have shape [n_items, n_metadata_fields]")
            if not np.issubdtype(metadata.dtype, np.integer):
                raise ValueError("item_features must contain local integer category IDs")
            if np.any(metadata < 0):
                raise ValueError("metadata categories must be nonnegative; use 0 for unknown")
            metadata = metadata.astype(np.int64, copy=True)

        self.metadata_cardinalities_ = (
            metadata.max(axis=0) + 1
            if metadata.shape[1]
            else np.empty(0, dtype=np.int64)
        )
        self.metadata_offsets_ = np.empty(metadata.shape[1], dtype=np.int64)
        offset = self.n_users_ + self.n_items_
        for field, cardinality in enumerate(self.metadata_cardinalities_):
            self.metadata_offsets_[field] = offset
            offset += int(cardinality)
        self.n_features_ = offset
        encoded = metadata + self.metadata_offsets_[None, :]
        item_ids = self.n_users_ + np.arange(self.n_items_, dtype=np.int64)
        self.item_feature_ids_ = np.column_stack((item_ids, encoded))
        self.n_metadata_fields_ = metadata.shape[1]

    def _negative_membership(self, users, items):
        keys = users.astype(np.int64, copy=False) * self.n_items_ + items
        positions = np.searchsorted(self._positive_keys, keys)
        matches = np.zeros(keys.shape, dtype=bool)
        valid = positions < self._positive_keys.size
        matches[valid] = self._positive_keys[positions[valid]] == keys[valid]
        return matches

    def _sample_negative_pairs(self, users, rng=None):
        if not hasattr(self, "_positive_keys"):
            raise RuntimeError("fit must initialize train interactions before sampling")
        rng = self._rng if rng is None else rng
        users = np.asarray(users, dtype=np.int64).reshape(-1)
        if np.any((users < 0) | (users >= self.n_users_)):
            raise ValueError("negative-sampling user IDs are out of bounds")
        users = users[self._positive_counts[users] < self.n_items_].copy()
        items = rng.integers(0, self.n_items_, size=users.size, dtype=np.int64)
        rejected = self._negative_membership(users, items)
        dense = self._positive_counts[users] > 0.8 * self.n_items_
        for user in np.unique(users[rejected & dense]):
            positions = np.flatnonzero(users == user)
            start, end = self._train.indptr[user : user + 2]
            candidates = np.setdiff1d(
                np.arange(self.n_items_, dtype=np.int64),
                self._train.indices[start:end],
                assume_unique=True,
            )
            items[positions] = rng.choice(candidates, size=positions.size)
        rejected = self._negative_membership(users, items)
        while np.any(rejected):
            items[rejected] = rng.integers(
                0, self.n_items_, size=int(rejected.sum()), dtype=np.int64
            )
            rejected = self._negative_membership(users, items)
        return users, items

    def fit(self, X, item_features=None, **kwargs):
        self._validate_parameters()
        device = kwargs.pop("device", "cpu")
        verbose = kwargs.pop("verbose", False)
        self.score_chunk_size_ = kwargs.pop("score_chunk_size", 8192)
        if kwargs:
            raise TypeError(f"Unexpected fit arguments: {', '.join(sorted(kwargs))}")
        if not isinstance(self.score_chunk_size_, (int, np.integer)) or self.score_chunk_size_ < 1:
            raise ValueError("score_chunk_size must be a positive integer")
        if not sparse.issparse(X) or X.ndim != 2:
            raise ValueError("X must be a two-dimensional sparse train matrix")
        train = sparse.csr_matrix(X, copy=True)
        train.sum_duplicates()
        train.eliminate_zeros()
        train.sort_indices()
        if min(train.shape) < 1 or not train.nnz:
            raise ValueError("X must contain at least one user, item, and nonzero interaction")
        if not np.all(np.isfinite(train.data)):
            raise ValueError("X must contain finite values")

        started = time.perf_counter()
        self.n_users_, self.n_items_ = train.shape
        self._train = train
        self._positive_counts = np.diff(train.indptr)
        positive_users = np.repeat(np.arange(self.n_users_, dtype=np.int64), self._positive_counts)
        positive_items = train.indices.astype(np.int64, copy=True)
        self._positive_keys = positive_users * self.n_items_ + positive_items
        self._rng = np.random.default_rng(self.random_state)
        self._prepare_metadata(item_features)
        self.device_ = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            self.module_ = _FMParameters(self.n_features_, self.n_factors)
        self.module_.to(self.device_)
        self._item_features_tensor = torch.as_tensor(
            self.item_feature_ids_, dtype=torch.long, device=self.device_
        )
        optimizer = torch.optim.Adam(
            [
                {"params": [self.module_.factors.weight, self.module_.linear.weight], "weight_decay": self.l2},
                {"params": [self.module_.bias], "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        self.loss_history_ = []
        self.objective_history_ = []
        self.epoch_seconds_ = []
        self.epoch_samples_ = []
        self.module_.train()
        requested_negative_users = np.repeat(positive_users, self.negative_ratio)
        for epoch in range(self.epochs):
            epoch_started = time.perf_counter()
            negative_users, negative_items = self._sample_negative_pairs(requested_negative_users)
            users = np.concatenate((positive_users, negative_users))
            items = np.concatenate((positive_items, negative_items))
            labels = np.concatenate(
                (np.ones(positive_users.size, dtype=np.float32), np.zeros(negative_users.size, dtype=np.float32))
            )
            order = self._rng.permutation(users.size)
            total_bce = 0.0
            for start in range(0, users.size, self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_users = torch.as_tensor(users[idx], dtype=torch.long, device=self.device_)
                batch_items = torch.as_tensor(items[idx], dtype=torch.long, device=self.device_)
                batch_labels = torch.as_tensor(labels[idx], dtype=torch.float32, device=self.device_)
                features = torch.cat((batch_users[:, None], self._item_features_tensor[batch_items]), dim=1)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(self.module_(features), batch_labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("FM training produced a non-finite loss")
                loss.backward()
                optimizer.step()
                total_bce += float(loss.detach().cpu()) * idx.size
            bce = total_bce / users.size
            with torch.no_grad():
                penalty = 0.5 * self.l2 * float(
                    (self.module_.factors.weight.square().sum() + self.module_.linear.weight.square().sum()).cpu()
                )
            self.loss_history_.append(bce)
            self.objective_history_.append(bce + penalty)
            self.epoch_seconds_.append(time.perf_counter() - epoch_started)
            self.epoch_samples_.append({"positives": int(positive_users.size), "negatives": int(negative_users.size)})
            if verbose:
                print(f"FM epoch {epoch + 1}/{self.epochs}: BCE={bce:.5f}")
        self.module_.eval()
        self._cache_item_scores()
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def _cache_item_scores(self):
        sums, bases = [], []
        with torch.inference_mode():
            for start in range(0, self.n_items_, self.score_chunk_size_):
                ids = self._item_features_tensor[start : start + self.score_chunk_size_]
                factors = self.module_.factors(ids)
                factor_sum = factors.sum(dim=1)
                within_item = 0.5 * (factor_sum.square() - factors.square().sum(dim=1)).sum(dim=1)
                linear = self.module_.linear(ids).sum(dim=(1, 2))
                sums.append(factor_sum)
                bases.append(linear + within_item)
            self._item_factor_sums = torch.cat(sums)
            self._item_base_scores = torch.cat(bases)

    def score(self, user_ids, history=None):
        if not hasattr(self, "module_"):
            raise RuntimeError("FactorizationMachine must be fitted before scoring")
        users = np.asarray(user_ids)
        if users.ndim != 1 or not np.issubdtype(users.dtype, np.integer):
            if users.ndim == 1 and users.size == 0:
                return np.empty((0, self.n_items_), dtype=np.float32)
            raise ValueError("user_ids must be a one-dimensional integer array")
        if np.any((users < 0) | (users >= self.n_users_)):
            raise ValueError("user_ids contain an unknown or out-of-range user")
        if history is not None and (len(history.shape) != 2 or history.shape[1] != self.n_items_):
            raise ValueError("history must have the same item dimension as the fitted train matrix")
        result = np.empty((users.size, self.n_items_), dtype=np.float32)
        with torch.inference_mode():
            user_tensor = torch.as_tensor(users.astype(np.int64, copy=False), device=self.device_)
            vectors = self.module_.factors(user_tensor)
            bias = self.module_.bias + self.module_.linear(user_tensor).squeeze(1)
            for start in range(0, self.n_items_, self.score_chunk_size_):
                end = min(start + self.score_chunk_size_, self.n_items_)
                logits = vectors @ self._item_factor_sums[start:end].T
                logits += bias[:, None] + self._item_base_scores[None, start:end]
                result[:, start:end] = logits.cpu().numpy()
        return result
