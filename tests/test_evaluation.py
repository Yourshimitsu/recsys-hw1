from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

try:
    from evaluation import evaluate, paired_bootstrap_difference
except ModuleNotFoundError:
    from hw1.evaluation import evaluate, paired_bootstrap_difference


class FixedScores:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = []

    def score(self, user_ids, history):
        self.calls.append((user_ids.copy(), history))
        return self.scores[user_ids]


class TestEvaluation(unittest.TestCase):
    def test_masks_highest_seen_score_and_uses_item_index_for_ties(self):
        history = sparse.csr_matrix([[1, 0, 0, 0]])
        truth = sparse.csr_matrix([[0, 0, 1, 0]])
        model = FixedScores([[100, 5, 5, 4]])
        result = evaluate(model, history, truth, k=2)
        discount_at_two = 1 / np.log2(3)
        self.assertAlmostEqual(result["ndcg_at_2"], discount_at_two)
        self.assertEqual(result["recall_at_2"], 1)
        self.assertEqual(result["precision_at_2"], 0.5)
        self.assertEqual(result["hit_rate_at_2"], 1)
        self.assertEqual(result["catalog_coverage_at_2"], 0.5)
        self.assertIs(model.calls[0][1], history)
        miss = evaluate(FixedScores([[100, 5, 5, 5]]), history, truth, k=1)
        self.assertEqual(miss["recall_at_1"], 0)

    def test_multirelevant_ideal_ranking_and_full_recall_denominator(self):
        history = sparse.csr_matrix([[1, 0, 0, 0, 0]])
        truth = sparse.csr_matrix([[0, 1, 1, 1, 0]])
        result = evaluate(FixedScores([[100, 3, 2, 1, 0]]), history, truth, k=2)
        self.assertEqual(result["ndcg_at_2"], 1)
        self.assertAlmostEqual(result["recall_at_2"], 2 / 3)
        self.assertEqual(result["precision_at_2"], 1)

    def test_keeps_zero_hit_users_and_excludes_users_without_history_or_truth(self):
        history = sparse.csr_matrix([
            [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0],
        ])
        truth = sparse.csr_matrix([
            [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0],
        ])
        model = FixedScores([[100, 3, 2, 1]] * 4)
        result = evaluate(model, history, truth, k=1, batch_size=1)
        self.assertEqual(result["n_users"], 2)
        np.testing.assert_array_equal(result["user_ids"], [0, 1])
        np.testing.assert_array_equal(result["per_user_recall"], [1, 0])
        np.testing.assert_array_equal(result["per_user_ndcg"], [1, 0])
        self.assertEqual(result["ndcg_at_1"], 0.5)
        self.assertEqual(result["hit_rate_at_1"], 0.5)
        self.assertEqual(len(model.calls), 2)

    def test_exhausted_candidates_never_filled_with_seen_or_nonfinite_scores(self):
        history = sparse.csr_matrix([[1, 1, 0, 0, 0, 0]])
        truth = sparse.csr_matrix([[0, 0, 1, 0, 0, 0]])
        result = evaluate(
            FixedScores([[100, 90, 2, np.inf, np.nan, -np.inf]]), history, truth,
        )
        self.assertEqual(result["n_users"], 1)
        self.assertEqual(result["ndcg_at_10"], 1)
        self.assertEqual(result["recall_at_10"], 1)
        self.assertEqual(result["precision_at_10"], 0.1)
        self.assertAlmostEqual(result["catalog_coverage_at_10"], 1 / 6)
        missing = evaluate(FixedScores([[1, 2, np.nan, np.inf, np.nan, -np.inf]]), history, truth)
        self.assertEqual(missing["n_users"], 1)
        self.assertEqual(missing["ndcg_at_10"], 0)
        self.assertEqual(missing["catalog_coverage_at_10"], 0)

    def test_ground_truth_does_not_change_candidates(self):
        history = sparse.csr_matrix([[1, 0, 0, 0, 0]])
        model = FixedScores([[100, 4, 3, 2, 1]])
        for target in (1, 4):
            truth = sparse.csr_matrix(([1], ([0], [target])), shape=(1, 5))
            result = evaluate(model, history, truth, k=2)
            self.assertEqual(result["catalog_coverage_at_2"], 2 / 5)
            self.assertEqual(result["recall_at_2"], float(target == 1))

    def test_rejects_overlap_dimensions_and_bad_score_shape(self):
        history = sparse.csr_matrix([[1, 0]])
        with self.assertRaises(ValueError):
            evaluate(FixedScores([[1, 2]]), history, history)
        with self.assertRaises(ValueError):
            evaluate(FixedScores([[1, 2]]), history, sparse.csr_matrix((2, 2)))
        with self.assertRaises(ValueError):
            evaluate(FixedScores([[1, 2, 3]]), history, sparse.csr_matrix([[0, 1]]))

    def test_explicit_sparse_zeros_do_not_create_interactions(self):
        history = sparse.csr_matrix(([1.0, 0.0], ([0, 0], [0, 1])), shape=(1, 3))
        truth = sparse.csr_matrix(([1.0, 0.0], ([0, 0], [1, 0])), shape=(1, 3))
        result = evaluate(FixedScores([[100, 2, 1]]), history, truth, k=1)
        self.assertEqual(result["n_users"], 1)
        self.assertEqual(result["recall_at_1"], 1)

    def test_empty_cohort_is_explicit_and_does_not_call_model(self):
        history = sparse.csr_matrix([[1, 0]])
        truth = sparse.csr_matrix((1, 2))
        model = FixedScores([[1, 2]])
        result = evaluate(model, history, truth)
        self.assertEqual(result["n_users"], 0)
        self.assertEqual(result["ndcg_at_10"], 0)
        self.assertEqual(result["user_ids"].size, 0)
        self.assertEqual(model.calls, [])


class TestPairedBootstrap(unittest.TestCase):
    def test_pairing_yields_zero_variance_for_constant_improvement(self):
        baseline = np.array([0.1, 0.8, 0.3, 0.5])
        improved = baseline + 0.1
        result = paired_bootstrap_difference(improved, baseline, n_bootstrap=100)
        self.assertAlmostEqual(result["mean_delta"], 0.1)
        self.assertAlmostEqual(result["ci_low"], 0.1)
        self.assertAlmostEqual(result["ci_high"], 0.1)

    def test_seed_reproducibility_and_validation(self):
        a, b = np.array([0.1, 0.5, 0.8]), np.array([0.2, 0.3, 0.7])
        first = paired_bootstrap_difference(a, b, seed=10, n_bootstrap=200)
        self.assertEqual(first, paired_bootstrap_difference(a, b, seed=10, n_bootstrap=200))
        self.assertLessEqual(first["ci_low"], first["mean_delta"])
        self.assertGreaterEqual(first["ci_high"], first["mean_delta"])
        for left, right in (([], []), ([1], [1, 2]), ([np.nan], [0])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                paired_bootstrap_difference(left, right)


if __name__ == "__main__":
    unittest.main()
