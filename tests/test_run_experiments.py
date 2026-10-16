import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import run_experiments as runner


class ExperimentIntegrationTests(unittest.TestCase):
    def test_complete_run_cache_and_interrupted_force(self):
        rng = np.random.default_rng(7)
        rows, columns, val_columns, test_columns = [], [], [], []
        for user in range(20):
            chosen = rng.choice(30, 7, replace=False)
            rows.extend([user] * 5)
            columns.extend(chosen[:5])
            val_columns.append(chosen[5])
            test_columns.append(chosen[6])
        def matrix(rr, cc):
            return sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(20, 30))
        train = matrix(rows, columns)
        validation = matrix(np.arange(20), val_columns)
        test = matrix(np.arange(20), test_columns)
        train_ts, val_ts = train.copy(), validation.copy()
        train_ts.data[:] = 100.
        val_ts.data[:] = 250.
        summary = dict(requirements_met=dict(interactions=True, users=True, items=True),
            train_cutoff_utc='1970-01-01T00:03:20Z', validation_cutoff_utc='1970-01-01T00:05:00Z')
        data = dict(train=train, validation=validation, test=test,
            train_timestamps=train_ts, validation_timestamps=val_ts, summary=summary,
            item_features=dict(language=np.arange(30) % 4, owner=np.arange(30) % 9))
        spaces = {'ItemKNN': [dict(k=3, shrinkage=0.)],
            'iALS': [dict(n_factors=4, alpha=5., l2=1., n_iterations=2)],
            'EASE': [dict(l2=10.)],
            'FM metadata': [dict(n_factors=4, learning_rate=.01, l2=1e-5,
                epochs=1, batch_size=128, negative_ratio=1)]}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(directory)
            with patch.object(runner, 'RESULTS', directory / 'results'), \
                 patch.object(runner, 'DEFAULT_CACHE', directory / 'cache'), \
                 patch.object(runner, 'processed_cache_valid', lambda: True), \
                 patch.object(runner, 'load_data', lambda: data), \
                 patch.object(runner, 'SEARCH_SPACES', spaces), \
                 patch.object(runner, '_fingerprint', lambda: 'isolated-integration-fixture'):
                result = runner.run_experiments(force=True, download=False)
                self.assertEqual(len(result), 7)
                self.assertEqual(result.n_users.nunique(), 1)
                with patch.object(runner, '_fit', side_effect=AssertionError('Cache should avoid training')):
                    self.assertEqual(len(runner.run_experiments(download=False)), 7)
                with patch.object(runner, '_fit', side_effect=RuntimeError('Controlled interruption')):
                    with self.assertRaises(RuntimeError):
                        runner.run_experiments(force=True, download=False)
                manifest = json.loads((directory / 'results/run_manifest.json').read_text())
                self.assertFalse(manifest['completed'])
                with patch.object(runner, '_fit', side_effect=RuntimeError('Must refit incomplete cache')):
                    with self.assertRaises(RuntimeError):
                        runner.run_experiments(download=False)


if __name__ == '__main__':
    unittest.main()
