from __future__ import annotations
import os
for _name, _value in {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                      "OMP_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4"}.items():
    os.environ.setdefault(_name, _value)
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from numba import set_num_threads
from data_pipeline import prepare_data, load_data, processed_cache_valid, DEFAULT_CACHE
from evaluation import evaluate, paired_bootstrap_difference
from models.item_knn import ItemKNN
from models.ease import EASE
from models.implicit_als import ImplicitALS
from models.recency_als import RecencyALS
from models.fm import FactorizationMachine

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SEED = 42
PRIMARY_METRIC = 'ndcg_at_10'

SEARCH_SPACES = {
    'ItemKNN': [dict(k=k, shrinkage=s) for k, s in itertools.product([25, 75, 150], [0., 10.])],
    'iALS': [dict(n_factors=f, alpha=a, l2=r, n_iterations=12)
             for f, a, r in [(16, 10., 1.), (16, 40., 10.), (32, 10., 1.), (32, 40., 10.)]],
    'EASE': [dict(l2=value) for value in [20., 100., 500.]],
    'FM metadata': [dict(n_factors=f, learning_rate=lr, l2=l2, epochs=epochs,
                         batch_size=2048, negative_ratio=3)
        for f, lr, l2, epochs in [(16, .01, 1e-5, 8), (32, .01, 1e-5, 12),
                                  (32, .03, 1e-4, 12), (16, .03, 1e-4, 16)]],
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _cutoff(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def _fingerprint():
    digest = hashlib.sha256()
    files = [ROOT / 'run_experiments.py', ROOT / 'data_pipeline.py', ROOT / 'evaluation.py', ROOT / 'config/data_config.json']
    files += sorted((ROOT / 'models').glob('*.py'))
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _metrics_only(value):
    return {key: (number.item() if isinstance(number, np.generic) else number)
            for key, number in value.items() if not isinstance(number, np.ndarray)}


class Popularity:
    def fit(self, X):
        self.popularity_ = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return self
    def score(self, user_ids, history):
        return np.broadcast_to(self.popularity_, (len(user_ids), len(self.popularity_))).copy()


def _make_model(name, params):
    if name == 'ItemKNN':
        return ItemKNN(**params)
    if name == 'EASE':
        return EASE(**params)
    if name == 'iALS':
        return ImplicitALS(**params, random_state=SEED)
    if name == 'Recency iALS':
        return RecencyALS(**params, random_state=SEED)
    if name.startswith('FM'):
        return FactorizationMachine(**params, use_metadata=name != 'FM IDs only', random_state=SEED)
    if name == 'Popularity':
        return Popularity()
    raise ValueError(name)


def _fit(model, name, history, features, timestamps, reference):
    started = time.perf_counter()
    if name == 'Recency iALS':
        model.fit(history, timestamps=timestamps, reference_time=reference)
    elif name.startswith('FM'):
        model.fit(history, item_features=features, device='cpu')
    else:
        model.fit(history)
    return time.perf_counter() - started


def run_experiments(force=False, download=True):
    RESULTS.mkdir(exist_ok=True)
    code_hash = _fingerprint()
    manifest_path = RESULTS / 'run_manifest.json'
    needed = ['test_metrics.csv', 'validation_trials.csv', 'selected_parameters.json',
              'dataset_summary.json', 'evaluation_details.json', 'source_archives.json']
    if not force and manifest_path.exists() and all((RESULTS / name).exists() for name in needed):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('completed') and manifest.get('code_sha256') == code_hash:
            print('Using completed experiments with matching source-code fingerprint.', flush=True)
            return pd.read_csv(RESULTS / 'test_metrics.csv')
    if not processed_cache_valid():
        prepare_data(download=download)
    data = load_data()
    summary = data['summary']
    if not all(summary['requirements_met'].values()):
        raise ValueError(f'Dataset does not satisfy assignment: {summary["requirements_met"]}')
    _write_json(manifest_path, dict(completed=False, code_sha256=code_hash, seed=SEED))
    X, V = data['train'], data['validation']
    if not V.nnz:
        raise ValueError('Validation split is empty')
    features = np.column_stack([data['item_features']['language'], data['item_features']['owner']])
    _write_json(RESULTS / 'dataset_summary.json', summary)
    archives = []
    for path in sorted((DEFAULT_CACHE / 'hours').glob('*.manifest.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        archives.append({key: value[key] for key in ['source_url', 'source_sha256',
                         'source_compressed_bytes', 'extracted_sha256', 'counts']})
    _write_json(RESULTS / 'source_archives.json', archives)
    torch.set_num_threads(4)
    set_num_threads(4)
    np.random.seed(SEED)
    ImplicitALS(n_factors=2, n_iterations=1).fit(sparse.csr_matrix([[1., 0.], [0., 1.]]))
    selected, trials = {}, []
    spaces = {name: list(values) for name, values in SEARCH_SPACES.items()}
    for name in ['ItemKNN', 'iALS', 'EASE', 'FM metadata', 'Recency iALS']:
        if name == 'Recency iALS':
            spaces[name] = [dict(**selected['iALS'], half_life_days=h, floor=f)
                           for h, f in itertools.product([.5, 2., 7.], [.25, .75])]
        best_value = -np.inf
        for number, params in enumerate(spaces[name], start=1):
            model = _make_model(name, params)
            seconds = _fit(model, name, X, features, data['train_timestamps'], _cutoff(summary['train_cutoff_utc']))
            metrics = evaluate(model, X, V)
            record = dict(model=name, trial=number, parameters=json.dumps(params, sort_keys=True),
                          fit_seconds=seconds, **_metrics_only(metrics))
            trials.append(record)
            print(f'{name} {number}/{len(spaces[name])}: validation NDCG@10={metrics[PRIMARY_METRIC]:.6f}, '
                  f'Recall@10={metrics["recall_at_10"]:.6f}, fit={seconds:.1f}s', flush=True)
            if metrics[PRIMARY_METRIC] > best_value:
                best_value = metrics[PRIMARY_METRIC]
                selected[name] = params.copy()
            pd.DataFrame(trials).to_csv(RESULTS / 'validation_trials.csv', index=False)
        print(f'Selected {name}: {selected[name]}', flush=True)
    selected['FM IDs only'] = selected['FM metadata'].copy()
    selected['Popularity'] = {}
    _write_json(RESULTS / 'selected_parameters.json', {'primary_metric': PRIMARY_METRIC,
        'seed': SEED, 'search_spaces': spaces, 'selected': selected,
        'test_not_used_for_selection': True, 'tie_break': 'first configuration in fixed declared order'})
    test = data['test']
    if not test.nnz:
        raise ValueError('Test split is empty')
    history = (X + V).tocsr()
    history.data[:] = 1.
    timestamps = (data['train_timestamps'] + data['validation_timestamps']).tocsr()
    test_records, per_user, loss_curves = [], {}, {}
    reference_cohort = None
    for name in ['Popularity', 'ItemKNN', 'iALS', 'EASE', 'FM metadata', 'Recency iALS', 'FM IDs only']:
        model = _make_model(name, selected[name])
        seconds = _fit(model, name, history, features, timestamps, _cutoff(summary['validation_cutoff_utc']))
        started = time.perf_counter()
        metrics = evaluate(model, history, test)
        if reference_cohort is None:
            reference_cohort = metrics['user_ids']
        else:
            np.testing.assert_array_equal(reference_cohort, metrics['user_ids'])
        per_user[name] = metrics['per_user_ndcg']
        test_records.append(dict(model=name, role='baseline' if name == 'Popularity' else (
            'ablation' if name == 'FM IDs only' else 'required'), fit_seconds=seconds,
            evaluation_seconds=time.perf_counter() - started,
            parameters=json.dumps(selected[name], sort_keys=True), **_metrics_only(metrics)))
        if name.startswith('FM'):
            loss_curves[name] = model.loss_history_
        print(f'Final {name}: test NDCG@10={metrics[PRIMARY_METRIC]:.6f}, Recall@10={metrics["recall_at_10"]:.6f}', flush=True)
    frame = pd.DataFrame(test_records)
    frame.to_csv(RESULTS / 'test_metrics.csv', index=False)
    details = {
        'protocol': 'Frozen training catalog; full-catalog ranking; remove all train+validation pairs; binary future-star relevance.',
        'n_validation_users': int(trials[0]['n_users']), 'n_test_users': int(len(reference_cohort)),
        'metadata_ablation': paired_bootstrap_difference(per_user['FM metadata'], per_user['FM IDs only'], seed=SEED),
        'recency_ablation': paired_bootstrap_difference(per_user['Recency iALS'], per_user['iALS'], seed=SEED),
        'fm_training_bce': loss_curves,
        'bootstrap_note': 'Paired user bootstrap, 1000 resamples, percentile 95% CI; conditional on one temporal split and seed.',
    }
    _write_json(RESULTS / 'evaluation_details.json', details)
    versions = {package: importlib.metadata.version(package) for package in
                ['numpy', 'scipy', 'pandas', 'matplotlib', 'numba', 'torch', 'nbformat', 'nbclient']}
    try:
        versions['orjson'] = importlib.metadata.version('orjson')
    except importlib.metadata.PackageNotFoundError:
        versions['orjson'] = 'not installed; standard-library json fallback'
    _write_json(manifest_path, dict(completed=True, code_sha256=code_hash, seed=SEED,
        primary_metric=PRIMARY_METRIC, versions=versions, python=platform.python_version(),
        platform=platform.platform(), fm_device='cpu', threads=4,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_archive_count=len(archives), test_used_after_selection=True))
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Refit and evaluate instead of reusing matching results')
    parser.add_argument('--no-download', action='store_true', help='Use only the locally prepared/extracted data')
    args = parser.parse_args()
    print(run_experiments(force=args.force, download=not args.no_download).to_string(index=False))
