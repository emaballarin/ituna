from pathlib import Path
import subprocess
import tempfile
import threading
import time
import warnings

import cebra
import numpy as np
import pytest
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import ituna
from ituna import config
from ituna import metrics
from ituna._backends.disk_cache import DiskCacheBackend
from ituna._backends.disk_cache import DiskCacheDistributedBackend


def get_estimator(estimator_cls):
    if estimator_cls == FastICA:
        estimator = FastICA(n_components=7, random_state=42, max_iter=200)
    elif estimator_cls == PCA:  # PCA
        estimator = PCA(n_components=7, random_state=42)

    elif estimator_cls == cebra.CEBRA:  # CEBRA
        estimator = cebra.CEBRA(
            model_architecture="offset1-model",
            batch_size=1,
            num_hidden_units=8,
            learning_rate=3e-4,
            temperature=1,
            output_dimension=7,
            max_iterations=1,
            distance="cosine",
            conditional="time_delta",
            device="cuda_if_available",
            verbose=True,
            time_offsets=1,
        )
    else:
        raise ValueError(f"Unsupported base model class: {estimator_cls}")

    return estimator


@pytest.mark.parametrize(
    "estimator_cls",
    [
        FastICA,
        PCA,
        cebra.CEBRA,
    ],
)
def test_multi_process(ica_data, estimator_cls):
    """Test that the DiskCacheDistributedBackend caches fitted models."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        backend_kwargs = {
            "trigger_type": "auto",
            "num_workers": 3,
            "fit_time_out": 60,
        }

        with config.config_context(
            DEFAULT_BACKEND="disk_cache_distributed",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS=backend_kwargs,
        ):
            estimator = get_estimator(estimator_cls)

            ensemble = ituna.ConsistencyEnsemble(
                estimator=estimator,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=2,
            )
            # First fit, should train and cache models
            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble.fit(ica_data)
                score1 = ensemble.score(ica_data)
                transform1 = ensemble.transform(ica_data)

            # Check that cache directory is not empty
            model_cache_dir = cache_dir / "trained_models"
            assert model_cache_dir.exists()
            cached_files = list(model_cache_dir.glob("*.pkl"))
            # +1 for cached consistency transform
            assert len(cached_files) == ensemble.random_states + 1

            estimator2 = get_estimator(estimator_cls)

            ensemble2 = ituna.ConsistencyEnsemble(
                estimator=estimator2,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=2,
            )

            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble2.fit(ica_data)
                score2 = ensemble2.score(ica_data)
                transform2 = ensemble2.transform(ica_data)

            np.testing.assert_allclose(score1, score2)
            np.testing.assert_allclose(transform1, transform2)
            assert isinstance(score1, (int, float, np.number))


def run_manual_worker(cache_dir: Path, sweep_name: str, poll_interval: float = 0.1, timeout: float = 10.0):
    """
    Poll for the sweep file and run the manual trigger command.
    """
    sweep_file = cache_dir / "sweep_data" / f"{sweep_name}.csv"
    start_time = time.time()

    while not sweep_file.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Sweep file {sweep_file} did not appear within {timeout} seconds.")
        time.sleep(poll_interval)

    # The file exists, now run the command.
    cmd = [
        "ituna-fit-distributed",
        "--sweep-name",
        sweep_name,
        "--cache-dir",
        str(cache_dir.resolve()),
        "--order-by",
        "random",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print("Manual worker command failed.")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        result.check_returncode()


@pytest.mark.parametrize(
    "estimator_cls",
    [
        FastICA,
        PCA,
        cebra.CEBRA,
    ],
)
def test_manual_trigger(ica_data, estimator_cls):
    """Test that the DiskCacheDistributedBackend caches fitted models."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        backend_kwargs = {
            "trigger_type": "manual",
            "sweep_type": "constant",
            "sweep_name": "test_manual_trigger",
            "fit_time_out": 60,
        }

        with config.config_context(
            DEFAULT_BACKEND="disk_cache_distributed",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS=backend_kwargs,
        ):
            estimator = get_estimator(estimator_cls)

            ensemble = ituna.ConsistencyEnsemble(
                estimator=estimator,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=2,
            )

            sweep_name = backend_kwargs["sweep_name"]

            # Create and start the worker thread
            worker_thread = threading.Thread(
                target=run_manual_worker,
                args=(cache_dir, sweep_name),
            )
            worker_thread.start()

            # First fit, should train and cache models
            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble.fit(ica_data)
                score1 = ensemble.score(ica_data)
                transform1 = ensemble.transform(ica_data)

            # Wait for the worker thread to finish
            worker_thread.join()

            # Check that cache directory is not empty
            model_cache_dir = cache_dir / "trained_models"
            assert model_cache_dir.exists()
            cached_files = list(model_cache_dir.glob("*.pkl"))
            # +1 for cached consistency transform
            assert len(cached_files) == ensemble.random_states + 1

            estimator2 = get_estimator(estimator_cls)

            ensemble2 = ituna.ConsistencyEnsemble(
                estimator=estimator2,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=2,
            )

            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble2.fit(ica_data)
                score2 = ensemble2.score(ica_data)
                transform2 = ensemble2.transform(ica_data)

            np.testing.assert_allclose(score1, score2)
            np.testing.assert_allclose(transform1, transform2)
            assert isinstance(score1, (int, float, np.number))


def test_distributed_fast_path_skips_sweep_when_all_models_already_cached(ica_data, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        def make_models():
            return [
                PCA(n_components=3, random_state=0),
                PCA(n_components=3, random_state=1),
            ]

        local_backend = DiskCacheBackend(cache_dir=cache_dir)
        local_backend.fit_models(make_models(), ica_data)

        distributed_backend = DiskCacheDistributedBackend(
            cache_dir=cache_dir,
            trigger_type="manual",
            sweep_type="constant",
            sweep_name="fast_path_test",
            fit_time_out=1,
        )

        def should_not_run(*args, **kwargs):
            raise AssertionError("distributed sweep registration should be skipped on full cache hit")

        monkeypatch.setattr(distributed_backend, "_store_model", should_not_run)
        monkeypatch.setattr(distributed_backend, "_add_to_sweep", should_not_run)
        monkeypatch.setattr(distributed_backend, "_trigger_sweep", should_not_run)

        trained_models = distributed_backend.fit_models(make_models(), ica_data)
        assert len(trained_models) == 2
        assert all(model is not None for model in trained_models)
        assert not (cache_dir / "sweep_data" / "fast_path_test.csv").exists()
