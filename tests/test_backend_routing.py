from pathlib import Path
import subprocess
import tempfile
import threading
import time
import warnings

import numpy as np
import pytest
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import ituna
from ituna import config
from ituna import metrics


def test_backend_route_specificity_resolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with config.config_context(
            DEFAULT_BACKEND="in_memory",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            config.register_backend_route(method="fit", backend="disk_cache")
            config.register_backend_route(model_class=metrics.ConsistencyTransform, backend="disk_cache_distributed", backend_kwargs={"trigger_type": "manual"})
            config.register_backend_route(method="fit", model_class=metrics.ConsistencyTransform, backend="datajoint", backend_kwargs={"schema_name": "x"})

            backend_name, kwargs = config.resolve_backend_route(method="fit", model_class=metrics.PairwiseConsistency)
            assert backend_name == "datajoint"
            assert kwargs["schema_name"] == "x"

            backend_name, kwargs = config.resolve_backend_route(method="fit", model_class=PCA)
            assert backend_name == "disk_cache"
            assert kwargs == {}

            backend_name, kwargs = config.resolve_backend_route(method="score", model_class=metrics.PairwiseConsistency)
            assert backend_name == "disk_cache_distributed"
            assert kwargs["trigger_type"] == "manual"


def test_consistency_transform_fit_route_is_cached_with_disk_cache(ica_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with config.config_context(
            DEFAULT_BACKEND="in_memory",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            config.register_backend_route(
                method="fit",
                model_class=metrics.ConsistencyTransform,
                backend="disk_cache",
            )

            ensemble = ituna.ConsistencyEnsemble(
                estimator=PCA(n_components=3, random_state=42),
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Linear(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=[0, 1],
            )
            ensemble.fit(ica_data)

            cached_files = list((cache_dir / "trained_models").glob("*.pkl"))
            # Base estimators were in-memory, but consistency transform should be cached.
            assert len(cached_files) == 1


def test_transform_calls_are_cached_in_disk_cache_backend(ica_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with config.config_context(
            DEFAULT_BACKEND="disk_cache",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            ensemble = ituna.ConsistencyEnsemble(
                estimator=PCA(n_components=3, random_state=42),
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Linear(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=[0, 1, 2],
            )
            ensemble.fit(ica_data)

            transform_cache_files_before = list((cache_dir / "transforms").glob("*"))
            assert len(transform_cache_files_before) >= len(ensemble.estimators_)

            # If transforms are recomputed instead of loaded from cache, this will crash.
            for estimator_model in ensemble.estimators_:
                estimator_model.transform = lambda X: (_ for _ in ()).throw(RuntimeError("transform recomputed"))

            cached_results = ensemble._transforms(ica_data)
            assert len(cached_results) == len(ensemble.estimators_)
            assert all(isinstance(result, np.ndarray) for result in cached_results)


def _run_manual_worker(cache_dir: Path, sweep_name: str, poll_interval: float = 0.1, timeout: float = 10.0):
    """Poll for the sweep file and run the distributed worker command."""
    sweep_file = cache_dir / "sweep_data" / f"{sweep_name}.csv"
    start_time = time.time()

    while not sweep_file.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Sweep file {sweep_file} did not appear within {timeout} seconds.")
        time.sleep(poll_interval)

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


@pytest.mark.parametrize("estimator_cls", [FastICA, PCA])
def test_tutorial_route_pattern_distributed_fit_disk_cache_transforms(ica_data, estimator_cls):
    """Integration test mirroring tutorial route-registration workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        sweep_name = "tutorial_route_pattern"
        with config.config_context(
            DEFAULT_BACKEND="disk_cache_distributed",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={
                "trigger_type": "manual",
                "sweep_type": "constant",
                "sweep_name": sweep_name,
                "fit_time_out": 60,
            },
            BACKEND_ROUTES={},
        ):
            config.register_backend_route(
                method="fit",
                model_class=metrics.ConsistencyTransform,
                backend="disk_cache",
            )
            config.register_backend_route(
                method="transform",
                model_class=estimator_cls,
                backend="disk_cache",
            )

            if estimator_cls == FastICA:
                estimator = FastICA(n_components=3, random_state=42, max_iter=200)
            else:
                estimator = PCA(n_components=3, random_state=42)

            ensemble = ituna.ConsistencyEnsemble(
                estimator=estimator,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=2,
            )

            worker_thread = threading.Thread(
                target=_run_manual_worker,
                args=(cache_dir, sweep_name),
            )
            worker_thread.start()

            with warnings.catch_warnings():
                if estimator_cls == FastICA:
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.decomposition._fastica",
                    )
                ensemble.fit(ica_data)
                _ = ensemble.score(ica_data)
                _ = ensemble.transform(ica_data)

            worker_thread.join()

            # Estimator fits (2) + consistency transform fit (1) should be cached.
            trained_model_files = list((cache_dir / "trained_models").glob("*.pkl"))
            assert len(trained_model_files) == 3

            # ConsistencyTransform fit should not be part of the distributed sweep.
            sweep_csv = cache_dir / "sweep_data" / f"{sweep_name}.csv"
            assert sweep_csv.exists()
            with open(sweep_csv, "r", encoding="utf-8") as f:
                line_count = len([line for line in f if line.strip()])
            # header + two estimator entries
            assert line_count == 3

            # Transform caching route should create transform cache files.
            transform_cache_files = list((cache_dir / "transforms").glob("*"))
            assert len(transform_cache_files) >= len(ensemble.estimators_)
