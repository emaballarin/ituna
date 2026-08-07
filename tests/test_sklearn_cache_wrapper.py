from pathlib import Path
import tempfile
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import check_estimator

import ituna
from ituna import metrics
import ituna.estimator


def test_cached_local_patch_preserves_estimator_type(ica_data):
    estimator = PCA(n_components=3, random_state=42)
    patched = ituna.sklearn.cached(estimator)
    assert patched is estimator
    assert isinstance(patched, PCA)
    patched.fit(ica_data)


def test_cached_local_patch_fit_uses_route_outside_ensemble(ica_data, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
            DEFAULT_BACKEND="in_memory",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            ituna.config.register_backend_route(
                method="fit",
                model_class=PCA,
                backend="disk_cache",
            )

            patched = ituna.sklearn.cached(PCA(n_components=3, random_state=42), methods=["fit"])
            patched.fit(ica_data)
            assert len(list((cache_dir / "trained_models").glob("*.pkl"))) == 1

            # If cache misses on second local fit, this patched method would crash.
            monkeypatch.setattr(PCA, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fit recomputed")))
            patched_again = ituna.sklearn.cached(PCA(n_components=3, random_state=42), methods=["fit"])
            patched_again.fit(ica_data)


def test_cached_local_patch_transform_uses_disk_cache(ica_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
            DEFAULT_BACKEND="disk_cache",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            patched = ituna.sklearn.cached(PCA(n_components=3, random_state=42), methods=["fit", "transform"])
            patched.fit(ica_data)
            first = patched.transform(ica_data)
            assert len(list((cache_dir / "transforms").glob("*"))) >= 1

            second = ituna.sklearn.cached(PCA(n_components=3, random_state=42), methods=["fit", "transform"]).fit(ica_data).transform(ica_data)
            np.testing.assert_allclose(first, second)


def test_enable_global_cache_for_regular_sklearn_usage(ica_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
            DEFAULT_BACKEND="disk_cache",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            ituna.config.register_backend_route(method="fit", model_class=PCA, backend="disk_cache")
            ituna.config.register_backend_route(method="transform", model_class=PCA, backend="disk_cache")

            ituna.sklearn.enable_global_cache([PCA], methods=["fit", "transform"])
            try:
                status = ituna.sklearn.get_global_cache_status()
                pca_key = f"{PCA.__module__}.{PCA.__qualname__}"
                assert sorted(status[pca_key]) == ["fit", "transform"]

                pca = PCA(n_components=3, random_state=42)
                pca.fit(ica_data)
                transformed = pca.transform(ica_data)
                assert transformed.shape[0] == ica_data.shape[0]
                assert len(list((cache_dir / "trained_models").glob("*.pkl"))) == 1
                assert len(list((cache_dir / "transforms").glob("*"))) >= 1
            finally:
                ituna.sklearn.disable_global_cache([PCA], methods=["fit", "transform"])

            assert ituna.sklearn.get_global_cache_status() == {}


def test_global_patch_predict_and_score_cache():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 4)
    y = X @ np.array([0.5, -1.2, 2.0, 0.3]) + 0.1 * rng.randn(120)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
            DEFAULT_BACKEND="disk_cache",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            ituna.config.register_backend_route(method="fit", model_class=LinearRegression, backend="disk_cache")
            ituna.config.register_backend_route(method="predict", model_class=LinearRegression, backend="disk_cache")
            ituna.config.register_backend_route(method="score", model_class=LinearRegression, backend="disk_cache")

            ituna.sklearn.enable_global_cache([LinearRegression], methods=["fit", "predict", "score"])
            try:
                model = LinearRegression()
                model.fit(X, y)
                preds = model.predict(X)
                score_val = model.score(X, y)
                assert preds.shape[0] == X.shape[0]
                assert isinstance(score_val, float)
                assert len(list((cache_dir / "transforms").glob("*"))) >= 2
            finally:
                ituna.sklearn.disable_global_cache([LinearRegression], methods=["fit", "predict", "score"])


def test_global_patch_does_not_recache_internal_indeterminacy_models(ica_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
            DEFAULT_BACKEND="disk_cache",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            ituna.config.register_backend_route(method="fit", model_class=LinearRegression, backend="disk_cache")
            ituna.sklearn.enable_global_cache([LinearRegression], methods=["fit"])
            try:
                ensemble = ituna.ConsistencyEnsemble(
                    estimator=PCA(n_components=3, random_state=42),
                    consistency_transform=metrics.PairwiseConsistency(
                        indeterminacy=metrics.Linear(),
                        symmetric=False,
                        include_diagonal=True,
                    ),
                    random_states=2,
                )
                ensemble.fit(ica_data)
                cached_files = list((cache_dir / "trained_models").glob("*.pkl"))
                # 2 estimator fits + 1 consistency transform fit; internal indeterminacy fits should not be globally recached.
                assert len(cached_files) == 3
            finally:
                ituna.sklearn.disable_global_cache([LinearRegression], methods=["fit"])


def test_sklearn_check_estimator_with_local_cached_patch():
    estimator = ituna.sklearn.cached(PCA(n_components=2, random_state=0), methods=["fit", "transform"])
    check_estimator(estimator)


def test_cached_warns_for_non_deterministic_estimators(monkeypatch):
    monkeypatch.setattr(ituna.estimator, "check_non_deterministic", lambda _estimator: True)
    with pytest.warns(UserWarning, match="non-deterministic"):
        ituna.sklearn.cached(PCA(n_components=2, random_state=0), methods=["fit"])


def test_cached_does_not_warn_for_deterministic_estimators(monkeypatch):
    monkeypatch.setattr(ituna.estimator, "check_non_deterministic", lambda _estimator: False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ituna.sklearn.cached(PCA(n_components=2, random_state=0), methods=["fit"])
    assert len(record) == 0


def test_cached_consistency_ensemble_mixed_backend_route_fit_cache(ica_data, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
            DEFAULT_BACKEND="in_memory",
            CACHE_DIR=cache_dir,
            BACKEND_KWARGS={},
            BACKEND_ROUTES={},
        ):
            ituna.config.register_backend_route(
                method="fit",
                model_class=ituna.ConsistencyEnsemble,
                backend="disk_cache",
            )

            ensemble = ituna.ConsistencyEnsemble(
                estimator=PCA(n_components=3, random_state=42),
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Linear(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=2,
            )
            cached_ensemble = ituna.sklearn.cached(ensemble, methods=["fit"])
            cached_ensemble.fit(ica_data)
            assert len(list((cache_dir / "trained_models").glob("*.pkl"))) == 1

            monkeypatch.setattr(
                ituna.ConsistencyEnsemble,
                "fit",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ensemble fit recomputed")),
            )
            second = ituna.sklearn.cached(
                ituna.ConsistencyEnsemble(
                    estimator=PCA(n_components=3, random_state=42),
                    consistency_transform=metrics.PairwiseConsistency(
                        indeterminacy=metrics.Linear(),
                        symmetric=False,
                        include_diagonal=True,
                    ),
                    random_states=2,
                ),
                methods=["fit"],
            )
            second.fit(ica_data)


def test_cached_consistency_ensemble_disk_cache_for_outer_and_inner_fit(ica_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        with ituna.config.config_context(
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
                random_states=2,
            )
            cached_ensemble = ituna.sklearn.cached(ensemble, methods=["fit"])
            cached_ensemble.fit(ica_data)
            assert len(list((cache_dir / "trained_models").glob("*.pkl"))) >= 1

            transformed = cached_ensemble.transform(ica_data)
            assert isinstance(transformed, metrics.PairwiseConsistencyArray)
