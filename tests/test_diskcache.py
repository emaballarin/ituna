from pathlib import Path
import tempfile
import warnings

import numpy as np
import pytest
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import ituna
from ituna import config
from ituna import metrics


@pytest.mark.parametrize(
    "estimator_cls",
    [
        FastICA,
        PCA,
    ],
)
def test_disk_cache_backend(ica_data, estimator_cls):
    """Test that the DiskCacheBackend caches fitted models."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        with config.config_context(DEFAULT_BACKEND="disk_cache", CACHE_DIR=cache_dir):
            if estimator_cls == FastICA:
                estimator = FastICA(n_components=3, random_state=42, max_iter=200)
            else:  # PCA
                estimator = PCA(n_components=3, random_state=42)

            ensemble = ituna.ConsistencyEnsemble(
                estimator=estimator,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=[0, 1, 2],
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
            assert len(cached_files) == len(ensemble.random_states) + 1

            # Second fit, should load from cache
            if estimator_cls == FastICA:
                estimator2 = FastICA(n_components=3, random_state=42, max_iter=200)
            else:  # PCA
                estimator2 = PCA(n_components=3, random_state=42)

            ensemble2 = ituna.ConsistencyEnsemble(
                estimator=estimator2,
                consistency_transform=metrics.PairwiseConsistency(
                    indeterminacy=metrics.Permutation(),
                    symmetric=False,
                    include_diagonal=True,
                ),
                random_states=[0, 1, 2],
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
