import numpy as np
import pytest
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

import ituna
from ituna import metrics

N_SAMPLES = 10000
N_FEATURES = 10


@pytest.mark.parametrize("n_components", [2, 3, 4])
@pytest.mark.parametrize(
    "indeterminancy_cls",
    [
        metrics.Permutation,
        metrics.Identity,
        metrics.Linear,
        metrics.Affine,
    ],
)
def test_pca_ensemble(data, n_components, indeterminancy_cls):
    n_features = data.shape[1]
    if n_components > n_features:
        pytest.skip(f"n_components ({n_components}) > n_features ({n_features})")
    n_models = 5
    pca = PCA(n_components=n_components)
    indeterminacy = indeterminancy_cls()
    ensemble = ituna.ConsistencyEnsemble(
        estimator=pca,
        consistency_transform=metrics.PairwiseConsistency(
            indeterminacy=indeterminacy,
            symmetric=False,
            include_diagonal=True,
        ),
        random_states=np.arange(n_models),
    )
    ensemble.fit(data)
    score = ensemble.score(data)

    assert isinstance(score, (int, float, np.number))

    np.testing.assert_almost_equal(score, 1.0)


@pytest.mark.parametrize("n_components", [3, 7])
@pytest.mark.parametrize("max_iter", [200, 1000])
@pytest.mark.parametrize(
    "indeterminancy_cls",
    [
        metrics.Permutation,
        metrics.Affine,
        metrics.Linear,
    ],
)
def test_fastica_ensemble(ica_data, n_components, max_iter, indeterminancy_cls):
    n_features = ica_data.shape[1]
    if n_components > n_features:
        pytest.skip(f"n_components ({n_components}) > n_features ({n_features})")

    n_models = 5
    fastica = FastICA(n_components=n_components, max_iter=max_iter, random_state=42)
    indeterminacy = indeterminancy_cls()
    ensemble = ituna.ConsistencyEnsemble(
        estimator=fastica,
        consistency_transform=metrics.PairwiseConsistency(
            indeterminacy=indeterminacy,
            symmetric=False,
            include_diagonal=True,
        ),
        random_states=np.arange(n_models),
    )

    # Suppress FastICA convergence warnings
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.decomposition._fastica")
        ensemble.fit(ica_data)
        score = ensemble.score(ica_data)

    assert isinstance(score, (int, float, np.number))

    # For ICA, scores may not be perfect 1.0 due to convergence issues
    # but diagonal should be close to 1.0
    np.testing.assert_array_less(0.8, score)


@pytest.mark.parametrize(
    "estimator_cls",
    [
        FastICA,
        PCA,
    ],
)
@pytest.mark.parametrize(
    "indeterminancy_cls",
    [
        metrics.Identity,
        metrics.Permutation,
    ],
)
def test_gridsearchcv_compatibility(ica_data, estimator_cls, indeterminancy_cls):
    """Test that ConsistencyEnsemble is compatible with sklearn's GridSearchCV."""

    # Skip Identity indeterminacy for FastICA as it's not meaningful
    if estimator_cls == FastICA and indeterminancy_cls == metrics.Identity:
        pytest.skip("Identity indeterminacy not meaningful for FastICA")

    # Create a small parameter grid to test
    if estimator_cls == FastICA:
        param_grid = {
            "estimator__n_components": [2, 3],
            "estimator__max_iter": [100, 200],
        }
        estimator = FastICA(random_state=42)
    else:  # PCA
        param_grid = {
            "estimator__n_components": [2, 3],
        }
        estimator = PCA(random_state=42)

    # Skip if n_components > n_features
    max_components = max(param_grid.get("estimator__n_components", [0]))
    if max_components > ica_data.shape[1]:
        pytest.skip(f"max n_components ({max_components}) > n_features ({ica_data.shape[1]})")

    indeterminacy = indeterminancy_cls()
    ensemble = ituna.ConsistencyEnsemble(
        estimator=estimator,
        consistency_transform=metrics.PairwiseConsistency(
            indeterminacy=indeterminacy,
            symmetric=False,
            include_diagonal=True,
        ),
        random_states=[0, 1, 2],  # Small ensemble for speed
    )

    # Create GridSearchCV instance
    grid_search = GridSearchCV(
        estimator=ensemble,
        param_grid=param_grid,
        cv=2,  # Small cv for speed
        scoring=None,  # Use default scoring
        verbose=0,
    )

    # Suppress warnings for FastICA convergence
    import warnings

    with warnings.catch_warnings():
        if estimator_cls == FastICA:
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.decomposition._fastica")

        # Fit should work without errors
        grid_search.fit(ica_data)

        # Should have best parameters
        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_estimator_")
        assert hasattr(grid_search, "best_score_")

        # Transform should work with best estimator
        transformed = grid_search.transform(ica_data)
        assert transformed.shape[0] == ica_data.shape[0]  # Number of samples
        assert transformed.shape[1] == grid_search.best_estimator_.estimator.n_components  # Number of components

        # Score should work
        score = grid_search.score(ica_data)
        assert isinstance(score, (int, float, np.number))


def test_set_params_random_states_takes_effect():
    """`random_states` is resolved on use, so set_params is not silently ignored."""
    ensemble = ituna.ConsistencyEnsemble(
        estimator=PCA(n_components=2),
        consistency_transform=metrics.PairwiseConsistency(indeterminacy=metrics.Linear()),
        random_states=3,
    )
    assert len(ensemble._init_estimators()) == 3

    ensemble.set_params(random_states=7)
    assert ensemble.get_params()["random_states"] == 7
    assert len(ensemble._init_estimators()) == 7

    ensemble.set_params(random_states=[0, 1])
    assert len(ensemble._init_estimators()) == 2


@pytest.mark.parametrize("random_states", [3, [0, 1, 2], np.arange(3)])
def test_random_states_is_stored_exactly_as_passed(random_states):
    """sklearn requires __init__ to leave its parameters untouched."""
    ensemble = ituna.ConsistencyEnsemble(
        estimator=PCA(n_components=2),
        consistency_transform=metrics.PairwiseConsistency(indeterminacy=metrics.Linear()),
        random_states=random_states,
    )
    stored = ensemble.get_params()["random_states"]
    assert type(stored) is type(random_states)
    assert np.all(stored == random_states)
    assert len(ensemble._init_estimators()) == 3


def test_invalid_random_states_still_rejected_at_construction():
    with pytest.raises(ValueError, match="Random states must be an integer or a list of integers"):
        ituna.ConsistencyEnsemble(
            estimator=PCA(n_components=2),
            consistency_transform=metrics.PairwiseConsistency(indeterminacy=metrics.Linear()),
            random_states="five",
        )
