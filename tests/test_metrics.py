import itertools

import numpy as np
import pytest

from ituna import metrics
from ituna.utils import dense_to_sparse

N_SAMPLES = 10000
N_FEATURES = 10


@pytest.mark.parametrize(
    "data_fixture",
    ["other_data", "permuted_data", "linear_mixed_data", "affine_mixed_data"],
)
def test_data_shape_vs_base(data, data_fixture, request):
    transformed = request.getfixturevalue(data_fixture)
    assert data.shape == transformed.shape
    assert not np.allclose(data, transformed, atol=1e-10)


@pytest.mark.parametrize(
    "data_fixture1,data_fixture2",
    list(itertools.combinations(["other_data", "permuted_data", "linear_mixed_data", "affine_mixed_data"], 2)),
)
def test_data_shape_combinations(data, data_fixture1, data_fixture2, request):
    data1 = request.getfixturevalue(data_fixture1)
    data2 = request.getfixturevalue(data_fixture2)
    assert data1.shape == data2.shape
    assert not np.allclose(data1, data2, atol=1e-10)


def test_permutation(data, permuted_data):
    permutation = metrics.Permutation()
    permutation.fit(data, permuted_data)
    np.testing.assert_almost_equal(permutation.score(data, permuted_data), 1.0)


def test_identity(data):
    identity = metrics.Identity()
    identity.fit(data, data)
    np.testing.assert_almost_equal(identity.score(data, data), 1.0)


def test_linear(data, linear_mixed_data):
    linear = metrics.Linear()
    linear.fit(data, linear_mixed_data)
    np.testing.assert_almost_equal(linear.score(data, linear_mixed_data), 1.0)


def test_affine(data, affine_mixed_data):
    affine = metrics.Affine()
    affine.fit(data, affine_mixed_data)
    np.testing.assert_almost_equal(affine.score(data, affine_mixed_data), 1.0)


@pytest.mark.parametrize(
    "init_metric",
    [
        metrics.Permutation,
        metrics.Identity,
        metrics.Linear,
        metrics.Affine,
    ],
)
def test_metric_consistency(data, other_data, init_metric):
    metric = init_metric()
    metric.fit(data, data)
    np.testing.assert_almost_equal(metric.score(data, data), 1.0)

    metric = init_metric()
    metric.fit(data, other_data)
    assert metric.score(data, other_data) < 0.9


INDETERMINACY_MAP = {
    "identity": metrics.Identity,
    "permutation": metrics.Permutation,
    "linear": metrics.Linear,
    "affine": metrics.Affine,
}

# n_estimators, n_samples, n_features, indeterminacy_type
API_TEST_PARAMS_BASE = [
    (100, 5, "identity"),
    (100, 10, "permutation"),
    (200, 8, "linear"),
    (50, 3, "affine"),
]
API_TEST_PARAMS = [(n_est, n_samp, n_feat, ind_type) for n_est in [1, 3] for n_samp, n_feat, ind_type in API_TEST_PARAMS_BASE]


@pytest.mark.parametrize("consistent_embeddings", API_TEST_PARAMS, indirect=True)
def test_consistency_fit_transform_api(consistent_embeddings):
    X, params = consistent_embeddings
    n_estimators, n_samples, n_features, indeterminacy_type = params
    indeterminacy_cls = INDETERMINACY_MAP[indeterminacy_type]

    transformer = metrics.PairwiseConsistency(indeterminacy=indeterminacy_cls())
    result = transformer.fit(X).transform(X)

    assert isinstance(result, metrics.PairwiseConsistencyArray)
    assert hasattr(result, "embeddings")
    assert hasattr(result, "aligned_embeddings")
    assert hasattr(result, "scores")
    assert hasattr(result, "reference_id")
    assert hasattr(result, "aligned_to_reference")

    assert result.shape == (n_samples, n_features)
    assert result.embeddings.shape == (n_estimators, n_samples, n_features)
    assert np.allclose(result.embeddings, X)

    score_indices, score_values = result.scores
    aligned_indices, aligned_values = result.aligned_embeddings

    assert score_indices.shape[1] == 2
    assert score_indices.shape[0] == len(score_values)
    assert aligned_indices.shape[1] == 2
    assert aligned_indices.shape[0] == aligned_values.shape[0]
    # aligned_values is empty for 1 estimator
    if n_estimators > 1:
        assert aligned_values.shape == (len(aligned_indices), n_samples, n_features)
    else:
        assert aligned_values.shape == (0,)

    assert isinstance(result.reference_id, (int, np.integer))
    assert result.aligned_to_reference.shape == (
        n_estimators,
        n_samples,
        n_features,
    )

    # With perfect data, scores should be close to 1.0
    assert np.allclose(score_values, 1.0)

    # Check mean embedding calculation
    assert np.allclose(result, np.nanmean(result.aligned_to_reference, axis=0))


@pytest.mark.parametrize("consistent_embeddings", API_TEST_PARAMS, indirect=True)
def test_consistency_fit_transform_equivalence(consistent_embeddings):
    X, params = consistent_embeddings
    _, _, _, indeterminacy_type = params
    indeterminacy_cls = INDETERMINACY_MAP[indeterminacy_type]

    transformer1 = metrics.PairwiseConsistency(indeterminacy=indeterminacy_cls())
    result1 = transformer1.fit_transform(X)

    transformer2 = metrics.PairwiseConsistency(indeterminacy=indeterminacy_cls())
    result2 = transformer2.fit(X).transform(X)

    assert np.allclose(result1, result2)
    assert np.allclose(result1.scores[1], result2.scores[1])
    assert result1.reference_id == result2.reference_id


@pytest.mark.parametrize("consistent_embeddings", API_TEST_PARAMS, indirect=True)
def test_consistency_score(consistent_embeddings):
    X, params = consistent_embeddings
    n_estimators, _, _, indeterminacy_type = params
    indeterminacy_cls = INDETERMINACY_MAP[indeterminacy_type]

    transformer = metrics.PairwiseConsistency(indeterminacy=indeterminacy_cls())
    transformer.fit(X)
    score = transformer.score(X)

    assert isinstance(score, float)
    if n_estimators > 1:
        assert np.isclose(score, 1.0)
    else:
        # with include_diagonal=False (default), score is nan for 1 estimator
        assert np.isnan(score)


@pytest.mark.parametrize("consistent_embeddings", API_TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("include_diagonal", [True, False])
def test_consistency_init_params(consistent_embeddings, symmetric, include_diagonal):
    X, params = consistent_embeddings
    n_estimators, _, _, indeterminacy_type = params
    indeterminacy_cls = INDETERMINACY_MAP[indeterminacy_type]

    transformer = metrics.PairwiseConsistency(
        indeterminacy=indeterminacy_cls(),
        symmetric=symmetric,
        include_diagonal=include_diagonal,
    )
    result = transformer.fit_transform(X)

    # result, result.aligned_to_reference should never contain nans
    assert not np.any(np.isnan(result))
    assert not np.any(np.isnan(result.aligned_to_reference))

    score_indices, score_values = result.scores

    # Check that scores are all 1.0
    assert np.allclose(score_values, 1.0)

    # Check for number of pairs
    n_pairs = len(score_values)
    if symmetric:
        if include_diagonal:
            assert n_pairs == n_estimators * (n_estimators + 1) / 2
        else:
            assert n_pairs == n_estimators * (n_estimators - 1) / 2
    else:
        if include_diagonal:
            assert n_pairs == n_estimators * n_estimators
        else:
            assert n_pairs == n_estimators * (n_estimators - 1)


def _test_get_reference(scores, expected_max_id):
    """Helper to test reference selection."""
    n_estimators = scores.shape[0]

    # Test max_score
    transformer_max = metrics.PairwiseConsistency(indeterminacy=metrics.Identity(), reference_selection="max_score")
    transformer_max.n_estimators_ = n_estimators
    sparse_scores = dense_to_sparse(scores)
    max_id = transformer_max._get_reference(sparse_scores)
    assert max_id == expected_max_id

    # Test min_score
    transformer_min = metrics.PairwiseConsistency(indeterminacy=metrics.Identity(), reference_selection="min_score")
    transformer_min.n_estimators_ = n_estimators
    min_sparse_scores = dense_to_sparse(scores * -1)
    min_id = transformer_min._get_reference(min_sparse_scores)
    assert min_id == expected_max_id

    # Test first
    transformer_first = metrics.PairwiseConsistency(
        indeterminacy=metrics.Identity(),
        reference_selection=0,
    )
    transformer_first.n_estimators_ = n_estimators
    first_id = transformer_first._get_reference(sparse_scores)
    assert first_id == 0


@pytest.mark.parametrize(
    "scores,expected_max_id",
    [
        # Case 1: Standard case - Non Symmetric
        (
            np.array(
                [
                    [np.nan, 1.0, 2.0, 1.0],
                    [0.0, np.nan, 2.0, 1.0],
                    [0.0, 1.0, np.nan, 1.0],
                    [0.0, 1.0, 2.0, np.nan],
                ]
            ),
            2,
        ),
        # Case 2: Symmetric matrix
        (
            np.array(
                [
                    [np.nan, 1.0, 2.0, 3.0],
                    [1.0, np.nan, 4.0, 5.0],
                    [2.0, 4.0, np.nan, 6.0],
                    [3.0, 5.0, 6.0, np.nan],
                ]
            ),
            3,
        ),
        # Case 3: All same scores
        (
            np.array(
                [
                    [np.nan, 1.0, 1.0],
                    [1.0, np.nan, 1.0],
                    [1.0, 1.0, np.nan],
                ]
            ),
            0,
        ),
        # Case 4: Single entry 1x1 matrix
        (
            np.array(
                [
                    [np.nan],
                ]
            ),
            0,
        ),
    ],
)
def test_reference_selection(scores, expected_max_id):
    """Test reference selection with various score matrices."""
    _test_get_reference(scores, expected_max_id)
    # same if diagonal is included
    scores_with_diagonal = scores.copy()
    scores_with_diagonal[np.eye(scores_with_diagonal.shape[0], dtype=bool)] = 10.0
    _test_get_reference(scores_with_diagonal, expected_max_id)


@pytest.mark.parametrize("consistent_embeddings", [(3, 100, 10, "permutation"), (3, 200, 8, "linear")], indirect=True)
def test_transform_single_space_by_source_id(consistent_embeddings):
    """A single embedding space can be aligned by source_id (regression: raised AttributeError)."""
    X, params = consistent_embeddings
    _, _, _, indeterminacy_type = params
    transformer = metrics.PairwiseConsistency(indeterminacy=INDETERMINACY_MAP[indeterminacy_type]())
    transformer.fit(X)

    for source_id in range(len(X)):
        aligned = transformer.transform([X[source_id]], source_id=source_id)
        assert aligned.shape == X[source_id].shape
        assert not np.any(np.isnan(aligned))
        assert aligned.reference_id == transformer.reference_id_

    # No i->i model is fitted by default, and aligning the reference to itself is the identity.
    reference = transformer.reference_id_
    assert np.allclose(transformer.transform([X[reference]], source_id=reference), X[reference])


@pytest.mark.parametrize("consistent_embeddings", [(3, 100, 10, "permutation")], indirect=True)
def test_transform_single_space_without_fitted_pair_raises_value_error(consistent_embeddings):
    """With symmetric=True only j >= i is fitted; the reverse direction must fail explicitly."""
    X, _ = consistent_embeddings
    transformer = metrics.PairwiseConsistency(indeterminacy=metrics.Permutation(), symmetric=True)
    transformer.fit(X)

    unavailable = [i for i in range(len(X)) if i > transformer.reference_id_]
    if not unavailable:
        pytest.skip("reference is the last estimator, so every pair is available")
    with pytest.raises(ValueError, match="No indeterminacy model found"):
        transformer.transform([X[unavailable[0]]], source_id=unavailable[0])


def _partially_consistent_embeddings(n_estimators=4, n_samples=400, n_features=6, seed=20260807):
    """Embeddings that agree only in part, so the consistency score sits strictly below 1.0.

    Perfectly consistent embeddings cannot detect diagonal inflation: with s = 1.0 the inflated
    value (1 - f) * s + f is also 1.0, so the test would pass whatever the implementation did.
    """
    rng = np.random.default_rng(seed)
    reference = rng.laplace(size=(n_samples, n_features)) / np.sqrt(2)
    embeddings = []
    for i in range(n_estimators):
        if i % 2:
            # recoverable: a signed permutation of the reference
            transform = np.eye(n_features)[rng.permutation(n_features)] * rng.choice([-1.0, 1.0], size=n_features)
        else:
            # not recoverable under Permutation: a dense rotation
            q, r = np.linalg.qr(rng.standard_normal((n_features, n_features)))
            transform = q * np.sign(np.diag(r))
        embeddings.append(reference @ transform)
    return embeddings


@pytest.mark.parametrize("symmetric", [True, False])
def test_score_ignores_the_diagonal(symmetric):
    """`include_diagonal` governs which models are fitted, never what the score averages.

    A self-alignment scores exactly 1.0 under any indeterminacy class, so averaging it in would
    report (1 - f) * s + f with f the diagonal's share of the fitted pairs -- 1 / n_estimators
    when symmetric is False and 2 / (n_estimators + 1) when it is True.
    """
    X = _partially_consistent_embeddings()
    scores = {}
    for include_diagonal in (False, True):
        transformer = metrics.PairwiseConsistency(
            indeterminacy=metrics.Permutation(),
            symmetric=symmetric,
            include_diagonal=include_diagonal,
        )
        transformer.fit(X)
        scores[include_diagonal] = transformer.score(X)

        # the diagonal is still fitted, and so still available for alignment
        n_pairs = len(transformer.indeterminacy_indices_)
        if include_diagonal:
            expected = len(X) * (len(X) + 1) / 2 if symmetric else len(X) * len(X)
        else:
            expected = len(X) * (len(X) - 1) / 2 if symmetric else len(X) * (len(X) - 1)
        assert n_pairs == expected

    # non-vacuous: with s == 1.0 the inflated value would also be 1.0 and this would prove nothing
    assert scores[False] < 0.9
    assert scores[True] == scores[False]

    # and the inflation the fix removes is exactly (1 - f) * s + f
    f = 2 / (len(X) + 1) if symmetric else 1 / len(X)
    assert not np.isclose(scores[False], (1 - f) * scores[False] + f)


@pytest.mark.parametrize("include_diagonal", [True, False])
@pytest.mark.parametrize("consistent_embeddings", [(1, 100, 10, "permutation")], indirect=True)
def test_score_is_nan_for_a_single_estimator(consistent_embeddings, include_diagonal):
    """Consistency is undefined for one estimator, whatever `include_diagonal` says."""
    X, _ = consistent_embeddings
    transformer = metrics.PairwiseConsistency(indeterminacy=metrics.Permutation(), include_diagonal=include_diagonal)
    transformer.fit(X)
    assert np.isnan(transformer.score(X))


def _anisotropic(n_samples=200, n_features=5, seed=20260807):
    """Anisotropic, non-symmetric source data.

    Isotropic or symmetric test data cannot separate the Procrustes orientation `Q = U V^T` from its
    transpose: both produce a valid orthogonal matrix and a plausible R2, and neither raises.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)) @ np.diag([3.0, 1.0, 0.5, 2.0, 0.2])


def _random_orthogonal(n_features=5, seed=20260807):
    """Draw an orthogonal matrix from the QR of a Gaussian."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    return q


def test_orthogonal_recovers_a_known_rotation():
    """Exact recovery on a known Q0, which is what pins the orientation of the Procrustes solution."""
    X = _anisotropic()
    Q0 = _random_orthogonal()
    orthogonal = metrics.Orthogonal().fit(X, X @ Q0)

    np.testing.assert_allclose(orthogonal.orthogonal_, Q0, atol=1e-8)
    np.testing.assert_allclose(orthogonal.score(X, X @ Q0), 1.0, atol=1e-12)


def test_orthogonal_fits_an_orthogonal_matrix_on_arbitrary_data():
    """The constraint must hold whatever the data, not only when a gauge relation exists."""
    rng = np.random.default_rng(1)
    X = _anisotropic()
    y = rng.normal(size=X.shape)
    orthogonal = metrics.Orthogonal().fit(X, y)

    np.testing.assert_allclose(orthogonal.orthogonal_.T @ orthogonal.orthogonal_, np.eye(X.shape[1]), atol=1e-10)


def test_orthogonal_prediction_is_an_isometry():
    """An orthogonal map preserves the Frobenius norm; this is the cheapest check that it really is one."""
    rng = np.random.default_rng(2)
    X = _anisotropic()
    orthogonal = metrics.Orthogonal().fit(X, rng.normal(size=X.shape))

    np.testing.assert_allclose(np.linalg.norm(orthogonal.predict(X), "fro"), np.linalg.norm(X, "fro"), rtol=1e-12)


def test_orthogonal_reflection_handling():
    """allow_reflection selects between O(L) and SO(L), and the difference must be visible in the score."""
    X = _anisotropic()
    reflection = np.diag([1.0, 1.0, 1.0, 1.0, -1.0])
    y = X @ reflection

    with_reflection = metrics.Orthogonal(allow_reflection=True).fit(X, y)
    np.testing.assert_allclose(with_reflection.orthogonal_, reflection, atol=1e-8)
    np.testing.assert_allclose(with_reflection.score(X, y), 1.0, atol=1e-12)

    rotations_only = metrics.Orthogonal(allow_reflection=False).fit(X, y)
    np.testing.assert_allclose(np.linalg.det(rotations_only.orthogonal_), 1.0, atol=1e-8)
    assert rotations_only.score(X, y) < 1.0


def test_orthogonal_is_strictly_stronger_than_linear():
    """The justification for the class: a general GL(L) move is a real violation, and only Orthogonal sees it.

    `A` is pinned deliberately. "Invertible, non-orthogonal, well-conditioned" does not imply the
    threshold -- at cond(A) = 1.2 the score is 0.9943 and at 1.05 it is 0.9995, both of which would
    pass a laxer bound while proving nothing. The bound stays; the matrix is what gets fixed.
    """
    X = _anisotropic()
    A = np.diag([3.0, 1.0, 1.0, 1.0, 1.0])
    y = X @ A

    np.testing.assert_allclose(metrics.Linear().fit(X, y).score(X, y), 1.0, atol=1e-10)
    assert metrics.Orthogonal().fit(X, y).score(X, y) < 0.99


def test_orthogonal_separation_survives_out_of_sample():
    """Both classes fit in-sample, so the separation must be shown to be data and not the constraint."""
    X = _anisotropic()
    y = X @ np.diag([3.0, 1.0, 1.0, 1.0, 1.0])
    split = X.shape[0] // 2

    linear = metrics.Linear().fit(X[:split], y[:split]).score(X[split:], y[split:])
    orthogonal = metrics.Orthogonal().fit(X[:split], y[:split]).score(X[split:], y[split:])

    assert linear > 0.99
    assert orthogonal < 0.99


@pytest.mark.parametrize("allow_reflection", [True, False])
def test_orthogonal_sklearn_clone_round_trip(allow_reflection):
    """`sklearn.base.clone` reconstructs from get_params, so __init__ must store its argument verbatim."""
    import sklearn.base

    original = metrics.Orthogonal(allow_reflection=allow_reflection)
    assert sklearn.base.clone(original).get_params()["allow_reflection"] is allow_reflection
    assert original.set_params(allow_reflection=not allow_reflection).allow_reflection is (not allow_reflection)


def test_orthogonal_never_beats_linear():
    """A theorem, and therefore the cheapest structural check available.

    `Linear` is OLS with fit_intercept=False: it decouples across output columns and minimises each
    column's residual over all of R^L. Since O(L) is a subset of R^{L x L}, no orthogonal map can beat
    it on any single column, and multioutput="uniform_average" inherits the inequality. A violation
    means a transposed `X^T y`, a reversed orientation, or a stray centring step -- each of which
    otherwise yields a valid orthogonal matrix, a plausible R2, and no exception.
    """
    rng = np.random.default_rng(20260807)
    for trial in range(50):
        n_samples = int(rng.integers(6, 60))
        n_features = int(rng.integers(2, 6))
        X = rng.normal(size=(n_samples, n_features)) * rng.uniform(0.1, 5.0, size=n_features)
        if trial % 3 == 0:
            q, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
            y = X @ q
        else:
            y = rng.normal(size=(n_samples, n_features)) * rng.uniform(0.1, 5.0, size=n_features)

        assert metrics.Linear().fit(X, y).score(X, y) >= metrics.Orthogonal().fit(X, y).score(X, y)


def test_orthogonal_coef_follows_the_sklearn_orientation():
    """`coef_` is the transposed view, so that predict(X) == X @ coef_.T as everywhere else in sklearn."""
    X = _anisotropic()
    orthogonal = metrics.Orthogonal().fit(X, X @ _random_orthogonal())

    np.testing.assert_allclose(orthogonal.coef_, orthogonal.orthogonal_.T, atol=0.0)
    np.testing.assert_allclose(orthogonal.predict(X), X @ orthogonal.coef_.T, atol=1e-12)


def test_orthogonal_coef_raises_before_fitting():
    """An unfitted estimator must not present a `coef_`, which is what check_is_fitted keys on."""
    with pytest.raises(AttributeError, match="not fitted"):
        metrics.Orthogonal().coef_


def test_orthogonal_predict_raises_before_fitting():
    """Predicting before fitting is a caller error and must say so."""
    with pytest.raises(ValueError, match="must be fitted"):
        metrics.Orthogonal().predict(_anisotropic())


@pytest.mark.parametrize(
    "X,y,message",
    [
        (np.zeros((10, 3)), np.zeros((10, 4)), "equal source and target shapes"),
        (np.zeros((10, 3)), np.zeros((8, 3)), "equal source and target shapes"),
        (np.zeros(10), np.zeros((10, 3)), "2-dimensional"),
        (np.zeros((10, 3)), np.zeros(10), "2-dimensional"),
    ],
)
def test_orthogonal_boundary_errors_raise_value_error(X, y, message):
    """ValueError, never AssertionError: the suite is required to pass under `python -O`, which strips assert."""
    with pytest.raises(ValueError, match=message):
        metrics.Orthogonal().fit(X, y)


def test_orthogonal_drops_into_pairwise_consistency():
    """End-to-end through the transform, which is also where @typeguard.typechecked gets a say."""
    X = _partially_consistent_embeddings()
    transformer = metrics.PairwiseConsistency(indeterminacy=metrics.Orthogonal())
    transformer.fit(X)

    score = transformer.score(X)
    assert np.isfinite(score)
    np.testing.assert_array_equal(np.asarray(transformer.transform(X)).shape, X[0].shape)


def test_orthogonal_detects_a_gl_move_that_linear_cannot():
    """The same separation as the unit test, but through the transform a consumer actually calls."""
    reference = _partially_consistent_embeddings(n_estimators=1)[0]
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(reference.shape[1], reference.shape[1])))
    contaminated = [reference, reference @ q, reference @ np.diag([3.0, 1.0, 1.0, 1.0, 1.0, 1.0])]

    scores = {}
    for name, indeterminacy in (("linear", metrics.Linear()), ("orthogonal", metrics.Orthogonal())):
        transformer = metrics.PairwiseConsistency(indeterminacy=indeterminacy)
        transformer.fit(contaminated)
        scores[name] = transformer.score(contaminated)

    np.testing.assert_allclose(scores["linear"], 1.0, atol=1e-10)
    assert scores["orthogonal"] < 0.8


def test_r2_score_mixin_does_not_depend_on_a_transitive_import():
    """`sklearn.metrics` must be imported explicitly, not inherited from whatever else pulled it in.

    `R2ScoreMixin.score` calls `sklearn.metrics.r2_score`. Before this was imported by name it
    resolved only because `sklearn.linear_model` happens to import `sklearn.metrics` transitively --
    an upstream refactor away from an AttributeError on the scoring path.
    """
    import subprocess
    import sys

    probe = "import numpy as np; import ituna.metrics as m; print(m.Identity().fit(np.eye(3), np.eye(3)).score(np.eye(3), np.eye(3)))"
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "1.0"


def test_mismatched_embedding_shapes_raise_value_error():
    """Shape validation must survive `python -O`, which strips assert and would let this through."""
    X = _partially_consistent_embeddings(n_estimators=3, n_features=6)
    transformer = metrics.PairwiseConsistency(indeterminacy=metrics.Identity())
    transformer.fit(X)

    with pytest.raises(ValueError, match="same shape"):
        transformer._transform_all([X[0], X[1], X[2][:, :3]])


def test_align_embeddings_count_mismatch_raises_value_error():
    """Same reasoning as above for the embedding count, which was also guarded only by assert."""
    X = _partially_consistent_embeddings(n_estimators=3)
    transformer = metrics.PairwiseConsistency(indeterminacy=metrics.Identity())
    transformer.fit(X)

    with pytest.raises(ValueError, match="must match number of indeterminacy models"):
        transformer._align_embeddings(X[:2])
