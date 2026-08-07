import warnings

import numpy as np
import pytest

from ituna import spectral

SEED = 20260807


def _rng():
    """Fresh generator, so test order cannot couple the fixtures."""
    return np.random.default_rng(SEED)


def _orthogonal(size, rng):
    """Draw a Haar-ish orthogonal matrix from the QR of a Gaussian."""
    matrix, _ = np.linalg.qr(rng.normal(size=(size, size)))
    return matrix


def _rotation(theta):
    """Planar rotation by theta."""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _well_separated(size, rng):
    """An operator whose eigenvalues are far apart, so optimal matching is unambiguous."""
    diagonal = np.diag(np.linspace(-0.9, 1.2, size))
    basis = _orthogonal(size, rng)
    return basis @ diagonal @ basis.T


def test_gauge_invariance():
    """K and Q K Q^T are the same operator up to the latent gauge, so they must not be told apart."""
    rng = _rng()
    operator = rng.normal(size=(6, 6)) / np.sqrt(6)
    basis = _orthogonal(6, rng)
    result = spectral.spectral_consistency([operator, basis @ operator @ basis.T])
    assert result.distance_matrix[0, 1] < 1e-8
    np.testing.assert_allclose(result.consistency, 1.0, atol=1e-8)


def test_ordering_invariance():
    """A permutation conjugate only relabels latent coordinates."""
    rng = _rng()
    operator = rng.normal(size=(6, 6)) / np.sqrt(6)
    permutation = np.eye(6)[rng.permutation(6)]
    result = spectral.spectral_consistency([operator, permutation @ operator @ permutation.T])
    assert result.distance_matrix[0, 1] < 1e-10


def test_chirality_is_invisible():
    """A reflected conjugate is a similarity, so it contributes exactly zero. Regression test for a documented limitation."""
    block = np.zeros((4, 4))
    block[:2, :2] = _rotation(0.7)
    block[2:, 2:] = np.diag([0.5, 0.9])
    reflection = np.diag([1.0, -1.0, 1.0, 1.0])
    result = spectral.spectral_consistency([block, reflection @ block @ reflection])
    assert result.distance_matrix[0, 1] < 1e-10


def test_sensitivity_to_a_uniform_shift():
    """K against K + eps I moves every eigenvalue by exactly eps, and raw distances must report exactly that."""
    rng = _rng()
    epsilon = 1e-3
    operator = _well_separated(6, rng)
    shifted = operator + epsilon * np.eye(6)

    raw = spectral.spectral_consistency([operator, shifted], normalise=False)
    np.testing.assert_allclose(raw.distance_matrix[0, 1], epsilon, rtol=1e-6)
    np.testing.assert_allclose(raw.max_distance_matrix[0, 1], epsilon, rtol=1e-6)
    np.testing.assert_array_equal(raw.scale, np.ones((2, 2)))

    # The default normalises, so the reported number is smaller by the pair scale; multiplying it
    # back must land on the raw value, which is the contract the `scale` field exists to provide.
    normalised = spectral.spectral_consistency([operator, shifted])
    assert normalised.scale[0, 1] > 1.0 or normalised.scale[0, 1] < 1.0
    np.testing.assert_allclose(
        normalised.distance_matrix[0, 1] * normalised.scale[0, 1],
        raw.distance_matrix[0, 1],
        rtol=1e-12,
    )


def test_complex_spectra_do_not_trip_the_dtype():
    """A purely rotational operator has no real eigenvalue; pairing it with a real-spectrum run must still work."""
    rotational = np.zeros((4, 4))
    rotational[:2, :2] = _rotation(0.4)
    rotational[2:, 2:] = _rotation(1.1)
    real_spectrum = np.diag([0.3, 0.6, -0.2, 0.9])

    both = spectral.spectral_consistency([rotational, real_spectrum])
    assert np.all(np.isfinite(both.distance_matrix))

    pair = spectral.spectral_consistency([rotational, rotational.copy()])
    assert pair.distance_matrix[0, 1] < 1e-12


def test_conditioning_warning_fires_on_a_non_normal_operator():
    """Bauer-Fike: a near-defective operator must be flagged, and the flag must be readable off the result."""
    non_normal = np.array([[1.0, 1e8], [0.0, 1 + 1e-9]])
    benign = np.diag([1.0, 0.5])

    with pytest.warns(RuntimeWarning, match="Bauer-Fike"):
        result = spectral.spectral_consistency([non_normal, benign])

    # Assert on the conditioning, not on the eigenvalues: they collapse numerically to exactly [1, 1]
    # for this operator, so an eigenvalue-based assertion would be testing LAPACK's round-off.
    assert result.eigvec_cond[0] > 1e6
    assert result.eigvec_cond[1] < 1e6


def test_conditioning_warning_is_silent_for_normal_operators():
    """A well-conditioned set must not warn, or the warning stops carrying information."""
    rng = _rng()
    operators = [_well_separated(5, rng) for _ in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spectral.spectral_consistency(operators)


def test_reference_is_the_medoid_not_the_outlier():
    """The medoid must come from the tight cluster, whatever the outlier's index."""
    rng = _rng()
    base = _well_separated(5, rng)
    operators = [base + 1e-6 * rng.normal(size=(5, 5)) for _ in range(3)]
    operators.append(base * 3.0 + np.eye(5))
    result = spectral.spectral_consistency(operators)
    assert result.reference_id != 3
    assert result.reference_id in (0, 1, 2)


@pytest.mark.parametrize(
    "bad_operators,message",
    [
        ([np.eye(3)], "at least 2"),
        ([np.eye(3), np.eye(4)], "share one shape"),
        ([np.eye(3), np.zeros((3, 4))], "square"),
        ([np.eye(3), np.zeros(3)], "2-dimensional"),
        ([np.eye(3), np.zeros((2, 2, 2))], "2-dimensional"),
        ([np.eye(3), np.eye(3, dtype=np.complex128)], "complex dtype"),
        ([np.eye(3), np.full((3, 3), np.nan)], "non-finite"),
        ([np.eye(3), np.full((3, 3), np.inf)], "non-finite"),
        ([np.zeros((0, 0)), np.zeros((0, 0))], "empty"),
    ],
)
def test_boundary_errors_raise_value_error(bad_operators, message):
    """Every boundary violation must raise ValueError, which survives `python -O` where assert does not."""
    with pytest.raises(ValueError, match=message):
        spectral.spectral_consistency(bad_operators)


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, np.inf, np.nan])
def test_invalid_dt_raises_value_error(bad_dt):
    """A sampling interval that is not strictly positive and finite is a caller error."""
    with pytest.raises(ValueError, match="strictly positive"):
        spectral.spectral_consistency([np.eye(3), np.eye(3)], dt=bad_dt)


def test_invariants_are_invariant_under_the_gauge():
    """singular_values, departure, spectral_radius and eigvec_cond must not move under K -> Q K Q^T."""
    rng = _rng()
    operator = rng.normal(size=(6, 6)) / np.sqrt(6)
    basis = _orthogonal(6, rng)
    result = spectral.spectral_consistency([operator, basis @ operator @ basis.T])

    np.testing.assert_allclose(result.singular_values[0], result.singular_values[1], atol=1e-10)
    np.testing.assert_allclose(result.departure[0], result.departure[1], atol=1e-10)
    np.testing.assert_allclose(result.spectral_radius[0], result.spectral_radius[1], atol=1e-10)
    np.testing.assert_allclose(result.eigvec_cond[0], result.eigvec_cond[1], rtol=1e-6)


def test_departure_is_zero_for_a_normal_operator_and_positive_otherwise():
    """Henrici departure must separate normal from non-normal, or it is not measuring non-normality."""
    rng = _rng()
    symmetric = _well_separated(5, rng)
    shear = np.eye(5)
    shear[0, 1] = 2.0
    with warnings.catch_warnings():
        # The shear is defective by construction, so the conditioning warning firing is the point.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = spectral.spectral_consistency([symmetric, shear])

    # Departure is a square root of a difference of O(1) quantities, so its resolution floor is
    # sqrt(eps) * ||K||_F, near 1e-8 -- not eps. A tighter tolerance would be testing round-off.
    np.testing.assert_allclose(result.departure[0], 0.0, atol=1e-6)
    assert result.departure[1] > 1.0


def test_assignments_mirror_as_the_inverse_permutation():
    """The lower triangle must invert the upper, not copy it -- a naive mirror raises nothing but is wrong."""
    rng = _rng()
    operator = rng.normal(size=(6, 6)) / np.sqrt(6)
    basis = _orthogonal(6, rng)
    result = spectral.spectral_consistency([operator, basis @ operator @ basis.T])

    np.testing.assert_array_equal(result.assignments[1, 0], np.argsort(result.assignments[0, 1]))
    np.testing.assert_array_equal(result.assignments[0, 0], np.arange(6))
    np.testing.assert_array_equal(result.assignments[1, 1], np.arange(6))


def test_degenerate_spectrum_still_matches_and_reports_its_worst_case():
    """Repeated eigenvalues make the assignment non-unique; the max distance is what keeps that visible."""
    rng = _rng()
    degenerate = np.diag([1.0, 1.0, 0.5, 0.5, 0.2, 0.2])
    basis = _orthogonal(6, rng)
    result = spectral.spectral_consistency([degenerate, basis @ degenerate @ basis.T], normalise=False)

    assert result.distance_matrix[0, 1] < 1e-10
    assert result.max_distance_matrix[0, 1] < 1e-10
    assert result.max_distance_matrix[0, 1] >= result.distance_matrix[0, 1]


def test_consistency_may_go_negative_and_is_not_clipped():
    """Spectra further apart than the typical eigenvalue modulus are a real finding, not something to clamp."""
    far_apart = [np.diag([0.01, 0.01, 0.01]), np.diag([5.0, -5.0, 5.0])]
    result = spectral.spectral_consistency(far_apart)
    assert result.consistency < 0.0


def test_zero_scale_falls_back_to_raw_distances():
    """A nilpotent pair has zero mean modulus; normalisation must be skipped and recorded, not divide by zero."""
    nilpotent = np.zeros((3, 3))
    nilpotent[0, 1] = 1.0
    other = np.zeros((3, 3))
    other[1, 2] = 1.0
    with warnings.catch_warnings():
        # A nilpotent operator is maximally defective, so the conditioning warning is expected here.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = spectral.spectral_consistency([nilpotent, other])
    np.testing.assert_array_equal(result.scale, np.ones((2, 2)))
    assert np.all(np.isfinite(result.distance_matrix))


def test_continuous_time_conversion_is_opt_in_and_correct():
    """log(lambda)/dt must invert exp(lambda dt) away from the branch cut, and stay absent by default."""
    dt = 0.1
    continuous_truth = np.array([-1.0, -0.5, -2.0])
    discrete = np.diag(np.exp(continuous_truth * dt))

    without = spectral.spectral_consistency([discrete, discrete.copy()])
    assert without.continuous_eigenvalues is None

    with_dt = spectral.spectral_consistency([discrete, discrete.copy()], dt=dt)
    recovered = np.sort(with_dt.continuous_eigenvalues[0].real)
    np.testing.assert_allclose(recovered, np.sort(continuous_truth), atol=1e-10)


def test_continuous_time_warns_near_the_nyquist_branch_cut():
    """An eigenvalue on the negative real axis has an unidentifiable imaginary part; that must be said out loud."""
    nyquist = np.diag([-0.8, 0.5, 0.3])
    with pytest.warns(RuntimeWarning, match="Nyquist"):
        spectral.spectral_consistency([nyquist, nyquist.copy()], dt=0.1)


def test_zero_eigenvalue_maps_to_minus_infinity():
    """A zero eigenvalue has no logarithm; -inf is the honest answer and must not come with a numpy warning."""
    singular = np.diag([0.0, 0.5, 0.3])
    result = spectral.spectral_consistency([singular, singular.copy()], dt=0.1)
    assert np.isneginf(result.continuous_eigenvalues[0].real).sum() == 1


def test_result_arrays_are_read_only():
    """A frozen result whose arrays are mutable is only half frozen, and it is cached and shared."""
    rng = _rng()
    operators = [_well_separated(4, rng) for _ in range(3)]
    result = spectral.spectral_consistency(operators)
    with pytest.raises(ValueError):
        result.distance_matrix[0, 1] = 0.0


def test_distance_matrix_is_symmetric_with_a_zero_diagonal():
    """Both matrices are defined on unordered pairs, so symmetry is part of the contract."""
    rng = _rng()
    operators = [_well_separated(4, rng) + 0.1 * rng.normal(size=(4, 4)) for _ in range(4)]
    result = spectral.spectral_consistency(operators)

    np.testing.assert_allclose(result.distance_matrix, result.distance_matrix.T, atol=0.0)
    np.testing.assert_allclose(result.max_distance_matrix, result.max_distance_matrix.T, atol=0.0)
    np.testing.assert_array_equal(np.diag(result.distance_matrix), np.zeros(4))
    assert np.all(result.max_distance_matrix >= result.distance_matrix)
