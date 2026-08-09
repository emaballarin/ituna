import warnings

import numpy as np
import pytest

from ituna import gauge
from ituna import metrics
from ituna import spectral

SEED = 20260809

# Upper triangular, distinct eigenvalues, emphatically not normal. Non-normality is load-bearing in
# every test below: for a normal operator many wrong conventions accidentally give the right answer.
NON_NORMAL = np.array([[0.9, 0.7, 0.1], [0.0, 0.4, 0.6], [0.0, 0.0, -0.5]])


def _rng():
    """Fresh generator, so test order cannot couple the fixtures."""
    return np.random.default_rng(SEED)


def _orthogonal(size, rng):
    """Draw a Haar-ish orthogonal matrix from the QR of a Gaussian."""
    matrix, _ = np.linalg.qr(rng.normal(size=(size, size)))
    return matrix


def _gauge_copy(operator, alignment):
    """The source-frame operator of a run whose latent is the reference latent seen through `alignment`.

    Inverse of what `pushforward` does, written independently of it so the two cannot agree by
    sharing a mistake.
    """
    return alignment @ operator @ np.linalg.inv(alignment)


def test_pushforward_recovers_the_reference_operator_from_known_gauge_copies():
    """The defining property: gauge copies of one operator all come back as that operator."""
    rng = _rng()
    alignments = [_orthogonal(3, rng) for _ in range(4)]
    copies = [_gauge_copy(NON_NORMAL, alignment) for alignment in alignments]

    for copy, alignment in zip(copies, alignments):
        np.testing.assert_allclose(gauge.pushforward(copy, alignment), NON_NORMAL, atol=1e-12)


def test_the_transposed_convention_is_caught_only_by_agreement():
    """🔴 The guard this module exists for, and the demonstration that the obvious test would not have worked.

    `Q^-1 K Q` and `Q K Q^-1` are both orthogonal similarities, so every invariant `ituna.spectral`
    computes is identical under either. Only agreement against a known original separates them.
    """
    rng = _rng()
    alignment = _orthogonal(3, rng)
    copy = _gauge_copy(NON_NORMAL, alignment)

    correct = gauge.pushforward(copy, alignment)
    transposed = alignment @ copy @ alignment.T  # the plausible mistake, valid-looking and wrong

    # Agreement separates them, and by a wide margin rather than at the tolerance.
    np.testing.assert_allclose(correct, NON_NORMAL, atol=1e-12)
    assert np.linalg.norm(transposed - NON_NORMAL) > 0.1

    # And this is why the naive property test is vacuous: both are similarity-equivalent to the
    # original, so the whole diagnostic surface of `ituna.spectral` reports them as indistinguishable.
    for candidate in (correct, transposed):
        result = spectral.spectral_consistency([NON_NORMAL, candidate])
        np.testing.assert_allclose(result.consistency, 1.0, atol=1e-9)
        np.testing.assert_allclose(result.singular_values[0], result.singular_values[1], atol=1e-10)
        np.testing.assert_allclose(result.departure[0], result.departure[1], atol=1e-9)


def test_pushforward_bridges_an_alignment_fitted_by_metrics_end_to_end():
    """The real pipeline: operators fitted from data in each run's own frame, aligned by `metrics.Orthogonal`."""
    rng = _rng()
    alignment = _orthogonal(3, rng)

    # Paired consecutive latents in the reference frame, and the same pair seen in another frame.
    reference = rng.normal(size=(256, 3))
    reference_next = reference @ NON_NORMAL
    source = reference @ alignment.T
    source_next = reference_next @ alignment.T

    # Each run fits its own operator from its own latents, knowing nothing about the other.
    source_operator, *_ = np.linalg.lstsq(source, source_next, rcond=None)
    fitted = metrics.Orthogonal().fit(X=source, y=reference)

    np.testing.assert_allclose(fitted.orthogonal_, alignment, atol=1e-10)
    np.testing.assert_allclose(gauge.pushforward(source_operator, fitted.orthogonal_), NON_NORMAL, atol=1e-10)


def test_pushforward_is_correct_for_a_non_orthogonal_alignment():
    """The solve is not decoration: under `metrics.Linear` the transpose would be wrong even in direction-correct form."""
    rng = _rng()
    alignment = _orthogonal(3, rng) @ np.diag([1.0, 2.5, 0.4])
    copy = _gauge_copy(NON_NORMAL, alignment)

    np.testing.assert_allclose(gauge.pushforward(copy, alignment), NON_NORMAL, atol=1e-10)
    # The orthogonal shortcut really does fail here, which is why it is not in the source.
    assert np.linalg.norm(alignment.T @ copy @ alignment - NON_NORMAL) > 0.1


def test_pushforward_of_the_identity_alignment_is_a_no_op():
    """A run already in the reference frame must not be touched at all."""
    np.testing.assert_allclose(gauge.pushforward(NON_NORMAL, np.eye(3)), NON_NORMAL, atol=0.0)


def test_a_permutation_alignment_relabels_rather_than_mixes():
    """`metrics.Permutation` produces a signed permutation, and pushing forward by it must be an exact relabelling."""
    rng = _rng()
    permutation = np.eye(3)[rng.permutation(3)] @ np.diag([1.0, -1.0, 1.0])
    copy = _gauge_copy(NON_NORMAL, permutation)

    pushed = gauge.pushforward(copy, permutation)
    np.testing.assert_allclose(pushed, NON_NORMAL, atol=1e-12)
    # Exactly relabelled: the multiset of entry magnitudes is preserved, which a general gauge breaks.
    np.testing.assert_allclose(np.sort(np.abs(copy).ravel()), np.sort(np.abs(NON_NORMAL).ravel()), atol=1e-12)


def test_a_singular_alignment_raises_rather_than_returning_something_finite():
    """A rank-deficient alignment defines no change of frame; it is a finding about the runs, not a numerical nuisance."""
    singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="singular"):
        gauge.pushforward(NON_NORMAL, singular)


def test_an_ill_conditioned_alignment_warns_but_still_returns():
    """Between invertible and singular there is a band where the answer is finite, plausible and unresolved."""
    alignment = np.diag([1.0, 1.0, 1e-10])
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        pushed = gauge.pushforward(NON_NORMAL, alignment)
    assert np.all(np.isfinite(pushed))


def test_a_well_conditioned_alignment_is_silent():
    """The warning must carry information, so it may not fire on the ordinary case."""
    rng = _rng()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gauge.pushforward(NON_NORMAL, _orthogonal(3, rng))


@pytest.mark.parametrize(
    "operator,alignment,message",
    [
        (np.zeros((3, 4)), np.eye(3), "square"),
        (np.eye(3), np.zeros((3, 4)), "square"),
        (np.zeros(3), np.eye(3), "2-dimensional"),
        (np.eye(3), np.zeros((2, 2, 2)), "2-dimensional"),
        (np.eye(3), np.eye(4), "share one shape"),
        (np.full((3, 3), np.nan), np.eye(3), "non-finite"),
        (np.eye(3), np.full((3, 3), np.inf), "non-finite"),
        (np.eye(3, dtype=np.complex128), np.eye(3), "complex dtype"),
        (np.zeros((0, 0)), np.zeros((0, 0)), "empty"),
    ],
)
def test_boundary_errors_raise_value_error(operator, alignment, message):
    """Every boundary violation raises ValueError, which survives `python -O` where assert does not."""
    with pytest.raises(ValueError, match=message):
        gauge.pushforward(operator, alignment)
