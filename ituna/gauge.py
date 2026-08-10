"""Moving a learned operator between the latent frames of independently trained runs.

:mod:`ituna.spectral` compares operators by computing only quantities the gauge cannot move, which
needs no alignment and no convention. This module does the complementary and more dangerous thing:
it *fixes* the gauge, carrying each run's operator into one common frame so that the operators
themselves -- not merely their invariants -- can be compared, averaged, or read as modes.

The gauge that acts on the operator is the same map that aligns the embeddings, so it is already
fitted by :class:`ituna.metrics.PairwiseConsistency`; nothing here re-estimates it.

The convention, which is the whole risk
---------------------------------------

Everything in iTuna is row-major: an embedding is ``(n_samples, n_features)`` and every indeterminacy
class predicts ``X @ alignment_``. This module follows that, so an operator is the matrix satisfying
``Z[t + 1] = Z[t] @ operator``, and an alignment is the matrix satisfying
``Z_reference = Z_source @ alignment``. Substituting the second into the first gives the only formula
here::

    operator_in_reference_frame = inv(alignment) @ operator @ alignment

🔴 **A transposed or inverted convention does not fail loudly, and the obvious test does not catch
it.** For an orthogonal alignment, ``Q^-1 K Q`` and ``Q K Q^-1`` are *both* orthogonal similarities
of ``K``: they have the same eigenvalues, the same singular values, the same departure from
normality and the same eigenvector conditioning. So a property test asserting "the invariants are
unchanged by the pushforward" passes on the bug, and every diagnostic in :mod:`ituna.spectral` is
blind to it -- which is exactly the property that makes that module work. The test with teeth is
*agreement*: push a set of known gauge copies of one non-normal operator into the common frame and
require that they come back equal to the original, which the wrong direction fails unless the gauge
happens to commute with the operator. That is
``tests/test_gauge.py::test_the_transposed_convention_is_caught_only_by_agreement``.

For the same reason this computes an explicit inverse solve rather than the transpose that would be
valid for the orthogonal case. The two agree whenever the alignment really is orthogonal, the solve
is additionally correct for :class:`ituna.metrics.Linear`, and writing ``alignment.T`` here would put
the trap above into the source where a reader would have to re-derive it to check.

Scope, and what is deliberately not here yet
--------------------------------------------

Only :func:`pushforward` is implemented. Extracting the per-run alignment matrices off a fitted
:class:`~ituna.metrics.PairwiseConsistency`, and averaging the pushed-forward operators into a
consensus operator, are not built. What that extractor needs from each fitted indeterminacy is now
settled, though: it reads ``alignment_``, which every class exposes and which is the map the class
actually applies -- so the remaining work is walking the fitted estimators, not deciding what to take
from them.

🔑 **Why this module, and not another gauge name in** :func:`ituna.spectral.spectral_consensus`.
That function raises :class:`NotImplementedError` for ``"signed_permutation"`` and
``"scaled_signed_permutation"``. That is a *not yet* and never a *cannot*, and the reason the work
has not simply been done there is worth stating, because the obvious reading -- that someone must
get round to writing more invariants -- points at the wrong file.

The invariants of the signed-permutation gauge are trivial to compute: the multiset of ``|K_ij|``,
the signed diagonal, the row and column norms. **Averaging them is the hard part.** It needs a
canonical coordinate labelling shared across runs, and recovering one from the operator alone is a
*quadratic* assignment problem. Eigenvalue matching is *linear* assignment, which is exactly why
:func:`~ituna.spectral.spectral_consensus` was cheap to build; the same trick does not transfer.

But the labelling already exists somewhere else. :class:`ituna.metrics.Permutation` recovers it from
the *embeddings*, where it is again a linear assignment, and it is implemented and tested already.
So the route is this module: fit the alignment on embeddings, :func:`pushforward` every operator into
the common frame, and average them entrywise. That uses every one of the extra invariants implicitly
-- entrywise agreement in a shared frame is strictly stronger than agreement of their multisets --
and costs nothing beyond what is already here. The scaled class needs a scale-bearing indeterminacy
class to fit its alignment with, and :class:`ituna.metrics.ScaledPermutation` now supplies one -- so
the two gauges are in the same position, both waiting on the operator consensus below and on nothing
peculiar to either.

Two constraints the unbuilt half will have to respect, recorded now because they are easy to miss
later:

- :class:`ituna.metrics.Affine` cannot be used. Its intercept does not act on an operator by
  conjugation at all, so there is no pushforward to compute, only a silent wrong answer. ✅ This one
  is no longer only a note: `Affine` is the single indeterminacy class that raises on ``alignment_``
  rather than returning one, so the mistake is unavailable rather than merely documented.
- :class:`ituna.metrics.Orthogonal` must be fitted with ``allow_reflection=True``. Two runs whose
  latents genuinely differ by a reflection are related by an element of ``O(L)`` that ``SO(L)``
  cannot represent; the alignment would then absorb the mismatch into a poor fit, and averaging
  operators across an unresolved discrete gauge corrupts them. Invariant comparison does not care --
  see :mod:`ituna.spectral` on chirality -- which is precisely why this is easy to overlook here.
"""

import warnings

import numpy as np


def _validate_square(matrix: np.ndarray, name: str) -> np.ndarray:
    """Check one matrix at the boundary and return it as float64."""
    array = np.asarray(matrix)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} has complex dtype {array.dtype}; only real matrices are accepted here, as elsewhere in iTuna")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got {array.ndim} dimension(s) with shape {array.shape}")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got shape {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} is empty; the latent dimension must be at least 1")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite entries")
    return array


def pushforward(
    operator: np.ndarray,
    alignment: np.ndarray,
    *,
    cond_warn_threshold: float = 1e8,
) -> np.ndarray:
    """Carry an operator from one run's latent frame into the frame it is aligned to.

    Both arguments are in iTuna's row-major convention: ``operator`` satisfies
    ``Z[t + 1] = Z[t] @ operator`` in the source run's own frame, and ``alignment`` satisfies
    ``Z_reference = Z_source @ alignment``, which is exactly the ``alignment_`` of a fitted
    indeterminacy class. The result satisfies the first relation in the reference frame. See the
    module docstring before changing either convention.

    🔴 **Pass ``alignment_`` and not a class's own matrix.** They are not always the same thing:
    :class:`ituna.metrics.Permutation` keeps its sign flips in a separate ``signs_``, so its
    ``permutation_matrix_`` is the gauge element with every flip discarded, and
    :class:`ituna.metrics.Linear` stores scikit-learn's *transposed* ``coef_``. Either substitution
    returns a finite, plausible operator that no invariant in :mod:`ituna.spectral` can distinguish
    from the right one -- see :attr:`ituna.metrics.Permutation.alignment_` for why -- so ``alignment_``
    is the only safe source.

    Parameters
    ----------
    operator : ndarray of shape (L, L)
        The source run's operator, in that run's own latent frame.
    alignment : ndarray of shape (L, L)
        The invertible map taking the source frame to the reference frame.
    cond_warn_threshold : float, default=1e8
        Emit a :class:`RuntimeWarning` when the alignment's condition number exceeds this. A
        near-singular alignment returns a finite, plausible and meaningless operator.

    Returns
    -------
    ndarray of shape (L, L)
        ``alignment^-1 @ operator @ alignment``.

    Raises
    ------
    ValueError
        If either argument is not a real, finite, non-empty square matrix, if their shapes differ,
        or if ``alignment`` is singular.

    Warns
    -----
    RuntimeWarning
        If ``alignment`` is invertible but ill-conditioned to ``cond_warn_threshold``.
    """
    operator = _validate_square(operator, "operator")
    alignment = _validate_square(alignment, "alignment")
    if operator.shape != alignment.shape:
        raise ValueError(f"operator and alignment must share one shape, got {operator.shape} and {alignment.shape}")

    condition = float(np.linalg.cond(alignment))
    if not np.isfinite(condition) or condition > 1.0 / np.finfo(np.float64).eps:
        raise ValueError(
            f"alignment is singular (condition number {condition:.3e}) and defines no change of frame. An alignment fitted by "
            "`ituna.metrics.Linear` can be rank-deficient when the two runs do not span the same subspace, which is a finding about the runs "
            "rather than something to regularise away here."
        )
    if condition > cond_warn_threshold:
        warnings.warn(
            f"alignment is ill-conditioned (condition number {condition:.3e} against a threshold of {cond_warn_threshold:g}). The pushed-forward "
            "operator is finite and plausible but carries that amplification, so treat it and anything averaged from it as unresolved.",
            RuntimeWarning,
            stacklevel=2,
        )

    # An explicit solve, never `alignment.T`: correct for a non-orthogonal alignment too, and it
    # keeps the direction of the conjugation readable rather than encoded in a transpose.
    return np.linalg.solve(alignment, operator @ alignment)
