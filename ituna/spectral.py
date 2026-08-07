"""Gauge-invariant spectral consistency for the operators of Koopman-structured latent models.

An encoder trained with an isotropy-enforcing objective is identifiable only up to an orthogonal
change of latent basis. That gauge acts on a learned transfer operator by similarity,
``K -> Q K Q^T``, so every quantity computed here is a similarity invariant and is therefore
comparable across independently trained runs *without* any alignment step. This answers a question
embedding alignment cannot: two runs that settled on different approximately-invariant subspaces can
still align well, because least squares happily fits a good linear map between overlapping but
distinct subspaces.

The module is deliberately standalone. It is not a :class:`ituna.metrics.ConsistencyTransform` and
cannot be passed as ``consistency_transform=``, because :class:`ituna.estimator.ConsistencyEnsemble`
feeds that slot the output of ``estimator.transform(X)`` -- embeddings, never operators. Pass plain
arrays instead, typically gathered off ``ensemble.estimators_``.

Limitations, none of which are papered over
-------------------------------------------

**Every similarity is invisible, chirality included.** Eigenvalues are invariants of the similarity
class of ``K``, so any ``K -> S K S^-1`` contributes exactly zero here. A reflected conjugate
``S R(theta) S = R(-theta)`` with ``S = diag(1, -1, 1, ...)`` is one instance: ``S`` is its own
inverse, so the map is a similarity and the eigenvalue multisets coincide. This is a statement about
characteristic polynomials and holds over the complex numbers as well; it is *not* a consequence of a
real spectrum being closed under conjugation. Recovering a residual handedness gauge needs eigenvector
coordinates, which are not gauge-invariant and are out of scope. This blindness is the same property
that makes the module work at all -- invariance to the ``O(L)`` gauge and blindness to chirality are
one fact, not two.

**Eigenvalues of non-normal operators are ill-conditioned.** By Bauer-Fike, for a diagonalisable
``K = V L V^-1`` every eigenvalue of ``K + E`` lies within ``kappa_2(V) ||E||_2`` of some eigenvalue
of ``K``. A large :attr:`SpectralConsistencyResult.eigvec_cond` therefore makes matched-eigenvalue
distances untrustworthy, and :attr:`~SpectralConsistencyResult.singular_values` and
:attr:`~SpectralConsistencyResult.departure` should be read instead. Runs above
``cond_warn_threshold`` raise a :class:`RuntimeWarning` naming themselves.

**Continuous-time conversion has a branch ambiguity.** ``log(lambda) / dt`` is single-valued only for
``|arg lambda| < pi``. Eigenvalues on or near the negative real axis sit at the Nyquist limit and
their imaginary part is not recoverable from a single sampling interval, so ``dt`` defaults to None
and supplying it warns when any eigenvalue comes close.

**This detects disagreement, not correctness.** Tightly clustered spectra are equally consistent with
all runs sharing one bias, including eigenvalues systematically polluted by compressing onto a
non-invariant subspace. Excluding that needs a residual test against the data, ResDMD-style, which is
out of scope here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import warnings

import numpy as np
import scipy.optimize


@dataclass(frozen=True)
class SpectralConsistencyResult:
    """Similarity-invariant summary of a set of transfer operators, as returned by :func:`spectral_consistency`.

    The dataclass is frozen and its arrays are marked non-writeable, so a result is safe to cache and
    to share between callers.

    Attributes
    ----------
    eigenvalues : ndarray of shape (H, L), complex
        Per-run spectrum as returned by :func:`numpy.linalg.eig`, in that routine's own order. No
        canonical ordering is imposed; pairwise comparison is by optimal matching instead.
    distance_matrix : ndarray of shape (H, H), float
        Mean matched eigenvalue distance for each unordered pair, symmetric with a zero diagonal.
        Divided by :attr:`scale` when ``normalise`` was requested.
    max_distance_matrix : ndarray of shape (H, H), float
        Worst matched eigenvalue distance for each unordered pair, symmetric, in the same units as
        :attr:`distance_matrix`, so ``distance_matrix <= max_distance_matrix`` holds elementwise.
    assignments : ndarray of shape (H, H, L), int
        Optimal-matching column indices: ``assignments[h, k, i]`` is the eigenvalue of run ``k``
        matched to eigenvalue ``i`` of run ``h``. The lower triangle holds the *inverse* permutation
        of the upper, not a copy of it, and the diagonal is the identity.
    scale : ndarray of shape (H, H), float
        Normalisation denominator actually applied per pair, or ``1.0`` where normalisation was
        skipped. ``distance_matrix * scale`` recovers raw distances in every branch.
    reference_id : int
        Medoid run, the argmin of the row means of :attr:`distance_matrix`.
    consistency : float
        ``1 - mean(offdiag(distance_matrix))``. May be negative, which is meaningful rather than a
        bug: the spectra are further apart than the typical eigenvalue magnitude. It is not clipped.
    eigvec_cond : ndarray of shape (H,), float
        ``kappa_2(V)`` of the eigenvector matrix of each operator, on LAPACK's unit-2-norm column
        scaling. Large values invalidate the matched distances -- see the module docstring.
    singular_values : ndarray of shape (H, L), float
        Singular values per run, descending.
    departure : ndarray of shape (H,), float
        Henrici departure from normality, ``sqrt(||K||_F^2 - sum |lambda|^2)``.
    spectral_radius : ndarray of shape (H,), float
        Largest eigenvalue modulus per run.
    normalised : bool
        Whether normalisation was requested. Individual pairs may still be unnormalised; consult
        :attr:`scale` for what was applied.
    continuous_eigenvalues : ndarray of shape (H, L), complex, or None
        ``log(lambda) / dt`` when ``dt`` was supplied, else None. A zero eigenvalue maps to ``-inf``.
    """

    eigenvalues: np.ndarray
    distance_matrix: np.ndarray
    max_distance_matrix: np.ndarray
    assignments: np.ndarray
    scale: np.ndarray
    reference_id: int
    consistency: float
    eigvec_cond: np.ndarray
    singular_values: np.ndarray
    departure: np.ndarray
    spectral_radius: np.ndarray
    normalised: bool
    continuous_eigenvalues: np.ndarray | None


def _validate_operators(operators: Sequence[np.ndarray]) -> np.ndarray:
    """Check the operator sequence at the boundary and stack it into one real (H, L, L) array."""
    candidates = list(operators)
    if len(candidates) < 2:
        raise ValueError(f"spectral consistency needs at least 2 operators to compare, got {len(candidates)}")

    arrays: list[np.ndarray] = []
    for index, operator in enumerate(candidates):
        array = np.asarray(operator)
        if np.iscomplexobj(array):
            raise ValueError(
                f"operator {index} has complex dtype {array.dtype}. Only real operators are accepted: every invariant computed here generalises to "
                "the complex case without difficulty, but iTuna's embeddings are real and so is any operator fitted on them, so complex input is "
                "outside the tested envelope and is more likely an already-diagonalised spectrum passed by mistake. Pass the operator itself."
            )
        if array.ndim != 2:
            raise ValueError(f"operator {index} must be 2-dimensional, got {array.ndim} dimension(s) with shape {array.shape}")
        if array.shape[0] != array.shape[1]:
            raise ValueError(f"operator {index} must be square, got shape {array.shape}")
        if array.shape[0] == 0:
            raise ValueError(f"operator {index} is empty; the latent dimension must be at least 1")
        if arrays and array.shape != arrays[0].shape:
            raise ValueError(f"all operators must share one shape, got {arrays[0].shape} for operator 0 and {array.shape} for operator {index}")
        array = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"operator {index} contains non-finite entries; an eigendecomposition of it is not meaningful")
        arrays.append(array)

    return np.stack(arrays)


def _validate_dt(dt: float | None) -> float | None:
    """Check that a supplied sampling interval is a strictly positive finite scalar."""
    if dt is None:
        return None
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"dt must be a strictly positive finite sampling interval, got {dt!r}")
    return value


def _match_pair(left: np.ndarray, right: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Optimally match two spectra in the complex plane, returning mean distance, worst distance and the assignment."""
    cost = np.abs(left[:, None] - right[None, :])
    rows, columns = scipy.optimize.linear_sum_assignment(cost)
    matched = cost[rows, columns]
    return float(matched.mean()), float(matched.max()), columns


def _henrici_departure(operator: np.ndarray, eigenvalues: np.ndarray) -> float:
    """Compute the Henrici departure from normality, clamped at zero against round-off."""
    residual = np.linalg.norm(operator, "fro") ** 2 - float((np.abs(eigenvalues) ** 2).sum())
    return float(np.sqrt(max(residual, 0.0)))


def _to_continuous(eigenvalues: np.ndarray, dt: float) -> np.ndarray:
    """Convert a discrete-time spectrum to continuous time, warning near the Nyquist branch cut."""
    near_cut = np.abs(np.angle(eigenvalues)) > 0.9 * np.pi
    if np.any(near_cut):
        runs = sorted({int(h) for h in np.nonzero(near_cut)[0]})
        warnings.warn(
            f"run(s) {runs} carry eigenvalues within 0.1*pi of the negative real axis, where log(lambda)/dt sits at the Nyquist limit and the "
            "imaginary part is not recoverable from a single sampling interval. Treat continuous_eigenvalues for those runs as unidentified.",
            RuntimeWarning,
            stacklevel=3,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        continuous = np.log(eigenvalues) / dt
    continuous[eigenvalues == 0] = complex(-np.inf, 0.0)
    return continuous


def _freeze(array: np.ndarray) -> np.ndarray:
    """Mark an array non-writeable so a frozen result is frozen all the way down."""
    array.flags.writeable = False
    return array


def spectral_consistency(
    operators: Sequence[np.ndarray],
    *,
    normalise: bool = True,
    dt: float | None = None,
    cond_warn_threshold: float = 1e6,
) -> SpectralConsistencyResult:
    """Compare a set of transfer operators through invariants of the latent gauge ``K -> Q K Q^T``.

    Parameters
    ----------
    operators : sequence of ndarray of shape (L, L)
        At least two real square operators of identical shape, one per run.
    normalise : bool, default=True
        Divide each pair's distances by the mean eigenvalue modulus of the two runs, making them
        dimensionless and comparable across latent dimensions and datasets. A pair whose scale is
        zero -- a nilpotent or zero operator -- is left unnormalised, with ``1.0`` recorded in
        :attr:`SpectralConsistencyResult.scale`.
    dt : float, optional
        Sampling interval. When given, the discrete spectrum is additionally reported in continuous
        time as ``log(lambda) / dt``. Defaults to None because that conversion is ambiguous near the
        Nyquist limit; see the module docstring.
    cond_warn_threshold : float, default=1e6
        Emit a :class:`RuntimeWarning` for any run whose eigenvector-matrix condition number exceeds
        this, since Bauer-Fike then makes the matched distances unreliable.

    Returns
    -------
    SpectralConsistencyResult
        Spectra, pairwise matched distances, the medoid run, a scalar consistency, and the
        per-run non-normality diagnostics that qualify all of the above.

    Raises
    ------
    ValueError
        If fewer than two operators are given, or any is non-square, ragged, empty, complex, or
        non-finite, or if ``dt`` is not a strictly positive finite scalar.
    """
    stacked = _validate_operators(operators)
    dt = _validate_dt(dt)
    n_runs, n_latent = stacked.shape[0], stacked.shape[1]

    eigenvalues = np.empty((n_runs, n_latent), dtype=np.complex128)
    eigvec_cond = np.empty(n_runs, dtype=np.float64)
    singular_values = np.empty((n_runs, n_latent), dtype=np.float64)
    departure = np.empty(n_runs, dtype=np.float64)

    for index, operator in enumerate(stacked):
        values, vectors = np.linalg.eig(operator)
        # LAPACK returns unit-2-norm eigenvector columns, which is the scaling kappa_2(V) assumes.
        eigenvalues[index] = values.astype(np.complex128, copy=False)
        eigvec_cond[index] = float(np.linalg.cond(vectors))
        singular_values[index] = np.linalg.svd(operator, compute_uv=False)
        departure[index] = _henrici_departure(operator, eigenvalues[index])

    ill_conditioned = sorted({int(h) for h in np.nonzero(eigvec_cond > cond_warn_threshold)[0]})
    if ill_conditioned:
        warnings.warn(
            f"run(s) {ill_conditioned} have eigenvector-matrix condition numbers above {cond_warn_threshold:g} "
            f"(max {eigvec_cond.max():.3e}). By Bauer-Fike their eigenvalues move by up to that factor times any "
            "perturbation, so distance_matrix and consistency are unreliable for them; read singular_values and departure instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    distance_matrix = np.zeros((n_runs, n_runs), dtype=np.float64)
    max_distance_matrix = np.zeros((n_runs, n_runs), dtype=np.float64)
    scale = np.ones((n_runs, n_runs), dtype=np.float64)
    assignments = np.tile(np.arange(n_latent, dtype=np.intp), (n_runs, n_runs, 1))
    modulus_mean = np.abs(eigenvalues).mean(axis=1)

    for left in range(n_runs):
        for right in range(left + 1, n_runs):
            mean_distance, worst_distance, columns = _match_pair(eigenvalues[left], eigenvalues[right])

            pair_scale = 1.0
            if normalise:
                candidate = 0.5 * (modulus_mean[left] + modulus_mean[right])
                # A zero scale means a nilpotent or zero operator: leave the pair in raw units, and
                # record 1.0, so distance_matrix * scale reconstructs raw distances in every branch.
                if candidate > 0.0:
                    pair_scale = float(candidate)

            distance_matrix[left, right] = distance_matrix[right, left] = mean_distance / pair_scale
            max_distance_matrix[left, right] = max_distance_matrix[right, left] = worst_distance / pair_scale
            scale[left, right] = scale[right, left] = pair_scale
            assignments[left, right] = columns
            # The reverse assignment is the inverse permutation, never a copy: mirroring `columns`
            # here would index run `left`'s eigenvalues by run `right`'s positions and raise nothing.
            assignments[right, left] = np.argsort(columns)

    off_diagonal = ~np.eye(n_runs, dtype=bool)
    consistency = float(1.0 - distance_matrix[off_diagonal].mean())
    reference_id = int(np.argmin(distance_matrix.sum(axis=1) / (n_runs - 1)))

    continuous_eigenvalues = None if dt is None else _freeze(_to_continuous(eigenvalues.copy(), dt))

    return SpectralConsistencyResult(
        eigenvalues=_freeze(eigenvalues),
        distance_matrix=_freeze(distance_matrix),
        max_distance_matrix=_freeze(max_distance_matrix),
        assignments=_freeze(assignments),
        scale=_freeze(scale),
        reference_id=reference_id,
        consistency=consistency,
        eigvec_cond=_freeze(eigvec_cond),
        singular_values=_freeze(singular_values),
        departure=_freeze(departure),
        spectral_radius=_freeze(np.abs(eigenvalues).max(axis=1)),
        normalised=normalise,
        continuous_eigenvalues=continuous_eigenvalues,
    )
