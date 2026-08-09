"""Gauge-invariant spectral consistency for the operators of Koopman-structured latent models.

An encoder trained with an isotropy-enforcing objective is identifiable only up to an orthogonal
change of latent basis. That gauge acts on a learned transfer operator by similarity,
``K -> Q K Q^T``, and every quantity computed here is invariant under it, so operators are
comparable across independently trained runs *without* any alignment step. This answers a question
embedding alignment cannot: two runs that settled on different approximately-invariant subspaces can
still align well, because least squares happily fits a good linear map between overlapping but
distinct subspaces.

Which gauge, and therefore which quantities
-------------------------------------------

The gauge is not a property of this module -- it is whatever the *anticollapse* half of the training
objective leaves standing, and it decides which of the numbers below mean anything across runs.

- **A decoder or autoencoder reconstruction, or a JEPA with a stop-gradient or EMA target.** Nothing
  constrains the latent metric, so the gauge is the whole of ``GL(L)`` and the spectrum is the only
  thing that survives it. Pass ``gauge="general_linear"`` to :func:`spectral_consensus`.
- **VICReg, or SIGReg against an isotropic Gaussian.** Both drive ``Cov(z) -> I``, and a Gaussian
  target has zero excess kurtosis, so it cannot pin a rotation: the gauge is ``O(L)`` and everything
  computed here survives it. This is the case the module is written for and its default.
- **SIGReg against a *product* non-Gaussian target.** The coordinate axes themselves are pinned and
  the gauge drops to the signed permutations, a finite subgroup of ``O(L)``. Everything here
  survives, and so do invariants no function of the spectrum can see. Not computed here.
- **The same without whitening: the classical ICA class.** The gauge is the scaled signed
  permutations. Not computed here, and ⚠️ **not a case where a smaller gauge means more
  invariants** -- a per-coordinate scaling is not orthogonal, so this group is *not* inside ``O(L)``
  and the norm-derived quantities below stop being invariant exactly as they do under ``GL(L)``.

Per quantity, since the split is what matters whenever the objective is not the second bullet:

- Invariant under **any** similarity, hence under every bullet above: :attr:`eigenvalues`,
  :attr:`spectral_radius`, :attr:`continuous_eigenvalues`, and therefore :attr:`distance_matrix`,
  :attr:`max_distance_matrix`, :attr:`assignments`, :attr:`scale`, :attr:`reference_id` and
  :attr:`consistency`, all of which are computed from eigenvalues alone. 🔑 **The comparison this
  module exists to make is therefore valid under ``GL(L)`` too** -- that is why
  :func:`spectral_consistency` takes no gauge argument.
- Invariant under **orthogonal** similarity only: :attr:`singular_values`, :attr:`departure` and
  :attr:`eigvec_cond`. Each is built from ``||K||_F`` or from the eigenvector matrix, neither of
  which a general change of basis preserves. Under a decoder or a plain JEPA objective these are
  properties of the particular representative that was handed over, not of the run, and averaging
  them across runs is meaningless. :func:`spectral_consensus` withholds them when told the gauge is
  ``GL(L)``, rather than trusting the caller to remember.

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
    """Gauge-invariant summary of a set of transfer operators, as returned by :func:`spectral_consistency`.

    The dataclass is frozen and its arrays are marked non-writeable, so a result is safe to cache and
    to share between callers.

    Most fields here are invariant under *any* similarity and so hold whatever pinned the latent;
    :attr:`singular_values`, :attr:`departure` and :attr:`eigvec_cond` need the gauge to be
    orthogonal. The module docstring has the split and the objectives that produce each case.

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
        **Orthogonal gauge only**: a general change of basis sends ``V -> G V`` and moves this.
    singular_values : ndarray of shape (H, L), float
        Singular values per run, descending. **Orthogonal gauge only**: these are invariants of
        ``K -> Q K Q^T`` and not of ``K -> G^-1 K G``.
    departure : ndarray of shape (H,), float
        Henrici departure from normality, ``sqrt(||K||_F^2 - sum |lambda|^2)``. **Orthogonal gauge
        only**, through ``||K||_F``; the subtracted term is a similarity invariant on its own.
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


@dataclass(frozen=True)
class SpectralConsensusResult:
    """Across-run average of the invariants in a :class:`SpectralConsistencyResult`, from :func:`spectral_consensus`.

    Where :class:`SpectralConsistencyResult` reports how far apart independently trained runs are,
    this reports the single estimate they jointly support and how tightly they support it. The
    dataclass is frozen and its arrays are marked non-writeable.

    ⚠️ **The dispersion fields bound the reducible half of the uncertainty and no more.** Averaging
    ``H`` runs shrinks what varies between them by ``sqrt(H)``; it does nothing to a bias they share.
    Every run compressing onto the same non-invariant subspace leaves :attr:`eigenvalues_scatter`
    small while the consensus is confidently wrong. This is the module docstring's "detects
    disagreement, not correctness" inherited one level up: a small scatter is evidence that the gauge
    was correctly quotiented out, never evidence that the operator is right.

    Attributes
    ----------
    gauge : str
        The gauge group the caller declared, which fixed what could legitimately be averaged.
    n_runs : int
        Number of runs averaged, ``H``.
    reference_id : int
        Medoid run inherited from the source result, whose eigenvalue order defines the slots.
    assignment : ndarray of shape (H, L), int
        The global labelling used, ``source.assignments[reference_id]``. Slot ``i`` of run ``h`` is
        eigenvalue ``assignment[h, i]`` of that run.
    matched_eigenvalues : ndarray of shape (H, L), complex
        Per-run spectra re-indexed into slots, so column ``i`` is one eigenvalue tracked across runs.
        Exposed so a caller can compute statistics this class does not.
    eigenvalues_mean : ndarray of shape (L,), complex
        Consensus spectrum, the per-slot mean.
    eigenvalues_scatter : ndarray of shape (L,), float
        Per-slot sample standard deviation, taken as a distance in the complex plane rather than
        componentwise: ``sqrt(sum_h |lambda_h - mean|^2 / (H - 1))``.
    eigenvalues_sem : ndarray of shape (L,), float
        ``eigenvalues_scatter / sqrt(H)``. Read the warning above before quoting it as an error bar.
    spectral_radius_mean, spectral_radius_sem : float
        Mean and standard error of the per-run spectral radius. Not the radius of
        :attr:`eigenvalues_mean`, which is a different and generally smaller number.
    conjugate_closure_residual : float
        Worst distance between :attr:`eigenvalues_mean` and its own conjugate under optimal matching,
        divided by :attr:`conjugate_closure_scale`. Zero exactly when the consensus multiset is
        closed under conjugation, hence realisable as the spectrum of a real operator. 🔑 **Reported,
        never enforced.** A non-zero value means the runs disagree about *structure* -- typically a
        complex pair in one run against two real eigenvalues in another -- and symmetrising it away
        would hide the one disagreement that cannot be averaged out.
    conjugate_closure_scale : float
        Denominator applied above, the mean modulus of the consensus spectrum, or ``1.0`` for a
        nilpotent consensus. Multiplying recovers the raw distance, as with
        :attr:`SpectralConsistencyResult.scale`.
    eigvec_cond_max : float
        Worst per-run eigenvector conditioning in the source result. A qualifier on everything above,
        by Bauer-Fike, and a property of the particular representatives rather than of the gauge
        class when that class is larger than ``O(L)``.
    continuous_eigenvalues_mean : ndarray of shape (L,), complex, or None
        Per-slot mean of ``log(lambda) / dt`` when the source result carried it, else None. This is
        the mean of the logarithms, not the logarithm of :attr:`eigenvalues_mean`; the former is the
        natural average of rates and the two differ. A zero eigenvalue in any run makes its slot
        ``-inf``, loudly rather than silently.
    singular_values_mean, singular_values_sem : ndarray of shape (L,), float, or None
        Mean and standard error of the descending singular values, which need no matching because
        the ordering is already canonical. **None under a gauge larger than ``O(L)``.**
    departure_mean, departure_sem : float, or None
        Mean and standard error of the Henrici departure from normality. **None under a gauge larger
        than ``O(L)``.**
    """

    gauge: str
    n_runs: int
    reference_id: int
    assignment: np.ndarray
    matched_eigenvalues: np.ndarray
    eigenvalues_mean: np.ndarray
    eigenvalues_scatter: np.ndarray
    eigenvalues_sem: np.ndarray
    spectral_radius_mean: float
    spectral_radius_sem: float
    conjugate_closure_residual: float
    conjugate_closure_scale: float
    eigvec_cond_max: float
    continuous_eigenvalues_mean: np.ndarray | None
    singular_values_mean: np.ndarray | None
    singular_values_sem: np.ndarray | None
    departure_mean: float | None
    departure_sem: float | None


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


_GAUGE_ORTHOGONAL = "orthogonal"
_GAUGE_GENERAL_LINEAR = "general_linear"

# Gauges that are recognised, that a latent objective in this family really does produce, and whose
# invariants are not computed here. Naming them separately keeps a caller who has one of these
# objectives from reading "unknown gauge" as a typo and silently falling back to `orthogonal`.
#
# 🔑 Both of these are a "not yet" and never a "cannot", and neither will be served by adding a gauge
# here -- see `ituna.gauge` for why the route runs through there instead.
_UNIMPLEMENTED_GAUGES = {
    "signed_permutation": (
        "a product non-Gaussian shape objective on a whitened latent pins the coordinate axes themselves, leaving only signed permutations. Every "
        "invariant computed here does survive it, so `orthogonal` is sound but wasteful: the smaller gauge additionally admits invariants no "
        "function of the spectrum can see -- the multiset of |K_ij|, the signed diagonal, the row and column norms -- and averaging only the "
        "spectral ones discards most of what it buys. Use `ituna.gauge` rather than waiting for a gauge name here; the module docstring explains "
        "why that is the cheaper route and not merely another one."
    ),
    "scaled_signed_permutation": (
        "the classical ICA class, a signed permutation composed with a per-coordinate scale, which is what a shape objective leaves standing when "
        "the latent is not whitened. It is NOT a subgroup of O(L), so singular_values, departure and eigvec_cond stop being invariant exactly as "
        "they do under GL(L); pass `general_linear` for a soundly conservative answer. What it adds over the bare spectrum are the diagonal-scaling "
        "invariants -- the diagonal of K up to permutation, and products around cycles -- none of which are computed here. As above, `ituna.gauge` "
        "is the route, once a scale-bearing indeterminacy class exists to fit the alignment with."
    ),
}


def _validate_gauge(gauge: str) -> str:
    """Resolve a gauge name, keeping a typo distinguishable from a group whose invariants are not built yet."""
    if gauge in (_GAUGE_ORTHOGONAL, _GAUGE_GENERAL_LINEAR):
        return gauge
    if gauge in _UNIMPLEMENTED_GAUGES:
        raise NotImplementedError(f"gauge {gauge!r} is recognised but not implemented here: {_UNIMPLEMENTED_GAUGES[gauge]}")
    raise ValueError(
        f"unknown gauge {gauge!r}; expected {_GAUGE_ORTHOGONAL!r} or {_GAUGE_GENERAL_LINEAR!r}. "
        f"Recognised but not implemented: {sorted(_UNIMPLEMENTED_GAUGES)}."
    )


def _run_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sample scatter and standard error along the leading run axis of a real array."""
    scatter = values.std(axis=0, ddof=1)
    return values.mean(axis=0), scatter, scatter / np.sqrt(values.shape[0])


def _conjugate_closure(spectrum: np.ndarray) -> tuple[float, float]:
    """Worst distance from a spectrum to its own conjugate under optimal matching, with the scale applied.

    The statistic is the worst matched distance and not the mean because closure is structural: one
    broken conjugate pair means the multiset is not the spectrum of any real operator, however well
    the remaining slots agree.
    """
    cost = np.abs(spectrum[:, None] - np.conjugate(spectrum)[None, :])
    rows, columns = scipy.optimize.linear_sum_assignment(cost)
    residual = float(cost[rows, columns].max())
    modulus_mean = float(np.abs(spectrum).mean())
    scale = modulus_mean if modulus_mean > 0.0 else 1.0
    return residual / scale, scale


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


def spectral_consensus(
    result: SpectralConsistencyResult,
    *,
    gauge: str = _GAUGE_ORTHOGONAL,
    closure_warn_threshold: float = 1e-6,
) -> SpectralConsensusResult:
    """Average the gauge invariants of a run ensemble into one estimate, with its across-run dispersion.

    Retraining the same model on the same data leaves two kinds of difference between the converged
    parameters: the gauge, which is exact and carries no information, and residual variance, which is
    noise. :func:`spectral_consistency` removes the first by computing only invariants. This function
    attacks the second the only way an ensemble allows -- by averaging -- and reports what is left.

    The averaging needs one thing the pairwise picture does not supply: a *global* labelling of
    eigenvalues, since optimal matching is computed per pair and is not transitive. The labelling used
    is the star through the medoid, ``result.assignments[result.reference_id]``, which the source
    result already computed. That is a star consensus and not the Fréchet mean of the spectra under
    matching; the two coincide when the runs agree and diverge when they do not, so read
    :attr:`SpectralConsistencyResult.distance_matrix` before reading this at all. A consensus over
    runs that disagree is a mean of unlike things, and nothing here can tell you that except the
    scatter it reports.

    Parameters
    ----------
    result : SpectralConsistencyResult
        Output of :func:`spectral_consistency`. Its spectra, matching and medoid are reused rather
        than recomputed, so this call performs no eigendecomposition.
    gauge : {"orthogonal", "general_linear"}, default="orthogonal"
        The group the training objective actually leaves standing, which decides what may be
        averaged. Under ``"general_linear"`` -- an autoencoder, or a JEPA with a stop-gradient or EMA
        target, neither of which constrains the latent metric -- the singular values and the Henrici
        departure are not invariants, and the corresponding fields are returned as None instead of as
        numbers that would look perfectly reasonable. See the module docstring for the mapping from
        anticollapse objective to gauge.
    closure_warn_threshold : float, default=1e-6
        Emit a :class:`RuntimeWarning` when the relative conjugate-closure residual exceeds this. The
        residual is at round-off for spectra that agree structurally and of order one when they do
        not, so the threshold sits far from both.

    Returns
    -------
    SpectralConsensusResult
        The consensus spectrum with per-slot scatter and standard error, the aggregates the declared
        gauge permits, and the closure diagnostic.

    Raises
    ------
    ValueError
        If ``gauge`` is not a recognised name.
    NotImplementedError
        If ``gauge`` names a group this module recognises but does not compute invariants for.

    Warns
    -----
    RuntimeWarning
        If the consensus spectrum is not closed under conjugation to ``closure_warn_threshold``, so
        it is not the spectrum of any real operator and at least one slot averages across a
        structural disagreement.
    """
    gauge = _validate_gauge(gauge)

    assignment = np.array(result.assignments[result.reference_id])
    matched_eigenvalues = np.take_along_axis(result.eigenvalues, assignment, axis=1)
    n_runs = matched_eigenvalues.shape[0]

    eigenvalues_mean = matched_eigenvalues.mean(axis=0)
    # Scatter as a distance in the plane, not per component: a real and an imaginary standard
    # deviation would describe an axis-aligned box, and the eigenvalue cloud has no reason to be one.
    variance = (np.abs(matched_eigenvalues - eigenvalues_mean) ** 2).sum(axis=0) / (n_runs - 1)
    eigenvalues_scatter = np.sqrt(variance)

    spectral_radius_mean, _, spectral_radius_sem = _run_statistics(result.spectral_radius)
    closure_residual, closure_scale = _conjugate_closure(eigenvalues_mean)

    if closure_residual > closure_warn_threshold:
        warnings.warn(
            f"the consensus spectrum is not closed under conjugation (relative residual {closure_residual:.3e} against a threshold of "
            f"{closure_warn_threshold:g}), so it is not the spectrum of any real operator. At least one slot averages a complex eigenvalue of one "
            "run against a real eigenvalue of another, which is a disagreement about structure rather than a difference the mean can absorb. Read "
            "matched_eigenvalues per slot rather than eigenvalues_mean.",
            RuntimeWarning,
            stacklevel=2,
        )

    continuous_eigenvalues_mean = None
    if result.continuous_eigenvalues is not None:
        continuous_eigenvalues_mean = _freeze(np.take_along_axis(result.continuous_eigenvalues, assignment, axis=1).mean(axis=0))

    singular_values_mean = singular_values_sem = None
    departure_mean = departure_sem = None
    if gauge == _GAUGE_ORTHOGONAL:
        # Singular values need no matching: descending order is already a canonical labelling.
        singular_mean, _, singular_sem = _run_statistics(result.singular_values)
        singular_values_mean, singular_values_sem = _freeze(singular_mean), _freeze(singular_sem)
        departure_stats = _run_statistics(result.departure)
        departure_mean, departure_sem = float(departure_stats[0]), float(departure_stats[2])

    return SpectralConsensusResult(
        gauge=gauge,
        n_runs=n_runs,
        reference_id=result.reference_id,
        assignment=_freeze(assignment),
        matched_eigenvalues=_freeze(matched_eigenvalues),
        eigenvalues_mean=_freeze(eigenvalues_mean),
        eigenvalues_scatter=_freeze(eigenvalues_scatter),
        eigenvalues_sem=_freeze(eigenvalues_scatter / np.sqrt(n_runs)),
        spectral_radius_mean=float(spectral_radius_mean),
        spectral_radius_sem=float(spectral_radius_sem),
        conjugate_closure_residual=closure_residual,
        conjugate_closure_scale=closure_scale,
        eigvec_cond_max=float(result.eigvec_cond.max()),
        continuous_eigenvalues_mean=continuous_eigenvalues_mean,
        singular_values_mean=singular_values_mean,
        singular_values_sem=singular_values_sem,
        departure_mean=departure_mean,
        departure_sem=departure_sem,
    )
