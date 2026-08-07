# iTuna: two additions for Koopman-structured latent models

Build brief for an agent working in a clone of `emaballarin/ituna`.

Scope: add an **orthogonal (Procrustes) indeterminacy class** and a **standalone spectral
consistency module**. Nothing else.

> **Revision note.** Three defects in the draft this document supersedes are corrected here, each
> marked **[CORRECTED]** at the point of change: the rationale for rejecting complex operators (4.2,
> 4.5) rested on a false claim about conjugate-closure; the test matrix in 3.5#5 was left unpinned,
> so its threshold did not follow from its own hypothesis; and 4.6#4 contradicted the default value
> of `normalise`. Every claim in section 1 has been re-read against the working tree and now carries
> a line number. Anything asserted here as measured was measured — the figures are reproducible from
> the tests in 3.5 and 4.6.

---

## 0. Context and motivation

iTuna measures empirical identifiability by training `H` seeds, aligning their embeddings under an
indeterminacy class, and reporting the post-alignment `R²`. Applied to a latent dynamical model

```
z_t = f(x_t),    z_{t+1} = K z_t
```

trained with an isotropy-enforcing objective (SIGReg or similar), two gaps appear.

1. **The gauge group is `O(L)`, and iTuna has no class for it.** The available classes are
   `Identity`, `Permutation` (signed permutations, `B_L ⊂ O(L)`), `Linear` (unconstrained
   least squares over all of `R^{L×L}`), and `Affine`. `Linear` tests a group strictly larger than
   the one the identifiability argument licenses: it scores `R² = 1` on a pair of runs differing by
   a general `A ∈ GL(L)`, which is a genuine violation of the isotropy constraint, not a gauge move.
   `Linear` measures *subspace* agreement; what is needed is *geometry* agreement.

2. **The operator `K` is never examined.** iTuna's API is sklearn-transformer-shaped — `fit(X)`,
   `transform(X)` — with no time indexing and no operator. But embedding consistency does not imply
   spectral consistency: two runs that converged to *different* approximately-Koopman-invariant
   subspaces can still score high `R²` on embeddings, because least squares will fit a good linear
   map between two overlapping-but-distinct subspaces. Distinguishing "same subspace, different
   gauge" from "different subspaces" requires comparing gauge-*invariants* of `K` directly.

These two additions close exactly those gaps and nothing else.

**The gap is measurable, and it is large.** On three embeddings where two are related by a genuine
`O(L)` gauge move and the third by `A = diag(3, 1, 1, 1)`, driven end-to-end through
`ConsistencyEnsemble`:

| indeterminacy | score |
|---|---|
| `Linear` | `1.000000000` — blind to the violation |
| `Orthogonal` | `0.475266414` |

---

## 1. Verified ground truth

Re-read against the working tree at `b064c24`. Line numbers are from that commit.

| Fact | Location |
|---|---|
| `R2ScoreMixin.score(X, y)` → `sklearn.metrics.r2_score(y, self.predict(X), multioutput="uniform_average")` | `ituna/metrics.py:14-18` |
| Indeterminacy contract: sklearn regressor with `fit(X, y)`, `predict(X)`, `score(X, y)`; instantiated via `sklearn.base.clone` | `ituna/metrics.py:336` |
| Pairwise fit is `estimator.fit(X=X[i], y=X[j])` per **ordered** pair — forward and backward maps are independent fits, *not* mutual inverses | `ituna/metrics.py:337` |
| `Identity` and `Permutation` derive from `BaseEstimator, RegressorMixin, R2ScoreMixin`; `Linear`/`Affine` derive from `LinearRegression` with `fit_intercept` `False`/`True` | `ituna/metrics.py:21, 40, 103, 113` |
| `ConsistencyEnsemble.fit` hardwires the embeddings into the consistency transform — the transform *only* ever receives embeddings | `ituna/estimator.py` |
| Fitted base models are exposed as `ensemble.estimators_` | `ituna/estimator.py:227` |
| `numpy` and `scipy` are already declared dependencies | `pyproject.toml:25-28` |

Two facts absent from earlier revisions of this brief, both load-bearing:

| Fact | Location | Consequence |
|---|---|---|
| `PairwiseConsistency` is decorated `@typeguard.typechecked`, and its `indeterminacy` parameter is annotated `sklearn.base.RegressorMixin` | `ituna/metrics.py:268, 283` | A new class is accepted **only** if it actually inherits `RegressorMixin`. The base-class list in 3.2 satisfies this, so non-goal 7.1 ("do not modify `PairwiseConsistency`") is achievable rather than aspirational |
| The upstream parity harness hardcodes `CLASSES = ["Identity", "Permutation", "Linear", "Affine"]` | `tools/upstream_parity/probe.py:26` | A new indeterminacy class is invisible to the parity check by construction. This is *why* section 5's "out of scope" holds — not an assumption about additive changes in general |

**Consequence for design.** A spectral consistency measure **cannot** be a `ConsistencyTransform`
subclass and **cannot** be passed as `consistency_transform=`, because `ConsistencyEnsemble.fit`
feeds that slot the output of `estimator.transform(X)`. It must be standalone, consuming a list of
operators obtained from `ensemble.estimators_`. Do not attempt to force it into the ABC.

---

## 2. Conventions

The repository carries `AGENTS.md` and `CLAUDE.md`. **Read both first; on any conflict they win over
this section.** The rules below are the defaults where the repository is silent.

- Python 3.14+, run under `python -O`. **Never use `assert` for validation in library code** — the
  optimiser strips it. Raise explicitly.
- Ruff for formatting and linting. `line-length = 160`, isort `force-single-line = true` with
  `typing` excluded from that rule.
- One-line `"""..."""` docstring on every module, class, and function.
- British English in comments, docstrings, and any prose.
- Validate at boundaries (shapes, dtypes, dimensionality); trust internal invariants.
- No new external dependencies. `numpy` and `scipy` suffice.
- Style split, decided — do not deliberate:
  - `Orthogonal` goes **into `ituna/metrics.py`** and matches that file's existing style
    (`typing.Optional`, `List`, `Tuple` where annotations are used at all).
  - The spectral code goes into a **new module `ituna/spectral.py`** and uses modern native
    type-hint syntax throughout (`int | None`, `tuple[int, ...]`, `collections.abc`).

---

## 3. Addition 1 — `Orthogonal` indeterminacy class

### 3.1 Mathematics

Solve the orthogonal Procrustes problem: given source `X ∈ R^{n×L}` and target `Y ∈ R^{n×L}`,

```
Q* = argmin_{Q ∈ O(L)} ‖X Q − Y‖_F
```

Since `‖XQ‖_F² = tr(Qᵀ Xᵀ X Q) = tr(Xᵀ X)` is constant over `O(L)`, the objective reduces to
maximising `tr(Qᵀ M)` with `M = Xᵀ Y`. Writing the SVD `M = U Σ Vᵀ`,

```
Q* = U Vᵀ
```

(Schönemann, *Psychometrika* 31(1), 1966.) Restricting to the rotation subgroup `SO(L)` instead
requires the sign correction `Q* = U diag(1, …, 1, det(U Vᵀ)) Vᵀ`.

**Do not centre `X` or `Y`.** Uncentred Procrustes is the correct test for a gauge acting linearly on
zero-mean latents. Translation freedom is `Affine`'s job; silently centring would test a larger group
and reintroduce the problem this class exists to fix.

### 3.2 Contract

```python
class Orthogonal(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, R2ScoreMixin):
    """Orthogonal indeterminacy: alignment restricted to O(L) via the Procrustes solution."""

    def __init__(self, *, allow_reflection: bool = True): ...
    def fit(self, X, y): ...  # sets self.orthogonal_ (L, L), self.is_fitted_
    def predict(self, X): ...  # returns X @ self.orthogonal_

    @property
    def coef_(self): ...  # self.orthogonal_.T -- sklearn's convention, see below
```

- `__init__` stores `allow_reflection` **verbatim** and does nothing else. `sklearn.base.clone`
  round-trips via `get_params()` and reconstructs; mutating a parameter in `__init__` breaks it.
- `allow_reflection=True` (default) → full `O(L)`, which is the SIGReg gauge group.
  `allow_reflection=False` → `SO(L)`.
- `fit` raises `ValueError` if `X.ndim != 2`, `y.ndim != 2`, or `X.shape != y.shape`. Procrustes over
  `O(L)` is defined only for equal source and target dimension — do **not** silently truncate or pad.
  (`Permutation` truncates samples to `min(n_X, n_y)` at `metrics.py:56`; do not copy that behaviour.
  Mismatched sample counts are a caller error here.)
- `predict` raises if not fitted.
- `score` is inherited unchanged from `R2ScoreMixin`, so it stays comparable with every other
  indeterminacy class.

**[CORRECTED] The fitted matrix is named `orthogonal_`, not `coef_`.** Earlier revisions stored the
Procrustes matrix as `coef_` with `predict = X @ coef_`. That inverts sklearn's own convention:
`LinearRegression.coef_` has shape `(n_targets, n_features)` and predicts `X @ coef_.T`, so
`Linear().coef_` and `Orthogonal().coef_` would have meant transposed things under one name. Nothing
inside `ituna` reads `.coef_` — verified by grep across `ituna/` — so no existing code breaks either
way, but a caller comparing the two classes' matrices silently would. `Permutation` already uses a
descriptive `permutation_matrix_` for exactly this reason. Expose `coef_` as a read-only property
returning `orthogonal_.T`, so both conventions are available and each is correct; raise
`AttributeError` from it when unfitted, which is what `sklearn.utils.validation.check_is_fitted`
expects.

### 3.3 Reference implementation of the core

```python
M = X.T @ y  # (L, L)
U, _, Vt = np.linalg.svd(M)
Q = U @ Vt
if not self.allow_reflection and np.linalg.det(Q) < 0:
    U[:, -1] *= -1
    Q = U @ Vt
self.orthogonal_ = Q
```

Rank-deficient `M` still yields an orthogonal `Q` (SVD of a square matrix always returns full
orthogonal factors); the maximiser is simply non-unique. Note this in the docstring; do not add a
rank guard.

### 3.4 Why this is also the less-biased estimator

In-sample `R²` on `H` runs is optimistically biased by roughly `p/n` per output dimension, `p` being
the number of free parameters per output. `Linear` fits `L` free parameters per output; `Orthogonal`
carries `L(L−1)/2` parameters in total across all `L` outputs, so by the usual degrees-of-freedom
heuristic the optimism is smaller by about a factor of two. This is a heuristic, not a theorem —
state it as such in any docstring that mentions it, or omit it.

### 3.4b [ADDED] What a low `Orthogonal` score does *not* prove

SIGReg and its relatives are **regularisers, not constraints**. The residual gauge is `O(L)` only to
the extent each run's latent covariance actually equals `I`. Two runs that agree perfectly up to
gauge, but whose covariances have drifted from `I`, are related by a near-orthogonal map with
non-unit singular values — and `Orthogonal` scores them below 1. That is not a geometry
disagreement; it is a measure of how far the isotropy objective was actually driven.

The consequence for the caller is concrete: **`Orthogonal` alone is not interpretable — the
`Linear − Orthogonal` gap is.** `Linear ≈ 1` with `Orthogonal ≪ 1` says the runs share a subspace but
not a geometry, which is either a real identifiability failure *or* slack isotropy, and the two are
not separable from the embeddings alone. Both numbers should be reported together. This belongs in
the class docstring, not only here.

### 3.5 Tests

Add to `tests/test_metrics.py`, which exists and covers `ituna/metrics.py`; follow its style (plain
test functions, `np.testing.assert_*`, fixtures from `tests/conftest.py`).

| # | Property | Assertion |
|---|---|---|
| 1 | Exact recovery | `Y = X @ Q0` with `Q0` from QR of a Gaussian and **anisotropic non-symmetric `X`** → `orthogonal_ ≈ Q0` (atol 1e-8), `score(X, Y) ≈ 1.0` |
| 2 | Orthogonality | `orthogonal_.T @ orthogonal_ ≈ I` (atol 1e-10) for arbitrary `X`, `Y` |
| 3 | Isometry | `‖predict(X)‖_F == ‖X‖_F` (rtol 1e-12) |
| 4 | Reflection handling | `Y = X @ diag(1,…,1,−1)`: recovered exactly with `allow_reflection=True`; `det(orthogonal_) ≈ +1` and `score < 1` with `allow_reflection=False` |
| 5 | **Strictly stronger than `Linear`** | **[CORRECTED]** `Y = X @ A` with **`A = diag(3, 1, 1, 1, 1)` specifically** → `Linear().fit(X, Y).score(X, Y) ≈ 1.0` while `Orthogonal().fit(X, Y).score(X, Y) < 0.99`. This test *is* the justification for the class; do not soften the threshold — pin `A` instead |
| 5b | Out-of-sample separation | Fit both on `X[:n//2]`, score both on `X[n//2:]` with the same `A`: `Linear > 0.99 > Orthogonal`. Answers pre-mortem 9.3 |
| 6 | sklearn plumbing | `sklearn.base.clone(Orthogonal(allow_reflection=False))` preserves the parameter; `get_params`/`set_params` round-trip |
| 6b | **Dominance invariant** | On arbitrary `X`, `Y` — including the real ensemble fixture of test 7 — `Linear().fit(X, Y).score(X, Y) ≥ Orthogonal().fit(X, Y).score(X, Y)`, with no tolerance slack. See below |
| 6c | `coef_` convention | `predict(X) ≈ X @ coef_.T` (sklearn's convention) and `coef_ ≈ orthogonal_.T`; `coef_` on an unfitted instance raises `AttributeError` |
| 7 | Drop-in | `PairwiseConsistency(indeterminacy=Orthogonal())` runs end-to-end inside `ConsistencyEnsemble` and returns a finite score |
| 8 | Boundary errors | Mismatched shapes and 1-D input raise `ValueError`, not `AssertionError` (must hold under `python -O`) |

**[CORRECTED] on test 5.** "Invertible, non-orthogonal, well-conditioned" does not imply the
threshold — it leaves `A` free, and the test flips on that choice. Measured on `n = 200`, `L = 5`:

| `A` | `Linear` | `Orthogonal` | against `< 0.99` |
|---|---|---|---|
| `diag(3,1,1,1,1)` | `1.000000` | `0.908552` | passes |
| shear, `cond ≈ 1.35` | `1.000000` | `0.955604` | passes |
| `diag(1.2,1,…)` | `1.000000` | `0.994311` | **fails** |
| `Q·diag(1.05,1,…)` | `1.000000` | `0.999529` | **fails** |

The instruction not to soften the threshold stands. The fix is to pin `A`, not to relax the bound.
Out-of-sample (test 5b) with `A = diag(3,1,1,1,1)`: `Linear = 1.000000`, `Orthogonal = 0.902810`, so
the separation is not an artefact of in-sample fitting.

Test 6b is a theorem, not a heuristic, and is therefore the cheapest structural check available.
`Linear` is OLS with `fit_intercept=False`, which decouples across output columns and minimises each
column's residual over all of `R^L`; `O(L) ⊂ R^{L×L}`, so no orthogonal map can beat it on any single
column, and `multioutput="uniform_average"` inherits the inequality. A violation means an orientation
error (`X Q` versus `Q X`), a transposed `M = Xᵀ Y`, or a stray centring step — all three of which
otherwise produce a valid orthogonal matrix, a plausible `R²`, and no exception. Confirmed empirically
at 0 violations over 400 randomised trials, worst excess `0.00e+00`.

---

## 4. Addition 2 — `ituna/spectral.py`

### 4.1 What it measures, and why it is separate

Every quantity here is invariant under the similarity action `K ↦ Q K Qᵀ` that the encoder gauge
induces on the operator. That invariance is the point: these numbers are comparable across runs
*without* alignment, so they answer the question embedding alignment cannot — did the runs find the
same dynamics, or merely correlated coordinates?

Standalone module. No `ConsistencyTransform` subclass. No changes to `estimator.py`.

### 4.2 Public surface

```python
def spectral_consistency(
    operators: Sequence[np.ndarray],
    *,
    normalise: bool = True,
    dt: float | None = None,
    cond_warn_threshold: float = 1e6,
) -> SpectralConsistencyResult: ...
```

`operators` is a sequence of `H` **real** square matrices of identical shape `(L, L)`. Raise
`ValueError` on ragged shapes, non-square input, `H < 2`, non-finite entries, or **complex dtype**.

**[CORRECTED] Why complex input is rejected.** The previous revision said the realness requirement
was load-bearing because "the chirality argument in 4.5 depends on the spectrum being closed under
conjugation, which holds for real matrices only." **That is false, and 4.5 is corrected below.**
Chirality blindness follows from *similarity* invariance — `S K S = S K S⁻¹` for `S = S⁻¹` — and
similar matrices have identical characteristic polynomials over any field. Measured directly: for a
complex `K`, the matched distance between `K` and `S K S` is `0.000e+00`. The guarantee does not fail
for complex `K`.

Reject complex input anyway, for the honest reason: every invariant here generalises to `C` without
difficulty, but iTuna's embeddings are real and so is any operator fitted on them, so complex input
is outside the tested envelope and is far more likely to be a caller passing an already-diagonalised
`Λ` than a deliberate choice. This is boundary validation, not a mathematical necessity — say so in
the error message and the docstring, and do not restate the withdrawn claim.

`SpectralConsistencyResult` is a frozen dataclass:

| field | shape | meaning |
|---|---|---|
| `eigenvalues` | `(H, L)` complex | per-run spectrum, `np.linalg.eig` |
| `distance_matrix` | `(H, H)` float | mean matched eigenvalue distance per unordered pair, symmetric, zero diagonal |
| `max_distance_matrix` | `(H, H)` float | worst matched eigenvalue distance per unordered pair, symmetric |
| `assignments` | `(H, H, L)` int | Hungarian column indices; **see the mirroring rule in 4.3** |
| `scale` | `(H, H)` float | per-pair normalisation denominator actually applied; `1.0` where normalisation was skipped — see 4.4 |
| `reference_id` | int | medoid run: `argmin` of row-mean of `distance_matrix` |
| `consistency` | float | `1 − mean(offdiag(distance_matrix))` on the already-normalised distances — see 4.4 |
| `eigvec_cond` | `(H,)` float | `κ₂(V)` from the eigendecomposition of each `K_h` |
| `singular_values` | `(H, L)` float | descending, per run |
| `departure` | `(H,)` float | Henrici departure from normality |
| `spectral_radius` | `(H,)` float | `max |λ|` per run |
| `normalised` | bool | whether `normalise` was requested |
| `continuous_eigenvalues` | `(H, L)` complex or `None` | `log(λ)/dt`, only when `dt` is given |

### 4.3 Matching

For each unordered pair `(h, h')` with `h < h'` build the cost matrix
`C[i, j] = |λ_i^{(h)} − λ_j^{(h')}|` in the complex plane and solve with
`scipy.optimize.linear_sum_assignment`. Record the mean and the maximum of the matched distances.

**Mirroring rule.** The distances are symmetric, so `distance_matrix` and `max_distance_matrix` may
be computed on the upper triangle and copied across. **`assignments` must not be copied.** The
assignment for `(h', h)` is the *inverse* permutation of the one for `(h, h')`, so the lower triangle
is `np.argsort(assignments[h, h_prime])`, not `assignments[h, h_prime]`. A naive mirror fills the
lower triangle with permutations that do not correspond to the pairs indexing them, and nothing
raises — the distances stay correct while `assignments` is silently wrong. Verified: for a random
pair the upper assignment `[0 1 3 4 5 2]` inverts to `[0 1 5 2 3 4]`, which is what solving the
transposed problem returns.

Casting note: `np.linalg.eig` returns a real array when a real matrix happens to have a real
spectrum. Cast unconditionally to `np.complex128` before building the cost matrix, or the pair
`(real-spectrum run, complex-spectrum run)` will broadcast inconsistently.

### 4.4 Normalisation

With `normalise=True`, divide each pairwise mean distance by

```
scale = 0.5 * (mean_i |λ_i^{(h)}| + mean_j |λ_j^{(h')}|)
```

so the distances are dimensionless and comparable across latent dimensions and datasets. Record the
denominator actually applied in the `scale` field of the result. Guard against `scale == 0` (a
nilpotent or zero `K`) by leaving that pair's distance unnormalised and storing `1.0` in `scale`, so
a reader can always reconstruct raw distances as `distance_matrix * scale` regardless of which
branch was taken. With `normalise=False`, `scale` is all ones.

`max_distance_matrix` is divided by the same per-pair `scale`, so the two matrices stay in the same
units and `distance_matrix <= max_distance_matrix` holds elementwise in every branch.

`consistency = 1 − mean off-diagonal normalised distance`. **It can go negative**, and that is
meaningful, not a bug — it means the runs' spectra are further apart than the typical eigenvalue
magnitude. Do not clip it. Document this on the field.

### 4.5 Documented limitations — implement them as tests, do not paper over them

- **[CORRECTED] Chirality is invisible — and so is every other similarity.** Eigenvalues are
  invariants of the *similarity* class of `K`, so any `K ↦ S K S⁻¹` contributes exactly zero to this
  metric. A reflected conjugate `S R(θ) S = R(−θ)` with `S = diag(1,−1,1,…)` is one instance:
  `S = S⁻¹`, so this is a similarity and the multisets coincide. **This holds for complex `K` as
  well** — it is a statement about characteristic polynomials, not about conjugate-closure of a real
  spectrum, and the earlier claim that the guarantee "fails for complex `K`" was wrong. Detecting a
  residual handedness gauge requires eigenvector coordinates, which are not gauge-invariant and are
  out of scope. Note that this blindness is the *same property* that makes the module work at all:
  invariance to the `O(L)` gauge and blindness to chirality are one fact, not two.
- **Eigenvalues of non-normal operators are ill-conditioned.** By Bauer–Fike, for diagonalisable
  `K = V Λ V⁻¹`, every eigenvalue of `K + E` lies within `κ₂(V) ‖E‖₂` of some eigenvalue of `K`.
  When `eigvec_cond` exceeds `cond_warn_threshold`, emit a `RuntimeWarning` naming the offending run
  indices and stating that matched-eigenvalue distances are unreliable and that `singular_values`
  and `departure` should be used instead. LAPACK returns unit-2-norm eigenvector columns, which is
  the normalisation `κ₂(V)` assumes — do not renormalise `V`.
- **Continuous-time conversion has a branch ambiguity.** `log(λ)/dt` is single-valued only for
  `|arg λ| < π`. Eigenvalues on or near the negative real axis sit at the Nyquist limit and their
  imaginary part is not recoverable from a single `Δt`. Keep `dt=None` as the default; when `dt` is
  supplied, warn if any `|arg λ| > 0.9π`. A zero eigenvalue has no logarithm — map it to `-inf`
  rather than letting numpy emit a divide-by-zero warning, and say so on the field.
- **This detects disagreement, not correctness.** Tightly clustered spectra across runs are
  consistent with all `H` runs sharing the same bias — including systematically polluted eigenvalues
  from compressing onto a non-invariant subspace. Ruling that out needs a residual test against the
  data (ResDMD-style), which is out of scope here. Say so in the module docstring.

### 4.6 Tests

| # | Property | Assertion |
|---|---|---|
| 1 | Gauge invariance | `K` and `Q K Qᵀ` for random `Q ∈ O(L)` → pairwise distance `< 1e-8` |
| 2 | Ordering invariance | Row/column-permuted conjugate `P K Pᵀ` → distance `< 1e-10` |
| 3 | Chirality blindness | Block-diagonal `K` with `R(θ)`, versus `S K S` with `S = diag(1,−1,1,…)` → distance `< 1e-10`. Encodes the known limitation as a regression test |
| 4 | Sensitivity | **[CORRECTED]** `K` versus `K + εI` with a **well-separated** spectrum and **`normalise=False`** → mean matched distance `≈ ε` (rtol 1e-6), and `max_distance_matrix ≈ ε` too. Also assert `distance_matrix * scale` reproduces the same value with `normalise=True` |
| 5 | Complex spectra | Purely rotational `K` (no real eigenvalues) runs without dtype error; mixed real/complex pair runs without dtype error |
| 6 | Conditioning warning | **[CORRECTED]** `[[1, 1e8], [0, 1+1e-9]]` triggers the `RuntimeWarning` and gives `eigvec_cond > cond_warn_threshold`. Assert on `eigvec_cond` and the warning — **not** on the eigenvalues, which collapse numerically to exactly `[1, 1]` |
| 7 | Reference selection | For `H−1` near-identical runs plus one outlier, `reference_id` is not the outlier |
| 8 | Boundary errors | Ragged shapes, non-square, `H < 2`, 1-D input, complex dtype, and non-finite entries raise `ValueError` under `python -O` |
| 9 | Invariants are invariants | `singular_values`, `departure`, `spectral_radius`, `eigvec_cond` unchanged (atol 1e-10) under `K ↦ Q K Qᵀ` |
| 10 | Assignment mirroring | For `H = 2`, `assignments[1, 0] == np.argsort(assignments[0, 1])` — catches the naive mirror in 4.3 |
| 11 | Degenerate spectra | A deliberate double eigenvalue still yields near-zero distance, and `max_distance_matrix` is asserted alongside the mean. Encodes pre-mortem 9.2 |

**[CORRECTED] on test 4.** The stated `≈ ε (rtol 1e-6)` holds on *raw* distances — measured
`1.000000000e-03` for `ε = 1e-3` — but `normalise=True` is the **default**, and it divides by
`scale = 0.6294`, yielding `1.5889e-03`. The test as previously written fails against its own
default. Pass `normalise=False`, or multiply back through the `scale` field.

---

## 5. Order of work

1. `ituna/spectral.py` **first**, with its tests.
2. `Orthogonal` in `ituna/metrics.py`, with its tests.
3. Export both from `ituna/__init__.py` (`Orthogonal` via `metrics`, which is already exported;
   add `spectral` to the module list and `__all__`).
4. Ruff clean, full `pytest tests -v` green, and green again under `python -O -m pytest`.
5. Re-run `tools/upstream_parity/compare.py` and confirm it still exits 0.

`tools/upstream_parity/` needs no *reconciliation*, because `probe.py:26` enumerates the four
original classes by name and cannot see a new one. It still needs to be **re-run**: `AGENTS.md`
requires it after any change under `ituna/metrics.py`, and "the harness should be blind to this" is a
prediction to verify, not to assume. Step 5 is not optional.

Spectral comes first because it needs no alignment machinery, so it is the cheaper of the two, and
its output determines whether `Orthogonal` is worth using: if the spectra disperse across runs, the
runs found different invariant subspaces and *no* embedding alignment class will make the comparison
meaningful.

## 6. Usage the additions must support

```python
from ituna import ConsistencyEnsemble, metrics, spectral

ensemble = ConsistencyEnsemble(
    estimator=my_koopman_encoder,
    consistency_transform=metrics.PairwiseConsistency(
        indeterminacy=metrics.Orthogonal(),
    ),
    random_states=H,
)
ensemble.fit(X)

embedding_consistency = ensemble.score(X)
operator_consistency = spectral.spectral_consistency([est.K_ for est in ensemble.estimators_])
```

The attribute name `K_` is illustrative — `spectral_consistency` takes plain arrays and must not
know anything about the estimator.

## 7. Non-goals

- Do **not** modify `PairwiseConsistency`, `ConsistencyEnsemble`, the backends, or `utils` *in
  service of these additions*. Section 1 establishes that neither addition requires it.
- Do **not** add an operator-alignment routine of the form `K_h ↦ A_h K_h A_h⁻¹`. The pairwise maps
  are independent least-squares fits, not mutual inverses, and `Linear.coef_` can be singular. That
  design needs a decision that is not settled here.
- Do **not** attempt a minimisation of `‖Qᵀ K_h Q − K_ref‖_F` over `O(L)`. It is a hard non-convex
  problem and is not needed: the invariants already answer the question.
- **[AMENDED] Pre-existing defects are now in scope**, by maintainer decision, but belong in their
  own commit so the additions stay separable from the repairs. All are crash-path only — none moves a
  number a correct run reports, so the repository invariant holds:
  - `metrics.py:18` calls `sklearn.metrics.r2_score`, but the module never imports
    `sklearn.metrics`; it resolves today only through `sklearn.linear_model`'s transitive import.
    Add the explicit import.
  - `metrics.py:368`, `:415` and `:604` validate with `assert`, which `python -O` strips — the very
    interpreter section 8 requires the suite to pass under. Convert to explicit raises. The
    duplicate-pair guard is an internal invariant `_iter_pairs` makes unreachable, so it becomes a
    `RuntimeError`; the two shape/count guards are boundary checks and become `ValueError`.
  - **`utils.py` carries the same defect, and it is the one that actually bites.** Five `assert`
    statements across `sparse_to_dense` and `dense_to_sparse` — both public — validate caller input.
    `tests/test_utils.py::test_assertions` asserted they raise `AssertionError`, so under
    `python -O` the checks vanished, the malformed input travelled on, and the test failed with an
    opaque `TypeError` from deep inside numpy. This was invisible until the `-O` criterion in
    section 8 was actually exercised; running the suite only under plain `python` hid it entirely.
    Converting these changes a public exception type from `AssertionError` to `ValueError`, which is
    the correct type for boundary validation and is what the test now asserts.

## 8. Acceptance criteria

- All tests in sections 3.5 and 4.6 pass under both `pytest` and `python -O -m pytest`.
- `ruff format --check .` and `ruff check .` report no findings across the repository.
- No new entries in `pyproject.toml` `dependencies`.
- Every added module, class, and function carries a one-line docstring.
- The limitations in 4.5 appear in the `ituna/spectral.py` module docstring, not only in tests.
- The `Linear − Orthogonal` gap caveat from 3.4b appears in the `Orthogonal` class docstring.
- `tools/upstream_parity/compare.py` exits 0.

## 9. Pre-mortem

Three ways this lands silently wrong, and the minimal diagnostic for each.

1. **Procrustes solved in the wrong orientation.** `Q = U Vᵀ` from `M = Xᵀ Y` aligns `X → Y`; the
   transposed convention aligns `Y → X`, and both produce a valid orthogonal matrix, a plausible
   `R²`, and no error. Test 3.5#1 (exact recovery of a *known* `Q0` on a non-symmetric `X`) catches
   it; a symmetric or isotropic test matrix will not. Confirmed: the transposed convention lands
   `1.62` away from `Q0` on an anisotropic `X`, and test 3.5#6b catches it independently.
2. **Hungarian assignment silently degenerates on near-degenerate spectra.** Repeated or clustered
   eigenvalues make the optimal assignment non-unique, and the reported mean distance stays near
   zero while `assignments` is arbitrary — masking a real difference in eigenvector structure.
   Confirmed: `diag(1,1,0.5,0.5,0.2,0.2)` against its own orthogonal conjugate returns the
   non-identity assignment `[1 2 3 4 5 0]` at distance `3.38e-16`. Diagnostic: assert on
   `max_distance_matrix` alongside the mean (tests 4.6#4 and 4.6#11).
3. **`Orthogonal` reported as an improvement when it is only a different bias.** Both `Linear` and
   `Orthogonal` score in-sample on the same data used to fit the alignment, so a lower `Orthogonal`
   score could reflect the tighter constraint rather than a real geometry violation. Test 3.5#5b
   settles it out-of-sample. Note this is *distinct* from the slack-isotropy confound in 3.4b, which
   no train/test split can remove.
