import abc
from collections.abc import Generator
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.optimize
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import typeguard

from ituna.utils import sparse_to_dense


class R2ScoreMixin:
    def score(self, X, y):
        """Compute R² score of the transformation."""
        y_pred = self.predict(X)
        return sklearn.metrics.r2_score(y, y_pred, multioutput="uniform_average")


class Identity(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, R2ScoreMixin):
    """
    Identity indeterminacy - no transformation needed.
    Used for models that are already identifiable (e.g., PCA).
    """

    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y):
        """Fit the identity transformation (no-op)."""
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Return X unchanged."""
        return X


class Permutation(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, R2ScoreMixin):
    """
    Permutation indeterminacy - handles sign and permutation of components.
    Used for models like FastICA where components can be permuted and have sign flips.
    """

    def __init__(self):
        self.permutation_matrix_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        """
        Find the optimal permutation matrix to align X with y.
        Uses Hungarian algorithm for optimal assignment.
        """
        # Ensure X and y have the same number of samples
        n_samples = min(X.shape[0], y.shape[0])
        X = X[:n_samples]
        y = y[:n_samples]

        # Handle sign flips first by taking absolute values
        X_abs = np.abs(X)
        y_abs = np.abs(y)

        # Compute cost matrix for Hungarian algorithm
        # Use negative correlation as cost (we want to maximize correlation)
        cost_matrix = np.zeros((X.shape[1], y.shape[1]))
        for i in range(X.shape[1]):
            for j in range(y.shape[1]):
                # Use negative correlation as cost
                corr = np.corrcoef(X_abs[:, i], y_abs[:, j])[0, 1]
                cost_matrix[i, j] = -corr if not np.isnan(corr) else 0

        # Find optimal assignment
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

        # Create permutation matrix
        self.permutation_matrix_ = np.zeros((X.shape[1], y.shape[1]))
        for i, j in zip(row_ind, col_ind):
            self.permutation_matrix_[i, j] = 1

        # Determine signs based on correlation
        signs = np.ones(y.shape[1])  # Use target shape
        for i, j in zip(row_ind, col_ind):
            corr = np.corrcoef(X[:, i], y[:, j])[0, 1]
            if not np.isnan(corr) and corr < 0:
                signs[j] = -1  # Use target column index j

        self.signs_ = signs
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Apply the fitted permutation and sign transformation."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Apply permutation and sign transformation
        X_permuted = X @ self.permutation_matrix_
        # Apply signs to the permuted columns
        return X_permuted * self.signs_.reshape(1, -1)


class Orthogonal(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, R2ScoreMixin):
    """
    Orthogonal indeterminacy - alignment restricted to O(L) by the Procrustes solution.

    Used for models whose objective fixes the latent basis only up to a rotation and reflection,
    such as an encoder trained to produce isotropic latents. `Linear` admits every invertible map
    and so scores 1.0 on a pair of runs differing by a general element of GL(L), which is a real
    violation of isotropy rather than a gauge move; this class tests agreement of *geometry* where
    `Linear` tests agreement of *subspace*.

    The alignment solves ``argmin_{Q in O(L)} ||X Q - y||_F``, whose maximiser is ``U V^T`` from the
    SVD of ``X^T y`` (Schoenemann, Psychometrika 31(1), 1966). `X` and `y` are deliberately not
    centred: translation freedom belongs to `Affine`, and centring here would silently test a larger
    group than the one this class exists to isolate. A rank-deficient ``X^T y`` still yields an
    orthogonal `Q`, since the SVD of a square matrix always returns full orthogonal factors; the
    maximiser is merely non-unique in that case.

    Interpreting the score
    ----------------------
    An isotropy-enforcing objective is usually a regulariser rather than a hard constraint, so the
    residual gauge is O(L) only to the extent each run's latent covariance really is the identity.
    Two runs that agree perfectly up to gauge, but whose covariances have drifted, are related by a
    near-orthogonal map with non-unit singular values, and this class scores them below 1.0. That is
    not a geometry disagreement -- it measures how far the isotropy objective was actually driven.

    The consequence is that this score is not interpretable on its own: the gap between it and
    `Linear` is. `Linear` near 1.0 with `Orthogonal` well below it says the runs share a subspace but
    not a geometry, which is either an identifiability failure or slack isotropy, and the embeddings
    alone cannot separate the two. Report both.

    By the usual degrees-of-freedom heuristic this is also the less optimistically biased of the two
    in-sample: `Linear` fits L free parameters per output where O(L) carries L(L-1)/2 in total. That
    is a heuristic, not a theorem.
    """

    def __init__(self, *, allow_reflection: bool = True):
        self.allow_reflection = allow_reflection

    def fit(self, X, y):
        """Solve the orthogonal Procrustes problem aligning X onto y."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Orthogonal alignment requires two 2-dimensional arrays, got shapes {X.shape} and {y.shape}")
        if X.shape != y.shape:
            raise ValueError(
                f"Orthogonal alignment is defined only for equal source and target shapes, got {X.shape} and {y.shape}. "
                f"Procrustes over O(L) needs matching feature dimensions, and mismatched sample counts are a caller error "
                f"here rather than something to truncate away."
            )

        U, _, Vt = np.linalg.svd(X.T @ y)
        Q = U @ Vt
        if not self.allow_reflection and np.linalg.det(Q) < 0:
            U[:, -1] *= -1
            Q = U @ Vt

        self.orthogonal_ = Q
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Apply the fitted orthogonal transformation."""
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model must be fitted before prediction")

        return np.asarray(X) @ self.orthogonal_

    @property
    def coef_(self) -> np.ndarray:
        """The fitted matrix in scikit-learn's orientation, so that ``predict(X) == X @ coef_.T``.

        `orthogonal_` is the canonical attribute and is right-multiplied, matching `Permutation`'s
        `permutation_matrix_`. This transposed view exists because `LinearRegression.coef_` has shape
        (n_targets, n_features) and predicts ``X @ coef_.T``; exposing the Procrustes matrix directly
        as `coef_` would have given `Linear().coef_` and `Orthogonal().coef_` transposed meanings
        under one name.
        """
        if not getattr(self, "is_fitted_", False):
            raise AttributeError("Orthogonal instance is not fitted yet; `coef_` is available after calling fit")

        return self.orthogonal_.T


class Linear(sklearn.linear_model.LinearRegression):
    """
    Linear indeterminacy - handles linear transformation of components.
    Used for models like PCA where components can be linearly transformed.
    """

    def __init__(self, *, copy_X=True, n_jobs=None, positive=False):
        super().__init__(fit_intercept=False, copy_X=copy_X, n_jobs=n_jobs, positive=positive)


class Affine(sklearn.linear_model.LinearRegression):
    """
    Affine indeterminacy - handles linear transformation of components with intercept.
    """

    def __init__(self, *, copy_X=True, n_jobs=None, positive=False):
        super().__init__(fit_intercept=True, copy_X=copy_X, n_jobs=n_jobs, positive=positive)


class PairwiseConsistencyArray(np.ndarray):
    """NumPy array subclass for consistency transform results.
    This class extends :class:`numpy.ndarray` to hold the mean aligned embeddings
    and additional metadata from the :class:`PairwiseConsistency`.
    This array itself contains the mean of aligned embeddings, with shape
    (n_samples, n_features).
    Attributes
    ----------
    embeddings : numpy.ndarray
        Original embeddings of shape (n_estimators, n_samples, n_features).
    aligned_embeddings : Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices (n_pairs, 2) and values
        (n_pairs, n_samples, n_features) of the aligned embeddings.
    scores : Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the indices (n_pairs, 2) and values (n_pairs,)
        of the alignment scores.
    reference_id : int
        ID of the reference estimator.
    aligned_to_reference : numpy.ndarray
        Embeddings aligned to the reference of shape (n_estimators, n_samples, n_features).
    """

    def __new__(
        cls,
        input_array,
        embeddings=None,
        aligned_embeddings=None,
        scores=None,
        reference_id=None,
        aligned_to_reference=None,
    ):
        """Construct a PairwiseConsistencyArray."""
        obj = np.asarray(input_array).view(cls)
        obj.embeddings = embeddings
        obj.aligned_embeddings = aligned_embeddings
        obj.scores = scores
        obj.reference_id = reference_id
        obj.aligned_to_reference = aligned_to_reference
        return obj

    def __array_finalize__(self, obj):
        """Finalize array creation, attaching attributes."""
        if obj is None:
            return
        self.embeddings = getattr(obj, "embeddings", None)
        self.aligned_embeddings = getattr(obj, "aligned_embeddings", None)
        self.scores = getattr(obj, "scores", None)
        self.reference_id = getattr(obj, "reference_id", None)
        self.aligned_to_reference = getattr(obj, "aligned_to_reference", None)


class ConsistencyTransform(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator, abc.ABC):
    """
    Base class for consistency transforms.
    A consistency transform takes multiple embedding spaces as input
    and finds a mapping from each input embedding space into a single reference embedding space
    .fit() finds the mapping
    .transform() maps a given embedding space into the reference space (i.e. performs alignment)
    .score() computes how good the mapping is
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
    ):
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()

        tags.non_deterministic = False

        return tags

    def fit(self, X, *args, **kwargs):
        """Fit the consistency transform.

        Parameters
        ----------
        X : array-like
            Multiple embedding spaces to fit the consistency transform on.
        """
        if len(X) == 0:
            raise ValueError("At least one embedding space must be provided")

        self.n_spaces_ = len(X)
        return self._fit(X, **kwargs)

    def fit_transform(self, X, **kwargs):
        """Fit the consistency transform and transform the embedding spaces."""
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    @abc.abstractmethod
    def _fit(self, X, **kwargs):
        """Fit the consistency transform."""
        pass

    def _validate_input(self, X, source_id):
        if len(X) == self.n_spaces_:
            if source_id is not None:
                raise ValueError("source_id should not be provided when transforming multiple embedding spaces")
            return

        if len(X) == 0:
            raise ValueError("At least one embedding space must be provided")
        if len(X) == 1:
            if source_id is None:
                raise ValueError("source_id must be provided when transforming a single embedding space")
        else:
            raise ValueError(f"Number of spaces ({len(X)}) must match the number passed to fit ({self.n_spaces_})")

    def transform(self, X, source_id: Optional[int] = None, **kwargs):
        """Transform a given embedding space into the reference space.

        Parameters
        ----------
        X : array-like
            Either a single embedding space (when source_id is provided) or
            multiple embedding spaces matching the number passed to fit().
        source_id : int, optional
            Required when X contains exactly one embedding space.
            Identifies which source mapping should be used based on the order
            in which spaces were passed to fit().
        """
        self._validate_input(X, source_id)

        return self._transform(X, source_id=source_id, **kwargs)

    @abc.abstractmethod
    def _transform(self, X, source_id: Optional[int] = None, **kwargs):
        """Transform a given embedding space into the reference space."""
        pass

    def score(self, X, source_id: Optional[int] = None, **kwargs) -> float:
        """Compute how good the mapping is."""
        self._validate_input(X, source_id)

        return self._score(X, source_id=source_id, **kwargs)

    @abc.abstractmethod
    def _score(self, X, source_id: Optional[int] = None, **kwargs):
        """Compute how good the mapping is."""
        pass


@typeguard.typechecked
class PairwiseConsistency(ConsistencyTransform):
    """Compute consistency among a set of estimators.
    This transformer assesses the consistency between multiple estimators by
    aligning their outputs (embeddings). It fits a series of indeterminacy models
    to map the embeddings from one estimator to another, and then uses these
    models to align and score the embeddings.
    The transformation results in a mean embedding, averaged over all estimators
    after alignment to a chosen reference estimator.
    """

    _fill_value: float = np.nan

    def __init__(
        self,
        indeterminacy: sklearn.base.RegressorMixin,
        symmetric: Optional[bool] = False,
        include_diagonal: Optional[bool] = False,
        reference_selection: Union[Literal["min_score", "max_score"], int] = "max_score",
        random_state: Optional[int] = None,
    ):
        """Initialize the PairwiseConsistency.
        Parameters
        ----------
        indeterminacy : sklearn.base.RegressorMixin
            A scikit-learn-compatible regressor used to model the indeterminacy
            between pairs of estimators. This regressor is fitted on the output of
            one estimator (X) to predict the output of another (y).
        symmetric : bool, default=False
            If True, assumes that the indeterminacy between estimator `i` and `j` is
            the same as between `j` and `i`. This reduces computation by fitting
            the indeterminacy model only for the upper triangular matrix of
            estimator pairs.
        include_diagonal : bool, default=False
            If True, includes the alignment of an estimator with itself in the
            computation. This is typically False.
        reference_selection : {"min_score", "max_score", int}, default="max_score"
            Method for selecting the reference estimator to which all other
            estimators' outputs are aligned.
            - "min_score": Select the estimator with the minimum mean alignment
              score as the reference.
            - "max_score": Select the estimator with the maximum mean alignment
              score as the reference.
            - int: Select the estimator with the given index as the reference.
        """
        super().__init__(random_state=random_state)
        self.indeterminacy = indeterminacy
        self.symmetric = symmetric
        self.include_diagonal = include_diagonal
        self.reference_selection = reference_selection

    def _fit(self, X, **kwargs):
        """Fit the indeterminacy models between all pairs of estimators.
        Parameters
        ----------
        X : multiple array-likes of shape ( n_samples, n_features)
            embedding spaces to fit the indeterminacy models on.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_estimators_ = len(X)

        self.indeterminacy_indices_ = []
        self.indeterminacies_ = []

        for i, j in self._iter_pairs():
            estimator = sklearn.base.clone(self.indeterminacy)
            self.indeterminacies_.append(estimator.fit(X=X[i], y=X[j]))
            self.indeterminacy_indices_.append([i, j])
        self.indeterminacies_ = np.array(self.indeterminacies_)
        self.indeterminacy_indices_ = np.array(self.indeterminacy_indices_).reshape(-1, 2)

        self.reference_id_ = self._get_reference(self._compute_scores_matrix(X))
        return self

    def _transform(self, X, source_id: Optional[int] = None, **kwargs) -> PairwiseConsistencyArray:
        if len(X) == self.n_spaces_:
            return self._transform_all(X, **kwargs)
        elif len(X) == 1:
            return self._transform_single(X[0], source_id=source_id, **kwargs)
        else:
            raise ValueError(f"Number of embeddings incompatible with number of indeterminacy models, got {len(X)} and {self.n_spaces_}")

    def _transform_all(self, X, **kwargs) -> PairwiseConsistencyArray:
        """Align embeddings and compute the mean.
        Parameters
        ----------
        X : multiple array-likes of shape ( n_samples, n_features)
            Data where each estimator's output is stored along the first axis.
        Returns
        -------
        X_transformed : PairwiseConsistencyArray of shape (n_samples, n_features)
            Mean of the aligned embeddings. See :class:`PairwiseConsistencyArray` for
            additional attributes.
        """

        reference_shape = X[self.reference_id_].shape
        X_shapes = [x.shape for x in X]
        if any(x_shape != reference_shape for x_shape in X_shapes):
            raise ValueError(f"All embeddings must have the same shape, got {X_shapes} and match reference shape {reference_shape}")

        aligned_embeddings = self._align_embeddings(X)
        aligned_embeddings = (aligned_embeddings[0], np.array(aligned_embeddings[1]))
        dense_aligned = sparse_to_dense(
            aligned_embeddings[0],
            aligned_embeddings[1],
            shape=(self.n_estimators_, self.n_estimators_, *reference_shape),
            symmetric=self.symmetric,
        )
        aligned_to_reference = dense_aligned[:, self.reference_id_]

        if not self.include_diagonal:
            # handle reference embedding, setting it to the original embeddings,
            # since in aligned_to_reference it is nan, because we usually don't align i->i
            # except if include_diagonal is True, then i->i is included and we don't need to do this
            aligned_to_reference[self.reference_id_] = X[self.reference_id_]
        mean_embedding = np.nanmean(aligned_to_reference, axis=0)

        scores = self._compute_scores_matrix(X)
        return PairwiseConsistencyArray(
            mean_embedding,
            embeddings=np.array(X),
            aligned_embeddings=aligned_embeddings,
            scores=scores,
            reference_id=self.reference_id_,
            aligned_to_reference=aligned_to_reference,
        )

    def _get_indeterminacy(
        self,
        source_id: int,
        target_id: int,
    ) -> sklearn.base.RegressorMixin:
        """Get the fitted indeterminacy model mapping source into target."""

        matches = np.where((self.indeterminacy_indices_ == np.array([source_id, target_id])).all(axis=1))[0]

        if len(matches) == 0:
            raise ValueError(
                f"No indeterminacy model found for source {source_id} and target {target_id}. "
                f"Pairs are fitted according to `symmetric` and `include_diagonal`: with "
                f"symmetric=True only pairs (i, j) with j >= i are fitted, and the models are not "
                f"invertible in general, so the reverse direction is unavailable."
            )
        if len(matches) > 1:
            # `_iter_pairs` yields each ordered pair exactly once, so this cannot occur. Raising
            # rather than asserting keeps the guard alive under `python -O`, which strips assert.
            raise RuntimeError(f"Multiple indeterminacy models found for source {source_id} and target {target_id}, this should not happen")

        return self.indeterminacies_[matches[0]]

    def _transform_single(self, X, source_id: int, **kwargs) -> PairwiseConsistencyArray:
        if source_id == self.reference_id_ and not self.include_diagonal:
            # No i->i model is fitted unless include_diagonal is set, and aligning the reference
            # to itself is the identity. `_transform_all` special-cases the reference the same way.
            aligned_X = X
        else:
            aligned_X = self._align_to(
                X,
                self._get_indeterminacy(
                    source_id,
                    target_id=self.reference_id_,
                ),
            )
        return PairwiseConsistencyArray(
            aligned_X,
            embeddings=X,
            aligned_embeddings=None,
            scores=None,
            reference_id=self.reference_id_,
        )

    def _score(self, X, **kwargs) -> float:
        """Compute the mean consistency score.
        Parameters
        ----------
        X : array-like of shape (n_estimators, n_samples, n_features)
            Data where each estimator's output is stored along the first axis.
        Returns
        -------
        score : float
            Mean consistency score across all estimator pairs, ignoring the diagonal.

            The diagonal is excluded whatever `include_diagonal` is set to. A self-alignment
            scores exactly 1.0 under any indeterminacy class, so averaging it in would shift
            the reported score to ``(1 - f) * s + f``, with ``f`` the diagonal's share of the
            fitted pairs -- ``1 / n_estimators`` when ``symmetric`` is False and
            ``2 / (n_estimators + 1)`` when it is True. `include_diagonal` governs which
            models are fitted and hence alignment, not what the score is averaged over.

            Returns NaN when no off-diagonal pair exists, i.e. for a single estimator, for
            which consistency is undefined.
        """
        if len(X) != self.n_spaces_:
            raise ValueError("PairwiseConsistency.score() is not defined for a different number of embedding spaces than passed to fit()")

        result = self._transform_all(X, **kwargs)
        indices, values = result.scores
        off_diagonal = values[indices[:, 0] != indices[:, 1]] if len(indices) else values
        if len(off_diagonal) == 0 or np.all(np.isnan(off_diagonal)):
            return float(self._fill_value)
        return float(np.nanmean(off_diagonal))

    def _iter_pairs(self) -> Generator[Tuple[int, int], None, None]:
        """Iterate over pairs of estimators according to the symmetric setting.
        Yields pairs (i, j) based on the symmetric and include_diagonal settings:
        - If symmetric=True, only yields pairs where j >= i (upper triangular)
        - If symmetric=False, yields all pairs (i, j)
        - If include_diagonal=False, skips pairs where i == j
        Yields
        ------
        Tuple[int, int]
            Pairs of estimator indices (i, j).
        """
        for i in range(self.n_estimators_):
            for j in range(i if self.symmetric else 0, self.n_estimators_):
                if i == j and not self.include_diagonal:
                    continue
                yield i, j

    def _align_to(
        self,
        X_source: np.ndarray,
        estimator: sklearn.base.RegressorMixin,
    ) -> np.ndarray:
        """Align source data using a fitted indeterminacy estimator.
        Parameters
        ----------
        X_source : np.ndarray
            The source embeddings to align.
        estimator : sklearn.base.RegressorMixin
            The fitted indeterminacy model to use for alignment.
        Returns
        -------
        np.ndarray
            The aligned embeddings.
        """
        return estimator.predict(X_source)

    def _align_embeddings(self, X, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Align all embeddings using every available fitted indeterminacy models producing a (sparse) matrix of aligned embeddings.
        Parameters
        ----------
        X : multiple array-likes of shape ( n_samples, n_features)
            embedding spaces to align.
        Returns
        -------
        aligned_embeddings : Tuple[np.ndarray, List[np.ndarray]]
            A tuple containing the indices (n_pairs, 2) and values n_pairs x (n_samples, n_features) of the aligned embeddings.
        """
        if len(X) != self.n_spaces_:
            raise ValueError(f"Number of embeddings must match number of indeterminacy models, got {len(X)} and {self.n_spaces_}")

        aligned_embeddings_values = []
        for (i, _), estimator in zip(self.indeterminacy_indices_, self.indeterminacies_):
            aligned_embeddings_values.append(self._align_to(X_source=X[i], estimator=estimator))

        return (self.indeterminacy_indices_, aligned_embeddings_values)

    def _score_alignment(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        estimator: sklearn.base.RegressorMixin,
    ) -> float:
        """Score the alignment between source and target data.
        Parameters
        ----------
        X_source : np.ndarray
            The source embeddings.
        X_target : np.ndarray
            The target embeddings.
        estimator : sklearn.base.RegressorMixin
            The fitted indeterminacy model.
        Returns
        -------
        float
            The alignment score.
        """
        return estimator.score(X_source, X_target)

    def _get_reference(self, scores: Tuple[np.ndarray, np.ndarray]) -> int:
        """Select the reference estimator based on scores."""
        if self.n_estimators_ == 1:
            return 0

        if isinstance(self.reference_selection, (int, np.integer)):
            return self.reference_selection

        indices, values = scores
        dense_scores = sparse_to_dense(
            indices,
            np.array(values),
            shape=(self.n_estimators_, self.n_estimators_),
            symmetric=self.symmetric,
        )
        mean_scores = np.nanmean(dense_scores, axis=0)

        if np.all(np.isnan(mean_scores)):
            return 0

        if self.reference_selection == "min_score":
            return np.nanargmin(mean_scores).item()
        elif self.reference_selection == "max_score":
            return np.nanargmax(mean_scores).item()
        else:
            raise ValueError(f"Invalid reference selection: {self.reference_selection}")

    def _compute_scores_matrix(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Compute alignment scores for all estimator pairs.
        Parameters
        ----------
        X : array-like of shape (n_estimators, n_samples, n_features)
            Data where each estimator's output is stored along the first axis.
        Returns
        -------
        scores : Tuple[np.ndarray, np.ndarray]
            A tuple containing the indices (n_pairs, 2) and values (n_pairs,)
            of the alignment scores.
        """
        score_values = []
        for (i, j), estimator in zip(self.indeterminacy_indices_, self.indeterminacies_):
            score_values.append(
                self._score_alignment(
                    X_source=X[i],
                    X_target=X[j],
                    estimator=estimator,
                )
            )
        return (self.indeterminacy_indices_, np.array(score_values))
