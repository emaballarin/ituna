import copy
from typing import List, Union

import numpy as np
import packaging.version
import sklearn.base
import sklearn.utils.metaestimators

from ituna import _backends as backends
from ituna import metrics


def generate_random_states(n: int) -> List[int]:
    """ "
    Generates a list of random seeds by multiplying range(n) with large prime number and taking modulo 2^32.

    Parameters
    ----------
    n : int
        Number of random states to generate.

    Returns
    -------
    random_states : List[int]
        List of random states.
    """

    large_prime = 65537
    max_int = 2**32 - 1

    return [(i * large_prime) % max_int for i in range(n)]


def check_version(estimator: sklearn.base.TransformerMixin):
    # NOTE(ppommer): required as a check for the old way of specifying tags
    # https://github.com/scikit-learn/scikit-learn/pull/29677#issuecomment-2334229165
    return packaging.version.parse(sklearn.__version__) < packaging.version.parse("1.6.dev")


def check_non_deterministic(estimator: sklearn.base.TransformerMixin) -> bool:
    """
    Check if the base model is non-deterministic using proper sklearn methods.
    Falls back to False if none of the tag methods are available.
    """
    if check_version(estimator):
        if hasattr(estimator, "__sklearn_tags__"):
            tags = estimator.__sklearn_tags__()
            return getattr(tags, "non_deterministic", False) if tags else False
        elif hasattr(estimator, "_get_tags"):
            tags = estimator._get_tags()
            return tags.get("non_deterministic", False)
        elif hasattr(estimator, "_more_tags"):
            more_tags = estimator._more_tags()
            return more_tags.get("non_deterministic", False)
    else:
        import sklearn.utils as sklearn_utils  # avoid shadowing the global sklearn module by aliasing

        tags = sklearn_utils.get_tags(estimator)
        return getattr(tags, "non_deterministic", False)

    return False


def clone_with_seed(
    estimator: sklearn.base.BaseEstimator,
    random_state: int,
) -> sklearn.base.BaseEstimator:
    """
    Create a model with a specific random seed.
    """
    non_deterministic = check_non_deterministic(estimator)
    cloned_estimator = sklearn.base.clone(estimator)
    if not non_deterministic and "random_state" in cloned_estimator.get_params():
        cloned_estimator.set_params(random_state=random_state)
    return cloned_estimator


class ConsistencyEnsemble(
    sklearn.base.TransformerMixin,
    sklearn.base.MetaEstimatorMixin,
    sklearn.base.BaseEstimator,
):
    """
    Ensemble of models that evaluates consistency across different random initializations.

    This ensemble fits multiple instances of the same model with different random states
    and evaluates their consistency using indeterminacy transformations.

    Parameters
    ----------
    estimator : sklearn.base.TransformerMixin
        The base model to be ensembled. Must have a random_state parameter.
    indeterminacy : sklearn.base.RegressorMixin
        The indeterminacy model used to transform between different model instances.
    random_states : list of int, optional
        Random states to use for the ensemble models. If None, generates 5 random states.

    Attributes
    ----------
    ensemble_models : list
        List of fitted base model instances with different random states.
    """

    def __init__(
        self,
        estimator: sklearn.base.TransformerMixin,
        consistency_transform: metrics.ConsistencyTransform = None,
        random_states: Union[int, List[int]] = 5,
    ):
        self.estimator: sklearn.base.TransformerMixin = estimator
        self.consistency_transform: metrics.ConsistencyTransform = consistency_transform
        self.random_states = random_states

        base_params = estimator.get_params()
        non_deterministic = check_non_deterministic(estimator)

        # check arguments
        if not non_deterministic:
            assert "random_state" in base_params, "Deterministic estimators must have a random_state parameter"

        if isinstance(self.random_states, np.ndarray):
            self.random_states = self.random_states.tolist()

        if isinstance(self.random_states, list) and non_deterministic:
            raise ValueError("List of random states is not supported for non-deterministic estimators, provide integer instead")

        # convert random_states integer to list
        if isinstance(self.random_states, int):
            self._random_states = generate_random_states(n=self.random_states)
        elif isinstance(self.random_states, list):
            self._random_states = self.random_states
        else:
            raise ValueError("Random states must be an integer or a list of integers")

    def _init_estimators(self) -> List[sklearn.base.BaseEstimator]:
        estimators = []
        for random_state in self._random_states:
            estimators.append(clone_with_seed(self.estimator, random_state))
        return estimators

    @property
    def _backend(self):
        """Backward-compatible backend accessor.

        Kept for private/test compatibility while backend selection is now resolved lazily.
        """
        return backends.get_backend()

    def __sklearn_tags__(self):
        # NOTE(ppommer): new way to specify tags (sklearn >= 1.6.dev)
        # Explicitly build the complete tag structure
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.get_tags.html
        import sklearn.utils._tags

        # Inherit core estimator properties
        tags = super().__sklearn_tags__()

        # Explicitly set properties from the sub-estimator
        sub_estimator_tags = sklearn.utils._tags.get_tags(self.estimator)
        tags.estimator_type = sub_estimator_tags.estimator_type
        tags.array_api_support = sub_estimator_tags.array_api_support
        tags.no_validation = sub_estimator_tags.no_validation
        tags.non_deterministic = sub_estimator_tags.non_deterministic
        tags.requires_fit = sub_estimator_tags.requires_fit

        # Deepcopy for objects to avoid reference sharing
        tags.target_tags = copy.deepcopy(sub_estimator_tags.target_tags)
        tags.transformer_tags = copy.deepcopy(sub_estimator_tags.transformer_tags)
        tags.classifier_tags = copy.deepcopy(sub_estimator_tags.classifier_tags)
        tags.regressor_tags = copy.deepcopy(sub_estimator_tags.regressor_tags)
        tags.input_tags = copy.deepcopy(sub_estimator_tags.input_tags)

        return tags

    @sklearn.utils.metaestimators.available_if(check_version)
    def _more_tags(self):
        # NOTE(ppommer): for backward compatibility (sklearn < 1.6.dev)
        # Override tags that are different from the inherited defaults
        # https://scikit-learn.org/1.3/developers/develop.html#estimator-tags

        # Inherit tag overrides from sub-estimator when available
        if hasattr(self.estimator, "_more_tags"):
            return self.estimator._more_tags()
        else:
            # No sub-estimator overrides, so we don't override anything
            return {}

    def set_params(self, **params):
        super().set_params(**params)
        return self

    def fit(self, *args, **kwargs):
        """
        Fit all ensemble models.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        estimator_backend = backends.get_backend(
            method="fit",
            model_class=self.estimator.__class__,
        )
        self.estimators_: sklearn.base.TransformerMixin = estimator_backend.fit_models(
            self._init_estimators(),
            *args,
            **kwargs,
        )

        embeddings = self._transforms(*args[:1])
        consistency_transform_backend = backends.get_backend(
            method="fit",
            model_class=self.consistency_transform.__class__,
        )
        self.consistency_transform_ = consistency_transform_backend.fit_models(
            [sklearn.base.clone(self.consistency_transform)],
            embeddings,
        )[0]
        return self

    def _transforms(self, X) -> List[np.ndarray]:
        """
        Computes transforms on X for all models

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : List[np.ndarray]
             List of n_estimators transformed data shaped (n_samples, n_features_out)
        """
        transform_backend = backends.get_backend(
            method="transform",
            model_class=self.estimator.__class__,
        )
        return transform_backend.call_models(self.estimators_, "transform", X)

    def transform(self, X):
        """
        Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data of the ensemble models.

        Returns
        -------
        consistency_embedding : PairwiseConsistencyArray of shape (n_samples, n_features)
            Mean of the embeddings aligned to the reference embedding. This array has additional attributes:
            - embeddings: original embeddings
            - aligned_embeddings: aligned embeddings
            - scores: alignment scores
            - reference_id: reference estimator id
            - aligned_to_reference: embeddings aligned to reference
        """
        return self.consistency_transform_.transform(self._transforms(X))

    def score(self, X, y=None) -> float:
        """
        Score the consistency of the ensemble models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score.
        y : array-like of shape (n_samples,), default=None
            Target values (ignored, present for API consistency).

        Returns
        -------
        score : float
            Consistency score of the ensemble models.
        """
        return self.consistency_transform_.score(self._transforms(X))
