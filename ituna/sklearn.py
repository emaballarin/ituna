import functools
from types import MethodType
from typing import Any, Dict, Iterable, Optional, Set, Tuple
import warnings

import sklearn.base

from ituna import _backends as backends
from ituna._cache_guard import is_global_cache_patch_suspended
from ituna._cache_guard import suspend_global_cache_patch

_PATCHED_METHODS: Dict[Tuple[type, str], Any] = {}
_PATCHED_INSTANCE_METHODS: Dict[Tuple[int, str], Any] = {}


def _resolve_methods(methods: Optional[Iterable[str]]) -> Set[str]:
    resolved = set(methods or ["fit", "transform"])
    unsupported = resolved.difference({"fit", "transform", "predict", "score"})
    if unsupported:
        raise ValueError(f"Unsupported patch methods: {sorted(unsupported)}. Supported methods are: ['fit', 'transform', 'predict', 'score']")
    return resolved


def _make_fit_instance_method(original_bound):
    @functools.wraps(original_bound)
    def patched(self, *args, **kwargs):
        if is_global_cache_patch_suspended():
            return original_bound(*args, **kwargs)
        backend = backends.get_backend(method="fit", model_class=self.__class__)
        with suspend_global_cache_patch():
            fitted_model = backend.fit_models([self], *args, **kwargs)[0]
        if fitted_model is not self:
            self.__dict__.update(getattr(fitted_model, "__dict__", {}))
        return self

    return patched


def _make_call_instance_method(method_name: str, original_bound):
    @functools.wraps(original_bound)
    def patched(self, *args, **kwargs):
        if is_global_cache_patch_suspended():
            return original_bound(*args, **kwargs)
        backend = backends.get_backend(method=method_name, model_class=self.__class__)
        with suspend_global_cache_patch():
            return backend.call_models([self], method_name, *args, **kwargs)[0]

    return patched


def _patch_estimator_instance(estimator: sklearn.base.BaseEstimator, methods: Set[str]):
    for method_name in methods:
        if not hasattr(estimator, method_name):
            raise ValueError(f"{estimator.__class__.__module__}.{estimator.__class__.__qualname__} does not implement {method_name}()")
        key = (id(estimator), method_name)
        if key in _PATCHED_INSTANCE_METHODS:
            continue
        original_bound = getattr(estimator, method_name)
        _PATCHED_INSTANCE_METHODS[key] = original_bound
        if method_name == "fit":
            patched = _make_fit_instance_method(original_bound)
        else:
            patched = _make_call_instance_method(method_name, original_bound)
        setattr(estimator, method_name, MethodType(patched, estimator))


def cached(
    estimator: sklearn.base.BaseEstimator,
    methods: Optional[Iterable[str]] = None,
):
    """Patch an estimator instance in place to use iTuna caching routes."""
    from ituna.estimator import check_non_deterministic

    if check_non_deterministic(estimator):
        warnings.warn(
            "ituna.sklearn.cached() was called for an estimator marked as non-deterministic by sklearn tags. "
            "Caching non-deterministic estimators may produce unintended side effects. Use with caution. "
            "Use the ConsistencyEnsemble wrapper to safely train multiple non-deterministic estimators with caching.",
            UserWarning,
            stacklevel=2,
        )
    resolved_methods = _resolve_methods(methods)
    _patch_estimator_instance(estimator, resolved_methods)
    return estimator


def _patch_fit_method(cls: type):
    key = (cls, "fit")
    if key in _PATCHED_METHODS:
        return
    if not hasattr(cls, "fit"):
        raise ValueError(f"{cls.__module__}.{cls.__qualname__} does not implement fit()")

    original = getattr(cls, "fit")
    _PATCHED_METHODS[key] = original

    @functools.wraps(original)
    def patched_fit(self, *args, **kwargs):
        if is_global_cache_patch_suspended():
            return original(self, *args, **kwargs)

        backend = backends.get_backend(method="fit", model_class=self.__class__)
        with suspend_global_cache_patch():
            fitted_model = backend.fit_models([self], *args, **kwargs)[0]
        if fitted_model is not self:
            self.__dict__.update(getattr(fitted_model, "__dict__", {}))
        return self

    setattr(cls, "fit", patched_fit)


def _patch_call_method(cls: type, method_name: str):
    key = (cls, method_name)
    if key in _PATCHED_METHODS:
        return
    if not hasattr(cls, method_name):
        raise ValueError(f"{cls.__module__}.{cls.__qualname__} does not implement {method_name}()")

    original = getattr(cls, method_name)
    _PATCHED_METHODS[key] = original

    @functools.wraps(original)
    def patched_call(self, *args, **kwargs):
        if is_global_cache_patch_suspended():
            return original(self, *args, **kwargs)

        backend = backends.get_backend(method=method_name, model_class=self.__class__)
        with suspend_global_cache_patch():
            return backend.call_models([self], method_name, *args, **kwargs)[0]

    setattr(cls, method_name, patched_call)


def enable_global_cache(
    model_classes: Iterable[type],
    methods: Optional[Iterable[str]] = None,
):
    """Globally patch selected methods for selected sklearn classes."""
    resolved_methods = _resolve_methods(methods)
    for cls in model_classes:
        for method_name in resolved_methods:
            if method_name == "fit":
                _patch_fit_method(cls)
            else:
                _patch_call_method(cls, method_name)


def disable_global_cache(
    model_classes: Optional[Iterable[type]] = None,
    methods: Optional[Iterable[str]] = None,
):
    """Restore original sklearn methods for previously patched classes."""
    resolved_methods = _resolve_methods(methods)
    requested_classes = set(model_classes) if model_classes is not None else None

    for (cls, method), original in list(_PATCHED_METHODS.items()):
        if requested_classes is not None and cls not in requested_classes:
            continue
        if method not in resolved_methods:
            continue
        setattr(cls, method, original)
        _PATCHED_METHODS.pop((cls, method), None)


def get_global_cache_status() -> Dict[str, list]:
    """Return current global patch state for debugging and UX introspection."""
    status = {}
    for cls, method in sorted(_PATCHED_METHODS.keys(), key=lambda x: (x[0].__module__, x[0].__qualname__, x[1])):
        class_path = f"{cls.__module__}.{cls.__qualname__}"
        status.setdefault(class_path, []).append(method)
    return status
