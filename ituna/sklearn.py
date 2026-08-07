from collections.abc import Iterable
import functools
from types import MethodType
from typing import Any, Dict, Optional, Set, Tuple
import warnings

import sklearn.base

from ituna import _backends as backends
from ituna._cache_guard import is_global_cache_patch_suspended
from ituna._cache_guard import suspend_global_cache_patch

SUPPORTED_METHODS = ("fit", "transform", "predict", "score")

# Maps (class, method) -> (original attribute, whether the class itself defined it).
# Instance-level patches are not tracked here: they are marked on the patched
# function and live in the instance __dict__, so they are freed with the estimator.
_PATCHED_METHODS: Dict[Tuple[type, str], Tuple[Any, bool]] = {}

_PATCH_MARKER = "_ituna_cache_patch"


def _resolve_methods(methods: Optional[Iterable[str]]) -> Set[str]:
    resolved = set(methods or ["fit", "transform"])
    unsupported = resolved.difference(SUPPORTED_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported patch methods: {sorted(unsupported)}. Supported methods are: {list(SUPPORTED_METHODS)}")
    return resolved


def _is_instance_patched(estimator: sklearn.base.BaseEstimator, method_name: str) -> bool:
    """Whether `method_name` is an iTuna patch sitting in the instance __dict__."""
    return getattr(estimator.__dict__.get(method_name), _PATCH_MARKER, False) is True


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
        if _is_instance_patched(estimator, method_name):
            continue
        original_bound = getattr(estimator, method_name)
        if method_name == "fit":
            patched = _make_fit_instance_method(original_bound)
        else:
            patched = _make_call_instance_method(method_name, original_bound)
        setattr(patched, _PATCH_MARKER, True)
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


def uncached(
    estimator: sklearn.base.BaseEstimator,
    methods: Optional[Iterable[str]] = None,
):
    """Remove iTuna cache patches applied to an estimator instance by :func:`cached`.

    Defaults to every supported method, so a bare ``uncached(estimator)`` undoes
    any :func:`cached` call regardless of which methods it patched.
    """
    targets = _resolve_methods(methods) if methods is not None else set(SUPPORTED_METHODS)
    for method_name in targets:
        if _is_instance_patched(estimator, method_name):
            delattr(estimator, method_name)
    return estimator


def _patch_fit_method(cls: type):
    key = (cls, "fit")
    if key in _PATCHED_METHODS:
        return
    if not hasattr(cls, "fit"):
        raise ValueError(f"{cls.__module__}.{cls.__qualname__} does not implement fit()")

    original = getattr(cls, "fit")
    _PATCHED_METHODS[key] = (original, "fit" in cls.__dict__)

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
    _PATCHED_METHODS[key] = (original, method_name in cls.__dict__)

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
    """Restore original sklearn methods for previously patched classes.

    Defaults to every supported method, so a bare ``disable_global_cache()``
    fully undoes any :func:`enable_global_cache` call.
    """
    resolved_methods = _resolve_methods(methods) if methods is not None else set(SUPPORTED_METHODS)
    requested_classes = set(model_classes) if model_classes is not None else None

    for (cls, method), (original, defined_on_cls) in list(_PATCHED_METHODS.items()):
        if requested_classes is not None and cls not in requested_classes:
            continue
        if method not in resolved_methods:
            continue
        if defined_on_cls:
            setattr(cls, method, original)
        else:
            # The class inherited this method; deleting the patch restores lookup
            # through the MRO. Assigning `original` back would instead pin a copy
            # of the base implementation onto the subclass, freezing it against
            # any later change to the base class.
            delattr(cls, method)
        _PATCHED_METHODS.pop((cls, method), None)


def get_global_cache_status() -> Dict[str, list]:
    """Return current global patch state for debugging and UX introspection."""
    status = {}
    for cls, method in sorted(_PATCHED_METHODS.keys(), key=lambda x: (x[0].__module__, x[0].__qualname__, x[1])):
        class_path = f"{cls.__module__}.{cls.__qualname__}"
        status.setdefault(class_path, []).append(method)
    return status
