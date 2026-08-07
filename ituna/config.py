# ituna/config.py
# This file holds the global configuration for the ituna package.

import contextlib
import copy

# The default backend to use. Can be 'in_memory', 'disk_cache', 'disk_cache_distributed', 'datajoint'.
DEFAULT_BACKEND = "in_memory"
BACKEND_KWARGS = dict()
BACKEND_ROUTES = dict()
CACHE_DIR = "backend_store"
FILE_LOCK_TIMEOUT = 30  # in seconds


def _class_to_path(model_class):
    if model_class is None:
        return None
    if isinstance(model_class, str):
        return model_class
    return f"{model_class.__module__}.{model_class.__qualname__}"


def _route_key(method=None, model_class=None):
    return (method, _class_to_path(model_class))


def _class_path_candidates(model_class):
    """Return class path candidates from most specific to least specific."""
    class_path = _class_to_path(model_class)
    if class_path is None:
        return [None]
    if isinstance(model_class, str):
        return [class_path]
    candidates = []
    for cls in model_class.__mro__:
        if cls is object:
            continue
        candidates.append(_class_to_path(cls))
    return candidates


def register_backend_route(method=None, model_class=None, backend=None, backend_kwargs=None):
    """Register a backend route override.

    Routes are resolved by specificity in this order:
    1. (method, model_class)
    2. (method, None)
    3. (None, model_class)
    4. DEFAULT_BACKEND + BACKEND_KWARGS
    """
    key = _route_key(method=method, model_class=model_class)
    route = {"backend": backend or DEFAULT_BACKEND}
    if backend_kwargs is not None:
        route["backend_kwargs"] = copy.deepcopy(backend_kwargs)
    BACKEND_ROUTES[key] = route


def remove_backend_route(method=None, model_class=None):
    """Remove a backend route override if it exists."""
    key = _route_key(method=method, model_class=model_class)
    BACKEND_ROUTES.pop(key, None)


def clear_backend_routes():
    """Remove all backend route overrides."""
    BACKEND_ROUTES.clear()


def resolve_backend_route(method=None, model_class=None):
    """Resolve backend name and kwargs for a method/class operation."""
    class_candidates = _class_path_candidates(model_class)
    route_candidates = []
    for class_path in class_candidates:
        route_candidates.append((method, class_path))
    route_candidates.append((method, None))
    for class_path in class_candidates:
        route_candidates.append((None, class_path))
    for key in route_candidates:
        if key in BACKEND_ROUTES:
            route = BACKEND_ROUTES[key]
            backend_name = route.get("backend", DEFAULT_BACKEND)
            route_kwargs = route.get("backend_kwargs", {})
            resolved_kwargs = copy.deepcopy(BACKEND_KWARGS)
            _deep_update(resolved_kwargs, route_kwargs)
            return backend_name, resolved_kwargs
    return DEFAULT_BACKEND, copy.deepcopy(BACKEND_KWARGS)


def _deep_update(target_dict, update_dict):
    """Recursively update nested dictionaries.

    Args:
        target_dict: The dictionary to update
        update_dict: The dictionary containing updates
    """
    for key, value in update_dict.items():
        if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
            _deep_update(target_dict[key], value)
        else:
            target_dict[key] = value


def set(**kwargs):
    """Set global configuration values by overwriting them.

    Args:
        **kwargs: Configuration key-value pairs to set.
                 All values will be strictly overwritten.
    """
    for key, value in kwargs.items():
        globals()[key.upper()] = value


def update(**kwargs):
    """Update global configuration values.

    Args:
        **kwargs: Configuration key-value pairs to update.
                 If key is 'BACKEND_KWARGS' or 'backend_kwargs', the dict will be updated recursively.
                 Otherwise, the global variable will be replaced.
    """
    for key, value in kwargs.items():
        if key.upper() == "BACKEND_KWARGS" or key.lower() == "backend_kwargs":
            # Update the BACKEND_KWARGS dict recursively instead of replacing it
            _deep_update(globals()["BACKEND_KWARGS"], value)
        else:
            # Replace the global variable
            set(**{key: value})


def get_config():
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}


@contextlib.contextmanager
def config_context(config_update_fn=set, verbose=False, **new_config):
    """Temporarily set config values."""
    old_config = copy.deepcopy(get_config())

    print(f"Storing old config: {old_config}") if verbose else None
    config_update_fn(**new_config)
    print(f"Using config: {get_config()}") if verbose else None
    try:
        yield
    finally:
        set(**old_config)
        print(f"Restored config: {get_config()}") if verbose else None


@contextlib.contextmanager
def update_config_context(verbose=False, **new_config):
    with config_context(config_update_fn=update, verbose=verbose, **new_config):
        yield


@contextlib.contextmanager
def set_config_context(verbose=False, **new_config):
    with config_context(config_update_fn=set, verbose=verbose, **new_config):
        yield
