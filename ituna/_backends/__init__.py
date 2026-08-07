import copy
from typing import Optional

from ituna._backends import utils
from ituna._backends.base import Backend
from ituna._backends.datajoint import DatajointBackend
from ituna._backends.disk_cache import DiskCacheBackend
from ituna._backends.disk_cache import DiskCacheDistributedBackend
from ituna._backends.in_memory import InMemoryBackend

# backend registry
_BACKENDS = {
    "in_memory": InMemoryBackend,
    "disk_cache": DiskCacheBackend,
    "disk_cache_distributed": DiskCacheDistributedBackend,
    "datajoint": DatajointBackend,
}


def _build_backend(backend_name: str, backend_kwargs: Optional[dict] = None):
    # delayed import so it uses the updated config
    from ituna import config

    if backend_kwargs is None:
        backend_kwargs = {}

    if backend_name not in _BACKENDS:
        raise ValueError(f"Unknown backend: '{backend_name}'. Available backends are: {list(_BACKENDS.keys())}")

    backend_factory = _BACKENDS[backend_name]

    kwargs = {}
    if backend_name == "disk_cache":
        if config.CACHE_DIR:
            kwargs["cache_dir"] = config.CACHE_DIR
    elif backend_name == "disk_cache_distributed":
        if config.CACHE_DIR:
            kwargs["cache_dir"] = config.CACHE_DIR
        kwargs.update(backend_kwargs)
    elif backend_name == "datajoint":
        if config.CACHE_DIR:
            kwargs["cache_dir"] = config.CACHE_DIR
        kwargs.update(backend_kwargs)
    return backend_factory(**kwargs)


# backend factory
def get_backend(backend_name: str = None, method: str = None, model_class=None):
    """
    Factory function to get a backend instance.

    If backend_name is None, it uses the default from the global config.
    """
    from ituna import config

    if backend_name is None:
        backend_name, backend_kwargs = config.resolve_backend_route(
            method=method,
            model_class=model_class,
        )
    else:
        backend_kwargs = copy.deepcopy(config.BACKEND_KWARGS)

    return _build_backend(backend_name=backend_name, backend_kwargs=backend_kwargs)


__all__ = [
    "get_backend",
    "Backend",
    "utils",
    "DiskCacheBackend",
    "DiskCacheDistributedBackend",
    "InMemoryBackend",
    "DatajointBackend",
]
