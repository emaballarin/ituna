import contextlib
import contextvars

_PATCH_SUSPENDED = contextvars.ContextVar("ituna_patch_suspended", default=False)


@contextlib.contextmanager
def suspend_global_cache_patch():
    """Temporarily suspend sklearn cache patch interception."""
    token = _PATCH_SUSPENDED.set(True)
    try:
        yield
    finally:
        _PATCH_SUSPENDED.reset(token)


def is_global_cache_patch_suspended() -> bool:
    return _PATCH_SUSPENDED.get()
