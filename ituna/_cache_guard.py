"""Re-entrancy guard for the ituna.sklearn method patches.

A backend that patched ``fit``/``transform`` has to call the estimator's original
implementation without the patch intercepting it again. The suspension has to hold
for everything that call reaches, including work the estimator hands to other
threads.
"""

import contextlib
import contextvars
import threading

_PATCH_SUSPENDED = contextvars.ContextVar("ituna_patch_suspended", default=False)

# A ContextVar alone is not enough. Threads start from an empty context, so a worker
# spawned by an estimator fitting under joblib's threading backend would read the
# default (False), re-enter the patch and recurse back into the backend. The process
# level depth below closes that hole: while any thread holds a suspension, every
# thread sees one.
#
# The cost is that concurrent unrelated ituna work in another thread also runs
# uncached for that window. That is the safe direction to fail -- it computes the
# correct result and only forgoes a cache hit, whereas re-entering the patch inside a
# worker corrupts control flow.
#
# Across fork the child inherits a non-zero depth and never decrements it, since the
# matching __exit__ runs in the parent. That is the wanted behaviour: a forked joblib
# worker should compute, not reach back into the cache.
_suspend_depth = 0
_suspend_depth_lock = threading.Lock()


@contextlib.contextmanager
def suspend_global_cache_patch():
    """Temporarily suspend sklearn cache patch interception, process-wide."""
    global _suspend_depth

    token = _PATCH_SUSPENDED.set(True)
    with _suspend_depth_lock:
        _suspend_depth += 1
    try:
        yield
    finally:
        with _suspend_depth_lock:
            _suspend_depth -= 1
        _PATCH_SUSPENDED.reset(token)


def is_global_cache_patch_suspended() -> bool:
    """Whether cache patch interception is currently suspended."""
    # Reading an int needs no lock; only the increment/decrement above is non-atomic.
    return _PATCH_SUSPENDED.get() or _suspend_depth > 0
