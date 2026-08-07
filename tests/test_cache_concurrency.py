"""Concurrency and crash-safety behaviour of the cache guard and the cache writers.

These paths are hard to exercise from the ordinary test suite: they need a second
thread, or a writer that dies part-way through. The cases below drive them directly.
Neither is covered by the higher level backend tests, so a regression here surfaces
only under real concurrent or interrupted use.
"""

import threading

import numpy as np
import pytest
from sklearn.decomposition import PCA

from ituna._backends import utils
from ituna._cache_guard import is_global_cache_patch_suspended
from ituna._cache_guard import suspend_global_cache_patch


def test_suspension_is_visible_to_threads_started_inside_it():
    """Threads start from an empty context, so a ContextVar alone would read False here."""
    observed = []

    with suspend_global_cache_patch():
        worker = threading.Thread(target=lambda: observed.append(is_global_cache_patch_suspended()))
        worker.start()
        worker.join()

    assert observed == [True]
    assert not is_global_cache_patch_suspended()


def test_suspension_nests_and_unwinds_completely():
    with suspend_global_cache_patch():
        assert is_global_cache_patch_suspended()
        with suspend_global_cache_patch():
            assert is_global_cache_patch_suspended()
        # The inner exit must not clear the outer suspension.
        assert is_global_cache_patch_suspended()
    assert not is_global_cache_patch_suspended()


def test_suspension_unwinds_when_the_body_raises():
    with pytest.raises(RuntimeError), suspend_global_cache_patch():
        raise RuntimeError("boom")
    assert not is_global_cache_patch_suspended()


def test_concurrent_suspensions_do_not_unwind_each_other():
    started = threading.Event()
    release = threading.Event()

    def hold():
        with suspend_global_cache_patch():
            started.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold)
    worker.start()
    started.wait(timeout=5)
    try:
        with suspend_global_cache_patch():
            pass
        # The other thread still holds a suspension, so it must still be in effect.
        assert is_global_cache_patch_suspended()
    finally:
        release.set()
        worker.join()

    assert not is_global_cache_patch_suspended()


def test_store_data_does_not_leave_a_partial_file_behind(tmp_path, monkeypatch):
    """A failed write must not leave a file that later existence checks accept."""
    target = tmp_path / "entry"

    def failing_save(path, data):
        # Simulate a writer dying after emitting part of the payload.
        with open(path, "wb") as handle:
            handle.write(b"truncated")
        raise OSError("no space left on device")

    monkeypatch.setitem(utils.SUPPORTED_DATA_TYPES["numpy"], "save_fn", failing_save)
    with pytest.raises(OSError):
        utils.store_data(target, np.arange(4.0))

    assert not target.with_suffix(".npy").exists()
    assert list(tmp_path.glob("*.tmp*")) == []

    # The cache is not poisoned: the next writer still stores the real payload.
    monkeypatch.undo()
    utils.store_data(target, np.arange(4.0))
    np.testing.assert_allclose(utils.load_data(target), np.arange(4.0))


def test_store_model_does_not_leave_a_partial_pickle_behind(tmp_path, monkeypatch):
    target = tmp_path / "model"
    model = PCA(n_components=2, random_state=0).fit(np.random.RandomState(0).randn(30, 4))

    def failing_dump(value, filename):
        with open(filename, "wb") as handle:
            handle.write(b"truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(utils.joblib, "dump", failing_dump)
    with pytest.raises(OSError):
        utils.store_model_pickle(target, model)

    assert not target.with_suffix(".pkl").exists()
    assert list(tmp_path.glob("*.tmp*")) == []

    monkeypatch.undo()
    utils.store_model_pickle(target, model)
    # Round-trips, which also pins the (value, filename) argument order of joblib.dump.
    np.testing.assert_allclose(utils.load_model(target).components_, model.components_)
