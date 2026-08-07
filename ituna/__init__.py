"""
ituna - Tune machine learning models for empirical identifiability and consistency
"""

from ituna import _backends
from ituna import config
from ituna import estimator
from ituna import metrics
from ituna import sklearn
from ituna import utils
from ituna.estimator import ConsistencyEnsemble

__all__ = [
    "ConsistencyEnsemble",
    "config",
    "estimator",
    "metrics",
    "sklearn",
    "utils",
    "_backends",
]


try:
    # Written at build time by hatch-vcs from the git tag; generated, not tracked.
    from ituna._version import __version__
except ImportError:  # a source tree that has never been built
    __version__ = "0.0.0.dev0"
