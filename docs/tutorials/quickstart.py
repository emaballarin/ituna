# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,language_info,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ituna
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.19
# ---

# %% [markdown]
# # Quickstart
#
# This notebook shows the smallest end-to-end iTuna workflow.

# %% [markdown]
# ## Optional: Enable disk caching (recommended)
#
# By default, iTuna uses the `in_memory` backend (no caching). For most users, enabling the `disk_cache`
# backend is a free win: repeated runs with the same model + data will **reuse cached fitted models**.
#
# The main reason *not* to use disk caching is if your sklearn estimator cannot be serialized
# (via pickle/joblib or a custom `.save()`/`.load()` mechanism).

# %%
from sklearn.datasets import make_blobs
from sklearn.decomposition import FastICA

import ituna

# Structured synthetic data from sklearn's dataset generators
X, _ = make_blobs(
    n_samples=1000,
    n_features=64,
    centers=8,
    cluster_std=3.0,
    random_state=0,
)

# Optional: persist fitted models across reruns.
ituna.config.set(DEFAULT_BACKEND="disk_cache", CACHE_DIR="./ituna_cache")

ensemble = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=16, random_state=0, max_iter=2000),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
        symmetric=False,
        include_diagonal=True,
    ),
    random_states=5,
)

ensemble.fit(X)
print("Consistency score:", ensemble.score(X))
emb = ensemble.transform(X)
print("Embedding shape:", emb.shape)
print("Scores:\n", ituna.utils.sparse_to_dense(*emb.scores, shape=(5, 5)))

# %%
