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
#     display_name: ituna-dev
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
#     version: 3.12.12
# ---

# %% [markdown]
# # Caching Standalone sklearn Estimators
#
# iTuna's caching backends are most commonly used through `ConsistencyEnsemble`, but you can also cache
# regular sklearn estimators directly.
#
# This is useful when:
#
# - You call `.fit()` repeatedly during exploratory analysis
# - You run hyperparameter searches where the same configuration may be revisited
# - You want to cache **predict/score/transform** results for expensive models
#
# Under the hood, iTuna routes calls through the configured backend (usually `disk_cache`).
#
# The main reason *not* to use disk caching is if your estimator cannot be serialized
# (via pickle/joblib or a custom `.save()`/`.load()` mechanism).

# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import ituna

import time

# %% [markdown]
# ## 1) Configure a persistent backend
#
# For caching across runs, use `disk_cache` (or `disk_cache_distributed`).

# %%
ituna.config.set(
    DEFAULT_BACKEND="disk_cache",
    CACHE_DIR="./sklearn_cache",
)
print(ituna.config.get_config())

# %% [markdown]
# ## 2) Instance-level caching (`ituna.sklearn.cached`)
#
# `ituna.sklearn.cached(estimator, methods=...)` patches the estimator **instance in-place**
# (type-preserving) so that selected methods route through iTuna's backends.
#
# Supported methods: `fit`, `transform`, `predict`, `score`.

# %%
X, y = make_regression(n_samples=2000, n_features=100, n_informative=5, tail_strength=0.8, noise=0.0, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = MLPRegressor(
    hidden_layer_sizes=(32,),
    max_iter=20000,
    random_state=0,
    n_iter_no_change=10000,
    tol=1e-6,
)

ituna.sklearn.cached(model, methods=["fit", "predict", "score"])

start = time.time()
model.fit(X_train, y_train)
print(f"Fit time: {time.time() - start:.5f}s")

pred = model.predict(X_test)
print("Pred shape:", pred.shape)
start = time.time()
score = model.score(X_test, y_test)
print(f"Score time: {time.time() - start:.5f}s")
print("R2:", score)

# %% [markdown]
# If you create a **new instance** with the same hyperparameters and data, it will reuse cached artifacts
# once you apply `cached(...)` to that instance as well:

# %%
model2 = MLPRegressor(
    hidden_layer_sizes=(32,),
    max_iter=20000,
    random_state=0,
    n_iter_no_change=10000,
    tol=1e-6,
)
# register the model2 to have caching
ituna.sklearn.cached(model2, methods=["fit", "predict", "score"])

start = time.time()
model2.fit(X_train, y_train)
print(f"Fit time: {time.time() - start:.5f}s")

pred = model2.predict(X_test)
print("Pred shape:", pred.shape)
start = time.time()
score = model2.score(X_test, y_test)
print(f"Score time: {time.time() - start:.5f}s")
print("R2:", score)

# %% [markdown]
# ## 3) Global caching (`enable_global_cache`)
#
# If you want caching to apply automatically to **all future instances** of a class, enable a global patch:

# %%
# Register the class MLPRegressor to have caching on the .fit() method
ituna.sklearn.enable_global_cache([MLPRegressor], methods=["fit"])

model3 = MLPRegressor(
    hidden_layer_sizes=(32,),
    max_iter=20000,
    random_state=1,
    n_iter_no_change=10000,
    tol=1e-6,
)

start = time.time()
model3.fit(X_train, y_train)  # cached automatically
print(f"Fit time: {time.time() - start:.5f}s")

print("Global cache status:", ituna.sklearn.get_global_cache_status())

# %%
model4 = MLPRegressor(
    hidden_layer_sizes=(32,),
    max_iter=20000,
    random_state=1,
    n_iter_no_change=10000,
    tol=1e-6,
)

start = time.time()
model4.fit(X_train, y_train)  # cached automatically
print(f"Fit time: {time.time() - start:.5f}s")

# %% [markdown]
# Restore original behavior:

# %%
ituna.sklearn.disable_global_cache([MLPRegressor], methods=["fit"])
print("Global cache status:", ituna.sklearn.get_global_cache_status())

# %%
model5 = MLPRegressor(
    hidden_layer_sizes=(32,),
    max_iter=20000,
    random_state=1,
    n_iter_no_change=10000,
    tol=1e-6,
)

start = time.time()
model5.fit(X_train, y_train)  # cached automatically
print(f"Fit time: {time.time() - start:.5f}s")

# %% [markdown]
# ## Notes / Caveats
#
# - Caching is most effective when your estimator's outputs are deterministic for a given `(params, data)`.
# - For nested iTuna usage (e.g., globally patched sklearn models used inside `ConsistencyEnsemble`),
#   iTuna suspends global patches internally to avoid double-caching loops.

# %% [markdown]
#
