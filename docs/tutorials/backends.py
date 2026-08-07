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
# # Backends: Caching and Distributed Computing
#
# Training multiple model instances for consistency evaluation can be computationally expensive. iTuna provides several backends to help:
#
# 1. **Disk caching** - Avoid re-training identical models
# 2. **Distributed execution** - Train models in parallel across multiple processes
# 3. **DataJoint integration** - Database-backed caching for team collaboration
#
# This tutorial covers how to configure and use these backends.

# %%
from sklearn.datasets import make_blobs
from sklearn.decomposition import FastICA

import ituna

# %%
# Sample data for all examples
X, _ = make_blobs(
    n_samples=1000,
    n_features=20,
    centers=6,
    cluster_std=2.0,
    random_state=42,
)

# %% [markdown]
# ## Default Backend: In-Memory
#
# By default, iTuna uses the `in_memory` backend, which trains all models fresh each time without caching.

# %%
# Check current configuration
print("Current config:", ituna.config.get_config())

# %% [markdown]
# ## Disk Cache Backend
#
# The `disk_cache` backend saves trained models to disk. If you run the same model on the same data again, it loads from cache instead of re-training.
#
# This is extremely useful during exploratory analysis when you're iterating on visualization or downstream analysis without changing the model.

# %%
# Enable disk caching globally
ituna.config.DEFAULT_BACKEND = "disk_cache"

print("Updated config:", ituna.config.get_config())

# %%
# Create and fit an ensemble - models will be cached
ensemble = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=500),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
    ),
    random_states=3,
)

# First run: trains and caches models
print("First run (training):")
ensemble.fit(X)
print(f"Score: {ensemble.score(X):.4f}")

# %%
# Second run: loads from cache (much faster)
print("\nSecond run (loading from cache):")
ensemble2 = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=500),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
    ),
    random_states=3,
)
ensemble2.fit(X)
print(f"Score: {ensemble2.score(X):.4f}")

# %% [markdown]
# ### Cache Invalidation
#
# The cache key is computed from:
# - Model class and all hyperparameters
# - Data hash
# - Random state
#
# If you change **any** hyperparameter, it's treated as a new model and will be trained fresh.

# %%
# Changing max_iter creates a new cache entry
ensemble3 = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=501),  # Different max_iter!
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
    ),
    random_states=3,
)

print("New hyperparameter - trains fresh:")
ensemble3.fit(X)
print(f"Score: {ensemble3.score(X):.4f}")

# %% [markdown]
# ### Custom Cache Directory
#
# By default, models are cached in `./backend_store`. You can customize this:

# %%
# Set custom cache directory
ituna.config.CACHE_DIR = "./my_model_cache"

print(f"Cache directory: {ituna.config.CACHE_DIR}")

# %% [markdown]
# ### Shared Caching
#
# The disk cache is robust to concurrent access, so you can:
#
# - Share a cache directory across multiple notebooks
# - Share a cache with collaborators (e.g., on a network drive)
#
# If someone has already trained a model with the same configuration on the same data, you'll load their cached model instead of re-training.

# %% [markdown]
# ## Using Context Managers
#
# Instead of changing global config, you can use context managers for temporary settings:

# %%
# Reset to default
ituna.config.DEFAULT_BACKEND = "in_memory"

# Use disk cache only within this block
with ituna.config.config_context(DEFAULT_BACKEND="disk_cache"):
    print("Inside context:", ituna.config.get_config()["DEFAULT_BACKEND"])
    ensemble.fit(X)

print("Outside context:", ituna.config.get_config()["DEFAULT_BACKEND"])

# %% [markdown]
# ## Distributed Backend
#
# The `disk_cache_distributed` backend trains models in parallel across multiple processes. This is useful when:
#
# - You have a multi-core machine and want to utilize all cores
# - Training many models (large `random_states` value)
#
# ### Auto Mode
#
# In `auto` mode, iTuna automatically spawns worker processes:

# %%
# Configure distributed backend with auto workers
ituna.config.DEFAULT_BACKEND = "disk_cache_distributed"
ituna.config.BACKEND_KWARGS = {
    "trigger_type": "auto",
    "num_workers": 4,  # Number of parallel processes
}

print("Distributed config:", ituna.config.get_config())

# %%
# Train with 10 random states in parallel
ensemble_parallel = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=500),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
    ),
    random_states=10,
)

ensemble_parallel.fit(X)
print(f"Score: {ensemble_parallel.score(X):.4f}")

# %% [markdown]
# ### Manual Mode (for HPC clusters)
#
# In `manual` mode, iTuna prints a command that you can run on external compute nodes (e.g., SLURM jobs). This is ideal for HPC environments.

# %%
# Configure manual distributed backend
ituna.config.DEFAULT_BACKEND = "disk_cache_distributed"
ituna.config.BACKEND_KWARGS = {
    "trigger_type": "manual",
    "sweep_type": "constant",
    "sweep_name": "my_experiment_sweep",
}

# When you call fit(), it will print the worker command
# and wait for external workers to complete the training

# %% [markdown]
# ### CLI Worker Commands
#
# iTuna provides command-line tools for running workers:
#
# ```bash
# # Local distributed backend
# ituna-fit-distributed --sweep-name <uuid> --cache-dir ./backend_store
#
# # With DataJoint backend
# ituna-fit-distributed-datajoint --sweep-name <uuid> --schema-name myschema
# ```
#
# These can be submitted as SLURM jobs or run on any machine with access to the cache.

# %% [markdown]
# ## DataJoint Backend
#
# For team collaboration with database-backed caching, use the DataJoint backend.
#
# ### Setup
#
# 1. Install DataJoint support:
#    ```bash
#    pip install ituna[datajoint]
#    ```
#
# 2. Configure database credentials in `.env` (see `.env.template`):
#    ```
#    DJ_HOST=your-database-host
#    DJ_USER=your-username
#    DJ_PASS=your-password
#    ```
#
# 3. Use the backend:

# %%
# DataJoint backend configuration (requires setup)
# config.DEFAULT_BACKEND = "datajoint"
# config.BACKEND_KWARGS = {
#     "trigger_type": "auto",
#     "num_workers": 4,
#     "schema_name": "my_ituna_schema",
# }

# %% [markdown]
# ## Advanced: Route Different Operations to Different Backends
#
# Routing becomes valuable when you want to squeeze out *all* avoidable recomputation. Common reasons:
#
# - **Expensive estimator `transform`**: your model is huge, and producing embeddings is a real compute step.
# - **Expensive consistency transforms / indeterminacies**: alignment and scoring is non-trivial (or uses heavy internal models).
# - **Lots of models**: large grid searches where even small overhead per model adds up.
#
# In those workflows, you often want different backend behavior per operation:
#
# - Base estimator `fit` runs via `disk_cache_distributed` in manual mode (register on login node, train via workers)
# - `ConsistencyTransform.fit` runs locally via `disk_cache` (fast and cached)
# - Optional: estimator `transform` calls also use `disk_cache` to avoid recomputing embeddings during collection passes
#
# You can configure this with `register_backend_route(method=..., model_class=..., backend=...)`.

# %%
# Route estimator fit -> distributed/manual, transform fit -> disk_cache
ituna.config.set(
    DEFAULT_BACKEND="disk_cache_distributed",
    CACHE_DIR="./my_model_cache",
    BACKEND_KWARGS={
        "trigger_type": "manual",
        "sweep_type": "constant",
        "sweep_name": "routing_demo_sweep",
        "fit_time_out": 1,
    },
    BACKEND_ROUTES={},
)

# Route all ConsistencyTransform fit calls to local disk cache
ituna.config.register_backend_route(
    method="fit",
    model_class=ituna.metrics.ConsistencyTransform,
    backend="disk_cache",
)

print("Resolved config:", ituna.config.get_config())

ensemble_routed = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=500),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
    ),
    random_states=5,
)

# In manual mode this prints the worker command and waits up to fit_time_out
# Base estimator fits are distributed; consistency transform fit is cached locally via disk_cache.
try:
    ensemble_routed.fit(X)
except TimeoutError:
    print("Expected during registration phase when workers are not running.")

# %% [markdown]
# ### Optional: Cache Estimator `transform` Calls
#
# For large sweeps, collection passes may spend significant time recomputing embeddings (`model.transform(X)`) across many cached estimators.
#
# You can route `transform` to `disk_cache` so repeated transform calls on the same model+data are loaded from cache.
#
# Use this only when transform outputs are deterministic for your estimator and input data.

# %%
from pathlib import Path

# Route FastICA transform calls to disk cache (optional)
ituna.config.register_backend_route(
    method="transform",
    model_class=FastICA,
    backend="disk_cache",
)

# Example flow:
# 1) Fit once with workers running
# 2) Run transform/score repeatedly in analysis notebooks
# 3) Repeated transform calls can load from ./my_model_cache/transforms

transform_cache_dir = Path(ituna.config.CACHE_DIR) / "transforms"
print("Transform cache directory:", transform_cache_dir)
print("Exists now:", transform_cache_dir.exists())

# %% [markdown]
# ## Performance Tips (Fast Reruns)
#
# iTuna's caching backends help you avoid re-training and (optionally) avoid recomputing embeddings.
# For full hyperparameter searches, you can go one step further: **persist the search itself**.
#
# ### Hyperparameter Search: Use Optuna Storage as a \"trial cache\"
#
# If you use [Optuna](https://optuna.org/), configure a persistent storage backend (e.g. SQLite).
# Then, rerunning your script/notebook can **resume** a study and skip already-completed trials entirely.
#
# Minimal pattern:
#
# ```python
# import optuna
# from optuna.trial import TrialState
#
# STORAGE = "sqlite:///ituna_optuna.db"
# STUDY_NAME = "my_sweep"
# N_TRIALS = 50
#
# def objective(trial):
#     # Suggest hyperparameters...
#     # Build + score an iTuna ConsistencyEnsemble...
#     return score
#
# study = optuna.create_study(
#     direction="maximize",
#     study_name=STUDY_NAME,
#     storage=STORAGE,
#     load_if_exists=True,
# )
#
# n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
# if n_complete < N_TRIALS:
#     study.optimize(objective, n_trials=N_TRIALS - n_complete, n_jobs=1)
# else:
#     print(f"Study already complete ({n_complete}/{N_TRIALS}). Skipping optimize().")
# ```
#
# Combine this with iTuna's caching backends and routing to make reruns close to instantaneous:
# - Optuna storage avoids recomputing completed trials.
# - iTuna caching avoids recomputing model fits / transforms inside a trial.

# %% [markdown]
# ## Summary
#
# | Backend | Use Case |
# |---------|----------|
# | `in_memory` | Quick experiments, no caching needed |
# | `disk_cache` | Iterative analysis, avoid re-training |
# | `disk_cache_distributed` | Large sweeps, multi-core/HPC workflows |
# | `datajoint` | Team collaboration, shared database |
#
# You can combine these with backend routes:
#
# ```python
# import ituna
# from sklearn.decomposition import FastICA
#
# ituna.config.set(
#     DEFAULT_BACKEND="disk_cache_distributed",
#     CACHE_DIR="./my_cache",
#     BACKEND_KWARGS={"trigger_type": "manual", "sweep_type": "constant", "sweep_name": "my_sweep"},
# )
#
# # Route consistency transform fits to local disk cache
# ituna.config.register_backend_route(
#     method="fit",
#     model_class=ituna.metrics.ConsistencyTransform,
#     backend="disk_cache",
# )
#
# # Optional: route transform calls to disk cache for repeated collection passes
# ituna.config.register_backend_route(
#     method="transform",
#     model_class=FastICA,
#     backend="disk_cache",
# )
# ```
#
# This pattern gives distributed/manual estimator training while keeping consistency + transform workloads cache-friendly during collection.

# %%
# Reset to defaults for clean state
ituna.config.DEFAULT_BACKEND = "in_memory"
ituna.config.BACKEND_KWARGS = {}
ituna.config.BACKEND_ROUTES = {}
ituna.config.CACHE_DIR = "backend_store"
