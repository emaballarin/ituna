# %% [markdown]
# # Core Concepts
#
# This tutorial covers the fundamental building blocks of iTuna:
#
# 1. **ConsistencyEnsemble** - The main class for evaluating model consistency
# 2. **Indeterminacy classes** - How to handle different types of model ambiguity
# 3. **Consistency scoring** - Measuring and interpreting consistency
# 4. **Working with embeddings** - Accessing aligned representations

# %%
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import ituna

# %% [markdown]
# ## ConsistencyEnsemble
#
# `ConsistencyEnsemble` is iTuna's main class. It wraps any sklearn-compatible transformer and:
#
# 1. Creates multiple clones of the base estimator
# 2. Fits each clone with a different random seed
# 3. Aligns the resulting embeddings under the specified indeterminacy
# 4. Computes consistency scores across all model pairs
#
# ### Requirements for the base estimator
#
# Your model must follow the sklearn API:
# - Implement `fit(X)` and `transform(X)` methods
# - Be clonable via `sklearn.base.clone()`
# - Accept a `random_state` parameter (for reproducibility)
#
# Most sklearn transformers work out of the box. For custom models, inherit from `sklearn.base.TransformerMixin` and `sklearn.base.BaseEstimator`.

# %% [markdown]
# ## Indeterminacy Classes
#
# Different representation learning algorithms are identifiable up to different classes of transformations. iTuna provides four built-in indeterminacy classes:
#
# | Class | Transformation | Example Models |
# |-------|---------------|----------------|
# | `Identity` | None (exact match) | Fully identifiable models |
# | `Permutation` | Sign flips + reordering | FastICA, sparse coding |
# | `Linear` | Linear transformation | PCA, factor analysis |
# | `Affine` | Linear + intercept | CEBRA, autoencoders |
#
# Choosing the correct indeterminacy class is crucial: if you pick one that's too restrictive, consistent models will appear inconsistent. If you pick one that's too permissive, you may miss genuine inconsistencies.

# %% [markdown]
# ### Example: FastICA with Permutation indeterminacy
#
# Independent Component Analysis (ICA) recovers independent sources from mixed signals. The recovered components are identifiable up to **permutation and sign flips** - we don't know which component is which, or whether it's flipped.

# %%
# Generate synthetic ICA data
np.random.seed(42)
n_samples = 2000
n_sources = 5

# Create independent sources
t = np.linspace(0, 10, n_samples)
sources = np.column_stack(
    [
        np.sin(2 * t),  # Sinusoid
        np.sign(np.sin(3 * t)),  # Square wave
        np.random.laplace(size=n_samples),  # Super-Gaussian
        np.random.uniform(-1, 1, n_samples),  # Uniform
        (t % 1) - 0.5,  # Sawtooth
    ]
)

# Mix the sources
mixing_matrix = np.random.randn(n_sources, n_sources)
X_ica = sources @ mixing_matrix.T
X_ica += 0.1 * np.random.randn(*X_ica.shape)  # Add noise

print(f"Data shape: {X_ica.shape}")

# %%
# Create a FastICA model
ica_model = FastICA(n_components=5, max_iter=1000)

# Wrap in ConsistencyEnsemble with Permutation indeterminacy
ica_ensemble = ituna.ConsistencyEnsemble(
    estimator=ica_model,
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
        symmetric=False,
        include_diagonal=True,
    ),
    random_states=5,  # Train 5 models with different seeds
)

# Fit the ensemble
ica_ensemble.fit(X_ica)

# Get consistency score
score = ica_ensemble.score(X_ica)
print(f"ICA Consistency score: {score:.4f}")

# %% [markdown]
# ### Example: PCA with Linear indeterminacy
#
# PCA finds orthogonal directions of maximum variance. The principal components are identifiable up to **linear transformations** (rotations and reflections within eigenspaces of equal variance).

# %%
# Generate data for PCA
np.random.seed(42)
X_pca = np.random.randn(1000, 20)

# Create PCA model
pca_model = PCA(n_components=5)

# Wrap in ConsistencyEnsemble with Linear indeterminacy
pca_ensemble = ituna.ConsistencyEnsemble(
    estimator=pca_model,
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Linear(),
        symmetric=False,
        include_diagonal=True,
    ),
    random_states=5,
)

pca_ensemble.fit(X_pca)
score = pca_ensemble.score(X_pca)
print(f"PCA Consistency score: {score:.4f}")

# %% [markdown]
# ## Understanding Consistency Scores
#
# The consistency score measures how well embeddings from different model instances align after accounting for the indeterminacy.
#
# - **Score = 1.0**: Perfect consistency - all models produce equivalent embeddings
# - **Score close to 1.0**: High consistency - models are reliably converging to the same solution
# - **Low score**: Models are finding different solutions, suggesting the representation may not be reproducible
#
# The score is computed as the R² between embeddings after fitting the indeterminacy transformation.

# %% [markdown]
# ## Working with Embeddings
#
# After fitting, you can access the embeddings and alignment information via `transform()`:

# %%
# Get embeddings with alignment metadata
embeddings = ica_ensemble.transform(X_ica)

print(f"Mean aligned embedding shape: {embeddings.shape}")
print(f"Number of individual model embeddings: {len(embeddings.embeddings)}")

# Access individual embeddings
for i, emb in enumerate(embeddings.embeddings):
    print(f"  Model {i} embedding shape: {emb.shape}")

# %%
# Access pairwise consistency scores
pairs, scores = embeddings.scores

print("\nPairwise consistency scores:")
for (i, j), s in zip(pairs, scores):
    print(f"  Model {i} -> Model {j}: {s:.4f}")


# %%
# or use built in utils to convert to dense matrix
score_matrix = ituna.utils.sparse_to_dense(
    *embeddings.scores,
    shape=(len(embeddings.embeddings), len(embeddings.embeddings)),
)
print("Scores:\n", score_matrix)

# %% [markdown]
# ## PairwiseConsistency Options
#
# The `PairwiseConsistency` transform has several options:
#
# - **`indeterminacy`**: The indeterminacy class to use for alignment
# - **`symmetric`**: If `True`, also compute j→i alignments (default: `False`)
# - **`include_diagonal`**: If `True`, include self-alignments i→i (default: `True`)

# %%
# Example with symmetric=True
symmetric_ensemble = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=1000),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=ituna.metrics.Permutation(),
        symmetric=True,  # Include both i->j and j->i
        include_diagonal=False,  # Exclude self-alignments
    ),
    random_states=3,
)

symmetric_ensemble.fit(X_ica)
emb = symmetric_ensemble.transform(X_ica)

pairs, scores = emb.scores
print(f"Number of pairwise comparisons: {len(pairs)}")
for (i, j), s in zip(pairs, scores):
    print(f"  Model {i} <-> Model {j}: {s:.4f}")

# %% [markdown]
# ## Custom Indeterminacy Classes
#
# You can also use any sklearn regressor as a custom indeterminacy class. The regressor is fitted to align embeddings from one model to another.
#
# For example, to use Ridge regression:

# %%
from sklearn.linear_model import Ridge

# Use Ridge regression as indeterminacy
ridge_ensemble = ituna.ConsistencyEnsemble(
    estimator=FastICA(n_components=5, max_iter=1000),
    consistency_transform=ituna.metrics.PairwiseConsistency(
        indeterminacy=Ridge(alpha=0.1),  # Any sklearn regressor works
        symmetric=False,
    ),
    random_states=3,
)

ridge_ensemble.fit(X_ica)
print(f"Consistency score with Ridge: {ridge_ensemble.score(X_ica):.4f}")

# %% [markdown]
# ## Summary
#
# Key takeaways:
#
# 1. **`ConsistencyEnsemble`** wraps any sklearn transformer to evaluate consistency
# 2. Choose the **indeterminacy class** based on your model's theoretical identifiability:
#    - `Permutation` for ICA-like models
#    - `Linear` for PCA-like models
#    - `Affine` for models like CEBRA
# 3. **Consistency scores** close to 1.0 indicate reproducible representations
# 4. Use **`transform()`** to access aligned embeddings and detailed pairwise scores
#
# Next, check out the [Backends tutorial](backends.ipynb) to learn about caching and distributed computation.
