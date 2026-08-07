# 🐟<b>i</b><span style="color: #3C92ED;">Tuna</span>

[![iTuna](https://img.shields.io/badge/repo-🐟_iTuna-3C92ED?logo=github&logoColor=white)](https://github.com/dynamical-inference/ituna)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://dynamical-inference.github.io/ituna/)
[![PyPI version](https://img.shields.io/pypi/v/ituna.svg?cacheSeconds=3600)](https://pypi.org/project/ituna/)
[![Python versions](https://img.shields.io/pypi/pyversions/ituna.svg?cacheSeconds=3600)](https://pypi.org/project/ituna/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Build](https://github.com/dynamical-inference/ituna/actions/workflows/build.yml/badge.svg)](https://github.com/dynamical-inference/ituna/actions/workflows/build.yml)

**Tune machine learning models for empirical identifiability and consistency**

## Why 🐟i<span style="color: #4D94E8;">Tuna</span>?

Applying machine learning to scientific data analysis often suffers from an **identifiability gap**: many models along the data-to-analysis pipeline lack statistical guarantees about the uniqueness of their learned representations. This means that re-running the same algorithm can yield different embeddings, making downstream interpretation unreliable without manual verification.

Identifiable representation learning addresses this by ensuring models recover representations that are unique up to a known class of transformations (permutation, linear, affine, etc.). However, even theoretically identifiable models need **empirical validation** to confirm they behave consistently in practice.

🐟i<span style="color: #4D94E8;">Tuna</span> closes this gap by providing a lightweight, model-agnostic framework to:

1. Train multiple instances of a model with different random seeds
2. Align their embeddings under the appropriate indeterminacy class
3. Measure how consistent the learned representations are

Think of it as a **unit test for reproducibility** of learned embeddings.

## Features

- **sklearn-compatible**: Works with any transformer implementing `fit`, `transform`, and standard sklearn conventions
- **Built-in indeterminacy classes**:
  - `Identity` - no transformation needed (model is already fully identifiable)
  - `Permutation` - handles sign flips and component reordering (e.g., FastICA)
  - `Linear` - linear transformation alignment (e.g., PCA)
  - `Affine` - linear transformation with intercept (e.g., CEBRA)
- **Consistency scoring**: Quantifies how stable embeddings are across runs
- **Embedding alignment**: Returns aligned embeddings for downstream analysis
- **Flexible backends**: In-memory, disk caching, distributed execution, and DataJoint support

## Installation

```bash
pip install ituna
```

or alternative install from source
```bash
pip install git+https://github.com/dynamical-inference/ituna.git
```

Optional extras:

```bash
pip install "git+https://github.com/dynamical-inference/ituna.git#egg=ituna[datajoint]"  # DataJoint backend for database-backed caching
pip install "git+https://github.com/dynamical-inference/ituna.git#egg=ituna[dev]"        # Development dependencies (pytest, etc.)
```

## Quickstart

```python
import numpy as np
from sklearn.decomposition import FastICA

from ituna import ConsistencyEnsemble, metrics

# Generate sample data
X = np.random.randn(1000, 64)

# Create a consistency ensemble
ensemble = ConsistencyEnsemble(
    estimator=FastICA(n_components=16, max_iter=500),
    consistency_transform=metrics.PairwiseConsistency(
        indeterminacy=metrics.Permutation(),  # FastICA is identifiable up to permutation
        symmetric=False,
        include_diagonal=True,
    ),
    random_states=5,  # Train 5 instances with different seeds
)

# Fit and evaluate
ensemble.fit(X)
print("Consistency score:", ensemble.score(X))

# Get aligned embeddings
emb = ensemble.transform(X)
print("Embedding shape:", emb.shape)
```

## Documentation

Full documentation is available at **[dynamical-inference.github.io/ituna](https://dynamical-inference.github.io/ituna/)**.

- **Quickstart notebook**: [`docs/tutorials/quickstart.ipynb`](docs/tutorials/quickstart.ipynb) - minimal working example
- **Core concepts**: [`docs/tutorials/core.ipynb`](docs/tutorials/core.ipynb) - in-depth walkthrough
- **Backends**: [`docs/tutorials/backends.ipynb`](docs/tutorials/backends.ipynb) - caching and distributed execution
- **sklearn caching**: [`docs/guides/sklearn_caching.ipynb`](docs/guides/sklearn_caching.ipynb) - cache standalone sklearn estimators (`fit`/`transform`/`predict`/`score`)

## Backends

🐟i<span style="color: #4D94E8;">Tuna</span> supports different backends for caching and distributed computation:

```python
from ituna import ConsistencyEnsemble, config, metrics
from sklearn.decomposition import FastICA

ensemble = ConsistencyEnsemble(
    estimator=FastICA(n_components=16, max_iter=500),
    consistency_transform=metrics.PairwiseConsistency(
        indeterminacy=metrics.Permutation(),
    ),
    random_states=10,
)

# Enable disk caching (avoids re-fitting identical models)
with config.config_context(DEFAULT_BACKEND="disk_cache"):
    ensemble.fit(X)

# Distributed execution with multiple workers
with config.config_context(
    DEFAULT_BACKEND="disk_cache_distributed",
    BACKEND_KWARGS={"trigger_type": "auto", "num_workers": 4},
):
    ensemble.fit(X)

# Advanced: route different operations to different backends
# (e.g. distributed estimator fit + locally cached consistency transforms)
# config.register_backend_route(method="fit", model_class=metrics.ConsistencyTransform, backend="disk_cache")
```

### CLI Commands

For large-scale experiments, use the command-line tools:

```bash
# Local distributed backend
ituna-fit-distributed --sweep-name <sweep-uuid> --cache-dir ./cache

# DataJoint backend
ituna-fit-distributed-datajoint --sweep-name <sweep-uuid> --schema-name myschema
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/dynamical-inference/ituna.git
cd ituna
pip install -e .[dev]

# Run tests
pytest tests -v

# Setup pre-commit hooks
pre-commit install
```

For the full development guide — branching conventions, code style, building docs, and the release process — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use 🐟i<span style="color: #4D94E8;">Tuna</span> in your research, please cite:

```bibtex
@software{ituna,
  author = {Schmidt, Tobias and Schneider, Steffen},
  title = {iTuna: Tune machine learning models for empirical identifiability and consistency},
  url = {https://github.com/dynamical-inference/ituna},
  version = {0.1.0},
}
```

## License

🐟i<span style="color: #4D94E8;">Tuna</span> is released under the [MIT License](./LICENSE). If you re-use parts of the iTuna code in your own package, please make sure to copy & paste the contents of the `LICENSE` file into a `NOTICE` in your repository.
