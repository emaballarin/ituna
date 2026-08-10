# 🐟<b>i</b><span style="color: #3C92ED;">Tuna</span>

[![iTuna](https://img.shields.io/badge/repo-🐟_iTuna-3C92ED?logo=github&logoColor=white)](https://github.com/dynamical-inference/ituna)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://dynamical-inference.github.io/ituna/)
[![PyPI version](https://img.shields.io/pypi/v/ituna.svg?cacheSeconds=3600)](https://pypi.org/project/ituna/)
[![Python versions](https://img.shields.io/pypi/pyversions/ituna.svg?cacheSeconds=3600)](https://pypi.org/project/ituna/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Build](https://github.com/emaballarin/ituna/actions/workflows/build.yml/badge.svg)](https://github.com/emaballarin/ituna/actions/workflows/build.yml)

**Tune machine learning models for empirical identifiability and consistency**

> **This is a fork** of [`dynamical-inference/ituna`](https://github.com/dynamical-inference/ituna),
> carrying bug fixes together with upstream's own unmerged backend-routing and transform-caching
> branch. It targets Python 3.14+, takes its version from git tags, and publishes wheels to GemFury
> rather than PyPI. Fixes here are meant to change what runs, not what a correct run reports;
> [`tools/upstream_parity/`](tools/upstream_parity/README.md) checks that against a clean upstream
> tree and records the result. See [`PROJECT.md`](PROJECT.md) for the rest.

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
    - `Orthogonal` - rotation and reflection only (e.g., an encoder trained to produce isotropic latents)
    - `Linear` - linear transformation alignment (e.g., PCA)
    - `Affine` - linear transformation with intercept (e.g., CEBRA)
- **Consistency scoring**: Quantifies how stable embeddings are across runs
- **Embedding alignment**: Returns aligned embeddings for downstream analysis
- **Operator-level consistency**: Compares and averages the gauge invariants of a learned linear predictor across runs, with no alignment step (`ituna.spectral`)
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

## Operator-level consistency and consensus

`ConsistencyEnsemble` compares _embeddings_, aligning each run to a reference under an indeterminacy
class. When the model also learns a linear transfer operator on the latent — a Koopman-style
predictor `z_{t+1} = K z_t` — there is a second, complementary route: compare invariants of `K` under
the gauge the training objective leaves standing, which needs no alignment step at all.

```python
from ituna import spectral

operators = [...]  # one (L, L) predictor per training run

result = spectral.spectral_consistency(operators)
print("agreement across runs:", result.consistency)

consensus = spectral.spectral_consensus(result)
print("consensus spectrum:", consensus.eigenvalues_mean)
print("across-run scatter:", consensus.eigenvalues_scatter)
```

`spectral_consistency` measures how far apart independently trained runs are. `spectral_consensus`
averages what they agree on, shrinking the run-to-run component of the uncertainty by `sqrt(H)` and
reporting what is left — it does nothing to a bias every run shares, and the docstrings say where
that line falls. Which quantities may legitimately be averaged is decided by the anticollapse half of
the training objective, not by this package:

| anticollapse objective                                            | residual gauge on `z` | invariant under it                                      |
| ----------------------------------------------------------------- | --------------------- | ------------------------------------------------------- |
| decoder / autoencoder, or JEPA with a stop-gradient or EMA target | `GL(L)`               | the spectrum                                            |
| VICReg, or SIGReg against an isotropic Gaussian                   | `O(L)`                | the spectrum, singular values, departure from normality |

Pass `gauge="general_linear"` for the first row: the aggregates that are not invariant there come
back as `None` rather than as numbers that would look perfectly reasonable. The module docstring
carries the full table — including the smaller gauges a product non-Gaussian target produces — and
states what each diagnostic does _not_ establish.

Where the operators themselves have to be compared rather than only their invariants, `ituna.gauge`
carries one run's operator into another's latent frame:

```python
from ituna import gauge, metrics

alignment = metrics.Orthogonal(allow_reflection=True).fit(X=source_latent, y=reference_latent)
common_frame = gauge.pushforward(source_operator, alignment.alignment_)
```

`alignment_` is the single matrix a fitted indeterminacy applies — `predict(X)` is `X @ alignment_` —
and it is what `pushforward` expects. Reach for it rather than for a class's own
attribute: `Permutation` keeps its sign flips in a separate `signs_` and `Linear` stores
scikit-learn's transposed `coef_`, and either substitution yields an operator that is wrong in a way
no invariant above can detect. `Affine` deliberately has no `alignment_` — a translation does not act
on an operator by conjugation, so there is nothing for it to return.

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
