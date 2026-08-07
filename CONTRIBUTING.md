# Contributing to 🐟<b>i</b><span style="color: #3C92ED;">Tuna</span>

Thank you for your interest in contributing to 🐟<b>i</b><span style="color: #3C92ED;">Tuna</span>! This guide covers everything from setting up your development environment to publishing a new release.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dynamical-inference/ituna.git
cd ituna

# Install in development mode with all dev dependencies
pip install -e .[dev]

# Setup pre-commit hooks (runs linting and formatting on every commit)
pre-commit install
```

Optional extras:

```bash
pip install -e .[datajoint]  # DataJoint backend support
pip install -r requirements.txt # install bundled third_party dependency
```

## Branching Conventions

We follow a **release candidate (RC) branch model**:

```
main              always releasable, protected
feature/*         individual features or improvements
fix/*             bug fixes
rc/x.y.z          release candidates, created when preparing a release
*-rc              alternative RC naming (e.g., v1.2.3-rc)
```

- **`main`** is the stable branch. Every commit on `main` should be in a releasable state. All changes go through pull requests.
- **`feature/*`** and **`fix/*`** branches are short-lived. Branch off `main`, open a PR back to `main`, and delete after merging.
- **`rc/*`** branches are created when preparing a release. See the Release Process section below for details.

## Making Changes

1. Create a branch from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b feature/my-feature
   ```

2. Make your changes, commit, and push:

   ```bash
   git add .
   git commit -m "Add my feature"
   git push -u origin feature/my-feature
   ```

3. Open a pull request to `main` on GitHub. The [build workflow](https://github.com/dynamical-inference/ituna/actions/workflows/build.yml) will run automatically to check tests, linting, and that the package builds.

4. Once the PR is reviewed and all checks pass, merge it.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting, configured in [`ruff.toml`](ruff.toml). Pre-commit hooks run these automatically, but you can also run them manually:

```bash
# Check formatting
ruff format --check .

# Auto-format
ruff format .

# Run linter
ruff check .

# Run linter with auto-fix
ruff check --fix .
```

Key style rules:
- Line length: 160 characters
- Import sorting: single-line imports, sorted within sections
- Python support: 3.8 - 3.14

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ -v --cov=ituna
```

### CI Testing Strategy

The CI pipeline tests different Python versions depending on the context:

- **Regular PRs**: Tests only Python **3.11** for fast feedback
- **Release candidate PRs** (branches starting with `rc/` or ending with `-rc`): Tests the full matrix **3.8 - 3.14**
- **Push to main**: Tests the full matrix **3.8 - 3.14**

This strategy keeps PR checks fast while ensuring comprehensive testing before releases.

## Documentation

The documentation is built using [Jupyter Book](https://jupyterbook.org/) version 1.x (version 2.x has a completely different build system and is not supported).

### Docs Structure

- `docs/tutorials/` -- tutorial notebooks
- `_config.yml` -- Jupyter Book configuration
- `_toc.yml` -- table of contents

### Local Build

```bash
# Install documentation dependencies (includes jupyter-book and jupytext)
pip install -e ".[docs]"

# Build the docs from the project root
jupyter-book build .

# The HTML output will be in _build/html/
# Open in browser:
open _build/html/index.html   # macOS
xdg-open _build/html/index.html  # Linux
```

### Editing Notebooks (Jupytext, Recommended)

We keep tutorial/demo notebooks paired as:

- `*.ipynb` (rendered by Jupyter Book)
- `*.py` in **percent** format (easy to diff/review)

To edit a notebook:

1. Edit the corresponding `*.py` percent file.
2. Sync back to the `*.ipynb`:

```bash
jupytext --sync docs/tutorials/<notebook>.ipynb
```

To pair a new notebook (creates/updates the `*.py` alongside the `*.ipynb`):

```bash
jupytext --set-formats ipynb,py:percent docs/tutorials/new_notebook.ipynb
jupytext --sync docs/tutorials/new_notebook.ipynb
```

### Local Server

To serve the docs locally with live preview:

```bash
cd _build/html
python -m http.server 8080
# Then open http://localhost:8080 in your browser
```

### Using Docker (Recommended)

For a consistent build environment with auto-rebuild on file changes:

```bash
./build.sh
```

This will:
1. Build the Docker image with all dependencies
2. Mount the current directory into the container
3. Build the docs and start a server at http://localhost:8000
4. Watch for changes to `.ipynb`, `.md`, `.yml`, and `.py` files and rebuild automatically

Press Ctrl+C to stop.

### Deploying to GitHub Pages

```bash
# Install required tools
pip install -e ".[docs]"

# Build and publish
jupyter-book build .
ghp-import -n -p -f _build/html
```

This creates/updates the `gh-pages` branch and pushes it to GitHub.

**Initial setup** (one-time): Go to repository Settings > Pages, set the source to the `gh-pages` branch (root `/`). The docs will be at: https://dynamical-inference.github.io/ituna/

## Release Process

The version is **derived from git tags** by `hatch-vcs`; there is no version literal anywhere in the
tree. `ituna/_version.py` is written at build time and is not tracked.

- A commit on `main` builds as `X.Y.Z.devN+g<sha>`, where `X.Y.Z` is one patch above the last tag.
- A commit that carries an annotated tag `vX.Y.Z` builds as exactly `X.Y.Z`.

Wheels are published to **GemFury**, not PyPI — the `ituna` name on PyPI belongs to upstream. Every
push to `main` uploads a dev wheel; a tag additionally creates a GitHub release.

### Version format

Stable `0.4.0`, alpha `0.4.0a1`, beta `0.4.0b1`. Tags carry a `v` prefix: `v0.4.0`, `v0.4.0a1`.

### Cutting a release

```bash
git checkout main && git pull

# Make sure the suite and the parity check pass first
ruff format --check . && ruff check . && pytest tests/ -v
python tools/upstream_parity/compare.py

git tag -s v0.4.0 -m "Release 0.4.0"
git push origin main v0.4.0
```

Pushing the tag runs the [build workflow](https://github.com/emaballarin/ituna/actions/workflows/build.yml),
which builds a clean wheel, uploads it to GemFury and opens a GitHub release.

### Verifying

```bash
pip install --index-url https://fury.ballarin.cc/pypi ituna==0.4.0
python -c "import ituna; print(ituna.__version__)"
```

> **Dev wheels sort above the release they follow.** `0.4.1.devN` is a higher version than `0.4.0`
> under PEP 440, so a resolver allowing pre-releases will prefer a dev wheel over the last tagged
> one. Pin exactly when that matters.

### Quick reference

| Step | Action |
|------|--------|
| Check | `ruff format --check . && ruff check . && pytest tests/ -v` |
| Parity | `python tools/upstream_parity/compare.py` |
| Tag | `git tag -s vx.y.z -m "Release x.y.z"` |
| Publish | `git push origin main vx.y.z` |
| Verify | `pip install --index-url https://fury.ballarin.cc/pypi ituna==x.y.z` |
