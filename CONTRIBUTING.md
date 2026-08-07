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

We use a two-stage release pipeline: a release candidate (RC) branch for staging, followed by a tag push for production.

### Version Format

- Stable: `0.4.0`
- Alpha: `0.4.0a1`
- Beta: `0.4.0b1`

Tags follow the same format prefixed with `v`: `v0.4.0`, `v0.4.0a1`, `v0.4.0b1`.

The version is defined in [`ituna/__init__.py`](ituna/__init__.py) as `__version__`.

### Step-by-step

#### 1. Prepare the release candidate

```bash
git checkout main && git pull
git checkout -b rc/0.4.0
```

- Bump `__version__` in [`ituna/__init__.py`](ituna/__init__.py)
- Make any last-minute fixes on this branch
- Commit and push:

  ```bash
  git add .
  git commit -m "Bump version to 0.4.0"
  git push -u origin rc/0.4.0
  ```

#### 2. Open a release PR

- [Create a PR](https://github.com/dynamical-inference/ituna/compare) from `rc/0.4.0` to `main`
- Add the `release` label to the PR

#### 3. Verify on TestPyPI

The [`publish` workflow](https://github.com/dynamical-inference/ituna/actions/workflows/publish.yml) will automatically build and upload to [TestPyPI](https://test.pypi.org/project/ituna/). Verify the staging version:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ituna==0.4.0
python -c "import ituna; print(ituna.__version__)"
```

> **Note:** If you push more commits to the RC branch, the TestPyPI version will **not** be automatically updated. Remove and re-add the `release` label to trigger a new upload.

#### 4. Merge

Once tests pass, the staging version looks good, and the PR is reviewed:
- Merge the PR **using rebase merging**
- Delete the `rc/0.4.0` branch

#### 5. Tag and publish

```bash
git checkout main && git pull
git tag v0.4.0
git push origin v0.4.0
```

Pushing the tag triggers the [`publish` workflow](https://github.com/dynamical-inference/ituna/actions/workflows/publish.yml), which builds and uploads the package to [PyPI](https://pypi.org/project/ituna/).

#### 6. Verify

```bash
pip install ituna==0.4.0
python -c "import ituna; print(ituna.__version__)"
```

### Quick Reference

| Step | Action |
|------|--------|
| Branch | `git checkout -b rc/x.y.z` |
| Bump version | Edit `ituna/__init__.py` |
| PR | Open PR to `main`, add `release` label |
| Staging | Verify on TestPyPI |
| Merge | Rebase merge, delete branch |
| Tag | `git tag vx.y.z && git push origin vx.y.z` |
| Verify | `pip install ituna==x.y.z` |
