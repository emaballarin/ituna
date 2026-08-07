# PROJECT.md — `ituna` (fork)

Project-specific context for anyone — person or agent — working in this repository.

## What the package is

`ituna` measures **empirical identifiability**: retrain the same estimator under several seeds, align
the resulting embeddings under a declared *indeterminacy class*, and score how consistent they are.
Making that class a first-class object — `Identity`, `Permutation`, `Linear`, `Affine` — turns
"identifiable up to what?" into a parameter rather than an assumption.

Public surface: `ConsistencyEnsemble` (sklearn-compatible, wraps any estimator),
`metrics.PairwiseConsistency`, the four indeterminacy classes, and a pluggable execution backend
(in-memory, disk cache, DataJoint).

Upstream is [`dynamical-inference/ituna`](https://github.com/dynamical-inference/ituna) by Tobias
Schmidt and Steffen Schneider (Helmholtz Munich), MIT-licensed. This repository is a fork of it.

## What the fork adds

Two things, and they are worth keeping separate:

| | range | scope | origin |
| --- | --- | --- | --- |
| backend routing and transform caching | `5aada31..4858961` | 36 files, +4148 / −338 | upstream's own branch, unmerged there |
| bug fixes | `4858961..HEAD`, 9 commits | 17 files, +758 / −80 | this fork |

The fix stratum is 287 changed lines across 7 library files, roughly 460 lines of added tests, and
some documentation. Only `metrics.py` (54 lines) touches the scoring path.

**Keep the two ranges separable** — do not rebase them together. They answer different questions
about where a behaviour came from, and that is cheap to preserve and expensive to reconstruct.

Roughly what the fixes address, newest first: a `PairwiseConsistency.transform(..., source_id=…)`
attribute typo that made a documented public call always raise; inverted argument order when storing
torch tensors; DataJoint backend construction being load-bearing (a pooling change reverted); cache
suspension not holding across threads and non-atomic cache writes; cached method calls keyed on
hyperparameters alone; `ituna.sklearn` patches outliving their targets; and drift between paired
docs notebooks and their `.py` twins.

## The invariant

**Fixes change what runs, not what a correct run reports.** Crashes, caching and concurrency are fair
game; the value a working call returns is not. Anyone adopting this fork should be able to keep
numbers they have already computed.

Where a fix genuinely must change a reported value, it is stated in the open rather than folded in
silently — and where upstream's default is unhelpful, prefer keeping the default and documenting the
hazard, since a caller can always pass something better explicitly.

This is checked rather than asserted: `tools/upstream_parity/` runs the same battery against a clean
upstream tree and this one and compares the results exactly. See
[`tools/upstream_parity/README.md`](tools/upstream_parity/README.md) for the method and the current
result.

## Layout

```
ituna/                  metrics.py (scoring + indeterminacy classes), estimator.py
                        (ConsistencyEnsemble), sklearn.py, config.py, _cache_guard.py,
                        _backends/{in_memory,disk_cache,datajoint}
tests/                  pytest
tools/upstream_parity/  the parity check and its result
docs/                   jupyter-book sources; tutorials are paired .py / .ipynb, kept in sync by a test
slurm/                  cluster launch scripts
third_party/            vendored wheels for optional backends
```

## Conventions

- **Python 3.14+.** The floor is deliberate and the classifiers, CI matrix and `requires-python` all
  agree on it.
- **The version comes from git tags** via `hatch-vcs`; there is no version literal to edit. Tagging
  `vX.Y.Z` produces a clean `X.Y.Z`; any other commit produces a `.devN` version.
- **Wheels go to GemFury**, on every push to `main` and on tags. Publishing to PyPI is deliberately
  not wired — that namespace belongs to upstream.
- **Formatting and linting are `ruff`**, configured by `ruff.toml`, and CI checks both.
- `scratch/` is a local workspace and is never committed.
