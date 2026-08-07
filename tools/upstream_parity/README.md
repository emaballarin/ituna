# Upstream parity check

This fork carries bug fixes on top of upstream
[`dynamical-inference/ituna`](https://github.com/dynamical-inference/ituna). Fixes should change
**what runs** — crashes, caching, concurrency — without changing **what a correct run reports**.
Anyone adopting the fork needs to know whether numbers they have already computed will move.

`compare.py` answers that by measurement rather than by reading the diff.

## Running it

```bash
python tools/upstream_parity/compare.py                    # vs upstream main at 5aada31, offline
python tools/upstream_parity/compare.py --mode upstream    # clone from upstream instead (needs network)
python tools/upstream_parity/compare.py --ref 4858961      # compare against the feature-branch merge
python tools/upstream_parity/compare.py --verbose          # list every explained difference
python tools/upstream_parity/compare.py --keep             # leave the temporary trees for inspection
```

Exit `0` — every difference is accounted for by a verified mechanism. `1` — at least one is not, or
a cell could not be compared. `3` — the comparison was vacuous and its result must not be read as a
pass. Takes about ten seconds.

## The verdict rule

A difference is a failure **unless it is explained by a named mechanism, and the mechanism is checked
rather than asserted**. Two are recognised:

- **`source_id`** — counted as explained only when the reference side really is the `AttributeError`
  below. If upstream ever starts returning a value there, this stops being a repaired crash and
  becomes a real difference, and the check will say so.
- **the diagonal**, on `include_diagonal=True` — counted as explained only when the observed pair
  satisfies the identity below *exactly*, per cell. A drift in that relation fails.

Everything else fails, including a cell that could not be compared in both trees.

## Method

`compare.py` materialises the reference tree at a pinned commit, then runs `probe.py` **once per tree
in a separate interpreter**, from a neutral working directory with `PYTHONPATH` stripped. Each probe
puts its tree at the front of `sys.path` and then verifies that `ituna.__file__` actually resolved
underneath it.

Values are compared as **hex floats** and **sha256 digests**, never as printed decimals — two floats
print identically long before their last bits agree. Per configuration: the consistency score, the
selected `reference_id`, a digest of the mean embedding, and digests of the fitted pair indices and
their per-pair scores.

The battery is 4 indeterminacy classes × 4 consistency regimes × `K ∈ {2,3,5}` × `symmetric ∈
{False,True}`, under both `include_diagonal` settings — 192 configurations per tree. The `rotation`
regime is the discriminating one: consistent under `Linear`/`Affine` and not under `Permutation`.

**If both trees hash the same `metrics.py`, the comparison refuses to report agreement** and exits
`3`. A differential check can pass because both sides resolved to the same code — an installed
package shadowing a local tree, a ref that did not check out, a stray `PYTHONPATH`. Every computation
runs, nothing raises, and the result is a perfect and meaningless match. Both arms therefore print
their resolved path, their `metrics.py` digest and their numpy version alongside the verdict.

## Result, 7 August 2026

Against upstream `5aada31`, on Python 3.14 with numpy 2.5.0. Arms confirmed distinct
(`metrics.py` sha256 `2e0f1230…` reference, `e6aebe3b…` fork).

| | cells | identical | explained | unexplained | incomparable |
| --- | --- | --- | --- | --- | --- |
| whole battery | 192 | 134 | 58 | **0** | 0 |
| **`include_diagonal=False`** | **96** | 80 | 16 | **0** | 0 |

Of the 58 explained, 42 are the diagonal identity and 16 are the repaired `source_id` crash.
**Every numeric field agrees across all 96 configurations** on the default path — score,
`reference_id`, mean embedding, pair indices, per-pair scores. The two mechanisms:

**1. `source_id` — a repaired crash.** Upstream's `PairwiseConsistency._get_indeterminancy` reads an
attribute name that `_fit` never sets, so the documented call
`transform([X], source_id=k)` raises `AttributeError` for every input. Nothing upstream exercises it
and `ConsistencyEnsemble` never reaches it. A call that always raised was not producing a number, so
repairing it cannot have moved one.

**2. `include_diagonal=True` — an exact, closed-form offset.** Upstream's `_score` averages
self-alignments, which score 1.0 by construction; this fork ignores the diagonal unconditionally,
matching that method's own docstring (*"Mean consistency score across all estimator pairs, ignoring
the diagonal"*). `include_diagonal` governs which alignment models are **fitted**, never what the
score averages over. The offset is exactly

$$\text{upstream} = (1-f)\cdot\text{fork} + f, \qquad f = \tfrac{1}{K}\ (\texttt{symmetric=False}),\quad f = \tfrac{2}{K+1}\ (\texttt{symmetric=True})$$

verified to 1e-12 across all four indeterminacy classes, three ensemble sizes and both `symmetric`
settings. Note that `include_diagonal` **defaults to `False` in both trees** — the default is
unchanged; only an explicit `True` is affected. Upstream's own `README.md` and `example.py` pass it.

## What a pass establishes

That these configurations agree. It is evidence, not proof: the battery is `probe.py` and nothing
else. It says nothing about the DataJoint or disk-cache backends, which it does not exercise.
