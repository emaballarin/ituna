# Design notes

Working documents: the reasoning behind changes, kept because the reasoning is not recoverable from
the diff. They are **not** user documentation — start from the top-level `README.md` and
`docs/tutorials/` for that — and they are **not** kept current. Each one is a record of what was
believed when it was written, so read the status line before quoting anything from it.

These sit outside `_toc.yml`, so the documentation build does not render them, and outside `ruff`'s
reach, since `ruff.toml` excludes `docs/**`. Code fences here are therefore unformatted and
unchecked; prefer pointing at the source over pasting it.

| note | status |
| ---- | ------ |
| [`gauge-and-spectral-additions.md`](gauge-and-spectral-additions.md) | **Spent.** The build brief for `metrics.Orthogonal` and `ituna/spectral.py`, both shipped at `057034d`. Two things in it still earn their keep: its verified-ground-truth table, and its note that `tools/upstream_parity/probe.py` cannot see a fork-only indeterminacy class. ⚠️ Its rationale for rejecting complex operators was measured **false** and is withdrawn — see `ituna/spectral.py` on chirality, which states the correct reason. |
| [`align-under-the-pinned-group.md`](align-under-the-pinned-group.md) | **Live, and acted on.** Align under the gauge group the objective leaves free, never a superset. Its missing rung shipped as `metrics.ScaledPermutation`; its chance-floor and held-out riders are in that class's docstring. ⚠️ Two figures in it do not generalise: `Orthogonal = 0.475266414` is one unseeded draw, and the `≈0.02` blindness gap is strongly `L`-dependent. Its "free **positive** diagonal" was resolved to **signs free** — the positive-only reading scores `(L − f)/L` on an exact gauge copy, so about 0.5 for random signs. That wording is not a slip, though: an *asymmetric* shape target really does pin sign, and the positive variants of both permutation classes were **considered and declined** rather than missed. `ScaledPermutation`'s docstring has the shape of that hole and the reasons, so it need not be re-derived to be dismissed again. |
