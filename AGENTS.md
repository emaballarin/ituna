# AGENTS.md

See [`PROJECT.md`](./PROJECT.md) for what this package is, how the fork relates to upstream, and the
conventions that apply here.

If you use the [`mindfunnel`](https://github.com/emaballarin/ccplugins/tree/main/plugins/mindfunnel)
plugin: `/mf:spinup` at the start of a session reads the auto-memory and produces a tight "where we
are + next action" brief; `/mf:dump` consolidates state before you close.

## Project-specific guidance

- **Read `tools/upstream_parity/README.md` before changing anything under `ituna/metrics.py`.** That
  file is on the scoring path, and this repository holds the invariant that a fix changes what runs
  and not what a correct run reports. The parity check is how that is verified; run it after any
  change there.

- **Keep the two commit ranges separable.** The backend-routing and transform-caching work came from
  upstream's own branch; the fixes are this fork's. `PROJECT.md` records the boundary. Do not rebase
  across it.

- **There is no version literal.** `hatch-vcs` derives the version from git tags and writes
  `ituna/_version.py` at build time, which is generated and untracked. To release, tag `vX.Y.Z`.

- **Tutorials are paired.** `docs/tutorials/*.py` and their `.ipynb` twins are kept in sync by
  `tests/test_docs_notebook_pairing.py`; edit one and reconcile the other, or the suite will say so.

- **Run what CI runs** before proposing a change:

  ```bash
  ruff format --check .
  ruff check .
  pytest tests/ -v
  ```
