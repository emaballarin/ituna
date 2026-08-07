## Checklist

> Full development and release documentation: [CONTRIBUTING.md](../CONTRIBUTING.md)

- [ ] The [tests workflow](https://github.com/emaballarin/ituna/actions/workflows/test.yml) passes —
      formatting, linting, the suite, and the upstream parity check.
- [ ] If anything under `ituna/metrics.py` changed, `python tools/upstream_parity/compare.py` still
      exits `0`. If it does not, the change moves a reported number: say so explicitly in the PR
      description and explain why that is intended.
- [ ] If a tutorial changed, its paired `.py` / `.ipynb` twin was reconciled
      (`tests/test_docs_notebook_pairing.py` will tell you).
- [ ] New behaviour has a test that fails without the change.

There is **no version to bump** — `hatch-vcs` derives it from git tags. Releases are cut by tagging
`main`; see [Release Process](../CONTRIBUTING.md#release-process).
