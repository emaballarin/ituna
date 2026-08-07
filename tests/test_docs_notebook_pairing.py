"""Every paired docs notebook must agree with its percent-format .py twin.

Jupyter Book runs with ``execute_notebooks: "off"``, so the committed .ipynb
outputs are exactly what readers see. When a pair drifts, ``jupytext --sync``
resolves it by modification time, which means whichever file a checkout happens
to touch last silently overwrites the other.
"""

import pathlib

import pytest

jupytext = pytest.importorskip("jupytext")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Pairs committed in a diverged state. Resolving one requires the author's intent,
# not a mechanical sync: the notebook holds the published outputs, so adopting the
# .py code without re-executing would show readers code that did not produce the
# results underneath it.
KNOWN_DRIFT = {
    "docs/tutorials/core": (
        "core.py has no jupytext header and is hand-maintained; it differs from core.ipynb "
        "only in import grouping and call formatting, with no semantic or output impact."
    ),
    "docs/tutorials/backends": (
        "backends.ipynb and backends.py were added in the same commit already disagreeing: "
        "random_state 420 vs 42, FastICA max_iter 1000/1001 vs 500/501, a parallel-training cell "
        "present only in the notebook, and a config-reset cell present only in the .py."
    ),
}


def _paired_notebooks():
    params = []
    for notebook in sorted((REPO_ROOT / "docs").rglob("*.ipynb")):
        if not notebook.with_suffix(".py").exists():
            continue
        key = notebook.relative_to(REPO_ROOT).with_suffix("").as_posix()
        marks = [pytest.mark.xfail(strict=True, reason=KNOWN_DRIFT[key])] if key in KNOWN_DRIFT else []
        params.append(pytest.param(notebook, id=key, marks=marks))
    return params


def _code_cells(path: pathlib.Path):
    return [cell.source.strip() for cell in jupytext.read(path).cells if cell.cell_type == "code"]


@pytest.mark.parametrize("notebook", _paired_notebooks())
def test_paired_python_file_matches_notebook(notebook: pathlib.Path):
    """A pair that agrees can be synced in either direction without losing content."""
    assert _code_cells(notebook) == _code_cells(notebook.with_suffix(".py"))
