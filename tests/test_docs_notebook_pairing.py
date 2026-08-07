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

# docs/tutorials/core and docs/tutorials/backends were introduced with their .py and
# .ipynb already disagreeing. Both .py files were regenerated from their notebook,
# because the notebook carries the outputs Jupyter Book publishes and re-executing to
# match the .py was not possible here. What that discarded, for the record:
#
#   core.py     -- had no jupytext header at all and was hand-maintained. Differences
#                  were confined to import grouping and call formatting, so nothing of
#                  substance was lost.
#
#   backends.py -- used random_state=42 (notebook: 420) and FastICA max_iter=500/501
#                  (notebook: 1000/1001), and its stale header claimed the "ituna"
#                  kernel on Python 3.10.19 while the notebook records
#                  "Python 3 (ipykernel)" on 3.12.12. It also ended with a cell absent
#                  from the notebook:
#
#                      # Reset to defaults for clean state
#                      ituna.config.DEFAULT_BACKEND = "in_memory"
#                      ituna.config.BACKEND_KWARGS = {}
#                      ituna.config.BACKEND_ROUTES = {}
#                      ituna.config.CACHE_DIR = "backend_store"
#
#                  That cell is worth restoring in the notebook: the tutorial assigns
#                  ituna.config global state throughout and otherwise leaves it dirty
#                  for anything the reader runs afterwards. Adding it needs a re-execution
#                  pass, so it is left to the notebook's author.


def _paired_notebooks():
    params = []
    for notebook in sorted((REPO_ROOT / "docs").rglob("*.ipynb")):
        if not notebook.with_suffix(".py").exists():
            continue
        key = notebook.relative_to(REPO_ROOT).with_suffix("").as_posix()
        params.append(pytest.param(notebook, id=key))
    return params


def _code_cells(path: pathlib.Path):
    return [cell.source.strip() for cell in jupytext.read(path).cells if cell.cell_type == "code"]


@pytest.mark.parametrize("notebook", _paired_notebooks())
def test_paired_python_file_matches_notebook(notebook: pathlib.Path):
    """A pair that agrees can be synced in either direction without losing content."""
    assert _code_cells(notebook) == _code_cells(notebook.with_suffix(".py"))
