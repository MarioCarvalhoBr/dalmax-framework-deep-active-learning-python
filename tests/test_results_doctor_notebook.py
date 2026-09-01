"""Tests for notebooks/results_doctor.ipynb.

Mirrors `tests/test_colab_notebook.py`'s structural-only approach (no
execution -- no GPU/Drive available here, `.claude/rules/data-safety.md`):
valid nbformat 4 JSON, no baked-in outputs, and the key CLI commands present
in the expected order. Unlike `colab_runbook.ipynb`, this notebook is
CPU-only (verify/migrate are pure filesystem operations), so no GPU
accelerator metadata is required or checked.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "results_doctor.ipynb"

# Key commands that must appear, in this relative order, across the
# notebook's code cells.
EXPECTED_COMMAND_ORDER = [
    "drive.mount",
    "poetry install",
    "make colab-setup",
    "results_doctor migrate-legacy --root results/ablations",
    "results_doctor migrate-legacy --root results/ablations --apply",
    "results_doctor verify --root results/ablations",
    "make ablation-report METHOD=rnhal",
    "make ablation-report METHOD=texhal",
]


def _load_notebook() -> dict:
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_notebook_is_valid_json_nbformat4() -> None:
    nb = _load_notebook()
    assert nb["nbformat"] == 4


def test_notebook_has_kernelspec_python3() -> None:
    nb = _load_notebook()
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


def test_no_cell_has_outputs() -> None:
    nb = _load_notebook()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == [], "code cell has baked-in outputs"
            assert cell.get("execution_count") is None


def test_every_code_cell_has_nonempty_source() -> None:
    nb = _load_notebook()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert source.strip() != "", "code cell has empty source"


def test_parameters_cell_has_expected_defaults() -> None:
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    first_source = "".join(code_cells[0]["source"])
    assert "DRIVE_ROOT" in first_source
    assert "METHOD" in first_source
    assert "APPLY_MIGRATION = False" in first_source


def test_migration_cell_is_guarded_by_apply_migration_flag() -> None:
    nb = _load_notebook()
    full_source = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "if APPLY_MIGRATION:" in full_source
    assert "--apply" in full_source


def test_key_commands_appear_in_expected_order() -> None:
    nb = _load_notebook()
    full_source = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )

    positions = []
    search_from = 0
    for command in EXPECTED_COMMAND_ORDER:
        idx = full_source.find(command, search_from)
        assert idx != -1, f"expected command {command!r} not found in notebook"
        positions.append(idx)
        search_from = idx + len(command)

    assert positions == sorted(positions), (
        "expected commands are not in the required order: "
        f"{EXPECTED_COMMAND_ORDER}"
    )


def test_notebook_has_no_gpu_accelerator_metadata() -> None:
    # This is a CPU-only notebook (filesystem checks + directory moves) --
    # unlike colab_runbook.ipynb, it must not claim a GPU accelerator.
    nb = _load_notebook()
    assert "accelerator" not in nb["metadata"]
