"""Tests for notebooks/colab_runbook.ipynb.

This notebook is the runnable companion to `COLAB_RUNBOOK.md` -- these tests
don't execute it (no GPU/Drive available here, see `.claude/rules/data-
safety.md`), they just check its structure and that it hasn't drifted into
an unusable state: valid JSON, no baked-in outputs, and the runbook's key
commands present in the expected order.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "colab_runbook.ipynb"

# The runbook commands that must appear, in this relative order, somewhere
# across the notebook's code cells (COLAB_RUNBOOK.md sections 1, 3, 4, 5, 6).
EXPECTED_COMMAND_ORDER = [
    "drive.mount",
    "poetry install",
    "make colab-setup",
    "make colab-check",
    # The campaign (section 6) is the only execution path.
    "make campaign-list",
    "make campaign-run",
    "make campaign-verify",
    "make campaign-report",
]


def _load_notebook() -> dict:
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_notebook_is_valid_json_nbformat4() -> None:
    nb = _load_notebook()
    assert nb["nbformat"] == 4


def test_notebook_has_at_least_20_cells() -> None:
    nb = _load_notebook()
    assert len(nb["cells"]) >= 20


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


def test_parameters_cell_defines_drive_root() -> None:
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    first_source = "".join(code_cells[0]["source"])
    assert "DRIVE_ROOT" in first_source


def test_parameters_cell_defines_campaign_part() -> None:
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    first_source = "".join(code_cells[0]["source"])
    assert 'PART = "all"' in first_source


def test_campaign_cells_use_the_part_parameter() -> None:
    nb = _load_notebook()
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("make campaign-run PART={PART}" in s for s in sources)
    assert any("make campaign-verify PART={PART}" in s for s in sources)


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


def test_notebook_has_colab_gpu_metadata() -> None:
    nb = _load_notebook()
    assert nb["metadata"]["accelerator"] == "GPU"
    # Hardware-agnostic: no specific GPU model is pinned in the notebook metadata.
    assert "gpuType" not in nb["metadata"]["colab"]
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
