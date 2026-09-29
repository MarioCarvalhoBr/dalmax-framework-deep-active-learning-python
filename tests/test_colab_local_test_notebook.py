"""Structure tests for notebooks/colab_local_test.ipynb (diagnostic notebook).

The notebook is not executed here (no Colab/Drive/GPU); we only check that it
stays a well-formed, output-free diagnostic with the expected probes.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "colab_local_test.ipynb"


def _load() -> dict:
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _code_sources() -> list[str]:
    return ["".join(c["source"]) for c in _load()["cells"] if c["cell_type"] == "code"]


def test_valid_nbformat4_python3_kernel() -> None:
    nb = _load()
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert "accelerator" not in nb["metadata"]


def test_no_outputs_and_nonempty_code_cells() -> None:
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            assert "".join(cell["source"]).strip() != ""


def test_drive_mount_is_inside_try_block() -> None:
    src = next(s for s in _code_sources() if "drive.mount" in s)
    assert src.index("try:") < src.index("drive.mount")


def test_probes_use_record_helper() -> None:
    sources = _code_sources()
    assert "def record(" in sources[0]
    assert sum("record(" in s for s in sources[1:]) >= 6


def test_summary_cell_is_last() -> None:
    last = _load()["cells"][-1]
    assert last["cell_type"] == "code"
    assert "RECOMMENDATION" in "".join(last["source"])
