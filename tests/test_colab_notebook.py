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
    "uv python install 3.12",
    "poetry env use",
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
    # Since 2026-11-07 the notebook defaulted to the KMH sweep (ADR 0012) with PART = "all", since 2026-11-10 to
    # hier_transfer (ADR 0013). Defaults since 2026-11-11 (ADR 0014): the re-run of the seven fixed paper-1
    # baselines, account A; the second account sets paper1_rerun_b.
    assert 'PART = "paper1_rerun_a"' in first_source
    assert 'PART = "paper1_rerun_b"' in first_source  # documented for the second account
    assert 'MANIFEST = ""' in first_source
    assert "resumes" in first_source


CHECK_RAISES = 'subprocess.run(["make", "campaign-archive-superseded", "CHECK=1"], cwd=REPO_DIR, check=True)'


def _code_sources() -> list[str]:
    return ["".join(c["source"]) for c in _load_notebook()["cells"] if c["cell_type"] == "code"]


def test_archive_cells_run_dry_then_move_then_check_before_the_campaign_run() -> None:
    sources = _code_sources()
    dry = sources.index("!make campaign-archive-superseded DRY=1")
    move = sources.index("!make campaign-archive-superseded")
    check = next(i for i, s in enumerate(sources) if CHECK_RAISES in s and "campaign-list" not in s)
    run_idx = next(i for i, s in enumerate(sources) if "make campaign-run" in s)
    assert dry < move < check < run_idx
    # ADR 0014 safeguard: the check RAISES (not a `!` shell line whose failure "Run all" ignores).
    assert not any(s.strip() == "!make campaign-archive-superseded CHECK=1" for s in sources)


def test_cell_right_before_the_run_raises_on_check_and_asserts_the_part_count() -> None:
    sources = _code_sources()
    run_idx = next(i for i, s in enumerate(sources) if "make campaign-run" in s)
    gate = sources[run_idx - 1]
    assert CHECK_RAISES in gate
    assert '"campaign-list", f"PART={PART}"' in gate and "check=True" in gate
    assert "assert (int(total.group(1)), int(total.group(2))) == (EXPECTED_RUNS, EXPECTED_GROUPS)" in gate


def test_parameters_cell_announces_the_part_and_the_expected_count() -> None:
    params = _code_sources()[0]
    assert 'RERUN_EXPECTED = {"paper1_rerun_a": (24, 8), "paper1_rerun_b": (39, 13)}' in params
    assert "SELECTED PART: {PART}" in params and "expect {EXPECTED_RUNS} runs" in params


def test_rerun_zip_cell_covers_both_parts_after_verify() -> None:
    sources = _code_sources()
    verify = sources.index("!make campaign-verify {MANIFEST_ARG} PART={PART}")
    zip_idx = next(i for i, s in enumerate(sources) if "RERUN_LEAF_GLOBS" in s)
    assert verify < zip_idx
    zip_cell = sources[zip_idx]
    assert '"paper1_rerun_a"' in zip_cell and '"paper1_rerun_b"' in zip_cell
    assert "zip -qr" in zip_cell
    for strategy in ("AdversarialBIM", "KMeansSampling", "KCenterGreedy", "LeastConfidenceDropout",
                     "MarginSamplingDropout", "EntropySamplingDropout", "BALDDropout"):
        assert strategy in zip_cell


def test_rerun_zip_cell_refuses_unless_check_and_verify_pass_and_ships_the_log() -> None:
    zip_cell = next(s for s in _code_sources() if "RERUN_LEAF_GLOBS" in s)
    check, verify, archive = (zip_cell.index(CHECK_RAISES), zip_cell.index('"campaign-verify"'),
                              zip_cell.index("zip -qr {zip_path}"))
    assert check < verify < archive
    assert "if n_ok != EXPECTED_RUNS and not ZIP_PARTIAL:" in zip_cell and "ZIP_PARTIAL = False" in zip_cell
    assert "{name}_{PART}.log" in zip_cell and '("campaign", "failures")' in zip_cell
    assert "!" not in zip_cell.replace("!=", "")  # every step is a raising subprocess call


def test_campaign_cells_use_the_part_parameter() -> None:
    nb = _load_notebook()
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # Since ADR 0012 the campaign cells also take the optional MANIFEST (KMH sweep).
    assert any("make campaign-run {MANIFEST_ARG} PART={PART}" in s for s in sources)
    assert any("make campaign-verify {MANIFEST_ARG} PART={PART}" in s for s in sources)
    params = next(s for s in sources if "MANIFEST_ARG = " in s)
    assert 'MANIFEST = ""' in params and "manifest_kmh_sweep.json" in params


def test_gpu_identity_check_runs_right_after_colab_check() -> None:
    nb = _load_notebook()
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    check_idx = next(i for i, s in enumerate(sources) if "make colab-check" in s)
    run_idx = next(i for i, s in enumerate(sources) if "make campaign-run" in s)
    gpu_idx = next(i for i, s in enumerate(sources) if "results/colab_check/" in s and "run_metadata.json" in s)
    assert check_idx < gpu_idx < run_idx
    # The old post-campaign check assumed a specific run group existed.
    assert not any("shared/random_nq100" in s for s in sources)


def test_progress_cell_present() -> None:
    nb = _load_notebook()
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("campaign.log" in s and "campaign-verify {MANIFEST_ARG} PART={PART}" in s for s in sources)


def test_every_make_target_in_notebook_exists() -> None:
    import re

    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9_-]+):", makefile, flags=re.M))
    nb = _load_notebook()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for target in re.findall(r"!make ([A-Za-z0-9_-]+)", "".join(cell["source"])):
            assert target in targets, f"notebook uses missing make target {target!r}"


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


def test_kernel_python_version_is_not_forced_to_3_12() -> None:
    sources = ["".join(c["source"]) for c in _load_notebook()["cells"] if c["cell_type"] == "code"]
    assert not any("sys.version_info" in s and "raise" in s for s in sources)
    # The venv interpreter is what gets asserted.
    assert any('startswith("3.12")' in s and "poetry" in s for s in sources)
