"""Tests for scripts/ablations/run_ablation_gpu_{0,1}.sh.

Three things are checked, all without spending any GPU time or touching the
real results/ tree:

1. Both scripts are syntactically valid bash (`bash -n`).
2. The `SKIP_EXISTING`/`DRY_RUN` logic added for Colab idempotency
   (`.claude/rules/data-safety.md` "results/ is append-only") behaves
   correctly for both `METHOD=rnhal` (default) and `METHOD=texhal`: given a
   temp results tree with exactly one fake `results.json` already present,
   running the script under `DRY_RUN=1` produces exactly one "SKIP (already
   completed)" line and one "DRY-RUN: ..." line per remaining (config, seed)
   triple -- and never invokes `poetry run python trainer.py` or
   ExperimentNotifier for real.
3. An invalid `METHOD` value exits 2 without running anything.

`DRY_RUN=1` makes this safe to run from the fast test suite: the script
still executes end-to-end (loop, glob checks, ExperimentNotifier guard) but
never shells out to `poetry`/`trainer.py`/`ExperimentNotifier/main.py`, so it
needs neither a GPU nor the real dataset nor working SMTP credentials.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    REPO_ROOT / "scripts" / "ablations" / "run_ablation_gpu_0.sh",
    REPO_ROOT / "scripts" / "ablations" / "run_ablation_gpu_1.sh",
]

N_SEEDS = 3

# (script, method) -> number of CONFIGS entries -- must match each script's
# CONFIGS_{RNHAL,TEXHAL} arrays so the expected SKIP/DRY-RUN counts below
# stay correct. texhal has one more §6.1 row (rep_q13) than rnhal, and it
# lives on GPU 1 -- see scripts/ablations/run_ablation_gpu_{0,1}.sh headers.
N_CONFIGS = {
    ("run_ablation_gpu_0.sh", "rnhal"): 5,
    ("run_ablation_gpu_0.sh", "texhal"): 5,
    ("run_ablation_gpu_1.sh", "rnhal"): 6,
    ("run_ablation_gpu_1.sh", "texhal"): 7,
}
FIRST_STUDY_CONFIG = {
    ("run_ablation_gpu_0.sh", "rnhal"): ("6_1", "rep_full"),
    ("run_ablation_gpu_0.sh", "texhal"): ("6_1", "rep_full"),
    ("run_ablation_gpu_1.sh", "rnhal"): ("6_1", "rep_spectral"),
    ("run_ablation_gpu_1.sh", "texhal"): ("6_1", "rep_q17"),
}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax_is_valid(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script,method",
    [(s, m) for s in SCRIPTS for m in ("rnhal", "texhal")],
    ids=lambda v: v.name if isinstance(v, Path) else v,
)
def test_skip_existing_and_dry_run(script: Path, method: str, tmp_path: Path) -> None:
    results_root = tmp_path / "ablations"
    study, config_name = FIRST_STUDY_CONFIG[(script.name, method)]
    seed = 1

    # Fake a completed (study, config, seed=1) leaf, matching the layout
    # dalmax/experiment/runner.py::results_dir_for produces (NIL/NE globbed
    # by the script, so any value works here).
    leaf = (
        results_root
        / study
        / config_name
        / "daninhas_full"
        / f"SEED_{seed}"
        / "NQ_100_NIL_100_NR_8_NE_10"
        / "RepresentationStrategy"
    )
    leaf.mkdir(parents=True)
    (leaf / "results.json").write_text("{}")

    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["METHOD"] = method
    env["RESULTS_ROOT"] = str(results_root)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    skip_lines = [line for line in proc.stdout.splitlines() if line.startswith("SKIP (already completed)")]
    dry_run_lines = [line for line in proc.stdout.splitlines() if line.startswith("DRY-RUN: CUDA_VISIBLE_DEVICES")]

    total_triples = N_CONFIGS[(script.name, method)] * N_SEEDS
    assert len(skip_lines) == 1, proc.stdout
    assert f"study={study} config={config_name} seed={seed}" in skip_lines[0]
    assert len(dry_run_lines) == total_triples - 1, proc.stdout

    # Never actually invoked poetry/trainer.py or a real ExperimentNotifier
    # email for real -- only the DRY-RUN echo lines above, no "FAILED"/
    # "finalizado" lines (which only appear after a real invocation).
    assert "FAILED:" not in proc.stdout
    assert "finalizado." not in proc.stdout
    assert "DRY-RUN: skipping ExperimentNotifier call." in proc.stdout


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_invalid_method_exits_2_without_running_anything(script: Path, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["METHOD"] = "bogus"
    env["DRY_RUN"] = "1"
    env["RESULTS_ROOT"] = str(tmp_path / "ablations")

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2, proc.stdout
    assert "METHOD must be" in proc.stderr
    assert "DRY-RUN:" not in proc.stdout
    assert not (tmp_path / "ablations").exists()
