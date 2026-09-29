"""Run metadata snapshot: what makes a results directory reproducible.

`.claude/rules/reproducibility.md` requires every experiment to be fully
determined by "params JSON + CLI args + seed + git commit". `snapshot()`
captures exactly that (via the resolved `ExperimentConfig`, not the raw JSON)
plus enough environment info (Python/torch versions, CUDA availability) to
diagnose hardware-nondeterminism gaps later. `write_run_metadata()` persists
it as `run_metadata.json` into a results directory, closing the gap noted in
`.specs/architecture/target-architecture.md` §8 (today only `results.json`,
`predictions.csv`, plots, and the log file are written; commit hash and
config snapshot are missing).
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from dalmax.config.loader import to_dict
from dalmax.config.schema import ExperimentConfig
from dalmax.experiment.environment import collect_environment

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit_hash() -> str | None:
    """Best-effort `git rev-parse HEAD`; `None` if git or a repo is unavailable.

    Runs with `cwd` pinned to the directory containing this file (git
    auto-discovers the enclosing repository from any subdirectory within it),
    so the result does not depend on the caller's current working directory.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def snapshot(config: ExperimentConfig, *, started_at: str | None = None) -> dict:
    """Build a JSON-serializable run metadata dict for `config`.

    Parameters
    ----------
    config:
        The fully-resolved `ExperimentConfig` for this run.
    started_at:
        ISO-8601 timestamp to record; defaults to "now" (UTC) if omitted.
    """
    return {
        "config": to_dict(config),
        "git_commit": _git_commit_hash(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(),
    }


def write_run_metadata(dir_results: str | Path, metadata: dict) -> Path:
    """Write `metadata` as `run_metadata.json` under `dir_results`.

    Creates `dir_results` (and parents) if it does not already exist. Returns
    the path written to.
    """
    dir_path = Path(dir_results)
    dir_path.mkdir(parents=True, exist_ok=True)
    out_path = dir_path / "run_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    return out_path
