"""Completeness check of one run's leaf results directory.

`check_leaf` verifies that a leaf
(`<dir_results>/<dataset>/SEED_<s>/NQ_*/<strategy>/`) carries the full artifact
set `dalmax.experiment.reporter.write_report` writes for a successful run.
Used by `dalmax.campaign` (skip-existing/resume and `verify`) and
`dalmax.reporting.campaign_report`. Importing this module has no side effects.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# The full artifact set a successful run writes, per
# dalmax/experiment/reporter.py::write_report and
# dalmax/experiment/run_metadata.py::write_run_metadata.
REQUIRED_FILES: tuple[str, ...] = (
    "results.json",
    "predictions.csv",
    "run_metadata.json",
    "confusion_matrix.pdf",
    "accuracy.pdf",
    "precision.pdf",
    "recall.pdf",
    "f1_score.pdf",
    "log-dalmax.log",
    "saved_model.pth",
)

# A saved_model.pth below this size is almost certainly the pre-KI-22-fix
# class-pickle (~900 bytes), not a real dalmax-checkpoint with weights -- see
# .specs/quality/known-issues.md KI-22 and ADR 0006.
SAVED_MODEL_MIN_BYTES = 10 * 1024 * 1024


def _parse_n_round(nq_dirname: str) -> int | None:
    """Parse `n_round` out of an `NQ_*_NIL_*_NR_<n>_NE_*` directory name."""
    match = re.search(r"_NR_(\d+)_", nq_dirname)
    return int(match.group(1)) if match else None


def check_leaf(leaf: Path) -> tuple[list[str], list[str]]:
    """Check one leaf directory against `REQUIRED_FILES`.

    Returns `(missing_files, warnings)`. `missing_files` is empty iff every
    required artifact is present; `warnings` flags a present-but-suspicious
    artifact (e.g. an unparsable `results.json`, an inconsistent round count,
    or a too-small `saved_model.pth`) without counting it as missing.
    """
    missing: list[str] = []
    warnings: list[str] = []

    for fname in REQUIRED_FILES:
        fpath = leaf / fname
        if not fpath.is_file():
            missing.append(fname)
            continue

        if fname == "results.json":
            try:
                payload = json.loads(fpath.read_text())
            except (OSError, json.JSONDecodeError):
                warnings.append("results.json: failed to parse")
                continue
            for key in ("all_acc", "all_f1_score", "all_f1_macro", "rounds"):
                if key not in payload:
                    warnings.append(f"results.json: missing key {key!r}")
            rounds = payload.get("rounds")
            n_round = _parse_n_round(leaf.parent.name)
            if n_round is not None and isinstance(rounds, list) and len(rounds) != n_round + 1:
                warnings.append(
                    f"results.json: len(rounds)={len(rounds)} != n_round+1={n_round + 1}"
                )
        elif fname == "predictions.csv":
            if fpath.stat().st_size == 0:
                warnings.append("predictions.csv: empty")
        elif fname == "run_metadata.json":
            try:
                meta = json.loads(fpath.read_text())
            except (OSError, json.JSONDecodeError):
                warnings.append("run_metadata.json: failed to parse")
                continue
            if "config" not in meta or "git_commit" not in meta:
                warnings.append("run_metadata.json: missing config/git_commit")
        elif fname == "saved_model.pth":
            size = fpath.stat().st_size
            if size < SAVED_MODEL_MIN_BYTES:
                warnings.append(
                    f"saved_model.pth: {size} bytes < 10 MB "
                    "(likely a pre-fix class-pickle, see KI-22/ADR 0006)"
                )

    return missing, warnings
