"""Completeness and integrity check of one run's leaf results directory.

`check_leaf` verifies that a leaf
(`<dir_results>/<dataset>/SEED_<s>/NQ_*/<strategy>/`) carries the full artifact
set `dalmax.experiment.reporter.write_report` writes for a successful run, and
that the machine-readable artifacts are intact. Used by `dalmax.campaign`
(skip-existing/resume and `verify`). Importing this module has no side effects.

Three outcomes per leaf (`LeafCheck`, 2026-11-12):

- `missing`: a required file is absent;
- `problems`: integrity problems, a file is present but not what a finished run
  writes (unparsable or empty JSON, a key the reporter always writes is absent,
  a round count that does not match the leaf's `NR_<n>`, an empty
  `predictions.csv` or `saved_model.pth`). The leaf is NOT complete: resume
  re-runs it into the same leaf (the run rewrites every file) and `verify`
  reports it INCOMPLETE with the reason;
- `warnings`: genuine soft warnings that do not make the leaf incomplete (a
  `saved_model.pth` smaller than a real checkpoint, the KI-22 hint).

Leaves of the earlier trees (before ADR 0016, no per-round files) are checked
only against `REQUIRED_FILES`; the round files are checked only when a manifest
lists them in `required_leaf_files` (`extra_required`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# Per-round history files (ADR 0016), written by every run since 2026-11-11. They are
# not in REQUIRED_FILES so that leaves of the earlier trees still verify; a manifest
# that requires them lists them in its `required_leaf_files` (the campaign on
# daninhas_full_fixed does), and `check_leaf(..., extra_required=...)` checks them.
ROUND_PREDICTIONS_FILE = "round_predictions.json"
ROUND_QUERIES_FILE = "round_queries.json"
ROUND_HISTORY_FILES: tuple[str, ...] = (ROUND_PREDICTIONS_FILE, ROUND_QUERIES_FILE)

# Keys `results.json` / `run_metadata.json` always carry when written by a finished run.
RESULTS_KEYS: tuple[str, ...] = ("all_acc", "all_f1_score", "all_f1_macro", "rounds")
RUN_METADATA_KEYS: tuple[str, ...] = ("config", "git_commit")

# Soft warning only: a saved_model.pth below this size is almost certainly the
# pre-KI-22-fix class-pickle (~900 bytes), not a real dalmax-checkpoint with weights
# (.specs/quality/known-issues.md KI-22, ADR 0006). It never makes a leaf incomplete;
# the integrity check of the checkpoint is "exists and is not empty".
SAVED_MODEL_MIN_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class LeafCheck:
    """Result of `check_leaf`: missing files, integrity problems, soft warnings."""

    missing: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True iff every required file is present and intact (warnings allowed)."""
        return not self.missing and not self.problems


def _parse_n_round(nq_dirname: str) -> int | None:
    """Parse `n_round` out of an `NQ_*_NIL_*_NR_<n>_NE_*` directory name."""
    match = re.search(r"_NR_(\d+)_", nq_dirname)
    return int(match.group(1)) if match else None


def _read_json_object(fpath: Path) -> tuple[dict[str, Any] | None, str | None]:
    """`(payload, None)` for a readable JSON object, else `(None, problem)`."""
    try:
        text = fpath.read_text()
    except OSError as exc:
        return None, f"{fpath.name}: unreadable ({exc.__class__.__name__})"
    if not text.strip():
        return None, f"{fpath.name}: empty"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, f"{fpath.name}: failed to parse (truncated or corrupt JSON)"
    if not isinstance(payload, dict):
        return None, f"{fpath.name}: not a JSON object"
    return payload, None


def _round_count_problem(name: str, rounds: Any, n_round: int | None) -> str | None:
    if n_round is not None and isinstance(rounds, list) and len(rounds) != n_round + 1:
        return f"{name}: {len(rounds)} rounds != n_round+1={n_round + 1}"
    return None


def _check_results_json(fpath: Path, n_round: int | None) -> list[str]:
    payload, problem = _read_json_object(fpath)
    if payload is None:
        return [problem or f"{fpath.name}: unreadable"]
    problems = [f"{fpath.name}: missing key {key!r}" for key in RESULTS_KEYS if key not in payload]
    count = _round_count_problem(fpath.name, payload.get("rounds"), n_round)
    return problems + ([count] if count else [])


def _check_run_metadata(fpath: Path) -> list[str]:
    payload, problem = _read_json_object(fpath)
    if payload is None:
        return [problem or f"{fpath.name}: unreadable"]
    return [f"{fpath.name}: missing key {key!r}" for key in RUN_METADATA_KEYS if key not in payload]


def _check_round_history(fpath: Path, n_round: int | None) -> list[str]:
    """Integrity problems of a present `round_predictions.json` / `round_queries.json`."""
    payload, problem = _read_json_object(fpath)
    if payload is None:
        return [problem or f"{fpath.name}: unreadable"]
    key = "predictions" if fpath.name == ROUND_PREDICTIONS_FILE else "rounds"
    rounds = payload.get(key)
    if not isinstance(rounds, list):
        return [f"{fpath.name}: missing key {key!r}"]
    count = _round_count_problem(fpath.name, rounds, n_round)
    return [count] if count else []


def _check_not_empty(fpath: Path) -> list[str]:
    return [f"{fpath.name}: empty (0 bytes)"] if fpath.stat().st_size == 0 else []


def check_leaf(leaf: Path, extra_required: tuple[str, ...] = ()) -> LeafCheck:
    """Check one leaf directory against `REQUIRED_FILES` plus `extra_required`.

    `extra_required` names further mandatory files, e.g. `ROUND_HISTORY_FILES`
    for the leaves of a manifest that lists them in `required_leaf_files`.
    See the module docstring for the split between `missing`, `problems` and
    `warnings`; `LeafCheck.complete` is what resume and `verify` act on.
    """
    unknown = sorted(set(extra_required) - set(ROUND_HISTORY_FILES))
    if unknown:
        raise ValueError(f"check_leaf: unknown extra required file(s) {unknown}; known: {ROUND_HISTORY_FILES}")

    missing: list[str] = []
    problems: list[str] = []
    warnings: list[str] = []
    n_round = _parse_n_round(leaf.parent.name)

    for fname in REQUIRED_FILES:
        fpath = leaf / fname
        if not fpath.is_file():
            missing.append(fname)
        elif fname == "results.json":
            problems.extend(_check_results_json(fpath, n_round))
        elif fname == "run_metadata.json":
            problems.extend(_check_run_metadata(fpath))
        elif fname == "predictions.csv":
            problems.extend(_check_not_empty(fpath))
        elif fname == "saved_model.pth":
            empty = _check_not_empty(fpath)
            problems.extend(empty)
            size = fpath.stat().st_size
            if not empty and size < SAVED_MODEL_MIN_BYTES:
                warnings.append(
                    f"saved_model.pth: {size} bytes < 10 MB "
                    "(likely a pre-fix class-pickle, see KI-22/ADR 0006)"
                )

    for fname in extra_required:
        fpath = leaf / fname
        if not fpath.is_file():
            missing.append(fname)
        else:
            problems.extend(_check_round_history(fpath, n_round))

    return LeafCheck(tuple(missing), tuple(problems), tuple(warnings))
