"""Superseded leaves of the campaign's re-run parts (ADR 0014): the ONE definition.

The seven paper-1 baselines fixed on 2026-11-11 (KI-41/KI-42/KI-45) are re-run by the parts
`RERUN_PARTS` into the SAME leaf paths as their first (degenerate) execution. A leaf left there by
the pre-fix code would be skipped as complete by `dalmax.campaign run` and counted as `OK` by
`verify`, so both the engine (`dalmax/campaign.py`: run guard, `STALE` verify status) and the
archive script (`scripts/campaign/archive_superseded_runs.py`) classify a leaf with
`classify_leaf` below, and nothing else.

A leaf of a re-run part is **superseded** when its `run_metadata.json`

- (a) has a `started_at` earlier than the fix cutoff `SUPERSEDED_BEFORE`, or
- (b) lacks the key `config.dataset.model_args`: the fixed code always writes it (the resolved
  `DatasetConfig.model_args`, `dalmax.config.loader.to_dict`), the pre-fix code never did.

A leaf without a readable `run_metadata.json` (or without a valid `started_at` in an otherwise
fixed-code file) is **interrupted** when it also has no `results.json` (the reporter writes
`results.json` last; the engine re-runs such a leaf as an incomplete one, it is not superseded),
and **unreadable** when it has `results.json` (a finished run of unknown provenance: blocking,
resolved by hand). Everything else is **current**.

Standard library only (the archive script imports this module without the project's
dependencies). Importing it has no side effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The campaign parts that re-run superseded leaves in place (ADR 0014). `dalmax.campaign.PARTS`,
# `scripts/campaign/build_manifest.py` and the archive script all take the names from here.
RERUN_PARTS: tuple[str, ...] = ("paper1_rerun_a", "paper1_rerun_b")

# UTC time the KI-41/KI-42/KI-45 code fix was finished (ADR 0014, `date -u` on 2026-11-11): every
# leaf of the re-run parts that started before it was produced by the degenerate code.
SUPERSEDED_BEFORE = "2026-11-11T19:12:28Z"

# Key path the fixed code always writes into run_metadata.json and the pre-fix code never wrote.
FIX_MARKER_KEYS: tuple[str, ...] = ("config", "dataset", "model_args")

METADATA_FILE = "run_metadata.json"
RESULTS_FILE = "results.json"

# Leaf verdicts.
CURRENT = "current"
SUPERSEDED = "superseded"
INTERRUPTED = "interrupted"
UNREADABLE = "unreadable"
BLOCKING: frozenset[str] = frozenset({SUPERSEDED, UNREADABLE})


@dataclass(frozen=True)
class LeafVerdict:
    """Classification of one existing leaf directory of a re-run part."""

    status: str  # CURRENT | SUPERSEDED | INTERRUPTED | UNREADABLE
    reason: str
    started_at: datetime | None = None

    @property
    def blocking(self) -> bool:
        """True when this leaf must leave the live tree before the part may run or verify OK."""
        return self.status in BLOCKING


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp with an explicit offset (`Z` accepted); naive is rejected."""
    stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no UTC offset")
    return stamp.astimezone(timezone.utc)


def default_cutoff() -> datetime:
    """`SUPERSEDED_BEFORE` as an aware UTC datetime."""
    return parse_utc(SUPERSEDED_BEFORE)


def format_utc(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def leaf_pattern(
    strategy_name: str, n_query: int, n_init_labeled: int, n_round: int, seed: int
) -> str:
    """Glob, relative to a group's `dir_results`, matching a job's leaf (any dataset folder / n_epoch)."""
    return f"*/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_*/{strategy_name}"


def _has_fix_marker(metadata: dict[str, Any]) -> bool:
    node: Any = metadata
    for key in FIX_MARKER_KEYS:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _without_metadata(leaf: Path, why: str) -> LeafVerdict:
    if (leaf / RESULTS_FILE).exists():
        return LeafVerdict(
            UNREADABLE,
            f"{why}; {RESULTS_FILE} present: a finished run of unknown provenance (resolve by hand)",
        )
    return LeafVerdict(
        INTERRUPTED, f"{why}; no {RESULTS_FILE}: an interrupted run, the engine re-runs it"
    )


def classify_leaf(leaf: Path, cutoff: datetime | None = None) -> LeafVerdict:
    """Classify an existing leaf directory of a re-run part (see the module docstring)."""
    limit = cutoff if cutoff is not None else default_cutoff()
    try:
        metadata = json.loads((leaf / METADATA_FILE).read_text())
        if not isinstance(metadata, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return _without_metadata(leaf, f"no readable {METADATA_FILE} ({exc})")

    started: datetime | None = None
    started_problem = ""
    try:
        started = parse_utc(metadata["started_at"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        started_problem = f"no valid started_at ({exc!r})"

    if started is not None and started < limit:
        return LeafVerdict(
            SUPERSEDED,
            f"started {format_utc(started)}, before the fix cutoff {format_utc(limit)}",
            started,
        )
    if not _has_fix_marker(metadata):
        return LeafVerdict(
            SUPERSEDED,
            f"{METADATA_FILE} lacks {'.'.join(FIX_MARKER_KEYS)} (written by the pre-fix code)",
            started,
        )
    if started is None:
        return _without_metadata(leaf, f"{METADATA_FILE} has {started_problem}")
    return LeafVerdict(CURRENT, f"started {format_utc(started)}", started)
