"""Read-only report of a campaign pilot (2026-11-12).

Before the long runs of a GPU part, `python -m dalmax.campaign run --pilot`
executes seed 1 of the part's VAAL and WAAL groups (their costs are FLOP
estimates) and of one cheap strategy. `python -m dalmax.campaign pilot-report`
then prints, per pilot leaf: the wall time (the trainer subprocess, from the
manifest's `campaign.log`, as in `files_config/campaign/gpu_time_estimates.json`)
against the estimate, the projected hours of that strategy over the selection,
the final weighted F1, every `strategy_diagnostics` series of `results.json`
(ADR 0015) and simple degeneracy flags. Nothing is written. This module does
not import `dalmax.campaign` (the campaign resolves the leaves and passes them in).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DONE_RE = re.compile(r"\] (\S+) seed=(\d+) DONE elapsed=(\d+)s")
TOTAL_TIME_RE = re.compile(r"Total time: ([0-9.]+) seconds")

# Degeneracy thresholds (reporting only). `pool_order_fraction` of an informative strategy is about
# 2n/|U| (0.03 at n=100 on a 7,000-image pool); at or above this value the queries concentrate at the
# head of the class-ordered pool, the KI-45 signature.
POOL_ORDER_FLAG = 0.5


@dataclass(frozen=True)
class PilotRow:
    """One pilot job, as resolved by the campaign."""

    label: str
    strategy: str
    leaf: Path | None
    campaign_log_seconds: float | None
    estimate_seconds: float | None
    runs_in_selection: int


def read_campaign_log_elapsed(log: Path, group_id: str, seed: int) -> float | None:
    """The LAST `DONE elapsed=<s>` of `group_id seed=<seed>` in a `campaign.log`, or None."""
    if not log.is_file():
        return None
    found = None
    for match in DONE_RE.finditer(log.read_text(errors="replace")):
        if match.group(1) == group_id and int(match.group(2)) == seed:
            found = float(match.group(3))
    return found


def read_total_time(leaf: Path) -> float | None:
    """`Total time: <s> seconds` of the run's own log (the round loop, without process start-up)."""
    log = leaf / "log-dalmax.log"
    if not log.is_file():
        return None
    match = TOTAL_TIME_RE.search(log.read_text(errors="replace"))
    return float(match.group(1)) if match else None


def read_results(leaf: Path) -> dict[str, Any] | None:
    try:
        return json.loads((leaf / "results.json").read_text())
    except (OSError, ValueError):
        return None


def degeneracy_flags(diagnostics: dict[str, list[Any]]) -> list[str]:
    """Human-readable warnings for collapsed selection signals (reporting only)."""
    flags = []
    for key, values in diagnostics.items():
        numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not numbers:
            continue
        if key == "pool_order_fraction" and max(numbers) >= POOL_ORDER_FLAG:
            flags.append(f"{key} reaches {max(numbers):.2f} (>= {POOL_ORDER_FLAG}): queries at the pool head (KI-45)")
        elif key.endswith("_n_unique") and min(numbers) <= 1:
            flags.append(f"{key} is {min(numbers)} in some round: constant scores")
        elif key.endswith("_std") and min(abs(v) for v in numbers) == 0:
            flags.append(f"{key} is 0 in some round: no score spread")
    return flags


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3g}"
    return str(value)


def _minutes(seconds: float | None) -> str:
    return f"{seconds / 60:.1f} min" if seconds is not None else "n/a"


def render_pilot_report(rows: Sequence[PilotRow]) -> str:
    lines = ["Pilot report (read only). Check the wall times against the estimates and the diagnostics for "
             "degeneracy before launching the full part; the full run skips these completed leaves."]
    for row in rows:
        lines.append("")
        lines.append(f"== {row.label} ({row.strategy})")
        if row.leaf is None:
            lines.append("   no leaf yet: run `make campaign-pilot` first")
            continue
        wall = row.campaign_log_seconds
        source = "campaign.log"
        if wall is None:
            wall, source = read_total_time(row.leaf), "log-dalmax.log Total time (no campaign.log entry)"
        line = f"   wall time {_minutes(wall)} [{source}]; estimate {_minutes(row.estimate_seconds)}"
        if wall is not None and row.estimate_seconds:
            ratio = wall / row.estimate_seconds
            line += (f"; measured/estimate = {ratio:.2f}; {row.strategy}: {row.runs_in_selection} runs in this "
                     f"selection = {row.runs_in_selection * wall / 3600:.1f} h at this pace "
                     f"(estimate {row.runs_in_selection * row.estimate_seconds / 3600:.1f} h)")
        lines.append(line)
        results = read_results(row.leaf)
        if results is None:
            lines.append("   results.json missing or unreadable (incomplete leaf?)")
            continue
        f1 = results.get("all_f1_score") or []
        lines.append("   weighted F1 per round: " + ", ".join(_fmt(v) for v in f1))
        diagnostics = results.get("strategy_diagnostics") or {}
        if not diagnostics:
            lines.append("   no strategy_diagnostics (this strategy reports none)")
        for key, values in diagnostics.items():
            lines.append(f"   {key}: " + ", ".join(_fmt(v) for v in values))
        for flag in degeneracy_flags(diagnostics):
            lines.append(f"   FLAG: {flag}")
    return "\n".join(lines)
