"""Archive the superseded leaves of the re-run parts into `results/old_results_broken/` (ADR 0014).

The seven paper-1 baselines fixed on 2026-11-11 (KI-41 MC dropout, KI-42 embeddings, KI-45 BIM
ties) are re-run into the SAME leaf paths as their first execution (parts `paper1_rerun_a` and
`paper1_rerun_b` of `files_config/campaign/manifest.json`). The campaign engine skips a complete
leaf, so the old leaves must leave the live tree first. This script MOVES every leaf of those parts
that `dalmax.superseded.classify_leaf` calls superseded (its `run_metadata.json` `started_at` is
earlier than the cutoff, or it lacks `config.dataset.model_args`, which only the fixed code writes;
the same definition the campaign engine's run guard and `verify` STALE status use) to
`results/old_results_broken/<path relative to results/>`, e.g.
`results/campaign/paper1/nq100/daninhas_full/SEED_1/NQ_100_NIL_100_NR_8_NE_10/AdversarialBIM/` ->
`results/old_results_broken/campaign/paper1/nq100/daninhas_full/SEED_1/NQ_100_NIL_100_NR_8_NE_10/AdversarialBIM/`.
It is the one sanctioned exception to "results/ is append-only" (`.claude/rules/data-safety.md`):
an explicit author instruction to keep the old runs, not to delete them.

Rules: nothing is ever deleted or overwritten; a leaf whose destination already exists is reported
and skipped (exit 1); a current leaf (started at or after the cutoff by the fixed code) is never
moved (safe to re-run once new runs exist); a leaf without a readable `run_metadata.json` is never
moved either: with `results.json` it is UNREADABLE (a finished run of unknown provenance, exit 1,
resolve by hand, COLAB_RUNBOOK.md section 6d), without it INTERRUPTED (not superseded: the
campaign re-runs it, exit 0); `dir_results` must lie under `results/` with no `..` component;
re-running is idempotent (an archived leaf is reported as already archived). One line per
moved leaf is appended to `results/old_results_broken/ARCHIVE_LOG.tsv`. Paths are handled relative
to the repo root and never resolved, so on Colab (`results/` is a symlink to Drive) the move is a
rename inside Drive and the logs show repo-relative paths.

Modes:
    python scripts/campaign/archive_superseded_runs.py             # move
    python scripts/campaign/archive_superseded_runs.py --dry-run   # report what would move
    python scripts/campaign/archive_superseded_runs.py --check     # move nothing; exit 1 if a live
                                                                   # leaf predates the cutoff
Standard library only. Makefile: `make campaign-archive-superseded [DRY=1|CHECK=1]`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# One definition of "superseded" (stdlib-only module), shared with dalmax/campaign.py.
from dalmax import superseded as sup  # noqa: E402
from dalmax.superseded import RERUN_PARTS, SUPERSEDED_BEFORE, parse_utc  # noqa: E402, F401

DEFAULT_MANIFEST = "files_config/campaign/manifest.json"
RESULTS_DIR = "results"
ARCHIVE_DIR = "results/old_results_broken"
ARCHIVE_LOG_NAME = "ARCHIVE_LOG.tsv"
ARCHIVE_LOG_HEADER = "utc_time\tsource\tdestination\treason"
REASON = "superseded by the KI-41/KI-42/KI-45 fix"

# Per-leaf outcomes.
MOVED = "moved"
WOULD_MOVE = "would move"
STALE = "stale"
CURRENT = "current"
ALREADY_ARCHIVED = "already archived"
CONFLICT = "conflict"
UNREADABLE = sup.UNREADABLE
INTERRUPTED = sup.INTERRUPTED
MISSING = "missing"
ERROR = "error"
FAILURES = frozenset({CONFLICT, UNREADABLE, ERROR})


class ArchiveError(ValueError):
    """Invalid manifest, cutoff or arguments (fail fast)."""


@dataclass(frozen=True)
class Outcome:
    status: str
    source: str  # repo-relative POSIX path (or the job label when there is no leaf)
    destination: str = ""
    detail: str = ""


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(o.status == status for o in self.outcomes)

    def failed(self, check: bool) -> bool:
        statuses = {o.status for o in self.outcomes}
        return bool(statuses & FAILURES) or (check and STALE in statuses)


def _rerun_groups(manifest: dict[str, Any], parts: Sequence[str]) -> list[dict[str, Any]]:
    groups = [g for g in manifest.get("groups", []) if g.get("part") in parts]
    for g in groups:
        rel = PurePosixPath(g["dir_results"])
        if rel.is_absolute() or not rel.parts or rel.parts[0] != RESULTS_DIR:
            raise ArchiveError(
                f"{g['id']}: dir_results {g['dir_results']!r} is not under {RESULTS_DIR}/"
            )
        if ".." in rel.parts:
            raise ArchiveError(f"{g['id']}: dir_results {g['dir_results']!r} has a '..' component")
    return groups


def _leaf_pattern(group: dict[str, Any], seed: int) -> str:
    """The leaf glob `dalmax.campaign.leaf_glob` also uses (any dataset folder / n_epoch)."""
    return sup.leaf_pattern(
        group["strategy_name"],
        int(group["n_query"]),
        int(group["n_init_labeled"]),
        int(group["n_round"]),
        seed,
    )


def _matches(root: Path, base: PurePosixPath, pattern: str) -> list[PurePosixPath]:
    directory = root / base
    if not directory.is_dir():
        return []
    return sorted(
        base / path.relative_to(directory).as_posix()
        for path in directory.glob(pattern)
        if path.is_dir()
    )


def _archived_path(source: PurePosixPath) -> PurePosixPath:
    return PurePosixPath(ARCHIVE_DIR) / source.relative_to(RESULTS_DIR)


def _jobs(manifest: dict[str, Any], parts: Sequence[str]) -> Iterator[tuple[dict[str, Any], int]]:
    default_seeds = manifest.get("seeds", [1, 2, 3])
    for group in _rerun_groups(manifest, parts):
        for seed in group.get("seeds", default_seeds):
            yield group, int(seed)


def _append_log(root: Path, source: PurePosixPath, destination: PurePosixPath) -> None:
    log = root / ARCHIVE_DIR / ARCHIVE_LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(log, "a", encoding="utf-8") as fh:
        if new:
            fh.write(ARCHIVE_LOG_HEADER + "\n")
        fh.write(f"{stamp}\t{source}\t{destination}\t{REASON}\n")


def _handle_leaf(root: Path, source: PurePosixPath, cutoff: datetime, mode: str) -> Outcome:
    destination = _archived_path(source)
    verdict = sup.classify_leaf(root / source, cutoff)
    if verdict.status == sup.CURRENT:
        return Outcome(CURRENT, str(source), detail=verdict.reason)
    if verdict.status in (sup.UNREADABLE, sup.INTERRUPTED):
        return Outcome(verdict.status, str(source), detail=verdict.reason)
    detail = verdict.reason
    if mode == "check":
        return Outcome(STALE, str(source), str(destination), detail)
    if os.path.lexists(root / destination):
        return Outcome(
            CONFLICT, str(source), str(destination), "destination already exists; not merged"
        )
    if mode == "dry-run":
        return Outcome(WOULD_MOVE, str(source), str(destination), detail)
    try:
        (root / destination).parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / source, root / destination)
    except OSError as exc:
        return Outcome(ERROR, str(source), str(destination), f"rename failed: {exc}")
    _append_log(root, source, destination)
    return Outcome(MOVED, str(source), str(destination), detail)


def archive(
    root: Path,
    manifest: dict[str, Any],
    *,
    cutoff: datetime,
    mode: str = "move",
    parts: Sequence[str] = RERUN_PARTS,
) -> Report:
    """Apply `mode` (`move`, `dry-run` or `check`) to every leaf of `parts` under `root`."""
    if mode not in ("move", "dry-run", "check"):
        raise ArchiveError(f"unknown mode {mode!r}")
    report = Report()
    for group, seed in _jobs(manifest, parts):
        base = PurePosixPath(group["dir_results"])
        pattern = _leaf_pattern(group, seed)
        live = _matches(root, base, pattern)
        for source in live:
            report.outcomes.append(_handle_leaf(root, source, cutoff, mode))
        if not live:
            label = f"{group['id']} seed={seed}"
            archived = _matches(root, _archived_path(base), pattern)
            status = ALREADY_ARCHIVED if archived else MISSING
            report.outcomes.append(Outcome(status, label, ", ".join(map(str, archived))))
    return report


def render(report: Report, *, mode: str, cutoff: datetime, parts: Sequence[str]) -> str:
    lines = []
    for o in report.outcomes:
        arrow = f" -> {o.destination}" if o.destination else ""
        extra = f" ({o.detail})" if o.detail else ""
        lines.append(f"{o.status.upper():<17} {o.source}{arrow}{extra}")
    moved = report.count(MOVED) + report.count(WOULD_MOVE)
    skipped = report.count(CONFLICT) + report.count(UNREADABLE) + report.count(ERROR)
    lines.append("")
    lines.append(
        f"Summary ({mode}; parts {', '.join(parts)}; cutoff {cutoff:%Y-%m-%dT%H:%M:%SZ}): "
        f"{'would move' if mode == 'dry-run' else 'moved'}={moved} stale={report.count(STALE)} "
        f"already_archived={report.count(ALREADY_ARCHIVED)} current={report.count(CURRENT)} "
        f"interrupted={report.count(INTERRUPTED)} skipped={skipped} missing={report.count(MISSING)}"
    )
    if mode == "check":
        lines.append(
            "CHECK FAILED: stale or unreadable leaves in the live tree (archive them first)"
            if report.failed(check=True)
            else "CHECK OK: no superseded or unreadable leaf of these parts in the live tree"
        )
    elif skipped:
        lines.append(
            "Some leaves were skipped (conflict / unreadable / error): see the lines above."
        )
    return "\n".join(lines)


def _load_manifest(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"cannot read manifest {path}: {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report what would move, move nothing")
    mode.add_argument(
        "--check",
        action="store_true",
        help="move nothing; exit 1 if a live leaf of the re-run parts is superseded or unreadable",
    )
    parser.add_argument(
        "--before",
        default=SUPERSEDED_BEFORE,
        help=f"cutoff, ISO-8601 with offset (default {SUPERSEDED_BEFORE})",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"campaign manifest (default {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--root", default=str(REPO_ROOT), help="repo root holding results/ (default: this repo)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mode = "check" if args.check else "dry-run" if args.dry_run else "move"
    try:
        cutoff = parse_utc(args.before)
        report = archive(Path(args.root), _load_manifest(args.manifest), cutoff=cutoff, mode=mode)
    except (ArchiveError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(render(report, mode=mode, cutoff=cutoff, parts=RERUN_PARTS))
    return 1 if report.failed(check=mode == "check") else 0


if __name__ == "__main__":
    sys.exit(main())
