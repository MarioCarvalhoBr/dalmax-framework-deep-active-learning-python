"""Verify and (once, with explicit approval) migrate `results/ablations/` trees.

Two subcommands, invoked as `python -m dalmax.reporting.results_doctor <cmd>`:

- `verify`: walks the expected `(study, config, seed)` grid from
  `dalmax.reporting.ablation_report.STUDY_CONFIGS_BY_METHOD` and reports, per
  triple, whether its leaf directory (`<root>/[<method>/]<study>/<config>/
  <dataset>/SEED_<seed>/NQ_*/RepresentationStrategy/`) exists and carries the
  full artifact set `dalmax.experiment.reporter.write_report` writes for a
  successful run. When checking `rnhal`, it also transparently looks at the
  LEGACY no-method root `<root>/<study>/<config>/...` (see
  `files_config/ablations/README.md` and `CLAUDE.md`'s "Per-method ablation
  layout" note) -- a triple satisfied there is reported OK, tagged "legacy
  layout".
- `migrate-legacy`: moves the already-executed legacy RNHAL tree
  (`<root>/{6_1,6_2,6_3}` plus the batch-level `gpu{0,1}_failures.log` /
  `colab.log` files) into the per-method layout (`<root>/rnhal/...`), at the
  user's explicit request. This is a `results/`-internal MOVE, not an edit or
  a delete -- `.claude/rules/data-safety.md`'s append-only policy is about
  never editing/deleting a past run's contents, which this preserves exactly
  (file bytes are untouched, only their parent directory changes). Default is
  a dry run; `--apply` performs the moves. Never overwrites an existing
  destination -- a conflict is skipped with a clear message and a nonzero
  exit code, and the operation is idempotent (a second `--apply` run after a
  successful migration reports "nothing to migrate").

CLI:
    poetry run python -m dalmax.reporting.results_doctor verify --root results/ablations --method rnhal
    poetry run python -m dalmax.reporting.results_doctor migrate-legacy --root results/ablations
    poetry run python -m dalmax.reporting.results_doctor migrate-legacy --root results/ablations --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dalmax.reporting.ablation_report import SHARED_RUN_SOURCE, STUDY_CONFIGS_BY_METHOD

# Seeds every ablation triple is swept over (see LAB_RUNBOOK.md / COLAB_RUNBOOK.md).
SEEDS: tuple[int, ...] = (1, 2, 3)

# The generic strategy every ablation config runs through (never the fixed
# SSRAE/VCTex presets -- see files_config/ablations/README.md).
STRATEGY_NAME = "RepresentationStrategy"

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

# The legacy (no-method-segment) RNHAL results tree, executed 2026-08-25/26 --
# see CLAUDE.md's "Per-method ablation layout + METHOD variable" note.
LEGACY_STUDY_DIRS: tuple[str, ...] = ("6_1", "6_2", "6_3")
LEGACY_BATCH_FILES: tuple[str, ...] = ("gpu0_failures.log", "gpu1_failures.log", "colab.log")

# Methods a legacy (no-method-segment) tree exists for -- only rnhal, since
# texhal never had a pre-reorganization root.
METHODS_WITH_LEGACY_LAYOUT: tuple[str, ...] = ("rnhal",)


# --- verify -------------------------------------------------------------------


@dataclass
class TripleStatus:
    """The verification result for one expected `(study, config, seed)` triple."""

    method: str
    study: str
    config: str
    seed: int
    status: str  # "OK" | "INCOMPLETE" | "MISSING"
    layout: str | None  # "current" | "legacy" | None (when MISSING)
    leaf: str | None = None
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "study": self.study,
            "config": self.config,
            "seed": self.seed,
            "status": self.status,
            "layout": self.layout,
            "leaf": self.leaf,
            "missing_files": self.missing_files,
            "warnings": self.warnings,
        }


def _find_leaf(base: Path, seed: int) -> Path | None:
    """Find `<base>/*/SEED_<seed>/NQ_*/RepresentationStrategy/`, any dataset
    folder name and any `NQ_*_NIL_*_NR_*_NE_*` combination (mirrors
    `dalmax.reporting.ablation_report.find_run_results`'s glob shape, scoped
    to one seed instead of every seed under `base`)."""
    if not base.is_dir():
        return None
    matches = sorted(base.glob(f"*/SEED_{seed}/NQ_*/{STRATEGY_NAME}"))
    return matches[0] if matches else None


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


def _check_triple(
    root: Path, method: str, study: str, config: str, seed: int
) -> TripleStatus:
    """Check one `(study, config, seed)` triple for `method`, preferring the
    current per-method layout and falling back to the legacy no-method-segment
    layout (rnhal only) when the current layout doesn't have it."""
    candidates: list[tuple[str, Path]] = []

    current_leaf = _find_leaf(root / method / study / config, seed)
    if current_leaf is not None:
        candidates.append(("current", current_leaf))

    if method in METHODS_WITH_LEGACY_LAYOUT:
        legacy_leaf = _find_leaf(root / study / config, seed)
        if legacy_leaf is not None:
            candidates.append(("legacy", legacy_leaf))

    if not candidates:
        return TripleStatus(
            method=method, study=study, config=config, seed=seed,
            status="MISSING", layout=None,
        )

    best: tuple[str, Path, list[str], list[str]] | None = None
    for layout_name, leaf in candidates:
        missing, warnings = check_leaf(leaf)
        if not missing:
            return TripleStatus(
                method=method, study=study, config=config, seed=seed,
                status="OK", layout=layout_name, leaf=str(leaf), warnings=warnings,
            )
        if best is None or len(missing) < len(best[2]):
            best = (layout_name, leaf, missing, warnings)

    assert best is not None  # candidates is non-empty and none were fully OK
    layout_name, leaf, missing, warnings = best
    return TripleStatus(
        method=method, study=study, config=config, seed=seed,
        status="INCOMPLETE", layout=layout_name, leaf=str(leaf),
        missing_files=missing, warnings=warnings,
    )


def verify_method(root: Path, method: str) -> list[TripleStatus]:
    """Verify every expected `(study, config, seed)` triple for `method`
    (one of `dalmax.reporting.ablation_report.STUDY_CONFIGS_BY_METHOD`'s
    keys)."""
    statuses: list[TripleStatus] = []
    for study, rows in STUDY_CONFIGS_BY_METHOD[method].items():
        for config, _label in rows:
            # ADR 0009: a row that is one run shared across methods is looked up
            # in the source method's tree, then reported under this method.
            source = SHARED_RUN_SOURCE.get((method, study, config))
            for seed in SEEDS:
                if source is None:
                    statuses.append(_check_triple(root, method, study, config, seed))
                    continue
                shared = _check_triple(root, source, study, config, seed)
                shared.method = method
                if shared.status == "OK":
                    shared.layout = "shared" if shared.layout == "current" else shared.layout
                statuses.append(shared)
    return statuses


@dataclass
class MethodCounts:
    ok: int = 0
    legacy: int = 0
    incomplete: int = 0
    missing: int = 0

    @property
    def total(self) -> int:
        return self.ok + self.legacy + self.incomplete + self.missing

    @property
    def all_ok(self) -> bool:
        return self.incomplete == 0 and self.missing == 0


def _count(statuses: list[TripleStatus]) -> MethodCounts:
    counts = MethodCounts()
    for s in statuses:
        if s.status == "OK" and s.layout in ("current", "shared"):
            counts.ok += 1
        elif s.status == "OK" and s.layout == "legacy":
            counts.legacy += 1
        elif s.status == "INCOMPLETE":
            counts.incomplete += 1
        else:
            counts.missing += 1
    return counts


def render_verify_report(all_statuses: dict[str, list[TripleStatus]]) -> str:
    """Render the per-method summary table for `verify`."""
    lines: list[str] = []
    for method, statuses in all_statuses.items():
        counts = _count(statuses)
        lines.append(f"=== Method: {method} ===")
        by_study: dict[str, list[TripleStatus]] = {}
        for s in statuses:
            by_study.setdefault(s.study, []).append(s)
        for study in sorted(by_study):
            lines.append(f"  {study}")
            by_config: dict[str, list[TripleStatus]] = {}
            for s in by_study[study]:
                by_config.setdefault(s.config, []).append(s)
            for config in by_config:
                cells = []
                for s in sorted(by_config[config], key=lambda x: x.seed):
                    if s.status == "OK" and s.layout == "legacy":
                        cells.append(f"seed{s.seed}: OK (legacy layout)")
                    elif s.status == "OK" and s.layout == "shared":
                        cells.append(f"seed{s.seed}: OK (shared run, rnhal tree)")
                    elif s.status == "OK":
                        cells.append(f"seed{s.seed}: OK")
                    elif s.status == "INCOMPLETE":
                        cells.append(
                            f"seed{s.seed}: INCOMPLETE (missing {', '.join(s.missing_files)})"
                        )
                    else:
                        cells.append(f"seed{s.seed}: MISSING")
                lines.append(f"    {config:<28} " + "  ".join(cells))
        lines.append(
            f"  Totals: OK={counts.ok} LEGACY={counts.legacy} "
            f"INCOMPLETE={counts.incomplete} MISSING={counts.missing} "
            f"(of {counts.total} expected triples)"
        )
        lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dalmax.reporting.results_doctor", description=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="Check an ablation results tree against the expected grid."
    )
    verify_parser.add_argument(
        "--root", type=str, default="results/ablations",
        help="Root directory to walk (default: results/ablations).",
    )
    verify_parser.add_argument(
        "--method", type=str, choices=["rnhal", "texhal", "all"], default="all",
        help="Which ablation suite to verify (default: all).",
    )
    verify_parser.add_argument(
        "--json", type=str, default=None, dest="json_out",
        help="Optional path to also write a machine-readable JSON dump.",
    )

    migrate_parser = subparsers.add_parser(
        "migrate-legacy",
        help="Move the legacy no-method-segment RNHAL tree into results/ablations/rnhal/.",
    )
    migrate_parser.add_argument(
        "--root", type=str, default="results/ablations",
        help="Root directory containing the legacy tree (default: results/ablations).",
    )
    migrate_parser.add_argument(
        "--apply", action="store_true",
        help="Perform the moves. Without this flag, only prints the planned moves (dry run).",
    )

    return parser


def _run_verify(args: argparse.Namespace) -> int:
    root = Path(args.root)
    methods = ["rnhal", "texhal"] if args.method == "all" else [args.method]

    all_statuses: dict[str, list[TripleStatus]] = {
        method: verify_method(root, method) for method in methods
    }

    print(render_verify_report(all_statuses))

    if args.json_out:
        payload = {
            method: [s.to_dict() for s in statuses]
            for method, statuses in all_statuses.items()
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote machine-readable dump to {json_path}")

    all_ok = all(_count(statuses).all_ok for statuses in all_statuses.values())
    return 0 if all_ok else 1


# --- migrate-legacy -------------------------------------------------------------


@dataclass
class MigrationPlan:
    moves: list[tuple[Path, Path]]
    conflicts: list[tuple[Path, Path]]


def plan_migration(root: Path) -> MigrationPlan:
    """Plan moving the legacy RNHAL tree (`<root>/{6_1,6_2,6_3}` plus the
    batch-level log files) into `<root>/rnhal/`. Never plans overwriting an
    existing destination -- those are reported as conflicts instead."""
    dest_root = root / "rnhal"
    moves: list[tuple[Path, Path]] = []
    conflicts: list[tuple[Path, Path]] = []

    for study in LEGACY_STUDY_DIRS:
        src = root / study
        if not src.exists():
            continue
        dst = dest_root / study
        (conflicts if dst.exists() else moves).append((src, dst))

    for fname in LEGACY_BATCH_FILES:
        src = root / fname
        if not src.is_file():
            continue
        dst = dest_root / fname
        (conflicts if dst.exists() else moves).append((src, dst))

    return MigrationPlan(moves=moves, conflicts=conflicts)


def apply_migration(plan: MigrationPlan, root: Path) -> None:
    """Perform every planned move in `plan` via `shutil.move`. Conflicts are
    never touched -- callers must have already reported them and are
    expected to exit nonzero regardless."""
    dest_root = root / "rnhal"
    dest_root.mkdir(parents=True, exist_ok=True)
    for src, dst in plan.moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _run_migrate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    plan = plan_migration(root)

    if not plan.moves and not plan.conflicts:
        print("Nothing to migrate.")
        return 0

    for src, dst in plan.conflicts:
        print(f"CONFLICT (skipped, destination already exists): {src} -> {dst}")

    verb = "Moving" if args.apply else "Would move"
    for src, dst in plan.moves:
        print(f"{verb}: {src} -> {dst}")

    if args.apply:
        apply_migration(plan, root)
        if plan.moves:
            print()
            print("Migration complete. Follow-ups:")
            print("  make ablation-report METHOD=rnhal")
            print(
                "  METHOD=rnhal SKIP_EXISTING=1 ablation scripts will now see "
                "these completed runs."
            )
    else:
        print()
        print("Dry run only -- re-run with --apply to perform these moves.")

    return 1 if plan.conflicts else 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "verify":
        return _run_verify(args)
    if args.command == "migrate-legacy":
        return _run_migrate(args)
    raise AssertionError(f"unreachable: unknown command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
