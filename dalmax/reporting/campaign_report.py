"""Report generator for the single campaign (`files_config/campaign/manifest.json`).

    python -m dalmax.reporting.campaign_report --root results/campaign_a100 \
        --out docs/results/campaign_a100

Reads the manifest's `tables` section (rows -> run ids; aliased rows resolve to
the ONE canonical run, so a value such as "TexHAL full" is identical in every
table of every paper) and writes, under `--out`:

- `<paper>/<table id>.md` and `.tex` -- paper 1: one table per n_query
  (Accuracy, Precision, Recall, weighted F1 -- the thesis columns -- plus
  macro F1) with the upper-bound row; papers 2/3: the 6.1/6.2/6.3 ablation
  tables (weighted + macro F1) and a comparison table. Values are the
  final-round mean (+/- population std) across seeds; a row with no runs is
  `TBD`. The LaTeX uses the same booktabs / `($\\pm$...)` style as
  `dalmax.reporting.ablation_report`.
- `summary.csv` -- one row per (paper, table, row) with every metric.
- `seed_audit.md` -- the seed-consistency audit (`campaign verify`'s check).
- `confusion_matrices/<run id>.csv|pdf` -- the mean confusion matrix across
  seeds (from each run's `predictions.csv`) for every run group with data.

`--root` rebases the manifest's `results_root` (default: use the manifest's own),
`--micro` selects the micro manifest (smoke).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dalmax.campaign import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MICRO_MANIFEST,
    REPO_ROOT,
    Manifest,
    RunGroup,
    audit_seed_consistency,
    expand_jobs,
    find_leaf,
    load_manifest,
    render_seed_audit,
)

TBD = "TBD"

# (key, results.json field, column header)
METRICS: tuple[tuple[str, str, str], ...] = (
    ("acc", "all_acc", "Accuracy"),
    ("precision", "all_precision", "Precision"),
    ("recall", "all_recall", "Recall"),
    ("f1_weighted", "all_f1_score", "F1 (weighted)"),
    ("f1_macro", "all_f1_macro", "F1 (macro)"),
)
METRIC_FIELD = {key: field for key, field, _ in METRICS}
METRIC_HEADER = {key: header for key, _, header in METRICS}
BENCHMARK_COLUMNS = [m[0] for m in METRICS]
ABLATION_COLUMNS = ["f1_weighted", "f1_macro"]
# `best_of.metric` in the manifest uses the results.json spelling.
BEST_OF_METRIC_KEY = {"f1_score": "f1_weighted", "f1_weighted": "f1_weighted", "f1_macro": "f1_macro",
                      "acc": "acc"}


# --- loading ------------------------------------------------------------------


@dataclass(frozen=True)
class RunData:
    seed: int
    leaf: Path
    final: dict[str, float]  # metric key -> last-round value


@dataclass(frozen=True)
class GroupSummary:
    group_id: str
    n_seeds: int
    mean: dict[str, float]
    std: dict[str, float]

    @property
    def has_data(self) -> bool:
        return self.n_seeds > 0


def rebase_group(manifest: Manifest, group: RunGroup, root: Path | None) -> RunGroup:
    """Point `group.dir_results` under `root` instead of the manifest's `results_root`."""
    if root is None:
        return group
    rel = Path(group.dir_results).relative_to(manifest.results_root)
    return replace(group, dir_results=str(Path(root).resolve() / rel))


def load_group_runs(group: RunGroup) -> list[RunData]:
    """One `RunData` per seed whose leaf has a parsable `results.json`."""
    runs: list[RunData] = []
    for seed in group.seeds:
        leaf = find_leaf(group, seed)
        if leaf is None or not (leaf / "results.json").is_file():
            continue
        try:
            payload = json.loads((leaf / "results.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        final: dict[str, float] = {}
        for key, field, _ in METRICS:
            values = payload.get(field)
            if values:
                final[key] = float(values[-1])
        if final:
            runs.append(RunData(seed, leaf, final))
    return runs


def summarize_group(group_id: str, runs: list[RunData]) -> GroupSummary:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in METRIC_FIELD:
        vals = [r.final[key] for r in runs if key in r.final]
        if vals:
            mean[key] = statistics.mean(vals)
            std[key] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return GroupSummary(group_id, len(runs), mean, std)


# --- tables -------------------------------------------------------------------


@dataclass(frozen=True)
class TableRow:
    label: str
    run: str  # resolved canonical run id ("" for an unresolved best_of)
    summary: GroupSummary | None


@dataclass(frozen=True)
class ResolvedTable:
    paper: str
    id: str
    title: str
    kind: str
    rows: tuple[TableRow, ...]

    @property
    def columns(self) -> list[str]:
        return BENCHMARK_COLUMNS if self.kind == "benchmark" else ABLATION_COLUMNS


def _cell(summary: GroupSummary | None, key: str) -> str:
    if summary is None or not summary.has_data or key not in summary.mean:
        return TBD
    return f"{summary.mean[key]:.4f} (+/-{summary.std[key]:.4f})"


def resolve_tables(manifest: Manifest, summaries: dict[str, GroupSummary]) -> list[ResolvedTable]:
    """Resolve every manifest table row to its canonical run's summary
    (alias -> canonical; `best_of` -> computed winner, name in the label)."""
    aliases = manifest.alias_by_id()
    resolved: list[ResolvedTable] = []
    for paper, paper_tables in manifest.tables.items():
        for table in paper_tables.get("tables", []):
            rows: list[TableRow] = []
            for row in table["rows"]:
                if "run" in row:
                    run = aliases[row["run"]].canonical if row["run"] in aliases else row["run"]
                    rows.append(TableRow(row["label"], run, summaries.get(run)))
                    continue
                spec = row["best_of"]
                key = BEST_OF_METRIC_KEY[spec.get("metric", "f1_score")]
                scored: list[tuple[str, GroupSummary]] = []
                for cand in spec["candidates"]:
                    canonical = aliases[cand["run"]].canonical if cand["run"] in aliases else cand["run"]
                    summary = summaries.get(canonical)
                    if summary is not None and summary.has_data and key in summary.mean:
                        scored.append((cand["name"], summary))
                if not scored:
                    rows.append(TableRow(f"{row['label']}: {TBD}", "", None))
                    continue
                name, winner = max(scored, key=lambda t: t[1].mean[key])
                rows.append(TableRow(f"{row['label']}: {name}", winner.group_id, winner))
            resolved.append(ResolvedTable(paper, table["id"], table["title"], table["kind"], tuple(rows)))
    return resolved


def render_markdown(table: ResolvedTable) -> str:
    cols = table.columns
    lines = [
        f"# {table.title}",
        "",
        "Final-round value, mean (+/- std) across seeds. `TBD` = no completed run.",
        "",
        "| Method | " + " | ".join(METRIC_HEADER[c] for c in cols) + " | n seeds |",
        "|---|" + "---|" * (len(cols) + 1),
    ]
    for row in table.rows:
        n = row.summary.n_seeds if row.summary else 0
        lines.append(f"| {row.label} | " + " | ".join(_cell(row.summary, c) for c in cols) + f" | {n} |")
    lines.append("")
    return "\n".join(lines)


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def render_latex(table: ResolvedTable) -> str:
    cols = table.columns
    header = " & ".join(METRIC_HEADER[c] for c in cols)
    lines = [
        r"% Auto-generated by dalmax.reporting.campaign_report -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_tex_escape(table.title)}: final-round value, mean $\pm$ std across seeds.}}",
        r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}",
        r"\toprule",
        rf"Method & {header} & $n$ seeds \\",
        r"\midrule",
    ]
    for row in table.rows:
        n = row.summary.n_seeds if row.summary else 0
        cells = " & ".join(_cell(row.summary, c).replace("+/-", r"$\pm$") for c in cols)
        lines.append(rf"{_tex_escape(row.label)} & {cells} & {n} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_summary_csv(tables: list[ResolvedTable], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["paper", "table", "row", "run", "n_seeds"]
    for key in METRIC_FIELD:
        header += [f"{key}_mean", f"{key}_std"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for table in tables:
            for row in table.rows:
                s = row.summary
                line: list[object] = [table.paper, table.id, row.label, row.run, s.n_seeds if s else 0]
                for key in METRIC_FIELD:
                    has = s is not None and key in s.mean
                    line += [s.mean[key] if has else "", s.std[key] if has else ""]
                writer.writerow(line)


# --- confusion matrices -------------------------------------------------------


def read_confusion_counts(predictions_csv: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    with open(predictions_csv, newline="") as fh:
        for rec in csv.DictReader(fh):
            key = (rec["Real Class"], rec["Predicted Class"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def mean_confusion_matrix(runs: list[RunData]) -> tuple[list[str], np.ndarray] | None:
    """Mean (over the runs that have a `predictions.csv`) confusion matrix of counts,
    rows = real class, columns = predicted class, classes = sorted union."""
    per_run = [read_confusion_counts(r.leaf / "predictions.csv") for r in runs if (r.leaf / "predictions.csv").is_file()]
    if not per_run:
        return None
    labels = sorted({c for counts in per_run for pair in counts for c in pair})
    index = {c: i for i, c in enumerate(labels)}
    total = np.zeros((len(labels), len(labels)))
    for counts in per_run:
        for (real, pred), n in counts.items():
            total[index[real], index[pred]] += n
    return labels, total / len(per_run)


def write_confusion_matrix(group_id: str, runs: list[RunData], out_dir: Path) -> bool:
    result = mean_confusion_matrix(runs)
    if result is None:
        return False
    labels, matrix = result
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = group_id.replace("/", "__")
    with open(out_dir / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["real\\predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *[f"{v:.4f}" for v in row]])

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Real class")
    ax.set_title(f"{group_id}\nmean over {len(runs)} seed(s) (cell = mean count)", fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="row-normalized")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    return True


# --- orchestration ------------------------------------------------------------


def generate_report(manifest: Manifest, root: Path | None, out_dir: Path) -> dict[str, int]:
    """Write every artifact; returns small counters (for the CLI summary / tests)."""
    groups = [rebase_group(manifest, g, root) for g in manifest.groups]
    runs_by_group = {g.id: load_group_runs(g) for g in groups}
    summaries = {gid: summarize_group(gid, runs) for gid, runs in runs_by_group.items()}

    tables = resolve_tables(manifest, summaries)
    for table in tables:
        paper_dir = out_dir / table.paper
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / f"{table.id}.md").write_text(render_markdown(table))
        (paper_dir / f"{table.id}.tex").write_text(render_latex(table))
    write_summary_csv(tables, out_dir / "summary.csv")

    rebased = replace(manifest, groups=tuple(groups))
    audits = audit_seed_consistency(expand_jobs(rebased))
    (out_dir / "seed_audit.md").write_text("```\n" + render_seed_audit(audits) + "\n```\n")

    n_cm = sum(
        write_confusion_matrix(gid, runs, out_dir / "confusion_matrices")
        for gid, runs in runs_by_group.items() if runs
    )
    return {
        "tables": len(tables),
        "groups_with_data": sum(1 for s in summaries.values() if s.has_data),
        "groups": len(summaries),
        "confusion_matrices": n_cm,
        "seed_audit_failures": sum(a.status == "FAIL" for a in audits),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=None, help=f"default {DEFAULT_MANIFEST}")
    parser.add_argument("--micro", action="store_true", help=f"use {DEFAULT_MICRO_MANIFEST}")
    parser.add_argument("--root", default=None, help="results root (default: the manifest's results_root)")
    parser.add_argument("--out", default="docs/results/campaign_a100")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = load_manifest(args.manifest or (DEFAULT_MICRO_MANIFEST if args.micro else DEFAULT_MANIFEST))
    root = Path(args.root) if args.root else None
    if root is not None and not root.is_absolute():
        root = Path.cwd() / root
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    stats = generate_report(manifest, root, out_dir)
    print(
        f"Wrote {stats['tables']} tables + summary.csv + seed_audit.md + {stats['confusion_matrices']} "
        f"confusion matrices to {out_dir} ({stats['groups_with_data']}/{stats['groups']} run groups have data; "
        f"seed-audit failures: {stats['seed_audit_failures']})"
    )


if __name__ == "__main__":
    main()
