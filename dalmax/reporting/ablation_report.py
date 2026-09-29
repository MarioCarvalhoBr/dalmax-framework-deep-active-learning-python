"""Aggregate ablation results into the paper's macro-F1 tables.

Walks `<root>/<study>/<config>/<dataset>/SEED_*/NQ_*/RepresentationStrategy/
results.json` (the layout `scripts/ablations/run_ablation_gpu_*.sh` and
`scripts/ablations/smoke_ablations.sh` both produce -- see
`.specs/experiments/ablation-study.md`'s "Materialized files" section),
reads each run's `all_f1_macro`/`all_f1_score` and `run_metadata.json`, and
writes, per `--out` directory:

Since 2026-08-30 this module is **method-aware** (`--method {rnhal,texhal}`,
default `rnhal`, matching `files_config/ablations/{rnhal,texhal}/` -- see
that folder's README.md and `.specs/experiments/papers-roadmap.md`): each
method has its own `STUDY_CONFIGS` (the two only differ in §6.1's config
basenames -- rnhal slices SSRAE `full`/`spatial`/`spectral`, texhal sweeps
VCTex's `Q` scale). `--root` should already point at the method's own
results subtree (e.g. `results/ablations/texhal`, not the shared parent),
since the config *basenames* inside a study directory are method-specific
for §6.1. `STUDY_CONFIGS` (module-level, no explicit method) is kept as an
alias for `STUDY_CONFIGS_BY_METHOD["rnhal"]`, unchanged, for backward
compatibility with existing callers/tests that predate the `--method` flag.

- `ablation_summary.csv`: one row per `(study, config)`, mean +/- std across
  seeds of both the **final-round** value and the **across-rounds mean**
  ("AUC" below -- rounds are evenly spaced, so the plain mean of the
  per-round values is proportional to the trapezoidal area under the
  accuracy-vs-round curve; it is not re-normalized by round spacing since
  every ablation config in this sweep shares the same `n_round`), for both
  macro F1 (primary metric per `.claude/rules/code-quality.md` and
  `.specs/experiments/ablation-study.md`) and weighted F1 (secondary).
- `ablation_6_1.md` / `.tex`, `ablation_6_2.md` / `.tex`, `ablation_6_3.md` /
  `.tex`: one booktabs LaTeX table (+ Markdown mirror) per sub-study, using
  each row's final-round macro F1 as the primary column and final-round
  weighted F1 as a secondary column. A config with zero discovered runs is
  rendered as `TBD` rather than omitted, so every row of
  `ablation-study.md`'s run tables always has a corresponding table row.

CLI: `python -m dalmax.reporting.ablation_report --root results/ablations/rnhal
--out docs/results/ablation_tables/rnhal --method rnhal` (or `--root
results/ablations/texhal --out docs/results/ablation_tables/texhal --method
texhal` for the TexHAL suite; `make ablation-report METHOD=texhal` wraps
this). `--method` defaults to `rnhal`, matching this module's pre-2026-08-30
behavior for callers that don't pass it.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

# study -> ordered list of (config_name, display_label), matching the exact
# run-table rows in .specs/experiments/ablation-study.md and the files under
# files_config/ablations/rnhal/ (see that folder's README.md). §6.2/§6.3
# config basenames are identical between rnhal and texhal; only §6.1 differs.
STUDY_CONFIGS_RNHAL: dict[str, list[tuple[str, str]]] = {
    "6_1": [
        ("rep_full", "Full"),
        ("rep_spatial", "Spatial-only"),
        ("rep_spectral", "Spectral-only"),
    ],
    "6_2": [
        ("hier_L1", "L=1, k=[50]"),
        ("hier_L2a", "L=2, k=[300,100]"),
        ("hier_L2b", "L=2, k=[100,50]"),
        ("hier_L3", "L=3, k=[300,100,50]"),
        ("hier_L4", "L=4, k=[300,100,50,25]"),
    ],
    "6_3": [
        ("stage_full", "RNHAL (full)"),
        ("stage_no_representation", "w/o representation module"),
        ("stage_no_hierarchy", "w/o hierarchical module"),
    ],
}

# TexHAL (paper 2, VCTex): §6.1 sweeps VCTex's own multi-scale hyperparameter
# `Q` instead of a spatial/spectral column-group split (slice_embedding is
# SSRAE-only) -- see files_config/ablations/README.md and
# .specs/experiments/ablation-study-texhal.md. `rep_q13`/`rep_q17` were added
# 2026-08-30 per the VCTex method authors (Fares & Ribas). §6.2/§6.3 mirror
# rnhal's basenames exactly.
STUDY_CONFIGS_TEXHAL: dict[str, list[tuple[str, str]]] = {
    "6_1": [
        ("rep_q5", "Q=5"),
        ("rep_q13", "Q=13"),
        ("rep_q17", "Q=17"),
        ("rep_full", "Q=[5,17] (full)"),
    ],
    "6_2": [
        ("hier_L1", "L=1, k=[50]"),
        ("hier_L2a", "L=2, k=[300,100]"),
        ("hier_L2b", "L=2, k=[100,50]"),
        ("hier_L3", "L=3, k=[300,100,50]"),
        ("hier_L4", "L=4, k=[300,100,50,25]"),
    ],
    "6_3": [
        ("stage_full", "TexHAL (full)"),
        ("stage_no_representation", "w/o representation module"),
        ("stage_no_hierarchy", "w/o hierarchical module"),
    ],
}

STUDY_CONFIGS_BY_METHOD: dict[str, dict[str, list[tuple[str, str]]]] = {
    "rnhal": STUDY_CONFIGS_RNHAL,
    "texhal": STUDY_CONFIGS_TEXHAL,
}

# Backward-compatible alias: pre-`--method` callers (and tests written before
# 2026-08-30) import `STUDY_CONFIGS` directly, always meaning the rnhal suite.
STUDY_CONFIGS: dict[str, list[tuple[str, str]]] = STUDY_CONFIGS_RNHAL

# ADR 0009 (2026-09-29): `stage_no_representation` is ONE run shared by both
# papers (byte-identical config), executed under the rnhal tree only. The
# texhal suite therefore has no results of its own for that row; it is
# resolved from the sibling rnhal tree instead:
# (method, study, config) -> method whose results tree holds the run.
SHARED_RUN_SOURCE: dict[tuple[str, str, str], str] = {
    ("texhal", "6_3", "stage_no_representation"): "rnhal",
}

STUDY_TITLES: dict[str, str] = {
    "6_1": "6.1 Representation ablation",
    "6_2": "6.2 Hierarchy ablation",
    "6_3": "6.3 Contribution of the two stages",
}


@dataclass(frozen=True)
class RunRecord:
    """One `results.json` (+ its sibling `run_metadata.json`, if present)."""

    seed: int
    dataset_folder: str
    all_f1_macro: list[float]
    all_f1_score: list[float]
    results_json_path: Path
    run_metadata: dict | None


@dataclass(frozen=True)
class MetricSummary:
    """Mean +/- std across seeds, or `None` if there were zero runs."""

    n_seeds: int
    final_mean: float | None
    final_std: float | None
    auc_mean: float | None
    auc_std: float | None

    @property
    def has_data(self) -> bool:
        return self.n_seeds > 0


@dataclass(frozen=True)
class ConfigSummary:
    study: str
    config: str
    label: str
    macro: MetricSummary
    weighted: MetricSummary


def find_run_results(root: Path, study: str, config: str) -> list[Path]:
    """Find every `results.json` for `(study, config)` under `root`.

    Matches `<root>/<study>/<config>/<dataset>/SEED_*/NQ_*/
    RepresentationStrategy/results.json` -- one file per seed (usually), any
    dataset folder name, any `NQ_*` budget/round/epoch combination.
    """
    base = root / study / config
    if not base.is_dir():
        return []
    return sorted(base.glob("*/SEED_*/NQ_*/RepresentationStrategy/results.json"))


def load_run(results_json_path: Path) -> RunRecord:
    """Load one `results.json` (+ sibling `run_metadata.json`, if present)."""
    payload = json.loads(results_json_path.read_text())

    run_metadata_path = results_json_path.parent / "run_metadata.json"
    run_metadata = (
        json.loads(run_metadata_path.read_text()) if run_metadata_path.is_file() else None
    )

    # dataset folder is two levels above SEED_*/NQ_*/RepresentationStrategy/results.json
    dataset_folder = results_json_path.parents[3].name

    return RunRecord(
        seed=int(payload["seed"]),
        dataset_folder=dataset_folder,
        all_f1_macro=[float(v) for v in payload.get("all_f1_macro", [])],
        all_f1_score=[float(v) for v in payload.get("all_f1_score", [])],
        results_json_path=results_json_path,
        run_metadata=run_metadata,
    )


def _summarize_metric(values_per_seed: list[list[float]]) -> MetricSummary:
    """Build a `MetricSummary` from one metric's per-seed round-value lists.

    `final` = the last round's value per seed; `auc` = the across-rounds mean
    per seed (see module docstring for why a plain mean is a valid AUC proxy
    here). Both are then reduced to mean +/- std **across seeds**. `std` uses
    the population standard deviation (`statistics.pstdev`) -- consistent
    with `dalmax/reporting/average_results.py`'s existing
    `numpy.std` (population, not sample) convention for averaging DalMax
    seeds.
    """
    values_per_seed = [v for v in values_per_seed if v]  # drop seeds with no recorded rounds
    n = len(values_per_seed)
    if n == 0:
        return MetricSummary(n_seeds=0, final_mean=None, final_std=None, auc_mean=None, auc_std=None)

    finals = [v[-1] for v in values_per_seed]
    aucs = [statistics.mean(v) for v in values_per_seed]

    final_std = statistics.pstdev(finals) if n > 1 else 0.0
    auc_std = statistics.pstdev(aucs) if n > 1 else 0.0

    return MetricSummary(
        n_seeds=n,
        final_mean=statistics.mean(finals),
        final_std=final_std,
        auc_mean=statistics.mean(aucs),
        auc_std=auc_std,
    )


def summarize_config(study: str, config: str, label: str, runs: list[RunRecord]) -> ConfigSummary:
    macro = _summarize_metric([r.all_f1_macro for r in runs])
    weighted = _summarize_metric([r.all_f1_score for r in runs])
    return ConfigSummary(study=study, config=config, label=label, macro=macro, weighted=weighted)


def build_summaries(
    root: Path,
    study_configs: dict[str, list[tuple[str, str]]] = STUDY_CONFIGS,
    shared_roots: dict[tuple[str, str], Path] | None = None,
) -> dict[str, list[ConfigSummary]]:
    """Build every `ConfigSummary` for every `(study, config)` in
    `study_configs` (default: `STUDY_CONFIGS`, i.e. the rnhal suite -- pass
    `STUDY_CONFIGS_BY_METHOD["texhal"]` for the texhal suite), whether or not
    any runs were actually found (a zero-run config still gets a
    `ConfigSummary` with `has_data=False`, so every table row is always
    present -- see module docstring)."""
    summaries: dict[str, list[ConfigSummary]] = {}
    for study, rows in study_configs.items():
        study_summaries = []
        for config, label in rows:
            config_root = (shared_roots or {}).get((study, config), root)
            run_paths = find_run_results(config_root, study, config)
            runs = [load_run(p) for p in run_paths]
            study_summaries.append(summarize_config(study, config, label, runs))
        summaries[study] = study_summaries
    return summaries


# --- Rendering ---------------------------------------------------------------


def _fmt(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "TBD"
    return f"{mean:.4f} (+/-{std:.4f})"


def write_summary_csv(summaries: dict[str, list[ConfigSummary]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "study",
        "config",
        "label",
        "n_seeds",
        "final_f1_macro_mean",
        "final_f1_macro_std",
        "final_f1_weighted_mean",
        "final_f1_weighted_std",
        "auc_f1_macro_mean",
        "auc_f1_macro_std",
        "auc_f1_weighted_mean",
        "auc_f1_weighted_std",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for study_summaries in summaries.values():
            for s in study_summaries:
                writer.writerow(
                    [
                        s.study,
                        s.config,
                        s.label,
                        s.macro.n_seeds,
                        s.macro.final_mean,
                        s.macro.final_std,
                        s.weighted.final_mean,
                        s.weighted.final_std,
                        s.macro.auc_mean,
                        s.macro.auc_std,
                        s.weighted.auc_mean,
                        s.weighted.auc_std,
                    ]
                )


def _render_markdown(study: str, study_summaries: list[ConfigSummary]) -> str:
    title = STUDY_TITLES[study]
    lines = [
        f"# {title}",
        "",
        "Final-round macro F1 (primary) and weighted F1 (secondary), mean (+/- std) across seeds.",
        "",
        "| Variant | Macro F1 | Weighted F1 | n seeds |",
        "|---|---|---|---|",
    ]
    for s in study_summaries:
        lines.append(
            f"| {s.label} | {_fmt(s.macro.final_mean, s.macro.final_std)} "
            f"| {_fmt(s.weighted.final_mean, s.weighted.final_std)} | {s.macro.n_seeds} |"
        )
    lines.append("")
    return "\n".join(lines)


def _escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def _render_latex(study: str, study_summaries: list[ConfigSummary]) -> str:
    title = STUDY_TITLES[study]
    lines = [
        r"% Auto-generated by dalmax.reporting.ablation_report -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_escape_latex(title)}: final-round macro F1 (primary) and weighted F1 "
        r"(secondary), mean $\pm$ std across seeds.}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variant & Macro F1 & Weighted F1 & $n$ seeds \\",
        r"\midrule",
    ]
    for s in study_summaries:
        macro_cell = _fmt(s.macro.final_mean, s.macro.final_std).replace("+/-", r"$\pm$")
        weighted_cell = _fmt(s.weighted.final_mean, s.weighted.final_std).replace("+/-", r"$\pm$")
        lines.append(
            f"{_escape_latex(s.label)} & {macro_cell} & {weighted_cell} & {s.macro.n_seeds} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_study_tables(summaries: dict[str, list[ConfigSummary]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for study, study_summaries in summaries.items():
        (out_dir / f"ablation_{study}.md").write_text(_render_markdown(study, study_summaries))
        (out_dir / f"ablation_{study}.tex").write_text(_render_latex(study, study_summaries))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="results/ablations",
        help="Root directory to walk (default: results/ablations).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="paper_drafts/ablation_tables",
        help="Output directory for the CSV/MD/TeX tables (default: paper_drafts/ablation_tables).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=sorted(STUDY_CONFIGS_BY_METHOD),
        default="rnhal",
        help=(
            "Which ablation suite's (study, config) basenames to look for under --root "
            "(default: rnhal). --root should already point at that method's own results "
            "subtree, e.g. results/ablations/texhal for --method texhal -- see "
            "files_config/ablations/README.md."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out)
    study_configs = STUDY_CONFIGS_BY_METHOD[args.method]

    shared_roots = {
        (study, config): root.parent / source
        for (method, study, config), source in SHARED_RUN_SOURCE.items()
        if method == args.method
    }
    summaries = build_summaries(root, study_configs, shared_roots)
    write_summary_csv(summaries, out_dir / "ablation_summary.csv")
    write_study_tables(summaries, out_dir)

    total_configs = sum(len(v) for v in summaries.values())
    total_with_data = sum(1 for v in summaries.values() for s in v if s.macro.has_data)
    print(
        f"Wrote ablation tables to {out_dir} "
        f"({total_with_data}/{total_configs} configs have discovered runs under {root})"
    )


if __name__ == "__main__":
    main()
