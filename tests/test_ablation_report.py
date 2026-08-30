"""Tests for `dalmax.reporting.ablation_report`.

Builds a synthetic `results/ablations/<study>/<config>/<dataset>/SEED_*/
NQ_*/RepresentationStrategy/results.json` tree in `tmp_path` (matching the
layout `scripts/ablations/run_ablation_gpu_*.sh` and
`scripts/ablations/smoke_ablations.sh` actually produce) and checks the
aggregation math, the zero-runs "TBD" path, and that every output file gets
written. See `.specs/experiments/ablation-study.md` for the real run tables
this mirrors.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dalmax.reporting.ablation_report import (
    STUDY_CONFIGS,
    STUDY_CONFIGS_BY_METHOD,
    STUDY_CONFIGS_TEXHAL,
    build_arg_parser,
    build_summaries,
    find_run_results,
    load_run,
    main,
    write_study_tables,
    write_summary_csv,
)


def _write_run(
    root: Path,
    study: str,
    config: str,
    seed: int,
    *,
    all_f1_macro: list[float],
    all_f1_score: list[float],
    dataset: str = "daninhas_full",
    nq_dir: str = "NQ_100_NIL_100_NR_8_NE_10",
) -> Path:
    leaf = (
        root
        / study
        / config
        / dataset
        / f"SEED_{seed}"
        / nq_dir
        / "RepresentationStrategy"
    )
    leaf.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": "DANINHAS",
        "strategy_name": "RepresentationStrategy",
        "n_init_labeled": 100,
        "n_query": 100,
        "n_round": len(all_f1_macro) - 1,
        "seed": seed,
        "all_acc": [0.5] * len(all_f1_macro),
        "all_precision": [0.5] * len(all_f1_macro),
        "all_recall": [0.5] * len(all_f1_macro),
        "all_f1_score": all_f1_score,
        "rounds": list(range(len(all_f1_macro))),
        "all_precision_macro": [0.5] * len(all_f1_macro),
        "all_recall_macro": [0.5] * len(all_f1_macro),
        "all_f1_macro": all_f1_macro,
    }
    results_json = leaf / "results.json"
    results_json.write_text(json.dumps(payload))
    (leaf / "run_metadata.json").write_text(json.dumps({"config": {"seed": seed}}))
    return results_json


# --- find_run_results / load_run --------------------------------------------


def test_find_run_results_returns_nothing_for_a_config_with_no_runs(tmp_path: Path):
    assert find_run_results(tmp_path, "6_1", "rep_full") == []


def test_find_run_results_finds_every_seed(tmp_path: Path):
    _write_run(tmp_path, "6_1", "rep_full", seed=1, all_f1_macro=[0.1, 0.2], all_f1_score=[0.1, 0.2])
    _write_run(tmp_path, "6_1", "rep_full", seed=2, all_f1_macro=[0.1, 0.3], all_f1_score=[0.1, 0.3])

    found = find_run_results(tmp_path, "6_1", "rep_full")

    assert len(found) == 2
    assert all(p.name == "results.json" for p in found)


def test_load_run_reads_metrics_and_dataset_folder(tmp_path: Path):
    path = _write_run(
        tmp_path, "6_1", "rep_full", seed=7, all_f1_macro=[0.4, 0.6], all_f1_score=[0.3, 0.5]
    )

    run = load_run(path)

    assert run.seed == 7
    assert run.dataset_folder == "daninhas_full"
    assert run.all_f1_macro == [0.4, 0.6]
    assert run.all_f1_score == [0.3, 0.5]
    assert run.run_metadata == {"config": {"seed": 7}}


def test_load_run_tolerates_missing_run_metadata(tmp_path: Path):
    path = _write_run(tmp_path, "6_1", "rep_full", seed=1, all_f1_macro=[0.5], all_f1_score=[0.4])
    (path.parent / "run_metadata.json").unlink()

    run = load_run(path)

    assert run.run_metadata is None


# --- summarize_config / build_summaries -------------------------------------


def test_build_summaries_computes_mean_and_std_across_seeds(tmp_path: Path):
    # seed 1: rounds [0.2, 0.4, 0.6] -> final=0.6, auc=mean(0.2,0.4,0.6)=0.4
    # seed 2: rounds [0.2, 0.4, 0.8] -> final=0.8, auc=mean(0.2,0.4,0.8)=0.4667
    _write_run(tmp_path, "6_1", "rep_full", seed=1, all_f1_macro=[0.2, 0.4, 0.6], all_f1_score=[0.1, 0.2, 0.3])
    _write_run(tmp_path, "6_1", "rep_full", seed=2, all_f1_macro=[0.2, 0.4, 0.8], all_f1_score=[0.1, 0.2, 0.5])

    summaries = build_summaries(tmp_path)
    rep_full = next(s for s in summaries["6_1"] if s.config == "rep_full")

    assert rep_full.macro.n_seeds == 2
    assert rep_full.macro.final_mean == pytest.approx((0.6 + 0.8) / 2)
    assert rep_full.macro.auc_mean == pytest.approx((0.4 + (0.2 + 0.4 + 0.8) / 3) / 2)
    assert rep_full.macro.final_std > 0
    assert rep_full.weighted.final_mean == pytest.approx((0.3 + 0.5) / 2)


def test_build_summaries_reports_tbd_for_a_config_with_no_runs(tmp_path: Path):
    summaries = build_summaries(tmp_path)

    for study_summaries in summaries.values():
        for s in study_summaries:
            assert s.macro.n_seeds == 0
            assert s.macro.final_mean is None
            assert not s.macro.has_data


def test_build_summaries_covers_every_run_table_row_even_with_partial_data(tmp_path: Path):
    _write_run(tmp_path, "6_2", "hier_L1", seed=1, all_f1_macro=[0.5], all_f1_score=[0.4])

    summaries = build_summaries(tmp_path)

    assert {c for c, _ in STUDY_CONFIGS["6_2"]} == {s.config for s in summaries["6_2"]}
    hier_l1 = next(s for s in summaries["6_2"] if s.config == "hier_L1")
    hier_l2a = next(s for s in summaries["6_2"] if s.config == "hier_L2a")
    assert hier_l1.macro.has_data
    assert not hier_l2a.macro.has_data


# --- output files -------------------------------------------------------------


def test_write_summary_csv_has_a_row_per_config(tmp_path: Path):
    _write_run(tmp_path, "6_1", "rep_full", seed=1, all_f1_macro=[0.5], all_f1_score=[0.4])
    summaries = build_summaries(tmp_path)
    out_csv = tmp_path / "out" / "ablation_summary.csv"

    write_summary_csv(summaries, out_csv)

    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    total_rows = sum(len(v) for v in STUDY_CONFIGS.values())
    assert len(rows) == total_rows
    rep_full_row = next(r for r in rows if r["config"] == "rep_full")
    assert rep_full_row["n_seeds"] == "1"
    assert float(rep_full_row["final_f1_macro_mean"]) == pytest.approx(0.5)
    hier_l1_row = next(r for r in rows if r["config"] == "hier_L1")
    assert hier_l1_row["final_f1_macro_mean"] == ""  # None -> empty CSV cell


def test_write_study_tables_writes_md_and_tex_for_every_study(tmp_path: Path):
    summaries = build_summaries(tmp_path)
    out_dir = tmp_path / "tables"

    write_study_tables(summaries, out_dir)

    for study in STUDY_CONFIGS:
        assert (out_dir / f"ablation_{study}.md").is_file()
        assert (out_dir / f"ablation_{study}.tex").is_file()

    md_6_1 = (out_dir / "ablation_6_1.md").read_text()
    assert "Spatial-only" in md_6_1
    assert "TBD" in md_6_1  # no runs written for this synthetic tree

    tex_6_2 = (out_dir / "ablation_6_2.tex").read_text()
    assert r"\begin{tabular}" in tex_6_2
    assert r"\toprule" in tex_6_2
    assert r"\bottomrule" in tex_6_2
    # every \caption{...} must be balanced -- regression test for a doubled
    # closing brace bug in _render_latex.
    assert tex_6_2.count("{") == tex_6_2.count("}")
    assert r"\caption{6.2 Hierarchy ablation" in tex_6_2


def test_tbd_row_renders_as_tbd_in_markdown_and_latex(tmp_path: Path):
    summaries = build_summaries(tmp_path)
    out_dir = tmp_path / "tables"
    write_study_tables(summaries, out_dir)

    md = (out_dir / "ablation_6_3.md").read_text()
    tex = (out_dir / "ablation_6_3.tex").read_text()

    assert "TBD" in md
    assert "TBD" in tex


def test_main_cli_writes_all_expected_files(tmp_path: Path):
    root = tmp_path / "results" / "ablations"
    _write_run(root, "6_1", "rep_full", seed=1, all_f1_macro=[0.5, 0.6], all_f1_score=[0.4, 0.5])
    out_dir = tmp_path / "paper_drafts" / "ablation_tables"

    main(["--root", str(root), "--out", str(out_dir)])

    assert (out_dir / "ablation_summary.csv").is_file()
    for study in STUDY_CONFIGS:
        assert (out_dir / f"ablation_{study}.md").is_file()
        assert (out_dir / f"ablation_{study}.tex").is_file()


# --- --method (2026-08-30 texhal support) -----------------------------------


def test_build_summaries_accepts_texhal_study_configs(tmp_path: Path):
    _write_run(tmp_path, "6_1", "rep_q5", seed=1, all_f1_macro=[0.3], all_f1_score=[0.2])

    summaries = build_summaries(tmp_path, STUDY_CONFIGS_TEXHAL)

    assert {c for c, _ in STUDY_CONFIGS_TEXHAL["6_1"]} == {s.config for s in summaries["6_1"]}
    rep_q5 = next(s for s in summaries["6_1"] if s.config == "rep_q5")
    assert rep_q5.macro.has_data
    # texhal's §6.1 has a 4th row (rep_q13) that rnhal's doesn't.
    assert "rep_q13" in {c for c, _ in STUDY_CONFIGS_TEXHAL["6_1"]}
    assert "rep_q13" not in {c for c, _ in STUDY_CONFIGS["6_1"]}


def test_main_cli_method_texhal_uses_texhal_config_names(tmp_path: Path):
    root = tmp_path / "results" / "ablations" / "texhal"
    _write_run(root, "6_1", "rep_q17", seed=1, all_f1_macro=[0.5], all_f1_score=[0.4])
    out_dir = tmp_path / "docs" / "ablation_tables" / "texhal"

    main(["--root", str(root), "--out", str(out_dir), "--method", "texhal"])

    md_6_1 = (out_dir / "ablation_6_1.md").read_text()
    assert "Q=17" in md_6_1
    assert "Q=13" in md_6_1  # zero-run row still rendered as TBD
    assert "Spatial-only" not in md_6_1


def test_main_cli_default_method_is_rnhal() -> None:
    args = build_arg_parser().parse_args([])
    assert args.method == "rnhal"
    assert STUDY_CONFIGS_BY_METHOD[args.method] is STUDY_CONFIGS
