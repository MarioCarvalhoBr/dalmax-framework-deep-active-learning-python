"""Tests for `dalmax.reporting.campaign_report` against a synthetic results tree
(no training): alias resolution (aliased rows show the canonical run's value),
`best_of` (winner computed, name printed), `TBD` rendering for rows without runs,
md/tex/csv output, mean confusion matrices, and the seed audit file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dalmax import campaign as C
from dalmax.reporting import campaign_report as R

SEEDS = (1, 2, 3)


def _group(tmp: Path, gid: str, strategy: str, part: str = "paper1") -> C.RunGroup:
    return C.RunGroup(
        id=gid, part=part, params_json="files_config/campaign/params_paper1_micro.json", strategy_name=strategy,
        n_query=100, n_init_labeled=100, n_round=8, seeds=SEEDS,
        dir_results=str(tmp / gid.replace("/", "_")) + "/", used_by=(),
    )


def _write_run(group: C.RunGroup, seed: int, f1: float, *, preds: list[tuple[str, str]]) -> None:
    leaf = Path(group.dir_results) / "daninhas_full" / f"SEED_{seed}" / "NQ_100_NIL_100_NR_8_NE_10" / group.strategy_name
    leaf.mkdir(parents=True)
    (leaf / "results.json").write_text(json.dumps({
        "all_acc": [0.1, f1], "all_precision": [0.1, f1], "all_recall": [0.1, f1],
        "all_f1_score": [0.1, f1], "all_f1_macro": [0.1, f1 - 0.1], "rounds": [0, 1],
    }))
    rows = ["Image Index,Real Class,Predicted Class,Correct,Path"]
    rows += [f"{i},{real},{pred},{real == pred},p{i}" for i, (real, pred) in enumerate(preds)]
    (leaf / "predictions.csv").write_text("\n".join(rows) + "\n")
    (leaf / "log-dalmax.log").write_text("Initial labeled idxs (sorted): [1, 2, 3]\n")


def _manifest(tmp: Path) -> C.Manifest:
    groups = (
        _group(tmp, "p1/RandomSampling/nq100", "RandomSampling"),
        _group(tmp, "p1/MarginSampling/nq100", "MarginSampling"),
        _group(tmp, "p1/KMH/nq100", "RepresentationStrategy"),
        _group(tmp, "texhal/6_3/stage_full", "RepresentationStrategy", "texhal"),
        _group(tmp, "texhal/6_2/hier_L1", "RepresentationStrategy", "texhal"),  # never run -> TBD
    )
    aliases = (
        C.Alias("texhal/6_1/rep_full", "texhal/6_3/stage_full", "x.json", "RepresentationStrategy", "same"),
        C.Alias("texhal/6_3/stage_no_representation", "p1/KMH/nq100", "x.json", "RepresentationStrategy", "same"),
    )
    tables = {
        "paper1": {"tables": [{"id": "paper1_nq100", "kind": "benchmark", "title": "P1 nq100", "rows": [
            {"label": "RandomSampling", "run": "p1/RandomSampling/nq100"},
            {"label": "MarginSampling", "run": "p1/MarginSampling/nq100"},
            {"label": "KMH (hier)", "run": "p1/KMH/nq100"},
        ]}]},
        "paper2": {"tables": [
            {"id": "6_1", "kind": "ablation", "title": "6.1", "rows": [
                {"label": "Q=[5,17] (full)", "run": "texhal/6_1/rep_full"}]},
            {"id": "6_2", "kind": "ablation", "title": "6.2", "rows": [
                {"label": "L=1", "run": "texhal/6_2/hier_L1"}]},
            {"id": "6_3", "kind": "ablation", "title": "6.3", "rows": [
                {"label": "TexHAL (full)", "run": "texhal/6_3/stage_full"},
                {"label": "w/o representation", "run": "texhal/6_3/stage_no_representation"}]},
            {"id": "comparison", "kind": "comparison", "title": "cmp", "rows": [
                {"label": "TexHAL", "run": "texhal/6_3/stage_full"},
                {"label": "Best paper-1", "best_of": {"candidates": [
                    {"run": "p1/RandomSampling/nq100", "name": "RandomSampling"},
                    {"run": "p1/MarginSampling/nq100", "name": "MarginSampling"},
                    {"run": "p1/KMH/nq100", "name": "KMH"}], "metric": "f1_score"}},
            ]},
        ]},
    }
    return C.Manifest(name="t", dataset_name="DANINHAS", results_root=str(tmp), seeds=SEEDS, primary_nq=100,
                      groups=groups, aliases=aliases, tables=tables)


def _populate(manifest: C.Manifest) -> None:
    values = {"p1/RandomSampling/nq100": 0.50, "p1/MarginSampling/nq100": 0.70, "p1/KMH/nq100": 0.60,
              "texhal/6_3/stage_full": 0.80}
    for gid, f1 in values.items():
        group = manifest.group_by_id()[gid]
        for seed in SEEDS:
            _write_run(group, seed, f1 + 0.01 * seed,
                       preds=[("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")] if seed != 3 else
                             [("a", "a"), ("a", "a"), ("b", "b"), ("b", "a")])


def test_report_end_to_end_alias_best_of_tbd_and_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "res")
    _populate(manifest)
    out = tmp_path / "out"

    stats = R.generate_report(manifest, None, out)

    assert stats["tables"] == 5 and stats["groups_with_data"] == 4 and stats["groups"] == 5
    assert stats["seed_audit_failures"] == 0

    p1 = (out / "paper1" / "paper1_nq100.md").read_text()
    assert "| MarginSampling | 0.7200 (+/-0.0082) |" in p1  # mean of .71,.72,.73; population std
    assert "F1 (macro)" in p1 and "Accuracy" in p1

    # Aliased rows show the SAME value as the canonical run they resolve to.
    p2_61 = (out / "paper2" / "6_1.md").read_text()
    p2_63 = (out / "paper2" / "6_3.md").read_text()
    full_cell = "0.8200 (+/-0.0082)"
    assert full_cell in p2_61 and f"| TexHAL (full) | {full_cell}" in p2_63
    kmh_cell = "0.6200 (+/-0.0082)"
    assert f"| w/o representation | {kmh_cell}" in p2_63
    assert "| KMH (hier) | " in p1 and kmh_cell in p1

    # A row with no runs renders TBD.
    p2_62 = (out / "paper2" / "6_2.md").read_text()
    assert "| L=1 | TBD | TBD | 0 |" in p2_62

    # best_of: winner computed (MarginSampling, 0.72) and its name printed.
    cmp_md = (out / "paper2" / "comparison.md").read_text()
    assert "Best paper-1: MarginSampling" in cmp_md and "0.7200" in cmp_md

    tex = (out / "paper2" / "6_3.tex").read_text()
    assert r"\toprule" in tex and r"$\pm$" in tex and "TexHAL (full)" in tex
    assert "TBD" in (out / "paper2" / "6_2.tex").read_text()

    with open(out / "summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    by_key = {(r["paper"], r["table"], r["row"]): r for r in rows}
    assert by_key[("paper2", "6_1", "Q=[5,17] (full)")]["run"] == "texhal/6_3/stage_full"
    assert by_key[("paper2", "6_1", "Q=[5,17] (full)")]["f1_weighted_mean"] == by_key[
        ("paper2", "6_3", "TexHAL (full)")]["f1_weighted_mean"]
    assert by_key[("paper2", "6_2", "L=1")]["f1_weighted_mean"] == ""

    assert "seed 1: PASS" in (out / "seed_audit.md").read_text()


def test_mean_confusion_matrix_is_averaged_over_seeds(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "res")
    _populate(manifest)
    out = tmp_path / "out"
    R.generate_report(manifest, None, out)

    cm_dir = out / "confusion_matrices"
    assert (cm_dir / "p1__RandomSampling__nq100.pdf").is_file()
    assert not (cm_dir / "texhal__6_2__hier_L1.csv").exists()  # no runs -> no matrix
    with open(cm_dir / "p1__RandomSampling__nq100.csv") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][1:] == ["a", "b"]
    # seeds 1,2: [[1,1],[0,2]]; seed 3: [[2,0],[1,1]] -> mean [[4/3, 2/3], [1/3, 5/3]]
    assert [float(v) for v in rows[1][1:]] == [1.3333, 0.6667]
    assert [float(v) for v in rows[2][1:]] == [0.3333, 1.6667]


def test_report_with_no_results_is_all_tbd(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "res")
    out = tmp_path / "out"
    stats = R.generate_report(manifest, None, out)
    assert stats["groups_with_data"] == 0 and stats["confusion_matrices"] == 0
    md = (out / "paper2" / "comparison.md").read_text()
    assert "Best paper-1: TBD" in md and "TBD" in (out / "paper1" / "paper1_nq100.md").read_text()


def test_root_rebase(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "res")
    _populate(manifest)
    moved = tmp_path / "moved"
    (tmp_path / "res").rename(moved)
    stats = R.generate_report(manifest, moved, tmp_path / "out")
    assert stats["groups_with_data"] == 4


def test_seed_audit_failure_is_reported(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "res")
    _populate(manifest)
    bad = manifest.group_by_id()["p1/KMH/nq100"]
    leaf = C.find_leaf(bad, 2)
    (leaf / "log-dalmax.log").write_text("Initial labeled idxs (sorted): [7, 8, 9]\n")
    stats = R.generate_report(manifest, None, tmp_path / "out")
    assert stats["seed_audit_failures"] == 1
    assert "seed 2: FAIL" in (tmp_path / "out" / "seed_audit.md").read_text()
