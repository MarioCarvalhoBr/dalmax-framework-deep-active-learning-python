"""Tests for `dalmax.reporting.results_doctor`.

Builds synthetic `results/ablations/[<method>/]<study>/<config>/<dataset>/
SEED_*/NQ_*/RepresentationStrategy/` leaves in `tmp_path` to exercise
`verify` and `migrate-legacy` without ever touching the real, on-disk
`results/ablations/` tree -- the migration function under test must only run
against a synthetic tree here (`.claude/rules/data-safety.md`: `results/` is
append-only and this session's migration must not be exercised against the
real local tree; the user runs the real migration on Colab/Drive
themselves).

One `@pytest.mark.dataset` test at the bottom *does* run `verify` (read-only)
against the real local `results/ablations` legacy tree to confirm the
already-executed 33-run RNHAL batch is fully OK.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dalmax.reporting.ablation_report import STUDY_CONFIGS_BY_METHOD
from dalmax.reporting.results_doctor import (
    REQUIRED_FILES,
    SAVED_MODEL_MIN_BYTES,
    build_arg_parser,
    check_leaf,
    main,
    plan_migration,
    verify_method,
)

STRATEGY_NAME = "RepresentationStrategy"


def _write_leaf(
    base: Path,
    seed: int,
    *,
    dataset: str = "daninhas_full",
    nq_dir: str = "NQ_100_NIL_100_NR_8_NE_10",
    n_round: int = 8,
    omit: tuple[str, ...] = (),
    saved_model_bytes: int = SAVED_MODEL_MIN_BYTES + 1024,
) -> Path:
    """Write a complete (unless `omit` names files to skip) leaf directory
    under `base` (a `<root>/[<method>/]<study>/<config>/` directory)."""
    leaf = base / dataset / f"SEED_{seed}" / nq_dir / STRATEGY_NAME
    leaf.mkdir(parents=True, exist_ok=True)

    if "results.json" not in omit:
        payload = {
            "dataset_name": "DANINHAS",
            "strategy_name": STRATEGY_NAME,
            "seed": seed,
            "all_acc": [0.5] * (n_round + 1),
            "all_f1_score": [0.5] * (n_round + 1),
            "all_f1_macro": [0.5] * (n_round + 1),
            "rounds": list(range(n_round + 1)),
        }
        (leaf / "results.json").write_text(json.dumps(payload))

    if "run_metadata.json" not in omit:
        (leaf / "run_metadata.json").write_text(
            json.dumps({"config": {"seed": seed}, "git_commit": "abc123"})
        )

    if "predictions.csv" not in omit:
        (leaf / "predictions.csv").write_text("Image Index,Real Class\n0,GRAMINEA\n")

    for pdf in ("confusion_matrix.pdf", "accuracy.pdf", "precision.pdf", "recall.pdf", "f1_score.pdf"):
        if pdf not in omit:
            (leaf / pdf).write_bytes(b"%PDF-1.4 fake")

    if "log-dalmax.log" not in omit:
        (leaf / "log-dalmax.log").write_text("log\n")

    if "saved_model.pth" not in omit:
        (leaf / "saved_model.pth").write_bytes(b"0" * saved_model_bytes)

    return leaf


# --- check_leaf ----------------------------------------------------------------


def test_check_leaf_complete_has_no_missing_or_warnings(tmp_path: Path) -> None:
    leaf = _write_leaf(tmp_path / "6_1" / "rep_full", seed=1)

    missing, warnings = check_leaf(leaf)

    assert missing == []
    assert warnings == []


def test_check_leaf_reports_missing_predictions_csv(tmp_path: Path) -> None:
    leaf = _write_leaf(tmp_path / "6_1" / "rep_full", seed=1, omit=("predictions.csv",))

    missing, _warnings = check_leaf(leaf)

    assert missing == ["predictions.csv"]


def test_check_leaf_warns_on_tiny_saved_model(tmp_path: Path) -> None:
    leaf = _write_leaf(tmp_path / "6_1" / "rep_full", seed=1, saved_model_bytes=900)

    missing, warnings = check_leaf(leaf)

    assert missing == []
    assert any("saved_model.pth" in w and "10 MB" in w for w in warnings)


def test_check_leaf_warns_on_round_count_mismatch(tmp_path: Path) -> None:
    # nq_dir claims NR_8 (9 rounds expected) but only 3 rounds are recorded.
    leaf = _write_leaf(tmp_path / "6_1" / "rep_full", seed=1, n_round=2, nq_dir="NQ_100_NIL_100_NR_8_NE_10")

    missing, warnings = check_leaf(leaf)

    assert missing == []
    assert any("len(rounds)" in w for w in warnings)


def test_check_leaf_warns_on_unparsable_results_json(tmp_path: Path) -> None:
    leaf = _write_leaf(tmp_path / "6_1" / "rep_full", seed=1, omit=("results.json",))
    (leaf / "results.json").write_text("{not json")

    missing, warnings = check_leaf(leaf)

    assert missing == []
    assert any("results.json" in w and "parse" in w for w in warnings)


def test_all_required_files_present_in_helper() -> None:
    # sanity: the test helper writes every file check_leaf checks for.
    assert set(REQUIRED_FILES) == {
        "results.json", "predictions.csv", "run_metadata.json",
        "confusion_matrix.pdf", "accuracy.pdf", "precision.pdf",
        "recall.pdf", "f1_score.pdf", "log-dalmax.log", "saved_model.pth",
    }


# --- verify_method ---------------------------------------------------------------


def test_verify_method_reports_missing_for_untouched_tree(tmp_path: Path) -> None:
    statuses = verify_method(tmp_path, "rnhal")

    expected_triples = sum(len(rows) for rows in STUDY_CONFIGS_BY_METHOD["rnhal"].values()) * 3
    assert len(statuses) == expected_triples
    assert all(s.status == "MISSING" for s in statuses)


def test_verify_method_finds_current_layout_leaf(tmp_path: Path) -> None:
    _write_leaf(tmp_path / "rnhal" / "6_1" / "rep_full", seed=1)

    statuses = verify_method(tmp_path, "rnhal")

    hit = next(s for s in statuses if s.study == "6_1" and s.config == "rep_full" and s.seed == 1)
    assert hit.status == "OK"
    assert hit.layout == "current"


def test_verify_method_falls_back_to_legacy_layout(tmp_path: Path) -> None:
    _write_leaf(tmp_path / "6_1" / "rep_full", seed=1)  # no rnhal/ segment

    statuses = verify_method(tmp_path, "rnhal")

    hit = next(s for s in statuses if s.study == "6_1" and s.config == "rep_full" and s.seed == 1)
    assert hit.status == "OK"
    assert hit.layout == "legacy"


def test_verify_method_prefers_current_over_legacy_when_both_ok(tmp_path: Path) -> None:
    _write_leaf(tmp_path / "6_1" / "rep_full", seed=1)  # legacy
    _write_leaf(tmp_path / "rnhal" / "6_1" / "rep_full", seed=1)  # current

    statuses = verify_method(tmp_path, "rnhal")

    hit = next(s for s in statuses if s.study == "6_1" and s.config == "rep_full" and s.seed == 1)
    assert hit.status == "OK"
    assert hit.layout == "current"


def test_verify_method_incomplete_when_leaf_exists_but_missing_files(tmp_path: Path) -> None:
    _write_leaf(tmp_path / "rnhal" / "6_1" / "rep_full", seed=1, omit=("predictions.csv",))

    statuses = verify_method(tmp_path, "rnhal")

    hit = next(s for s in statuses if s.study == "6_1" and s.config == "rep_full" and s.seed == 1)
    assert hit.status == "INCOMPLETE"
    assert "predictions.csv" in hit.missing_files


def test_verify_method_texhal_has_no_legacy_fallback(tmp_path: Path) -> None:
    # A texhal leaf written at the "legacy" (no-method-segment) path must
    # NOT be picked up -- only rnhal has a legacy layout.
    _write_leaf(tmp_path / "6_1" / "rep_q5", seed=1)

    statuses = verify_method(tmp_path, "texhal")

    hit = next(s for s in statuses if s.study == "6_1" and s.config == "rep_q5" and s.seed == 1)
    assert hit.status == "MISSING"


# --- CLI verify exit codes ----------------------------------------------------


def test_cli_verify_exit_0_when_fully_migrated_and_complete(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    for study, rows in STUDY_CONFIGS_BY_METHOD["rnhal"].items():
        for config, _label in rows:
            for seed in (1, 2, 3):
                _write_leaf(root / "rnhal" / study / config, seed=seed)

    exit_code = main(["verify", "--root", str(root), "--method", "rnhal"])

    assert exit_code == 0


def test_cli_verify_exit_0_when_satisfied_via_legacy_layout(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    for study, rows in STUDY_CONFIGS_BY_METHOD["rnhal"].items():
        for config, _label in rows:
            for seed in (1, 2, 3):
                _write_leaf(root / study / config, seed=seed)  # legacy layout only

    exit_code = main(["verify", "--root", str(root), "--method", "rnhal"])

    assert exit_code == 0


def test_cli_verify_exit_1_when_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    exit_code = main(["verify", "--root", str(root), "--method", "rnhal"])

    assert exit_code == 1


def test_cli_verify_writes_json_dump(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_leaf(root / "rnhal" / "6_1" / "rep_full", seed=1)
    json_out = tmp_path / "verify.json"

    main(["verify", "--root", str(root), "--method", "rnhal", "--json", str(json_out)])

    payload = json.loads(json_out.read_text())
    assert "rnhal" in payload
    assert any(row["config"] == "rep_full" and row["seed"] == 1 for row in payload["rnhal"])


def test_build_arg_parser_verify_defaults() -> None:
    args = build_arg_parser().parse_args(["verify"])
    assert args.root == "results/ablations"
    assert args.method == "all"
    assert args.json_out is None


def test_build_arg_parser_migrate_defaults() -> None:
    args = build_arg_parser().parse_args(["migrate-legacy"])
    assert args.root == "results/ablations"
    assert args.apply is False


# --- migrate-legacy: plan / dry-run / apply / conflicts / idempotency --------


def _write_legacy_tree(root: Path) -> None:
    for study, rows in STUDY_CONFIGS_BY_METHOD["rnhal"].items():
        for config, _label in rows:
            for seed in (1, 2, 3):
                _write_leaf(root / study / config, seed=seed)
    (root / "gpu0_failures.log").write_text("")
    (root / "gpu1_failures.log").write_text("")
    (root / "colab.log").write_text("done\n")


def test_plan_migration_finds_every_legacy_item(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_legacy_tree(root)

    plan = plan_migration(root)

    move_srcs = {src.name for src, _dst in plan.moves}
    assert move_srcs == {"6_1", "6_2", "6_3", "gpu0_failures.log", "gpu1_failures.log", "colab.log"}
    assert plan.conflicts == []


def test_plan_migration_empty_for_untouched_root(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    root.mkdir(parents=True)

    plan = plan_migration(root)

    assert plan.moves == []
    assert plan.conflicts == []


def test_cli_migrate_dry_run_does_not_move_anything(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_legacy_tree(root)

    exit_code = main(["migrate-legacy", "--root", str(root)])

    assert exit_code == 0
    assert (root / "6_1").is_dir()  # untouched
    assert not (root / "rnhal").exists()


def test_cli_migrate_apply_moves_everything(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_legacy_tree(root)

    exit_code = main(["migrate-legacy", "--root", str(root), "--apply"])

    assert exit_code == 0
    assert not (root / "6_1").exists()
    assert not (root / "gpu0_failures.log").exists()
    assert (root / "rnhal" / "6_1" / "rep_full").is_dir()
    assert (root / "rnhal" / "gpu0_failures.log").is_file()
    assert (root / "rnhal" / "colab.log").read_text() == "done\n"
    # contents of a moved leaf are untouched -- same files, same bytes.
    moved_leaf = next((root / "rnhal" / "6_1" / "rep_full").glob("*/SEED_1/NQ_*/RepresentationStrategy"))
    assert (moved_leaf / "results.json").is_file()


def test_cli_migrate_apply_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_legacy_tree(root)
    main(["migrate-legacy", "--root", str(root), "--apply"])

    exit_code = main(["migrate-legacy", "--root", str(root), "--apply"])

    assert exit_code == 0


def test_cli_migrate_conflict_skips_and_exits_nonzero(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_legacy_tree(root)
    # Pre-create a conflicting destination for one study.
    (root / "rnhal" / "6_1").mkdir(parents=True)

    exit_code = main(["migrate-legacy", "--root", str(root), "--apply"])

    assert exit_code == 1
    assert (root / "6_1").exists()  # not moved -- conflict, left in place
    # the non-conflicting items still moved.
    assert not (root / "6_2").exists()
    assert (root / "rnhal" / "6_2").is_dir()


def test_cli_migrate_never_touches_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "results" / "ablations"
    _write_legacy_tree(root)
    sibling = tmp_path / "results" / "dalmax1"
    sibling.mkdir(parents=True)
    (sibling / "marker.txt").write_text("do not touch")

    main(["migrate-legacy", "--root", str(root), "--apply"])

    assert (sibling / "marker.txt").read_text() == "do not touch"


# --- real local tree (read-only) ----------------------------------------------


@pytest.mark.dataset
def test_verify_real_local_legacy_rnhal_tree_is_fully_ok() -> None:
    """The already-executed 33-run RNHAL batch lives at the legacy root
    `results/ablations/{6_1,6_2,6_3}` (see CLAUDE.md). This test only reads
    it -- never migrates or modifies it (`.claude/rules/data-safety.md`)."""
    root = Path(__file__).resolve().parent.parent / "results" / "ablations"
    if not root.is_dir():
        pytest.skip("results/ablations not present in this checkout")

    statuses = verify_method(root, "rnhal")
    legacy_ok = [s for s in statuses if s.status == "OK" and s.layout == "legacy"]
    not_ok = [s for s in statuses if s.status != "OK"]

    assert not_ok == [], f"expected the legacy batch to be fully complete, found: {not_ok}"
    assert len(legacy_ok) == 33


def test_verify_method_texhal_no_representation_row_resolves_to_the_rnhal_tree(tmp_path: Path) -> None:
    # ADR 0009: one shared run; texhal has no results of its own for that row.
    _write_leaf(tmp_path / "rnhal" / "6_3" / "stage_no_representation", seed=1)

    statuses = verify_method(tmp_path, "texhal")

    hit = next(
        s for s in statuses
        if s.study == "6_3" and s.config == "stage_no_representation" and s.seed == 1
    )
    assert hit.status == "OK"
    assert hit.layout == "shared"
    assert hit.method == "texhal"
    miss = next(s for s in statuses if s.config == "stage_no_representation" and s.seed == 2)
    assert miss.status == "MISSING"
