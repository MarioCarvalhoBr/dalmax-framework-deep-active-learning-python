"""`scripts/campaign/archive_superseded_runs.py` (ADR 0014) on temporary trees only: never
against the real `results/`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD = "2026-10-01T10:00:00.000000+00:00"
NEW = "2026-11-12T08:00:00.000000+00:00"
LEAF_FILES = ("results.json", "predictions.csv", "log-dalmax.log", "saved_model.pth")


def _script():
    path = REPO_ROOT / "scripts/campaign/archive_superseded_runs.py"
    spec = importlib.util.spec_from_file_location("archive_superseded_runs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["archive_superseded_runs"] = module
    spec.loader.exec_module(module)
    return module


A = _script()


def _group(gid: str, part: str, strategy: str, nq: int = 5) -> dict:
    return {
        "id": gid,
        "part": part,
        "strategy_name": strategy,
        "n_query": nq,
        "n_init_labeled": 10,
        "n_round": 1,
        "seeds": [1, 2],
        "dir_results": f"results/campaign/paper1/nq{nq}/",
    }


MANIFEST = {
    "seeds": [1, 2],
    "groups": [
        _group("p1/KMeansSampling/nq5", "paper1_rerun_a", "KMeansSampling"),
        _group("p1/BALDDropout/nq5", "paper1_rerun_b", "BALDDropout"),
        _group("p1/RandomSampling/nq5", "paper1", "RandomSampling"),
    ],
}


def _leaf_rel(strategy: str, seed: int, nq: int = 5) -> str:
    return f"results/campaign/paper1/nq{nq}/daninhas_full/SEED_{seed}/NQ_{nq}_NIL_10_NR_1_NE_10/{strategy}"


def _write_leaf(root: Path, rel: str, started_at: str | None, tag: str = "old") -> Path:
    leaf = root / rel
    leaf.mkdir(parents=True)
    for name in LEAF_FILES:
        (leaf / name).write_text(f"{tag}:{name}")
    if started_at is not None:
        (leaf / "run_metadata.json").write_text(
            json.dumps({"started_at": started_at, "config": {}})
        )
    return leaf


def _archived(rel: str) -> str:
    return "results/old_results_broken/" + rel.removeprefix("results/")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    _write_leaf(tmp_path, _leaf_rel("KMeansSampling", 1), OLD)
    _write_leaf(tmp_path, _leaf_rel("KMeansSampling", 2), NEW, tag="new")
    _write_leaf(tmp_path, _leaf_rel("BALDDropout", 1), OLD)
    _write_leaf(tmp_path, _leaf_rel("RandomSampling", 1), OLD)  # part paper1: never touched
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    return tmp_path


def _run(root: Path, *extra: str) -> int:
    return A.main(["--root", str(root), "--manifest", str(root / "manifest.json"), *extra])


def test_moves_old_leaves_preserving_the_path_and_logs_them(tree: Path, capsys) -> None:
    assert _run(tree) == 0
    out = capsys.readouterr().out
    assert "moved=2" in out and "current=1" in out and "missing=1" in out
    for strategy in ("KMeansSampling", "BALDDropout"):
        rel = _leaf_rel(strategy, 1)
        assert not (tree / rel).exists()
        moved = tree / _archived(rel)
        assert [(moved / n).read_text() for n in LEAF_FILES] == [f"old:{n}" for n in LEAF_FILES]
    # The new run (after the cutoff) and the paper1 part are untouched.
    assert (
        tree / _leaf_rel("KMeansSampling", 2) / "results.json"
    ).read_text() == "new:results.json"
    assert (tree / _leaf_rel("RandomSampling", 1)).is_dir()
    log = (tree / "results/old_results_broken/ARCHIVE_LOG.tsv").read_text().splitlines()
    assert log[0] == "utc_time\tsource\tdestination\treason" and len(log) == 3
    for line in log[1:]:
        stamp, source, destination, reason = line.split("\t")
        assert stamp.endswith("Z") and destination == _archived(source)
        assert reason == "superseded by the KI-41/KI-42/KI-45 fix"
        assert not source.startswith("/") and str(tree) not in line


def test_rerun_is_idempotent(tree: Path, capsys) -> None:
    assert _run(tree) == 0
    capsys.readouterr()
    assert _run(tree) == 0
    out = capsys.readouterr().out
    assert "moved=0" in out and "already_archived=2" in out and "current=1" in out
    assert len((tree / "results/old_results_broken/ARCHIVE_LOG.tsv").read_text().splitlines()) == 3


def test_existing_destination_is_never_overwritten_or_merged(tree: Path, capsys) -> None:
    rel = _leaf_rel("KMeansSampling", 1)
    (tree / _archived(rel)).mkdir(parents=True)
    (tree / _archived(rel) / "keep.txt").write_text("earlier archive")
    assert _run(tree) == 1
    out = capsys.readouterr().out
    assert "CONFLICT" in out and "skipped=1" in out
    assert (tree / rel / "results.json").read_text() == "old:results.json"  # source left in place
    assert sorted(p.name for p in (tree / _archived(rel)).iterdir()) == ["keep.txt"]
    assert (tree / _archived(_leaf_rel("BALDDropout", 1))).is_dir()  # the other leaf still moved


def test_leaf_without_readable_metadata_is_skipped(tmp_path: Path, capsys) -> None:
    _write_leaf(tmp_path, _leaf_rel("KMeansSampling", 1), None)
    bad = _write_leaf(tmp_path, _leaf_rel("BALDDropout", 1), None)
    (bad / "run_metadata.json").write_text("{not json")
    naive = _write_leaf(tmp_path, _leaf_rel("BALDDropout", 2), None)
    (naive / "run_metadata.json").write_text(json.dumps({"started_at": "2026-10-01T10:00:00"}))
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    assert _run(tmp_path) == 1
    out = capsys.readouterr().out
    assert out.count("UNREADABLE") == 3 and "moved=0" in out
    assert (tmp_path / _leaf_rel("KMeansSampling", 1)).is_dir()
    assert not (tmp_path / "results/old_results_broken").exists()


def test_dry_run_moves_nothing(tree: Path, capsys) -> None:
    assert _run(tree, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "WOULD MOVE" in out and "would move=2" in out
    assert (tree / _leaf_rel("KMeansSampling", 1)).is_dir()
    assert not (tree / "results/old_results_broken").exists()


def test_check_fails_on_stale_leaves_and_passes_after_archiving(tree: Path, capsys) -> None:
    assert _run(tree, "--check") == 1
    out = capsys.readouterr().out
    assert "CHECK FAILED" in out and out.count("STALE") == 2
    assert _leaf_rel("KMeansSampling", 1) in out
    assert (tree / _leaf_rel("KMeansSampling", 1)).is_dir()  # check moves nothing
    assert _run(tree) == 0
    capsys.readouterr()
    assert _run(tree, "--check") == 0
    assert "CHECK OK" in capsys.readouterr().out


def test_before_override_keeps_leaves_started_at_or_after_it(tree: Path, capsys) -> None:
    assert _run(tree, "--before", OLD) == 0  # at the cutoff: not moved
    out = capsys.readouterr().out
    assert "moved=0" in out and "current=3" in out


def test_move_works_through_a_symlinked_results_dir(tmp_path: Path, capsys) -> None:
    repo, drive = tmp_path / "repo", tmp_path / "drive" / "results"
    drive.mkdir(parents=True)
    repo.mkdir()
    (repo / "results").symlink_to(drive)  # what make colab-setup does on Colab
    _write_leaf(repo, _leaf_rel("KMeansSampling", 1), OLD)
    (repo / "manifest.json").write_text(json.dumps(MANIFEST))
    assert _run(repo) == 0
    rel = _leaf_rel("KMeansSampling", 1)
    assert (drive / "old_results_broken" / rel.removeprefix("results/") / "results.json").is_file()
    assert (repo / "results").is_symlink() and not (repo / rel).exists()
    log = (drive / "old_results_broken/ARCHIVE_LOG.tsv").read_text()
    assert str(drive) not in log and str(tmp_path) not in log and rel in log
    assert str(drive) not in capsys.readouterr().out


@pytest.mark.parametrize("before", ["2026-11-11T19:12:28", "yesterday"])
def test_invalid_cutoff_fails_fast(tree: Path, before: str) -> None:
    assert _run(tree, "--before", before) == 2


def test_dir_results_outside_results_is_rejected(tmp_path: Path) -> None:
    bad = {
        "groups": [dict(_group("x", "paper1_rerun_a", "KMeansSampling"), dir_results="elsewhere/")]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(bad))
    assert _run(tmp_path) == 2


def test_real_manifest_covers_the_63_rerun_leaves(tmp_path: Path, capsys) -> None:
    # Against an EMPTY tmp root: nothing to move, every job reported missing.
    assert A.main(["--root", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "missing=63" in out and "would move=0" in out
    manifest = json.loads((REPO_ROOT / A.DEFAULT_MANIFEST).read_text())
    assert {g["part"] for g in manifest["groups"]} >= set(A.RERUN_PARTS)
    assert A.parse_utc(A.SUPERSEDED_BEFORE).isoformat() == "2026-11-11T19:12:28+00:00"


def test_makefile_target_exposes_dry_and_check() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "campaign-archive-superseded:" in makefile
    target = makefile.split("campaign-archive-superseded:", 1)[1].split("\n\n", 1)[0]
    assert "archive_superseded_runs.py" in target and "--dry-run" in target and "--check" in target
