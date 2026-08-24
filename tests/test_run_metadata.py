"""Tests for `dalmax.experiment.run_metadata`.

Uses `tmp_path` for `write_run_metadata` (never writes into the real
`results/` tree, see `.claude/rules/data-safety.md`). The git commit hash is
best-effort: `None` is an acceptable value (e.g. outside a git checkout), but
inside this repo's own git checkout it should resolve to the real `HEAD`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from dalmax.config.schema import (
    DatasetConfig,
    EmbeddingConfig,
    ExperimentConfig,
    OptimizerArgs,
    SelectionConfig,
    TrainArgs,
)
from dalmax.experiment.run_metadata import snapshot, write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_config() -> ExperimentConfig:
    dataset = DatasetConfig(
        name="DANINHAS",
        data_dir="DATA/daninhas_full/",
        n_classes=5,
        n_epoch=10,
        n_drop=10,
        train_args=TrainArgs(batch_size=256, num_workers=4),
        test_args=TrainArgs(batch_size=256, num_workers=4),
        optimizer_args=OptimizerArgs(lr=0.05, momentum=0.3),
        embedding=EmbeddingConfig(),
        selection=SelectionConfig(method="flat_closest", hierarchy=None),
    )
    return ExperimentConfig(
        dataset=dataset,
        strategy_name="RandomSampling",
        seed=1,
        n_init_labeled=100,
        n_query=100,
        n_round=8,
        dir_results="results/dalmax1/",
        device="cpu",
        params_json_path="files_config/benchmark/params_df_gpu_0.json",
    )


def test_snapshot_contains_expected_top_level_keys():
    metadata = snapshot(_make_config())

    assert set(metadata) == {
        "config",
        "git_commit",
        "python_version",
        "torch_version",
        "cuda_available",
        "started_at",
    }
    assert metadata["config"]["dataset"]["name"] == "DANINHAS"
    assert isinstance(metadata["cuda_available"], bool)
    assert isinstance(metadata["python_version"], str)
    assert isinstance(metadata["torch_version"], str)


def test_snapshot_git_commit_is_none_or_a_40_char_hex_string():
    metadata = snapshot(_make_config())
    commit = metadata["git_commit"]
    # Best-effort: None is fine (e.g. outside a git checkout); this repo IS a
    # git checkout, so we also accept the real thing.
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_snapshot_git_commit_matches_head_inside_this_repo():
    is_git_repo = (REPO_ROOT / ".git").exists()
    if not is_git_repo:
        return
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    ).stdout.strip()
    metadata = snapshot(_make_config())
    assert metadata["git_commit"] == (expected or None)


def test_snapshot_started_at_defaults_to_a_timestamp_string():
    metadata = snapshot(_make_config())
    assert isinstance(metadata["started_at"], str)
    assert len(metadata["started_at"]) > 0


def test_snapshot_started_at_uses_provided_value():
    metadata = snapshot(_make_config(), started_at="2026-08-23T00:00:00+00:00")
    assert metadata["started_at"] == "2026-08-23T00:00:00+00:00"


def test_write_run_metadata_creates_file_and_directories(tmp_path):
    dir_results = tmp_path / "some" / "nested" / "results_dir"
    metadata = snapshot(_make_config())

    out_path = write_run_metadata(dir_results, metadata)

    assert out_path == dir_results / "run_metadata.json"
    assert out_path.is_file()
    written = json.loads(out_path.read_text())
    assert written == metadata


def test_write_run_metadata_accepts_string_path(tmp_path):
    dir_results = str(tmp_path / "str_path_dir")
    metadata = snapshot(_make_config())

    out_path = write_run_metadata(dir_results, metadata)

    assert out_path.is_file()
    assert out_path.parent == Path(dir_results)
