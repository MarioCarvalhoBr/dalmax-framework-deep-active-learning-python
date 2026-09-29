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
from types import SimpleNamespace

import torch

from dalmax.config.schema import (
    DatasetConfig,
    EmbeddingConfig,
    ExperimentConfig,
    OptimizerArgs,
    SelectionConfig,
    TrainArgs,
)
from dalmax.experiment import environment as env_mod
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
        params_json_path="files_config/campaign/params_paper1.json",
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
        "environment",
        "determinism",
    }
    assert metadata["determinism"]["seed"] == 1
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


def test_environment_block_on_cpu_has_expected_keys():
    env = snapshot(_make_config())["environment"]
    assert set(env) == {
        "os",
        "machine",
        "python",
        "torch",
        "gpus",
        "cuda_visible_devices",
        "current_device",
        "runtime",
    }
    assert env["os"]["system"]
    assert env["python"]["version"]
    if not torch.cuda.is_available():
        assert env["gpus"] == []
    assert json.loads(json.dumps(env)) == env


def _fake_cuda(monkeypatch):
    props = SimpleNamespace(total_memory=42505207808, major=8, minor=0, multi_processor_count=108)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA Fake GPU 40GB")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_environment_fake_gpu_with_nvidia_smi(monkeypatch):
    _fake_cuda(monkeypatch)

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            cmd, 0, stdout="0, NVIDIA Fake GPU 40GB, 535.104.05, 40960\n", stderr=""
        )

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
    env = env_mod.collect_environment()
    gpu = env["gpus"][0]
    assert gpu["name"] == "NVIDIA Fake GPU 40GB"
    assert gpu["total_memory_gb"] == round(42505207808 / 1024**3, 2)
    assert gpu["compute_capability"] == "8.0"
    assert gpu["multi_processor_count"] == 108
    assert gpu["driver_version"] == "535.104.05"
    assert gpu["memory_total_mib"] == 40960
    assert gpu["nvidia_smi_name"] == "NVIDIA Fake GPU 40GB"
    assert env["current_device"] == 0
    assert "Fake GPU" in env_mod.summarize_environment(env)
    json.dumps(env)


def test_environment_fake_gpu_without_nvidia_smi(monkeypatch):
    _fake_cuda(monkeypatch)

    def missing(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(env_mod.subprocess, "run", missing)
    gpu = env_mod.collect_environment()["gpus"][0]
    assert gpu["driver_version"] is None
    assert gpu["memory_total_mib"] is None
    assert gpu["name"] == "NVIDIA Fake GPU 40GB"


def test_record_nondeterministic_ops_updates_determinism_block(tmp_path):
    from dalmax.experiment.run_metadata import record_nondeterministic_ops_in_metadata

    write_run_metadata(tmp_path, snapshot(_make_config()))
    before = json.loads((tmp_path / "run_metadata.json").read_text())
    assert before["determinism"]["nondeterministic_op_warnings"] == []

    record_nondeterministic_ops_in_metadata(tmp_path, ["cumsum_cuda_kernel"])
    after = json.loads((tmp_path / "run_metadata.json").read_text())
    assert after["determinism"]["nondeterministic_op_warnings"] == ["cumsum_cuda_kernel"]
    assert after["config"] == before["config"] and after["environment"] == before["environment"]
