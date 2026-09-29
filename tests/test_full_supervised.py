"""Tests for the `FullSupervised` upper bound (paper 1; NOT active learning).

Unit level: registry membership, `ExperimentConfig` rejecting `n_round != 0`,
`query()` raising, and the runner overriding `n_init_labeled` with the real pool
size (with a warning) so the results directory reads `NIL_<pool>`. Integration
level (`dataset` + `slow`): an end-to-end CPU run on the micro dataset (806-image
pool) that must produce a single-round `results.json` and the normal artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dalmax.config.schema import (
    FULL_SUPERVISED_STRATEGY,
    ConfigError,
    DatasetConfig,
    EmbeddingConfig,
    ExperimentConfig,
    OptimizerArgs,
    SelectionConfig,
    TrainArgs,
)
from dalmax.experiment.runner import resolve_full_supervised_config, results_dir_for
from dalmax.query_strategies.full_supervised import FullSupervised
from dalmax.query_strategies.registry import STRATEGY_REGISTRY, build_strategy

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config(strategy: str = FULL_SUPERVISED_STRATEGY, *, n_round: int = 0, n_init: int = 100) -> ExperimentConfig:
    dataset = DatasetConfig(
        name="DANINHAS", data_dir="DATA/daninhas_micro/", n_classes=5, n_epoch=1, n_drop=1,
        train_args=TrainArgs(batch_size=1, num_workers=0), test_args=TrainArgs(batch_size=1, num_workers=0),
        optimizer_args=OptimizerArgs(lr=0.1, momentum=0.1), embedding=EmbeddingConfig(),
        selection=SelectionConfig(method="flat_closest", hierarchy=None),
    )
    return ExperimentConfig(
        dataset=dataset, strategy_name=strategy, seed=1, n_init_labeled=n_init, n_query=100,
        n_round=n_round, dir_results="results/_t/", device="cpu", params_json_path="p.json",
    )


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, msg: str) -> None:
        self.messages.append(msg)


class _Pool:
    n_pool = 806


def test_registered_and_builds_a_full_supervised_strategy() -> None:
    assert FULL_SUPERVISED_STRATEGY in STRATEGY_REGISTRY
    strategy = build_strategy(FULL_SUPERVISED_STRATEGY, _Pool(), None, _config(), _Logger(), None)
    assert type(strategy) is FullSupervised


def test_query_must_never_be_called() -> None:
    with pytest.raises(RuntimeError, match="upper bound"):
        FullSupervised(_Pool(), None, _Logger()).query(10)


@pytest.mark.parametrize("n_round", [1, 8])
def test_config_rejects_active_learning_rounds(n_round: int) -> None:
    with pytest.raises(ConfigError, match="n_round must be 0"):
        _config(n_round=n_round)


def test_n_round_zero_is_accepted() -> None:
    assert _config(n_round=0).n_round == 0


def test_runner_overrides_n_init_labeled_with_pool_size_and_warns() -> None:
    logger = _Logger()
    resolved = resolve_full_supervised_config(_config(n_init=100), _Pool(), logger)
    assert resolved.n_init_labeled == 806
    assert any("overriding n_init_labeled=100" in m for m in logger.messages)
    assert "NIL_806_NR_0" in results_dir_for(resolved)


def test_no_warning_when_n_init_already_equals_pool_size() -> None:
    logger = _Logger()
    resolved = resolve_full_supervised_config(_config(n_init=806), _Pool(), logger)
    assert resolved.n_init_labeled == 806
    assert logger.messages == []


def test_other_strategies_are_left_untouched() -> None:
    config = _config("RandomSampling", n_round=8, n_init=100)
    assert resolve_full_supervised_config(config, object(), _Logger()) is config


@pytest.mark.dataset
@pytest.mark.slow
def test_micro_end_to_end_one_round_full_pool(tmp_path: Path) -> None:
    micro_train = REPO_ROOT / "DATA" / "daninhas_micro" / "train"
    if not micro_train.is_dir():
        pytest.skip("DATA/daninhas_micro not present")
    out = tmp_path / "ub"
    proc = subprocess.run(
        [sys.executable, "tools/trainer.py", "--params_json", "files_config/params_micro.json",
         "--dataset_name", "DANINHAS", "--strategy_name", FULL_SUPERVISED_STRATEGY,
         "--n_init_labeled", "10", "--n_query", "100", "--n_round", "0", "--seed", "1",
         "--device", "cpu", "--dir_results", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    leaf = next(out.glob("*/SEED_1/NQ_100_NIL_*_NR_0_NE_1/FullSupervised"))
    pool = sum(1 for _ in micro_train.rglob("*") if _.is_file())
    assert f"NIL_{pool}_" in str(leaf)
    results = json.loads((leaf / "results.json").read_text())
    assert results["rounds"] == [0]
    assert results["n_init_labeled"] == pool
    meta = json.loads((leaf / "run_metadata.json").read_text())
    assert meta["config"]["n_init_labeled"] == pool
    for name in ("predictions.csv", "confusion_matrix.pdf", "saved_model.pth", "log-dalmax.log"):
        assert (leaf / name).is_file()
