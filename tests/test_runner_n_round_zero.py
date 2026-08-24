"""Tests for `dalmax.experiment.runner.ExperimentRunner` with `n_round=0`.

LOW finding fix: `dalmax.config.schema.ExperimentConfig` used to reject
`n_round <= 0`, but the historical legacy `demo.py` (renamed `trainer.py` 2026-08-23) always allowed `--n_round 0` (train
and evaluate exactly once, no active-learning query rounds at all). Now that
`ExperimentConfig` allows `n_round >= 0` (see `tests/test_config.py`), this
file verifies `ExperimentRunner`/`dalmax.experiment.reporter.write_report`
actually handle that value correctly end to end:

- the round loop (`for rd in range(1, config.n_round + 1)`) never executes,
  so `strategy.query`/`strategy.update` are never called;
- exactly one data point (round 0) is recorded and plotted;
- the confusion matrix is built from round 0's predictions.

Uses a stub dataset/net/strategy (monkeypatched into
`dalmax.experiment.runner`) rather than the real DANINHAS/CIFAR10 pipeline,
so this stays a fast, dependency-free unit test -- the real end-to-end
`n_round>=1` path already has dedicated coverage in
`tests/test_golden_run.py` (marked `dataset`+`slow`, needs the real dataset
on disk).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from dalmax.config.schema import (
    DatasetConfig,
    EmbeddingConfig,
    ExperimentConfig,
    OptimizerArgs,
    SelectionConfig,
    TrainArgs,
)
from dalmax.experiment import reporter as reporter_module
from dalmax.experiment import runner as runner_module
from dalmax.experiment.reporter import write_report
from dalmax.experiment.runner import ExperimentRunner


class _FakeLogger:
    def warning(self, msg: str) -> None:
        pass


class _FakeNet:
    """Minimal stand-in for `dalmax.models.base.DeepLearning`."""

    def train(self, data) -> None:
        pass

    def predict(self, data):
        return torch.tensor([0, 1, 0, 1, 0])

    def save_model(self, path: str, *, model_name, class_names, img_size, extra=None) -> None:
        with open(path, "w") as f:
            f.write("fake checkpoint")


class _FakeStrategy:
    """Minimal stand-in for `dalmax.query_strategies.base.Strategy`.

    `query`/`update` raise if ever called: with `n_round=0` the round loop
    body must never execute, so calling either is a test failure, not just
    an assertion at the end.
    """

    def __init__(self, dataset, net, logger) -> None:
        self.dataset = dataset
        self.net = net
        self.logger = logger

    def info(self) -> None:
        pass

    def train(self) -> None:
        self.net.train(None)

    def predict(self, data):
        return self.net.predict(data)

    def query(self, n: int):
        raise AssertionError("strategy.query() must not be called when n_round=0")

    def update(self, pos_idxs, neg_idxs=None) -> None:
        raise AssertionError("strategy.update() must not be called when n_round=0")

    def save_model(self, dir_results: str, *, model_name, class_names, img_size, extra=None) -> None:
        self.net.save_model(
            dir_results + "/saved_model.pth",
            model_name=model_name,
            class_names=class_names,
            img_size=img_size,
            extra=extra,
        )


class _FakeDataset:
    """Minimal stand-in for `dalmax.data.datasets.Data`: only the
    attributes/methods `ExperimentRunner.run` and
    `dalmax.experiment.reporter.write_report` actually read/call."""

    def __init__(self) -> None:
        self.labeled_idxs = np.zeros(20, dtype=bool)
        self.Y_test = torch.tensor([0, 1, 0, 1, 0])
        self.Z_test_paths = [f"p{i}.png" for i in range(5)]

    def initialize_labels(self, n_init_labeled: int, strategy_name: str) -> None:
        self.labeled_idxs[:n_init_labeled] = True

    def get_classes_names(self) -> list[str]:
        return ["a", "b"]

    def get_class_name(self, idx) -> str:
        return "a" if int(idx) == 0 else "b"

    def get_test_data(self):
        return None

    def get_size_pool_unlabeled(self) -> int:
        return int((~self.labeled_idxs).sum())

    def get_size_bucket_labeled(self) -> int:
        return int(self.labeled_idxs.sum())

    def get_size_train_data(self) -> int:
        return len(self.labeled_idxs)

    def get_size_test_data(self) -> int:
        return len(self.Y_test)

    def cal_test_acc(self, preds) -> float:
        return float((self.Y_test == preds).float().mean().item())

    def calc_metrics(self, preds) -> dict:
        return {
            "precision_weighted": 1.0,
            "recall_weighted": 1.0,
            "f1_weighted": 1.0,
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "f1_macro": 1.0,
        }


def _make_config(tmp_path, n_round: int) -> ExperimentConfig:
    dataset = DatasetConfig(
        name="FAKE",
        data_dir="DATA/fake/",
        n_classes=2,
        n_epoch=1,
        n_drop=1,
        train_args=TrainArgs(batch_size=1, num_workers=0),
        test_args=TrainArgs(batch_size=1, num_workers=0),
        optimizer_args=OptimizerArgs(lr=0.1, momentum=0.1),
        embedding=EmbeddingConfig(),
        selection=SelectionConfig(method="flat_closest", hierarchy=None),
    )
    return ExperimentConfig(
        dataset=dataset,
        strategy_name="RandomSampling",
        seed=1,
        n_init_labeled=5,
        n_query=2,
        n_round=n_round,
        dir_results=str(tmp_path / "results"),
        device="cpu",
        params_json_path="params.json",
    )


def _run_with_stubs(config: ExperimentConfig, monkeypatch) -> runner_module.RunResult:
    monkeypatch.setattr(runner_module, "get_dataset", lambda cfg: _FakeDataset())
    monkeypatch.setattr(runner_module, "get_network", lambda cfg, device: _FakeNet())
    monkeypatch.setattr(
        runner_module,
        "build_strategy",
        lambda name, dataset, net, cfg, logger, rng: _FakeStrategy(dataset, net, logger),
    )
    return ExperimentRunner(config, _FakeLogger()).run()


def test_runner_n_round_zero_runs_round_zero_only(tmp_path, monkeypatch):
    config = _make_config(tmp_path, n_round=0)

    result = _run_with_stubs(config, monkeypatch)

    # Exactly one recorded round (round 0); the `for rd in range(1, 0 + 1)`
    # loop body never executed (proven by _FakeStrategy.query/update raising
    # if called -- if the loop had run, this test would have errored, not
    # just failed an assertion).
    assert result.all_rounds == [0]
    assert len(result.all_acc) == 1
    assert len(result.all_precision) == 1
    assert len(result.all_recall) == 1
    assert len(result.all_f1_score) == 1
    assert len(result.all_precision_macro) == 1
    assert len(result.all_recall_macro) == 1
    assert len(result.all_f1_macro) == 1
    assert result.final_preds is not None
    assert torch.equal(result.final_preds, torch.tensor([0, 1, 0, 1, 0]))


def test_write_report_with_n_round_zero_plots_a_single_point_and_writes_confusion_matrix(
    tmp_path, monkeypatch
):
    config = _make_config(tmp_path, n_round=0)
    result = _run_with_stubs(config, monkeypatch)

    # `config.dataset.name` ("FAKE") is not a registered dataset, since this
    # test uses a stub dataset/net/strategy instead of the real DANINHAS/
    # CIFAR10 pipeline (see module docstring) -- stub out the one call in
    # `write_report` that would otherwise look it up in
    # `dalmax.data.registry.DATASET_REGISTRY`.
    monkeypatch.setattr(reporter_module, "get_img_size", lambda name: 128)

    path_logger = tmp_path / "in_progress.log"
    path_logger.write_text("log contents")

    write_report(result, str(path_logger))

    dir_results = result.dir_results

    assert os.path.exists(os.path.join(dir_results, "confusion_matrix.pdf"))
    assert os.path.exists(os.path.join(dir_results, "accuracy.pdf"))
    assert os.path.exists(os.path.join(dir_results, "precision.pdf"))
    assert os.path.exists(os.path.join(dir_results, "recall.pdf"))
    assert os.path.exists(os.path.join(dir_results, "f1_score.pdf"))
    assert os.path.exists(os.path.join(dir_results, "saved_model.pth"))
    assert os.path.exists(os.path.join(dir_results, "log-dalmax.log"))

    with open(os.path.join(dir_results, "results.json")) as f:
        payload = json.load(f)
    assert payload["rounds"] == [0]
    assert payload["n_round"] == 0
    assert len(payload["all_acc"]) == 1
    # Confusion matrix (and predictions.csv) must be built from round 0's
    # predictions, which is all there is when n_round=0: final_preds is
    # never overwritten by a query/update round that didn't happen.
    with open(os.path.join(dir_results, "predictions.csv")) as f:
        predictions_csv = f.read()
    assert predictions_csv.count("\n") == len(result.dataset.Y_test) + 1  # header + 5 rows
