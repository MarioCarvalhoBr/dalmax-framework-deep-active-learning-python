"""Per-round history files (ADR 0016): `round_predictions.json` and `round_queries.json`.

Every run writes, next to the unchanged `results.json` / `predictions.csv`, the test-set
predictions of every round and the pool indices labeled in every round (round 0 = the
initial labeled set) with their true classes and image paths, so any metric can later be
recomputed on any subset of the test set (the gap that KI-46 exposed). Stub dataset /
net / strategy as in `tests/test_runner_n_round_zero.py`: fast, no dataset on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
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
from dalmax.reporting.leaf_check import ROUND_HISTORY_FILES, check_leaf

N_POOL = 12
Y_TEST = [0, 1, 2, 1, 0]
# Predictions of round 0, 1, 2 (the stub net returns them in turn).
PREDICTIONS = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 2, 1, 0]]
# What the stub strategy queries in rounds 1 and 2 (unsorted on purpose).
QUERIES = [[7, 3], [11, 5]]


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, msg: str) -> None:
        self.lines.append(msg)


class _Net:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, data) -> torch.Tensor:
        preds = torch.tensor(PREDICTIONS[self.calls])
        self.calls += 1
        return preds


class _Dataset:
    def __init__(self) -> None:
        self.n_pool = N_POOL
        self.labeled_idxs = np.zeros(N_POOL, dtype=bool)
        self.Y_train = np.array([i % 3 for i in range(N_POOL)])
        self.Z_train_paths = [f"DATA/x/train/c{i % 3}/img{i}.png" for i in range(N_POOL)]
        self.Y_test = torch.tensor(Y_TEST)
        self.Z_test_paths = [f"DATA/x/test/c{y}/t{i}.png" for i, y in enumerate(Y_TEST)]

    def initialize_labels(self, n_init_labeled: int, strategy_name: str) -> None:
        self.labeled_idxs[[9, 0]] = True

    def get_classes_names(self) -> list[str]:
        return ["c0", "c1", "c2"]

    def get_class_name(self, idx) -> str:
        return f"c{int(idx)}"

    def get_test_data(self):
        return None

    def cal_test_acc(self, preds) -> float:
        return float((self.Y_test == preds).float().mean().item())

    def calc_metrics(self, preds) -> dict:
        return dict.fromkeys(
            ("precision_weighted", "recall_weighted", "f1_weighted", "precision_macro", "recall_macro", "f1_macro"),
            0.5,
        )


class _Strategy:
    def __init__(self, dataset: _Dataset, net: _Net) -> None:
        self.dataset = dataset
        self.net = net
        self.rounds = 0

    def info(self) -> None:
        pass

    def train(self) -> None:
        pass

    def predict(self, data) -> torch.Tensor:
        return self.net.predict(data)

    def query(self, n: int) -> np.ndarray:
        idxs = np.array(QUERIES[self.rounds])
        self.rounds += 1
        return idxs

    def update(self, pos_idxs, neg_idxs=None) -> None:
        self.dataset.labeled_idxs[pos_idxs] = True

    def diagnostics(self) -> dict:
        return {}

    def save_model(self, dir_results: str, *, model_name, class_names, img_size, extra=None) -> None:
        Path(dir_results, "saved_model.pth").write_text("stub")


def _config(tmp_path: Path) -> ExperimentConfig:
    dataset = DatasetConfig(
        name="FAKE", data_dir="DATA/x/", n_classes=3, n_epoch=1, n_drop=1,
        train_args=TrainArgs(batch_size=1, num_workers=0), test_args=TrainArgs(batch_size=1, num_workers=0),
        optimizer_args=OptimizerArgs(lr=0.1, momentum=0.1), embedding=EmbeddingConfig(),
        selection=SelectionConfig(method="flat_closest", hierarchy=None),
    )
    return ExperimentConfig(
        dataset=dataset, strategy_name="RandomSampling", seed=1, n_init_labeled=2, n_query=2, n_round=2,
        dir_results=str(tmp_path / "results"), device="cpu", params_json_path="params.json",
    )


@pytest.fixture()
def leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, runner_module.RunResult, _Logger]:
    monkeypatch.setattr(runner_module, "get_dataset", lambda cfg: _Dataset())
    monkeypatch.setattr(runner_module, "get_network", lambda cfg, device: _Net())
    monkeypatch.setattr(runner_module, "build_strategy",
                        lambda name, dataset, net, cfg, logger, rng: _Strategy(dataset, net))
    monkeypatch.setattr(reporter_module, "get_img_size", lambda name: 128)
    logger = _Logger()
    result = ExperimentRunner(_config(tmp_path), logger).run()
    log = tmp_path / "in_progress.log"
    log.write_text("log")
    write_report(result, str(log))
    return Path(result.dir_results), result, logger


def test_round_predictions_hold_every_round_in_test_order(leaf) -> None:
    path, result, _ = leaf
    payload = json.loads((path / "round_predictions.json").read_text())
    assert payload["format"] == "dalmax-round-predictions" and payload["version"] == 1
    assert payload["rounds"] == [0, 1, 2] and payload["predictions"] == PREDICTIONS
    assert payload["test_labels"] == Y_TEST and payload["test_paths"] == result.dataset.Z_test_paths
    assert payload["class_names"] == ["c0", "c1", "c2"]
    # The last round equals the unchanged predictions.csv.
    csv_rows = (path / "predictions.csv").read_text().splitlines()[1:]
    assert [row.split(",")[2] for row in csv_rows] == [f"c{p}" for p in PREDICTIONS[-1]]
    # Re-scoring a subset later (here: the test images of class 1) needs nothing else.
    subset = [i for i, y in enumerate(payload["test_labels"]) if y == 1]
    assert [sum(r[i] == 1 for i in subset) for r in payload["predictions"]] == [0, 2, 2]


def test_round_queries_hold_the_initial_set_and_every_query(leaf) -> None:
    path, result, logger = leaf
    payload = json.loads((path / "round_queries.json").read_text())
    assert payload["format"] == "dalmax-round-queries" and payload["n_pool"] == N_POOL
    rounds = payload["rounds"]
    assert [r["round"] for r in rounds] == [0, 1, 2]
    assert [r["kind"] for r in rounds] == ["initial", "query", "query"]
    assert [r["pool_idxs"] for r in rounds] == [[0, 9], sorted(QUERIES[0]), sorted(QUERIES[1])]
    dataset = result.dataset
    for entry in rounds:
        assert entry["labels"] == [int(dataset.Y_train[i]) for i in entry["pool_idxs"]]
        assert entry["paths"] == [dataset.Z_train_paths[i] for i in entry["pool_idxs"]]
    # Same sets as the log lines the seed audit and the thesis scripts parse (unchanged).
    assert "Initial labeled idxs (sorted): [0, 9]" in logger.lines
    assert "Round 1 query_idxs (sorted): [3, 7]" in logger.lines


def test_leaf_check_accepts_the_files_and_results_json_is_unchanged(leaf) -> None:
    path, _, _ = leaf
    missing, warnings = check_leaf(path, ROUND_HISTORY_FILES)
    assert missing == []
    assert not [w for w in warnings if "round_" in w]
    payload = json.loads((path / "results.json").read_text())
    assert "round_predictions" not in payload and "round_queries" not in payload
    assert payload["rounds"] == [0, 1, 2]
