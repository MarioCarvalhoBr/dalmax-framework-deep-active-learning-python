"""`ExperimentRunner`: the round-loop half of `demo.py`'s former 289-line
`main()` (see `.specs/architecture/target-architecture.md` §10 and
`.claude/rules/code-quality.md` "single responsibility").

`ExperimentRunner(config, logger).run()` reproduces `demo.py`'s training
loop **exactly in the same order of RNG consumption** as the pre-Phase-2
`demo.py` (seed -> dataset -> net -> strategy -> initial labeling -> round
0 -> rounds loop), which is what keeps the Phase 1
`tests/golden/random_sampling_micro_seed1.json` fixture bit-identical
(`RandomSampling` never touches SSRAE/VCTex/embedding code, so nothing in
this refactor changes its RNG trace). Reporting (plots, `results.json`,
`predictions.csv`, model save, log move) is deliberately NOT done here —
see `dalmax.experiment.reporter.write_report`, which consumes the
`RunResult` this module returns.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dalmax.config.schema import ExperimentConfig
from dalmax.data.registry import get_dataset
from dalmax.experiment.run_metadata import snapshot, write_run_metadata
from dalmax.models.registry import get_network
from dalmax.query_strategies.base import Strategy
from dalmax.query_strategies.registry import build_strategy
from dalmax.seeding import seed_everything


@dataclass
class RunResult:
    """Everything `dalmax.experiment.reporter.write_report` needs to persist
    a finished run, plus the metric history a caller might want directly."""

    config: ExperimentConfig
    dir_results: str
    dataset: Any  # dalmax.data.datasets.Data — not type-hinted to avoid importing it here
    strategy: Strategy
    class_names: list[str]
    final_preds: Any  # last round's predictions (torch.Tensor), for the confusion matrix/CSV
    all_rounds: list[int] = field(default_factory=list)
    all_acc: list[float] = field(default_factory=list)
    all_precision: list[float] = field(default_factory=list)
    all_recall: list[float] = field(default_factory=list)
    all_f1_score: list[float] = field(default_factory=list)
    all_precision_macro: list[float] = field(default_factory=list)
    all_recall_macro: list[float] = field(default_factory=list)
    all_f1_macro: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def results_dir_for(config: ExperimentConfig) -> str:
    """Build the results directory for `config`, matching the layout
    documented in `CLAUDE.md` and `.claude/rules/spec-sync.md`:
    `{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/`.
    """
    base = config.dir_results
    if not base.endswith("/"):
        base += "/"
    dataset_folder = os.path.basename(config.dataset.data_dir.rstrip("/"))
    base += (
        f"{dataset_folder}/SEED_{config.seed}/"
        f"NQ_{config.n_query}_NIL_{config.n_init_labeled}_NR_{config.n_round}_NE_{config.dataset.n_epoch}/"
    )
    return base + f"{config.strategy_name}/"


class ExperimentRunner:
    """Runs one active-learning experiment end to end (no reporting)."""

    def __init__(self, config: ExperimentConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def run(self) -> RunResult:
        config = self.config
        logger = self.logger

        dir_results = results_dir_for(config)
        os.makedirs(dir_results, exist_ok=True)
        logger.warning(f"Results directory: {dir_results}")

        write_run_metadata(dir_results, snapshot(config))

        # Seed every global RNG source exactly once, before any
        # dataset/model/strategy object is constructed (`dalmax.seeding`
        # module docstring) — this replaces demo.py's scattered
        # `np.random.seed`/`torch.manual_seed`/`cudnn.enabled = False`.
        rng: np.random.Generator = seed_everything(config.seed)

        logger.warning(f"device: {config.device}")

        dataset = get_dataset(config)
        net = get_network(config, config.device)
        strategy = build_strategy(config.strategy_name, dataset, net, config, logger, rng)

        result = RunResult(
            config=config,
            dir_results=dir_results,
            dataset=dataset,
            strategy=strategy,
            class_names=dataset.get_classes_names(),
            final_preds=None,
        )

        start_time = time.time()

        # `Data.initialize_labels` only shuffles the pool and marks the
        # initial labeled ids; `RepresentationStrategy` computes/caches its
        # own embeddings on demand (see `dalmax/data/datasets.py::
        # Data.initialize_labels`'s docstring — the legacy SSRAE/VCTex
        # feature-map extraction this used to also trigger was dead code,
        # removed in refactor Phase 4).
        dataset.initialize_labels(config.n_init_labeled, config.strategy_name)
        logger.warning(
            "Initial labeled idxs (sorted): "
            f"{sorted(int(i) for i in np.where(dataset.labeled_idxs)[0])}"
        )

        logger.warning("Round 0")
        strategy.info()
        strategy.train()
        preds = strategy.predict(dataset.get_test_data())
        self._record_round(result, dataset, preds, round_idx=0)
        result.final_preds = preds

        logger.warning(f"Round 0 testing accuracy: {result.all_acc[-1]}")
        logger.warning(f"Round 0 precision: {result.all_precision[-1]}")
        logger.warning(f"Round 0 recall: {result.all_recall[-1]}")
        logger.warning(f"Round 0 f1_score: {result.all_f1_score[-1]}")

        for rd in range(1, config.n_round + 1):
            logger.warning("==========================================================================>")
            logger.warning(f"Round {rd}")

            query_idxs = strategy.query(config.n_query)
            logger.warning(
                f"Round {rd} query_idxs (sorted): {sorted(int(i) for i in query_idxs)}"
            )

            strategy.update(query_idxs)
            strategy.info()
            strategy.train()

            preds = strategy.predict(dataset.get_test_data())
            self._record_round(result, dataset, preds, round_idx=rd)
            result.final_preds = preds

            logger.warning(f"Round {rd} testing accuracy: {result.all_acc[-1]}")
            logger.warning(f"Round {rd} precision: {result.all_precision[-1]}")
            logger.warning(f"Round {rd} recall: {result.all_recall[-1]}")
            logger.warning(f"Round {rd} f1_score: {result.all_f1_score[-1]}")
            logger.warning(f"Local Accuracies: {result.all_acc}")
            logger.warning(f"Local Rounds: {result.all_rounds}")

        result.elapsed_seconds = time.time() - start_time
        logger.warning(f"Total time: {result.elapsed_seconds} seconds")
        return result

    @staticmethod
    def _record_round(result: RunResult, dataset, preds, *, round_idx: int) -> None:
        # `all_acc` uses the manual tensor-comparison accuracy (`cal_test_acc`),
        # exactly as `demo.py` always has; `calc_metrics` (weighted + macro)
        # supplies every other metric, added in this refactor without
        # touching the pre-existing `calc_metrics_sklearn` method.
        acc = dataset.cal_test_acc(preds)
        metrics = dataset.calc_metrics(preds)

        result.all_acc.append(float(acc))
        result.all_precision.append(float(metrics["precision_weighted"]))
        result.all_recall.append(float(metrics["recall_weighted"]))
        result.all_f1_score.append(float(metrics["f1_weighted"]))
        result.all_precision_macro.append(float(metrics["precision_macro"]))
        result.all_recall_macro.append(float(metrics["recall_macro"]))
        result.all_f1_macro.append(float(metrics["f1_macro"]))
        result.all_rounds.append(round_idx)
