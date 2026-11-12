"""`write_report`: the reporting half of the historical `demo.py`'s former `main()`
(confusion matrix PDF, per-metric plots, `results.json`, `predictions.csv`,
model checkpoint, log file move) — see `.specs/architecture/
target-architecture.md` §10 and `.claude/rules/code-quality.md`
"single responsibility".

Filenames and `results.json`/`predictions.csv` schemas are unchanged from
the historical `demo.py` (`.claude/rules/spec-sync.md`: changing the results directory
layout or `results.json` schema requires a spec update; this module keeps
every legacy key and only *adds* `all_precision_macro`/`all_recall_macro`/
`all_f1_macro`, so no existing consumer of `results.json` breaks).

Two additive files (ADR 0016) keep the per-round history so that any metric can be
recomputed later on any subset of the test set or of the queried samples (when KI-46
was found, only the last round's `predictions.csv` existed):

- `round_predictions.json`: the predicted class index of every test image at every
  round, with the test images' paths and true class indices (test-set order).
- `round_queries.json`: per round, the pool indices labeled in that round (round 0 =
  the initial labeled set), sorted, with their true class indices and image paths.
  Pool indices follow the loaders' sorted order (KI-47; `run_metadata.json`'s
  `data_order` hashes it); the paths stay the identifier to compare runs by
  (`dalmax.data_order.relative_image_path`), and the campaign's seed audit does.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dalmax.data.registry import get_img_size
from dalmax.experiment.run_metadata import snapshot
from dalmax.experiment.runner import RunResult
from dalmax.reporting.leaf_check import ROUND_PREDICTIONS_FILE, ROUND_QUERIES_FILE

ROUND_PREDICTIONS_FORMAT = "dalmax-round-predictions"
ROUND_QUERIES_FORMAT = "dalmax-round-queries"
ROUND_HISTORY_VERSION = 1


def _plot_confusion_matrix(cm, class_names: list[str], dir_results: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=45)
    plt.tight_layout()
    plt.savefig(f"{dir_results}/confusion_matrix.pdf")
    plt.close(fig)


def _plot_metric(rounds: list[int], values: list[float], title: str, ylabel: str, filename: str) -> None:
    plt.figure()
    plt.plot(rounds, values, marker="o")
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(filename)
    plt.close()


def _write_results_json(result: RunResult, dir_results: str) -> str:
    config = result.config
    payload = {
        "dataset_name": config.dataset.name,
        "strategy_name": config.strategy_name,
        "n_init_labeled": config.n_init_labeled,
        "n_query": config.n_query,
        "n_round": config.n_round,
        "seed": config.seed,
        "all_acc": result.all_acc,
        "all_precision": result.all_precision,
        "all_recall": result.all_recall,
        "all_f1_score": result.all_f1_score,
        "rounds": result.all_rounds,
        # New in Phase 2 (macro-averaged metrics, see .claude/rules/code-quality.md
        # and .specs/architecture/refactor-plan.md Phase 2): every legacy key
        # above is unchanged, these are additive.
        "all_precision_macro": result.all_precision_macro,
        "all_recall_macro": result.all_recall_macro,
        "all_f1_macro": result.all_f1_macro,
    }
    # ADR 0015: strategy-specific per-round reporting (e.g. CEAL's pseudo-labels);
    # absent for every strategy that reports nothing, so their schema is unchanged.
    diagnostics = result.strategy.diagnostics()
    if diagnostics:
        payload["strategy_diagnostics"] = diagnostics
    json_path = os.path.join(dir_results, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)
    return json_path


def _write_predictions_csv(result: RunResult, dir_results: str) -> str:
    dataset = result.dataset
    preds = result.final_preds
    rows = []
    for i in range(len(dataset.Y_test)):
        real_class = dataset.Y_test[i].item()
        predicted_class = preds[i].item()
        real_class_name = dataset.get_class_name(real_class)
        predicted_class_name = dataset.get_class_name(predicted_class)
        correct = 1 if real_class == predicted_class else 0
        path = dataset.Z_test_paths[i]
        rows.append([i, real_class_name, predicted_class_name, correct, path])

    predictions_df = pd.DataFrame(
        rows, columns=["Image Index", "Real Class", "Predicted Class", "Correct", "Path"]
    )
    predictions_csv_path = os.path.join(dir_results, "predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    return predictions_csv_path


def _write_compact_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def _write_round_predictions(result: RunResult, dir_results: str) -> str:
    """`round_predictions.json`: every round's test-set predictions (module docstring)."""
    dataset = result.dataset
    payload = {
        "format": ROUND_PREDICTIONS_FORMAT,
        "version": ROUND_HISTORY_VERSION,
        "class_names": list(result.class_names),
        "rounds": list(result.all_rounds),
        "test_paths": [str(path) for path in dataset.Z_test_paths],
        "test_labels": [int(y) for y in dataset.Y_test],
        "predictions": result.round_predictions,
    }
    path = os.path.join(dir_results, ROUND_PREDICTIONS_FILE)
    _write_compact_json(path, payload)
    return path


def _write_round_queries(result: RunResult, dir_results: str) -> str:
    """`round_queries.json`: the pool indices labeled in each round (module docstring)."""
    dataset = result.dataset
    rounds = []
    for round_idx, idxs in enumerate(result.round_queries):
        rounds.append({
            "round": round_idx,
            "kind": "initial" if round_idx == 0 else "query",
            "pool_idxs": idxs,
            "labels": [int(dataset.Y_train[i]) for i in idxs],
            "paths": [str(dataset.Z_train_paths[i]) for i in idxs],
        })
    payload = {
        "format": ROUND_QUERIES_FORMAT,
        "version": ROUND_HISTORY_VERSION,
        "class_names": list(result.class_names),
        "n_pool": int(dataset.n_pool),
        "rounds": rounds,
    }
    path = os.path.join(dir_results, ROUND_QUERIES_FILE)
    _write_compact_json(path, payload)
    return path


def _build_checkpoint_extra(result: RunResult) -> dict:
    """Provenance recorded alongside the trained model's weights (the
    `"extra"` field of `dalmax.models.checkpoint.save_checkpoint`): strategy
    name, seed, dataset name, git commit, torch/Python versions, and a save
    timestamp — reusing `dalmax.experiment.run_metadata.snapshot`, the same
    helper `run_metadata.json` is built from, so this never drifts from that
    provenance logic (`.claude/rules/reproducibility.md`)."""
    config = result.config
    meta = snapshot(config)
    return {
        "strategy_name": config.strategy_name,
        "seed": config.seed,
        "dataset_name": config.dataset.name,
        "git_commit": meta["git_commit"],
        "python_version": meta["python_version"],
        "torch_version": meta["torch_version"],
        "saved_at": meta["started_at"],
    }


def write_report(result: RunResult, path_logger: str) -> None:
    """Persist plots, `results.json`, `predictions.csv`, the trained model,
    and the run's log file into `result.dir_results`.

    Parameters
    ----------
    result:
        The `RunResult` returned by `dalmax.experiment.runner.ExperimentRunner.run()`.
    path_logger:
        Path to the in-progress log file (`dalmax.logging_utils.get_path_logger()`),
        moved into `result.dir_results` as `log-dalmax.log`, exactly as
        the historical `demo.py` always has.
    """
    dir_results = result.dir_results
    dataset = result.dataset

    cm = confusion_matrix(dataset.Y_test, result.final_preds)
    _plot_confusion_matrix(cm, result.class_names, dir_results)

    _plot_metric(result.all_rounds, result.all_acc, "Accuracy", "Accuracy", f"{dir_results}/accuracy.pdf")
    _plot_metric(result.all_rounds, result.all_precision, "Precision", "Precision", f"{dir_results}/precision.pdf")
    _plot_metric(result.all_rounds, result.all_recall, "Recall", "Recall", f"{dir_results}/recall.pdf")
    _plot_metric(result.all_rounds, result.all_f1_score, "F1-Score", "F1-Score", f"{dir_results}/f1_score.pdf")

    os.rename(path_logger, os.path.join(dir_results, "log-dalmax.log"))

    result.strategy.save_model(
        dir_results,
        model_name=result.config.dataset.name,
        class_names=result.class_names,
        img_size=get_img_size(result.config.dataset.name),
        extra=_build_checkpoint_extra(result),
    )

    predictions_csv_path = _write_predictions_csv(result, dir_results)
    _write_round_predictions(result, dir_results)
    _write_round_queries(result, dir_results)
    # `results.json` is written LAST: it is the "run finished" marker that
    # `dalmax.campaign.job_done` (skip-existing/resume) relies on, so a run
    # killed mid-report can never look complete.
    json_path = _write_results_json(result, dir_results)

    print(f"Dados salvos em {json_path}")
    print(f"Predictions saved in {predictions_csv_path}")
