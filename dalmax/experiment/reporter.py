"""`write_report`: the reporting half of `demo.py`'s former `main()`
(confusion matrix PDF, per-metric plots, `results.json`, `predictions.csv`,
model checkpoint, log file move) — see `.specs/architecture/
target-architecture.md` §10 and `.claude/rules/code-quality.md`
"single responsibility".

Filenames and `results.json`/`predictions.csv` schemas are unchanged from
`demo.py` (`.claude/rules/spec-sync.md`: changing the results directory
layout or `results.json` schema requires a spec update; this module keeps
every legacy key and only *adds* `all_precision_macro`/`all_recall_macro`/
`all_f1_macro`, so no existing consumer of `results.json` breaks).
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dalmax.experiment.runner import RunResult


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
        `demo.py` always has.
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

    result.strategy.save_model(dir_results)

    json_path = _write_results_json(result, dir_results)
    predictions_csv_path = _write_predictions_csv(result, dir_results)

    print(f"Dados salvos em {json_path}")
    print(f"Predictions saved in {predictions_csv_path}")
