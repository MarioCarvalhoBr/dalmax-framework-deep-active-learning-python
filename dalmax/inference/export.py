"""Shared CSV export for prediction results — used by both `predict.py`
(CLI) and `dalmax/inference/gui.py`'s "Export CSV" button, so the two tools
never drift into two slightly different CSV schemas
(`.claude/rules/code-quality.md` "no duplicated boilerplate").
"""

from __future__ import annotations

import csv
from pathlib import Path

from dalmax.inference.predictor import PredictionRow


def write_predictions_csv(
    rows: list[PredictionRow], class_names: list[str], out_path: str | Path
) -> None:
    """Write `rows` to `out_path` as `predictions.csv`: `Image Index,
    Predicted Class, Confidence, Path`, plus one probability column per
    class (header = `class_names`, in order)."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Index", "Predicted Class", "Confidence", "Path", *class_names])
        for i, row in enumerate(rows):
            writer.writerow(
                [i, row.predicted_class, row.confidence, row.path]
                + [row.probabilities[name] for name in class_names]
            )
