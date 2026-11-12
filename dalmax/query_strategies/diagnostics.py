"""Per-round degeneracy diagnostics of the DeepAL+ ports (ADR 0015).

Everything here is reporting only: values are appended to `results.json`'s
`strategy_diagnostics` (via `Strategy.diagnostics()`) and never feed selection or
training. Where a diagnostic needs the true labels of unlabeled samples it is pure
reporting of the simulation.

`pool_order_fraction` is the KI-45 signature: the fraction of a query that lies among
the first `2 n` unlabeled pool positions. The DANINHAS pool is stored class by class,
so a strategy whose scores collapse and falls back to pool order scores 1.0; an
informative strategy on a pool of thousands scores near `2 n / |U|`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


class RoundDiagnostics:
    """Named per-round series, `add(key=value, ...)` appends one value to each key."""

    def __init__(self) -> None:
        self._series: dict[str, list[Any]] = {}

    def add(self, **values: Any) -> None:
        for key, value in values.items():
            self._series.setdefault(key, []).append(_jsonable(value))

    def as_dict(self) -> dict[str, list[Any]]:
        return {key: list(values) for key, values in self._series.items()}


def _jsonable(value: Any) -> Any:
    """Plain Python numbers; NaN/inf become `None` (JSON has no NaN)."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def pool_order_fraction(query_idxs: np.ndarray, unlabeled_idxs: np.ndarray, n: int) -> float:
    """Fraction of `query_idxs` among the first `2 n` unlabeled pool positions."""
    if len(query_idxs) == 0:
        return 0.0
    head = set(np.asarray(unlabeled_idxs)[: 2 * n].tolist())
    return sum(int(i) in head for i in np.asarray(query_idxs)) / len(query_idxs)


def score_spread(scores: np.ndarray) -> dict[str, float | int]:
    """Mean, population std and number of distinct values of a score vector."""
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(scores.mean()) if len(scores) else math.nan,
        "std": float(scores.std()) if len(scores) else math.nan,
        "n_unique": int(len(np.unique(scores))),
    }


def binary_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """ROC AUC of `positive > negative` (0.5 = chance); NaN if a side is empty."""
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return math.nan
    y = np.r_[np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    return float(roc_auc_score(y, np.r_[positive_scores, negative_scores]))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; NaN when undefined (fewer than 2 points or a constant)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan
    return float(spearmanr(a, b).statistic)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN when undefined."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])
