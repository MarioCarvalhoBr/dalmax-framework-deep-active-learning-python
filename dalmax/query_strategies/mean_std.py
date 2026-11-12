"""MeanSTD: mean over classes of the per-class standard deviation of the
MC-dropout softmax (Kampffmeyer et al., 2016; DeepAL+ `query_strategies/mean_std.py`).

For `T = n_drop` MC-dropout passes kept separate
(`DeepLearning.predict_prob_dropout_split`), the score of a sample is
`mean_c std_t p_t(c)`, with the population standard deviation (`ddof = 0`, as
DeepAL+'s `np.std`); the `n` highest are queried. Needs an active dropout layer
(`model_args.dropout_p`); without one the query raises `MCDropoutError` (KI-41).

Deviation from DeepAL+: the ranking is a stable sort on the negated scores
(equal scores keep pool order) instead of torch's unstable descending sort, so a
query is reproducible bit for bit. Exact ties need identical dropout statistics
and do not occur in practice except for byte-identical images (KI-46).
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Strategy


def mean_std_scores(probs: torch.Tensor) -> torch.Tensor:
    """`mean_c std_t probs[t, :, c]` (population std) for `probs` `[T, n_samples, n_classes]`."""
    return probs.to(torch.float64).std(dim=0, unbiased=False).mean(dim=1)


class MeanSTD(Strategy):
    """Query the `n` unlabeled samples with the largest mean per-class std."""

    def __init__(self, dataset, net, logger) -> None:
        super().__init__(dataset, net, logger)
        self.n_drop: int = net.params["n_drop"]

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        scores = mean_std_scores(probs).numpy()
        return unlabeled_idxs[np.argsort(-scores, kind="stable")[:n]]
