"""MeanSTD: mean over classes of the per-class standard deviation of the
MC-dropout softmax (Kampffmeyer et al., 2016; DeepAL+ `query_strategies/mean_std.py`).

For `T = n_drop` MC-dropout passes kept separate
(`DeepLearning.predict_prob_dropout_split`), the score of a sample is
`mean_c std_t p_t(c)`, with the population standard deviation (`ddof = 0`, as
DeepAL+'s `np.std`); the `n` highest are queried. Needs an active dropout layer
(`model_args.dropout_p`); without one the query raises `MCDropoutError` (KI-41).

Deviation from DeepAL+: exact ties are broken at random with the generator injected
by `build_strategy` (`derive_seed(config.seed, MEAN_STD_SEED_TAG)`, via
`ranking.rank_descending`) instead of torch's unstable descending sort, so a query is
reproducible and never falls back to pool order if the scores collapse (the KI-45
failure mode). Reports the score mean/std and the KI-45 pool-order fraction per
round (`diagnostics()`, reporting only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import Strategy
from .diagnostics import RoundDiagnostics, pool_order_fraction, score_spread
from .ranking import rank_descending

# `dalmax.seeding.derive_seed` tag of the tie-break generator.
MEAN_STD_SEED_TAG = "mean_std_ties"


def mean_std_scores(probs: torch.Tensor) -> torch.Tensor:
    """`mean_c std_t probs[t, :, c]` (population std) for `probs` `[T, n_samples, n_classes]`."""
    return probs.to(torch.float64).std(dim=0, unbiased=False).mean(dim=1)


class MeanSTD(Strategy):
    """Query the `n` unlabeled samples with the largest mean per-class std."""

    def __init__(self, dataset, net, logger, *, rng: np.random.Generator) -> None:
        super().__init__(dataset, net, logger)
        self.n_drop: int = net.params["n_drop"]
        self.rng = rng
        self._diag = RoundDiagnostics()

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        scores = mean_std_scores(probs).numpy()
        picked = unlabeled_idxs[rank_descending(scores, self.rng)[:n]]
        spread = score_spread(scores)
        self._diag.add(
            mean_std_score_mean=spread["mean"],
            mean_std_score_std=spread["std"],
            pool_order_fraction=pool_order_fraction(picked, unlabeled_idxs, n),
        )
        return picked

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()
