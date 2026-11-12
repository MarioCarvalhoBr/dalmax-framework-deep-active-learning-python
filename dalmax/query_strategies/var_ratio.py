"""VarRatio: variation ratios over MC-dropout passes (ADR 0015).

Definition (Freeman, 1965; used for deep active learning by Gal, Islam &
Ghahramani, 2017): with `T` stochastic forward passes, the variation ratio of a
sample is `1 - f_m / T`, where `f_m` is the number of passes that predict the
modal class. It is 0 when every pass agrees and grows with disagreement.

Deviation from DeepAL+ (`query_strategies/var_ratio.py`), deliberate: DeepAL+
computes `1 - max softmax` of ONE deterministic pass, which ranks exactly like
`LeastConfidence` (a duplicate row). Here the `T = n_drop` passes are MC-dropout
passes (`DeepLearning.predict_prob_dropout_split`, the KI-41 machinery), so the
model must carry an active dropout layer (`model_args.dropout_p`, e.g.
`files_config/campaign/params_paper1_mc_dropout.json`); without one the query
raises `MCDropoutError`.

The scores take at most `T + 1` distinct values, so many samples tie; exact ties
are broken uniformly at random with the generator injected by `build_strategy`
(`derive_seed(config.seed, VAR_RATIO_SEED_TAG)`, one per run, advancing every
round), via `dalmax.query_strategies.ranking.rank_descending`.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Strategy
from .ranking import rank_descending

# `dalmax.seeding.derive_seed` tag of the tie-break generator.
VAR_RATIO_SEED_TAG = "var_ratio_ties"


def variation_ratios(probs: torch.Tensor) -> torch.Tensor:
    """`1 - f_m / T` per sample for `probs` of shape `[T, n_samples, n_classes]`.

    The class predicted by a pass is its `argmax` (lowest class index on an exact
    tie inside a pass).
    """
    n_passes, _, n_classes = probs.shape
    votes = torch.nn.functional.one_hot(probs.argmax(dim=2), n_classes).sum(dim=0)
    modal_count = votes.max(dim=1).values
    return 1.0 - modal_count.to(torch.float64) / n_passes


class VarRatio(Strategy):
    """Query the `n` unlabeled samples with the highest variation ratio."""

    def __init__(self, dataset, net, logger, *, rng: np.random.Generator) -> None:
        super().__init__(dataset, net, logger)
        self.n_drop: int = net.params["n_drop"]
        self.rng = rng

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        scores = variation_ratios(probs).numpy()
        return unlabeled_idxs[rank_descending(scores, self.rng)[:n]]
