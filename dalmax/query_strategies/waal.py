"""WAAL: Wasserstein adversarial active learning (Shui et al., AISTATS 2020), port of
DeepAL+ `query_strategies/waal.py` (ADR 0015).

Training (every round, round 0 included) is the WAAL min-max training of the injected
`dalmax.models.waal.WAALTrainer` on the labeled set and the unlabeled pool, instead of
plain cross-entropy training; the evaluated and saved model is the ordinary task model
(feature extractor + classifier), so `saved_model.pth` and the inference tools work as
for every strategy. The critic is kept by the strategy for the query only.

Query: `0.5 * ||log p||_2 + 0.5 * ||log p||_1 - selection * critic(z)` over the
unlabeled pool, the `n` LOWEST first (DeepAL+ and the original); a stable sort, so
exact ties keep pool order.
"""

from __future__ import annotations

import numpy as np

from dalmax.models.waal import Critic, WAALTrainer

from .base import Strategy


class WAAL(Strategy):
    """WAAL training plus the uncertainty-minus-critic query."""

    def __init__(
        self, dataset, net, logger, *, trainer: WAALTrainer, selection: float
    ) -> None:
        super().__init__(dataset, net, logger)
        self.trainer = trainer
        self.selection = selection
        self.critic: Critic | None = None

    def train(self) -> None:
        self.logger.warning("Training the DAL strategy (WAAL min-max on labeled + unlabeled)")
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        _unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        self.critic = self.trainer.fit(self.net, labeled_data, unlabeled_data)

    def query(self, n: int) -> np.ndarray:
        if self.critic is None:
            raise RuntimeError("WAAL.query() called before train()")
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores = self.trainer.score_pool(self.net, self.critic, unlabeled_data, self.selection)
        return unlabeled_idxs[np.argsort(scores.numpy(), kind="stable")[:n]]
