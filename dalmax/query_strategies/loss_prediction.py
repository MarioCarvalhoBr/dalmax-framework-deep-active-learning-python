"""LossPredictionLoss: learning loss for active learning (Yoo & Kweon, CVPR 2019),
port of DeepAL+ `query_strategies/loss_prediction.py` (ADR 0015).

Training is delegated to the injected `dalmax.models.loss_prediction.LossPredictionTrainer`
(task model + loss-prediction module trained jointly, same `n_epoch` budget as every
other strategy); the query takes the `n` unlabeled samples with the highest predicted
loss, exact ties broken at random with the generator injected by `build_strategy`
(`derive_seed(config.seed, LOSS_PREDICTION_TIES_SEED_TAG)`, `ranking.rank_descending`),
so a collapsed module never degenerates into pool order (KI-45). See
`dalmax/models/loss_prediction.py` for every deviation from DeepAL+.

Reporting only (`diagnostics()`): after each training, the Spearman correlation
between the predicted loss and the true cross-entropy on the labeled set (does the
module rank losses at all?); at each query, the std and number of distinct predicted
losses on the pool and the KI-45 pool-order fraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dalmax.models.loss_prediction import LossNet, LossPredictionTrainer

from .base import Strategy
from .diagnostics import RoundDiagnostics, pool_order_fraction, score_spread, spearman
from .ranking import rank_descending

# `dalmax.seeding.derive_seed` tag of the query tie-break generator.
LOSS_PREDICTION_TIES_SEED_TAG = "loss_prediction_ties"


class LossPredictionLoss(Strategy):
    """Query the unlabeled samples whose predicted training loss is highest."""

    def __init__(
        self,
        dataset,
        net,
        logger,
        *,
        trainer: LossPredictionTrainer,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(dataset, net, logger)
        self.trainer = trainer
        self.rng = rng
        self.module: LossNet | None = None
        self._diag = RoundDiagnostics()

    def train(self) -> None:
        self.logger.warning("Training the DAL strategy (task model + loss-prediction module)")
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.module = self.trainer.fit(self.net, labeled_data)
        predicted, true_loss = self.trainer.predict_losses(
            self.net, self.module, labeled_data, with_true_loss=True
        )
        self._diag.add(
            lpl_labeled_spearman=spearman(predicted.numpy(), true_loss.numpy()),
            lpl_n_labeled=len(labeled_data),
        )

    def query(self, n: int) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("LossPredictionLoss.query() called before train()")
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores = self.trainer.predict_losses(self.net, self.module, unlabeled_data).numpy()
        picked = unlabeled_idxs[rank_descending(scores, self.rng)[:n]]
        spread = score_spread(scores)
        self._diag.add(
            lpl_pool_score_std=spread["std"],
            lpl_pool_score_n_unique=spread["n_unique"],
            pool_order_fraction=pool_order_fraction(picked, unlabeled_idxs, n),
        )
        return picked

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()
