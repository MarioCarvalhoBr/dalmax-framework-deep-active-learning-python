"""LossPredictionLoss: learning loss for active learning (Yoo & Kweon, CVPR 2019),
port of DeepAL+ `query_strategies/loss_prediction.py` (ADR 0015).

Training is delegated to the injected `dalmax.models.loss_prediction.LossPredictionTrainer`
(task model + loss-prediction module trained jointly, same `n_epoch` budget as every
other strategy); the query takes the `n` unlabeled samples with the highest predicted
loss. See `dalmax/models/loss_prediction.py` for every deviation from DeepAL+.
"""

from __future__ import annotations

import numpy as np

from dalmax.models.loss_prediction import LossNet, LossPredictionTrainer

from .base import Strategy


class LossPredictionLoss(Strategy):
    """Query the unlabeled samples whose predicted training loss is highest."""

    def __init__(self, dataset, net, logger, *, trainer: LossPredictionTrainer) -> None:
        super().__init__(dataset, net, logger)
        self.trainer = trainer
        self.module: LossNet | None = None

    def train(self) -> None:
        self.logger.warning("Training the DAL strategy (task model + loss-prediction module)")
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.module = self.trainer.fit(self.net, labeled_data)

    def query(self, n: int) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("LossPredictionLoss.query() called before train()")
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores = self.trainer.predict_losses(self.net, self.module, unlabeled_data).numpy()
        return unlabeled_idxs[np.argsort(-scores, kind="stable")[:n]]
