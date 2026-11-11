import numpy as np
import torch

from .base import Strategy


class EntropySamplingDropout(Strategy):
    """Entropy sampling on the mean of `n_drop` MC-dropout passes (DeepAL port).

    `p * log(p)` is computed as `torch.xlogy(p, p)` so a probability of exactly 0
    contributes 0 instead of NaN (KI-41, ADR 0014).
    """

    def __init__(self, dataset, net, logger) -> None:
        super().__init__(dataset, net, logger)
        self.n_drop = net.params['n_drop']

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        uncertainties = torch.xlogy(probs, probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
