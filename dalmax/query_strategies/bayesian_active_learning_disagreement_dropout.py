import numpy as np
import torch

from .base import Strategy


class BALDDropout(Strategy):
    """BALD with MC dropout (DeepAL port): query the `n` samples with the largest
    mutual information `H[mean_k p_k] - mean_k H[p_k]` over `n_drop` dropout passes.

    `p * log(p)` is computed as `torch.xlogy(p, p)` so a probability of exactly 0
    contributes 0 instead of NaN (KI-41, ADR 0014); the model must carry an active
    dropout layer (`DeepLearning.predict_prob_dropout_split` fails fast otherwise).
    """

    def __init__(self, dataset, net, logger) -> None:
        super().__init__(dataset, net, logger)
        self.n_drop = net.params['n_drop']

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        uncertainties = bald_scores(probs)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


def bald_scores(probs: torch.Tensor) -> torch.Tensor:
    """`mean_k H[p_k] - H[mean_k p_k]` (= minus the mutual information) per sample,
    for `probs` of shape `[n_drop, n_samples, n_classes]`; ascending order = most
    informative first, as in DeepAL (`entropy2 - entropy1`)."""
    pb = probs.mean(0)
    entropy1 = -torch.xlogy(pb, pb).sum(1)
    entropy2 = -torch.xlogy(probs, probs).sum(2).mean(0)
    return entropy2 - entropy1
