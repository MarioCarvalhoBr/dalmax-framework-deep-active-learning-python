"""Shared per-sample query loop for the DeepAL adversarial-distance strategies.

Source: DeepAL (Huang, 2021), https://github.com/ej0cl6/deep-active-learning,
`query_strategies/adversarial_bim.py` and `adversarial_deepfool.py`. Both
originals share an identical `query`; this module holds it once.
"""

import time
from abc import abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from .base import Strategy


class PerSampleAdversarialStrategy(Strategy):
    """Rank unlabeled samples by the L2^2 size of the smallest adversarial
    perturbation found for each one, and query the `n` smallest.

    Deviations from the original DeepAL `query`, and only these: the model is
    not moved to CPU and back (a hardcoded `.cuda()` in the original); the
    loop runs on the model's own device instead, and the model's previous
    train/eval mode is restored afterwards. The math is unchanged.
    """

    @abstractmethod
    def cal_dis(self, x: torch.Tensor) -> float:
        """Return `(eta * eta).sum()` for one sample `x` (no batch dim)."""

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        clf = self.net.clf
        prev_modes = [(m, m.training) for m in clf.modules()]
        clf.eval()
        start = time.time()
        dis = np.zeros(unlabeled_idxs.shape)
        try:
            for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
                x, y, idx = unlabeled_data[i]
                dis[i] = self.cal_dis(x)
        finally:
            for module, mode in prev_modes:
                module.training = mode
        self.logger.warning(
            f"{type(self).__name__}: processed {len(unlabeled_idxs)} unlabeled samples "
            f"in {time.time() - start:.1f}s"
        )
        return unlabeled_idxs[dis.argsort()[:n]]

    def _device(self) -> torch.device:
        return next(self.net.clf.parameters()).device
