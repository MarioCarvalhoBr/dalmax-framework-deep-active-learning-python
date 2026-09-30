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
    train/eval mode is restored afterwards. The math is unchanged, except the
    non-convergence handling documented on the subclasses (KI-39): samples
    that cannot reach the boundary get distance `inf` and are ranked last.
    """

    def _begin_query(self) -> None:
        """Hook: reset per-query counters (default: nothing)."""

    def _query_summary(self) -> str:
        """Hook: describe non-converged samples for the per-query log line."""
        return "no breakdown available"

    @abstractmethod
    def cal_dis(self, x: torch.Tensor) -> float:
        """Return `(eta * eta).sum()` for one sample `x` (no batch dim).

        Returns `float("inf")` when the sample could not be moved to the
        decision boundary (see the subclasses and KI-39).
        """

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        clf = self.net.clf
        prev_modes = [(m, m.training) for m in clf.modules()]
        clf.eval()
        start = time.time()
        dis = np.zeros(unlabeled_idxs.shape)
        self._begin_query()
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
        n_inf = int(np.isinf(dis).sum())
        if n_inf:
            self.logger.warning(
                f"{type(self).__name__}: {n_inf}/{len(dis)} samples did not reach the decision "
                f"boundary (distance=inf, ranked last): {self._query_summary()}"
            )
        # argsort places inf last; if fewer than n finite distances exist the
        # remaining picks come from the inf samples in argsort order.
        return unlabeled_idxs[dis.argsort(kind="stable")[:n]]

    def _device(self) -> torch.device:
        return next(self.net.clf.parameters()).device
