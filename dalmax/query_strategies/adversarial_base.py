"""Shared per-sample query loop for the DeepAL adversarial-distance strategies.

Source: DeepAL (Huang, 2021), https://github.com/ej0cl6/deep-active-learning,
`query_strategies/adversarial_bim.py` and `adversarial_deepfool.py`. Both
originals share an identical `query`; this module holds it once.

Tie-break (KI-45, ADR 0014): the original ranks with `dis.argsort()`, which
breaks ties in pool-index order. BIM distances take few discrete values (every
step moves each input element by exactly `eps`), so many samples tie, and the
DANINHAS pool is stored class by class: the selection degenerated into "the
lowest unlabeled indices". `rank_by_distance` breaks ties uniformly at random
with a generator derived from the experiment seed; without ties the order is
exactly the stable argsort, so continuous distances (DeepFool) are unaffected.
"""

import time
from abc import abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from .base import Strategy

# `dalmax.seeding.derive_seed` tag of the tie-break generator, injected by
# `dalmax.query_strategies.registry.build_strategy`.
TIE_BREAK_SEED_TAG = "adversarial_ties"


def rank_by_distance(dis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of `dis` in ascending order, exact ties in a uniformly random order.

    Primary key `dis`, secondary key a random permutation drawn from `rng` (one
    `permutation(len(dis))` call per ranking, so the generator advances every
    query round). `inf` (did not reach the boundary, KI-39) still ranks after
    every finite distance; with no ties the result equals
    `dis.argsort(kind="stable")`.
    """
    return np.lexsort((rng.permutation(len(dis)), dis))


class PerSampleAdversarialStrategy(Strategy):
    """Rank unlabeled samples by the L2^2 size of the smallest adversarial
    perturbation found for each one, and query the `n` smallest.

    Deviations from the original DeepAL `query`, and only these: the model is
    not moved to CPU and back (a hardcoded `.cuda()` in the original); the
    loop runs on the model's own device instead, and the model's previous
    train/eval mode is restored afterwards. The math is unchanged, except the
    non-convergence handling documented on the subclasses (KI-39): samples
    that cannot reach the boundary get distance `inf` and are ranked last;
    and the tie-break (KI-45): equal distances are ordered at random by
    `rng`, see `rank_by_distance`.

    `rng` is the tie-break generator, created once per run by
    `dalmax.query_strategies.registry.build_strategy` from
    `derive_seed(config.seed, TIE_BREAK_SEED_TAG)`.
    """

    def __init__(self, dataset, net, logger, *, rng: np.random.Generator) -> None:
        super().__init__(dataset, net, logger)
        self.rng = rng

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
        # inf ranks last; if fewer than n finite distances exist the remaining
        # picks come from the inf samples, in random order (they all tie).
        return unlabeled_idxs[rank_by_distance(dis, self.rng)[:n]]

    def _device(self) -> torch.device:
        return next(self.net.clf.parameters()).device
