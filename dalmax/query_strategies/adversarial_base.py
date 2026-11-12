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
exactly the stable argsort.

Only `AdversarialBIM` receives that generator. `AdversarialDeepFool` (not re-run,
its first-execution runs must stay reproducible) keeps the pool-order tie-break
(`rng=None`): its distances are continuous, and its only exact ties are between
byte-identical duplicate images of the pool (1,269 of the 8,086 DANINHAS
training files have a twin, always in the same class), whose order does not
change what is selected (same image, same label), only which copy.
"""

import time
from abc import abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from .base import Strategy
from .ranking import rank_ascending

# `dalmax.seeding.derive_seed` tag of the tie-break generator, injected by
# `dalmax.query_strategies.registry.build_strategy`.
TIE_BREAK_SEED_TAG = "adversarial_ties"


def rank_by_distance(dis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of `dis` in ascending order, exact ties in a uniformly random order.

    Primary key `dis`, secondary key a random permutation drawn from `rng` (one
    `permutation(len(dis))` call per ranking, so the generator advances every
    query round). `inf` (did not reach the boundary, KI-39) still ranks after
    every finite distance; with no ties the result equals
    `dis.argsort(kind="stable")`. Generalised (ADR 0015) as
    `dalmax.query_strategies.ranking.rank_ascending`, which this delegates to.
    """
    return rank_ascending(dis, rng)


class PerSampleAdversarialStrategy(Strategy):
    """Rank unlabeled samples by the L2^2 size of the smallest adversarial
    perturbation found for each one, and query the `n` smallest.

    Deviations from the original DeepAL `query`, and only these: the model is
    not moved to CPU and back (a hardcoded `.cuda()` in the original); the
    loop runs on the model's own device instead, and the model's previous
    train/eval mode is restored afterwards. The math is unchanged, except the
    non-convergence handling documented on the subclasses (KI-39): samples
    that cannot reach the boundary get distance `inf` and are ranked last;
    and the tie-break (KI-45): with a generator `rng`, equal distances are
    ordered at random (`rank_by_distance`); with `rng=None`, in pool order
    (`argsort(kind="stable")`, the first-execution behaviour).

    `rng` must be passed explicitly. For `AdversarialBIM` it is created once
    per run by `dalmax.query_strategies.registry.build_strategy` from
    `derive_seed(config.seed, TIE_BREAK_SEED_TAG)` and advances across rounds.
    """

    def __init__(self, dataset, net, logger, *, rng: np.random.Generator | None) -> None:
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
        # picks come from the inf samples (they all tie: random or pool order).
        order = dis.argsort(kind="stable") if self.rng is None else rank_by_distance(dis, self.rng)
        return unlabeled_idxs[order[:n]]

    def _device(self) -> torch.device:
        return next(self.net.clf.parameters()).device
