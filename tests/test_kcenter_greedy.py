"""`KCenterGreedy` on near-duplicate embeddings (ADR 0014): float cancellation in the
pairwise squared distances must not turn into NaN (and a NaN row must not win the
`argmax`, which is what the pre-fix `np.sqrt` of a slightly negative value caused)."""

from __future__ import annotations

import logging

import numpy as np
import torch

from dalmax.query_strategies.kcenter_greedy import KCenterGreedy


class _Dataset:
    def __init__(self, n_pool: int, labeled: np.ndarray) -> None:
        self.n_pool = n_pool
        self.labeled_idxs = np.zeros(n_pool, dtype=bool)
        self.labeled_idxs[labeled] = True

    def get_train_data(self):
        return self.labeled_idxs.copy(), None


class _Net:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = torch.from_numpy(embeddings)

    def get_embeddings(self, data) -> torch.Tensor:
        return self._embeddings.clone()


def _near_duplicate_pool(
    n_labeled: int = 20, n_far: int = 5, dim: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Labeled points, unlabeled near-duplicates of them, and a few far unlabeled points."""
    rng = np.random.default_rng(0)
    labeled = (rng.random((n_labeled, dim)) * 10).astype(np.float32)
    near = labeled + rng.normal(0, 1e-4, labeled.shape).astype(np.float32)
    far = (rng.random((n_far, dim)) * 10 + 20).astype(np.float32)
    embeddings = np.concatenate([labeled, near, far])
    return embeddings, np.arange(n_labeled)


def test_pairwise_distances_of_near_duplicates_go_negative_in_float32() -> None:
    # Precondition of the test below: without the clamp these entries hit np.sqrt.
    embeddings, _ = _near_duplicate_pool()
    d = embeddings @ embeddings.T
    sq = d.diagonal().reshape(-1, 1).copy()
    d = -2 * d + sq + sq.T
    assert (d < 0).any()


def test_kcenter_greedy_selects_far_points_without_nan() -> None:
    embeddings, labeled = _near_duplicate_pool()
    strategy = KCenterGreedy(
        _Dataset(len(embeddings), labeled), _Net(embeddings), logging.getLogger("t")
    )
    with np.errstate(invalid="raise"):  # a sqrt of a negative number would raise here
        picked = strategy.query(5)
    # The five far points (indices 40..44) are the k-center choices; never a near-duplicate.
    assert sorted(picked.tolist()) == [40, 41, 42, 43, 44]
