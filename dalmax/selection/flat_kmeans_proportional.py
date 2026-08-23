"""Flat k-means selection: random picks per cluster, proportional to size.

New strategy required by `.specs/experiments/ablation-study.md` §6.3 ("without
hierarchical module" row): flat k-means with a cluster count that is
*independent* of the query budget, and a **random** (not closest-to-centroid)
pick within each cluster, with each cluster's quota proportional to its
relative size. This is deliberately different from `FlatKMeansClosest`
(`k = budget`, one closest sample per cluster) — see
`.specs/architecture/target-architecture.md` §6 for the comparison table.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from dalmax.selection.base import SelectionStrategy

_MAX_KMEANS_SEED = 2**32 - 1


def _largest_remainder_quotas(sizes: np.ndarray, total: int) -> np.ndarray:
    """Round `sizes / sizes.sum() * total` to integers summing exactly to `total`.

    Uses the largest-remainder (Hamilton) method: take the floor of each
    proportional share, then hand out the leftover units one at a time to
    the clusters with the largest fractional remainder, skipping clusters
    that are already at capacity (`quota == size`).

    Precondition: `0 <= total <= sizes.sum()` (guaranteed by the caller,
    which only calls this when `budget < len(ids) == sizes.sum()`), so
    there is always room to place every leftover unit without exceeding any
    cluster's size.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    pool = int(sizes.sum())
    if pool == 0 or total <= 0:
        return np.zeros_like(sizes)

    raw = sizes.astype(np.float64) / pool * total
    quotas = np.floor(raw).astype(np.int64)
    quotas = np.minimum(quotas, sizes)  # defensive: floor should already respect this
    remainder = total - int(quotas.sum())

    fractional = raw - quotas
    order = np.argsort(-fractional)

    # A single pass over clusters sorted by fractional part suffices in the
    # typical case; loop defensively (bounded) in case some top candidates
    # are already at capacity due to floating-point edge cases.
    guard = 0
    while remainder > 0 and guard < 2 * len(sizes) + 1:
        for c in order:
            if remainder <= 0:
                break
            if quotas[c] < sizes[c]:
                quotas[c] += 1
                remainder -= 1
        guard += 1

    return quotas


class FlatKMeansProportionalRandom(SelectionStrategy):
    """`k = n_clusters or budget` flat k-means; random picks proportional to
    cluster size.

    Parameters
    ----------
    n_clusters:
        Number of k-means clusters. If `None`, defaults to `budget` at
        `select()` time (matching `FlatKMeansClosest`'s cluster count, but
        with a different — random, proportional — picking rule). Ablation
        6.3 requires this to be settable independently of `budget`.
    n_init:
        Number of k-means initializations.
    """

    def __init__(self, n_clusters: int | None = None, n_init: int = 10):
        self.n_clusters = n_clusters
        self.n_init = n_init

    def select(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = len(ids)
        budget = min(budget, n)
        if budget <= 0:
            return np.array([], dtype=ids.dtype)
        if budget == n:
            return np.array(ids, copy=True)

        k = min(self.n_clusters or budget, n)
        seed = int(rng.integers(0, _MAX_KMEANS_SEED))
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=self.n_init)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_sizes = np.bincount(cluster_labels, minlength=k)

        quotas = _largest_remainder_quotas(cluster_sizes, budget)

        selected_chunks = []
        for c in range(k):
            quota = int(quotas[c])
            if quota <= 0:
                continue
            cluster_positions = np.flatnonzero(cluster_labels == c)
            picked = rng.choice(cluster_positions, size=quota, replace=False)
            selected_chunks.append(picked)

        selected_positions = (
            np.concatenate(selected_chunks) if selected_chunks else np.array([], dtype=np.int64)
        )
        return ids[selected_positions]
