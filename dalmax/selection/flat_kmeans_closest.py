"""Flat k-means selection: pick the sample closest to each centroid.

Replicates the logic of `core/query_strategies/ssrae_kmeans_sampling.py`
(`SSRAEKmeansSampling`), but fixes its coupling to a hardcoded
`random_state=3` (see `.specs/architecture/target-architecture.md` §6):
the k-means seed here is derived from the caller-supplied `rng`, so two
calls seeded from different experiment seeds produce different clusterings
(and therefore, generically, different selections), while two calls seeded
from the same experiment seed are fully reproducible.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from dalmax.selection.base import SelectionStrategy

_MAX_KMEANS_SEED = 2**32 - 1


class FlatKMeansClosest(SelectionStrategy):
    """`k = budget` flat k-means; select the point closest to each centroid.

    Parameters
    ----------
    n_init:
        Number of k-means initializations (`sklearn.cluster.KMeans(n_init=...)`).
    """

    def __init__(self, n_init: int = 10):
        self.n_init = n_init

    def select(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = len(ids)
        k = min(budget, n)
        if k <= 0:
            return np.array([], dtype=ids.dtype)
        if k == n:
            # Nothing to choose: every point is selected.
            return np.array(ids, copy=True)

        seed = int(rng.integers(0, _MAX_KMEANS_SEED))
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=self.n_init)
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_

        selected_mask = np.zeros(n, dtype=bool)
        selected_positions = np.empty(k, dtype=np.int64)
        for i in range(k):
            distances = np.linalg.norm(embeddings - centroids[i], axis=1)
            # Exclude already-picked points so two centroids never claim the
            # same closest sample (guarantees `k` distinct picks).
            distances[selected_mask] = np.inf
            closest_idx = int(np.argmin(distances))
            selected_mask[closest_idx] = True
            selected_positions[i] = closest_idx

        return ids[selected_positions]
