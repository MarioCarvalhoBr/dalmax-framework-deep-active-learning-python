"""Abstract base class for batch-selection strategies.

A ``SelectionStrategy`` decides *which* points to query next, given their
embeddings. It is deliberately decoupled from *how* the embeddings were
computed (that is the `dalmax.embeddings.EmbeddingProvider` concern) and
from the deep-learning model/dataset machinery — see
`.specs/architecture/target-architecture.md` §3/§6.

All strategies are stateless functions of their inputs: they never touch
global RNG state (`random`, `np.random`) directly except where an
underlying vendored dependency leaves no alternative (see
`hierarchical_kmeans.py`'s docstring), and even then they seed that global
state deterministically from the caller-supplied `rng` so results are
reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SelectionStrategy(ABC):
    """Selects a batch of ids to query from a pool of embedded points.

    Implementations must be pure with respect to their arguments: the same
    `(embeddings, ids, budget)` triple, driven by two `rng` instances seeded
    identically, must produce identical results; different seeds should
    (generically) produce different results.
    """

    @abstractmethod
    def select(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select up to `budget` ids from `ids`.

        Parameters
        ----------
        embeddings:
            Array of shape ``(N, D)``, float32, row-aligned with `ids`
            (`embeddings[i]` is the embedding of `ids[i]`).
        ids:
            Array of shape ``(N,)`` of integer sample ids.
        budget:
            Number of ids to select. If `budget >= N`, all ids are
            returned (order is implementation-defined but must remain a
            permutation of `ids`, with no duplicates).
        rng:
            NumPy `Generator` used for every random decision made by the
            strategy (cluster-seed derivation, random picks within a
            cluster, ...). Passing two `Generator`s seeded from the same
            seed must yield the same selection.

        Returns
        -------
        np.ndarray
            Array of selected ids, a subset of `ids` with no duplicates,
            of length ``min(budget, len(ids))``.
        """
        raise NotImplementedError
