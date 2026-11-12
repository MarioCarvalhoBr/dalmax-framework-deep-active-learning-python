"""KCenterGreedyPCA: `KCenterGreedy` on PCA-reduced embeddings (DeepAL+
`query_strategies/kcenter_greedy_pca.py`, ADR 0015).

When the embedding width exceeds `n_components` (default 50; 2048 for the DANINHAS
ResNet-50), the embeddings of the whole pool (labeled and unlabeled, as in DeepAL+)
are projected onto their first `n_components` principal components, then the
shared greedy loop `kcenter_greedy.k_center_greedy` runs on them.

Deviations from DeepAL+, deliberate:

- PCA gets `random_state` derived from the run seed
  (`derive_seed(config.seed, KCENTER_PCA_SEED_TAG)`, injected by `build_strategy`);
  scikit-learn picks the randomized SVD solver for this shape, which DeepAL+ left
  unseeded.
- The reduced embeddings are kept in float32 (DeepAL+ casts them to float16 before
  the pairwise distances, losing precision and creating spurious ties), and the
  squared distances are clamped at 0 before the square root, exactly as in our
  `KCenterGreedy` (ADR 0014).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .base import Strategy
from .kcenter_greedy import k_center_greedy

# `dalmax.seeding.derive_seed` tag of the PCA random state.
KCENTER_PCA_SEED_TAG = "kcenter_pca"


def reduce_embeddings(embeddings: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    """PCA to `n_components` dimensions when the embeddings are wider, float32 output."""
    if embeddings.shape[1] > n_components:
        embeddings = PCA(n_components=n_components, random_state=random_state).fit_transform(
            embeddings
        )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


class KCenterGreedyPCA(Strategy):
    """Greedy k-center selection on the first `n_components` principal components."""

    def __init__(
        self, dataset, net, logger, *, n_components: int, random_state: int
    ) -> None:
        super().__init__(dataset, net, logger)
        self.n_components = n_components
        self.random_state = random_state

    def query(self, n: int) -> np.ndarray:
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data).numpy()
        reduced = reduce_embeddings(embeddings, self.n_components, self.random_state)
        return k_center_greedy(reduced, labeled_idxs, n)
