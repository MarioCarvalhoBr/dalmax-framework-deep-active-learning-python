import numpy as np
from tqdm import tqdm

from .base import Strategy


def k_center_greedy(embeddings: np.ndarray, labeled_mask: np.ndarray, n: int) -> np.ndarray:
    """Greedy k-center (core-set) selection of `n` pool positions (DeepAL loop).

    `embeddings` holds one row per pool sample and `labeled_mask` marks the labeled
    ones (not modified). Each step picks the unlabeled sample farthest from its
    nearest labeled-or-picked sample. Returns the picked pool positions in
    increasing order (as the original `labeled ^ picked` mask did).

    Shared by `KCenterGreedy` and `KCenterGreedyPCA` (ADR 0015); the arithmetic is
    the loop `KCenterGreedy.query` always ran, moved here unchanged.
    """
    labeled_idxs = labeled_mask.copy()
    n_pool = len(labeled_idxs)
    dist_mat = np.matmul(embeddings, embeddings.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(n_pool, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    # Float cancellation in |a|^2 + |b|^2 - 2ab can leave tiny negative squared
    # distances (NaN after the sqrt, ADR 0014); clamp them to 0 first.
    np.maximum(dist_mat, 0, out=dist_mat)
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

    for _i in tqdm(range(n), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n_pool)[~labeled_idxs][q_idx_]
        labeled_idxs[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

    return np.arange(n_pool)[(labeled_mask ^ labeled_idxs)]


class KCenterGreedy(Strategy):
    """Greedy k-center (core-set) selection on the model embeddings (DeepAL port);
    for DANINHAS the embeddings are the 2048-d trained pooled features (KI-42)."""

    def __init__(self, dataset, net, logger) -> None:
        super().__init__(dataset, net, logger)

    def query(self, n: int) -> np.ndarray:
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()
        return k_center_greedy(embeddings, labeled_idxs, n)
