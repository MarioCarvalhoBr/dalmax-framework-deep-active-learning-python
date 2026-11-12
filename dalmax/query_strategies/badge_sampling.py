"""BadgeSampling: BADGE (Ash et al., ICLR 2020), port of DeepAL+
`query_strategies/badge_sampling.py` / `nets.py::get_grad_embeddings` (ADR 0015).

Gradient embedding of a sample with penultimate features `h` (the input of the
final linear layer, `e1`) and softmax `p`, using the predicted class `y_hat` as
pseudo-label: the gradient of the cross-entropy with respect to the last layer's
weights, `g_c = (p_c - [c == y_hat]) * h` for every class `c`, concatenated
class-major (dimension `n_classes * dim(h)`, 5 x 2048 = 10,240 for the DANINHAS
ResNet-50). The batch is chosen by k-means++ seeding on these embeddings: the
first point is the one with the largest gradient norm, every next point is drawn
with probability proportional to `D(x)^2`, `D(x)` its distance to the nearest point
already chosen.

Deviations from DeepAL+, deliberate (the method is unchanged):

- The embeddings are computed batched in torch (`gradient_embeddings`), not with
  two nested per-sample Python loops.
- The distances are computed from the rank-one structure of the embeddings,
  `<g_x, g_y> = <a_x, a_y> <h_x, h_y>` with `a = p - onehot(y_hat)` (the speed-up
  the BADGE authors adopted later, Zhang et al. 2023, appendix D), in float64,
  one O(n * dim(h)) pass per pick, clamped at 0 before the square root; neither
  the 10,240-d matrix nor a pairwise matrix is built. `gradient_embeddings` is kept
  for testing that the two agree.
- The D^2 draws come from the injected `np.random.Generator`
  (`derive_seed(config.seed, BADGE_SEED_TAG)`, one per run), not scipy's global
  `rv_discrete` state; chosen points get probability 0 (DeepAL+ re-draws until an
  unchosen index comes up). If every unchosen point is at distance 0 from the chosen
  set (duplicate images, KI-46), the next point is drawn uniformly among the
  unchosen ones (DeepAL+ stops in `pdb.set_trace()`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import Strategy
from .diagnostics import RoundDiagnostics, pool_order_fraction

# `dalmax.seeding.derive_seed` tag of the k-means++ generator.
BADGE_SEED_TAG = "badge"


def _residuals(probs: torch.Tensor) -> torch.Tensor:
    """`a = p - onehot(argmax p)`: the per-class factor of the gradient embedding."""
    residuals = probs.clone()
    residuals[torch.arange(len(probs)), probs.argmax(dim=1)] -= 1.0
    return residuals


def gradient_embeddings(probs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """BADGE gradient embeddings `[n, n_classes * dim]`, class-major blocks,
    `block_c = (p_c - [c == argmax p]) * h` (DeepAL+'s layout and sign)."""
    residuals = _residuals(probs)
    return (residuals[:, :, None] * features[:, None, :]).reshape(len(probs), -1)


def kmeans_plus_plus(
    residuals: np.ndarray, features: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding of `n` points on the embeddings `g_x = a_x (x) h_x`.

    `residuals` `[N, C]` and `features` `[N, D]` are the two factors (float64 is
    used internally). Returns the chosen positions in selection order.
    """
    a = np.asarray(residuals, dtype=np.float64)
    h = np.asarray(features, dtype=np.float64)
    n_points = len(a)
    if n > n_points:
        raise ValueError(f"cannot choose {n} points out of {n_points}")
    sq_norms = (a * a).sum(axis=1) * (h * h).sum(axis=1)

    chosen = np.zeros(n_points, dtype=bool)
    order: list[int] = []
    min_d2 = np.full(n_points, np.inf)
    index = int(np.argmax(sq_norms))
    while True:
        chosen[index] = True
        order.append(index)
        if len(order) == n:
            return np.asarray(order, dtype=np.int64)
        cross = (a @ a[index]) * (h @ h[index])
        d2 = np.maximum(sq_norms + sq_norms[index] - 2.0 * cross, 0.0)
        np.minimum(min_d2, d2, out=min_d2)
        weights = np.where(chosen, 0.0, min_d2)
        total = weights.sum()
        if total > 0:
            # D(x)^2 / sum D^2: `min_d2` already holds squared distances.
            index = int(rng.choice(n_points, p=weights / total))
        else:
            index = int(rng.choice(np.flatnonzero(~chosen)))


class BadgeSampling(Strategy):
    """Query `n` unlabeled samples by k-means++ on their BADGE gradient embeddings."""

    def __init__(self, dataset, net, logger, *, rng: np.random.Generator) -> None:
        super().__init__(dataset, net, logger)
        self.rng = rng
        self._diag = RoundDiagnostics()

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        logits, features = self.net.predict_logits_and_embeddings(unlabeled_data)
        residuals = _residuals(torch.softmax(logits.to(torch.float64), dim=1))
        chosen = kmeans_plus_plus(residuals.numpy(), features.numpy(), n, self.rng)
        picked = unlabeled_idxs[chosen]
        self._diag.add(pool_order_fraction=pool_order_fraction(picked, unlabeled_idxs, n))
        return picked
