"""AdversarialBIM: faithful per-sample port of the original DeepAL strategy.

Source: DeepAL (Huang, 2021), https://github.com/ej0cl6/deep-active-learning.
Method: Basic Iterative Method (Kurakin et al., 2016) used as an
active-learning distance as in DFAL (Ducoffe & Precup, 2018).
"""

import torch
import torch.nn.functional as F

from .adversarial_base import PerSampleAdversarialStrategy


class AdversarialBIM(PerSampleAdversarialStrategy):
    """Query the samples closest to the decision boundary under BIM.

    For each unlabeled sample, `eta` starts at 0 and, while the predicted
    class still equals the original prediction, takes a step
    `eta += eps * sign(grad)` of the cross-entropy (w.r.t. the original
    prediction). The distance is `(eta * eta).sum()`; the `n` smallest are
    queried.

    Deviations from the original DeepAL algorithm (both are required for
    termination, observed on the real campaign on 2026-09-30, KI-39):

    * The original loop is uncapped. For a sample predicted with probability
      ~1.0 in float32 the cross-entropy gradient w.r.t. the input is exactly
      zero, so `sign(0) = 0`, `eta` never changes, the prediction never flips
      and the loop never ends. Here `max_iter` defaults to 50 (the cap the
      original DeepFool uses) and the loop also stops as soon as the gradient
      is all zeros (no progress possible).
    * A sample whose loop ends WITHOUT flipping the prediction (cap reached or
      zero gradient) could not reach the decision boundary within the budget,
      so it returns `float("inf")` (maximally far, never queried before a
      converged sample). Samples that flip return `(eta * eta).sum()` as in
      the original.

    One `logger.warning` summary per query reports how many samples did not
    flip (capped / zero-gradient).
    """

    def __init__(self, dataset, net, logger, eps: float = 0.05, max_iter: int = 50) -> None:
        super().__init__(dataset, net, logger)
        self.eps = eps
        self.max_iter = max_iter
        self._n_capped = 0
        self._n_zero_grad = 0

    def _begin_query(self) -> None:
        self._n_capped = 0
        self._n_zero_grad = 0

    def _query_summary(self) -> str:
        return (
            f"{self._n_capped} hit the {self.max_iter}-iteration cap, "
            f"{self._n_zero_grad} had a zero gradient (saturated prediction)"
        )

    def cal_dis(self, x: torch.Tensor) -> float:
        device = self._device()
        nx = torch.unsqueeze(x, 0).to(device)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape, device=device)

        out, e1 = self.net.clf(nx + eta)
        py = out.max(1)[1]
        ny = out.max(1)[1]
        n_iter = 0
        while py.item() == ny.item():
            if n_iter >= self.max_iter:
                self._n_capped += 1
                return float("inf")
            loss = F.cross_entropy(out, ny)
            loss.backward()

            if nx.grad is None or not bool(nx.grad.data.any()):
                self._n_zero_grad += 1
                return float("inf")

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            out, e1 = self.net.clf(nx + eta)
            py = out.max(1)[1]
            n_iter += 1

        return (eta * eta).sum().item()
