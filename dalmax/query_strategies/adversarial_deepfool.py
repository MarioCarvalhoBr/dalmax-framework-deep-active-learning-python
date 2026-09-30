"""AdversarialDeepFool: faithful per-sample port of the original DeepAL strategy.

Source: DeepAL (Huang, 2021), https://github.com/ej0cl6/deep-active-learning.
Method: DeepFool (Moosavi-Dezfooli et al., 2016) used as an active-learning
distance as in DFAL (Ducoffe & Precup, 2018).
"""

import numpy as np
import torch

from .adversarial_base import PerSampleAdversarialStrategy


class AdversarialDeepFool(PerSampleAdversarialStrategy):
    """Query the samples closest to the decision boundary under DeepFool.

    For each unlabeled sample, up to `max_iter` multi-class DeepFool steps:
    for every non-predicted class `k`, `w_k = grad_k - grad_py`,
    `f_k = out_k - out_py`; the class with minimal `|f_k| / ||w_k||` gives the
    step `r = value / ||w|| * w`. The distance is `(eta * eta).sum()`; the `n`
    smallest are queried. As in the original, a sample still not flipped after
    `max_iter` steps returns its `(eta * eta).sum()` anyway.

    Deviation from the original (KI-39, same failure family as the BIM hang
    observed on the real campaign on 2026-09-30): with a saturated float32
    prediction the gradients vanish (`||w_k|| == 0`, or NaN), so no finite
    `value_i` exists and the original step is undefined. Classes with a
    zero/NaN norm are skipped; if no class is usable in an iteration the loop
    stops and the sample returns `float("inf")` (could not be moved, treated
    as maximally far).
    """

    def __init__(self, dataset, net, logger, max_iter: int = 50) -> None:
        super().__init__(dataset, net, logger)
        self.max_iter = max_iter
        self._n_zero_grad = 0

    def _begin_query(self) -> None:
        self._n_zero_grad = 0

    def _query_summary(self) -> str:
        return f"{self._n_zero_grad} had no usable gradient (saturated prediction)"

    def cal_dis(self, x: torch.Tensor) -> float:
        device = self._device()
        nx = torch.unsqueeze(x, 0).to(device)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape, device=device)

        out, e1 = self.net.clf(nx + eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0
        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                norm_wi = wi.detach().cpu().flatten().norm().item()
                if not np.isfinite(norm_wi) or norm_wi == 0.0:
                    continue
                value_i = np.abs(fi.item()) / norm_wi
                if np.isfinite(value_i) and value_i < value_l:
                    ri = value_i / norm_wi * wi
                    value_l = value_i

            if ri is None:
                self._n_zero_grad += 1
                return float("inf")

            eta += ri.clone()
            nx.grad.data.zero_()
            out, e1 = self.net.clf(nx + eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta * eta).sum().item()
