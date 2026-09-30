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

    As in the original, the `while` loop has NO iteration cap by default.
    `max_iter` (default None = original behaviour) is an optional safety cap
    that is not part of the original algorithm. A single `logger.warning` is
    emitted the first time any sample exceeds 1000 iterations so a
    pathological sample is visible in the log instead of silently hanging.
    """

    _WARN_AFTER_ITERS = 1000

    def __init__(self, dataset, net, logger, eps: float = 0.05, max_iter: int | None = None) -> None:
        super().__init__(dataset, net, logger)
        self.eps = eps
        self.max_iter = max_iter
        self._warned_long = False

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
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            out, e1 = self.net.clf(nx + eta)
            py = out.max(1)[1]

            n_iter += 1
            if n_iter > self._WARN_AFTER_ITERS and not self._warned_long:
                self._warned_long = True
                self.logger.warning(
                    f"AdversarialBIM: a sample exceeded {self._WARN_AFTER_ITERS} iterations "
                    "without flipping its prediction (loop is uncapped unless max_iter is set)"
                )
            if self.max_iter is not None and n_iter >= self.max_iter:
                break

        return (eta * eta).sum().item()
