"""Loss-prediction module and its joint training (Yoo & Kweon, "Learning Loss for
Active Learning", CVPR 2019), port of DeepAL+ `nets_lossprediction.py` (ADR 0015).

`LossNet`: for each of the four residual-stage maps of the backbone, global average
pooling, a fully connected layer to `interm_dim` (128) and a ReLU; the four vectors
are concatenated and a last fully connected layer outputs the predicted loss.

`LossPredictionTrainer.fit` replaces `DeepLearning.train` for `LossPredictionLoss`:
the task model (`DeepLearning.build_model`, same initial weights and the same
shuffled loader as every other strategy) and a fresh `LossNet` are trained jointly
for `n_epoch` epochs on `CE + weight * ranking_loss`; during the first
`floor(detach_fraction * n_epoch)` epochs the module's gradient also reaches the
backbone, afterwards the stage maps are detached (the paper's "stop the gradient
after 120 of 200 epochs").

Deviations from DeepAL+, deliberate:

- Epoch budget: `n_epoch` epochs, the same as every other strategy. DeepAL+ trains
  `n_epoch + lpl_epoch` (20 extra) epochs and detaches after epoch 20 (i.e. the
  module's gradient reaches the backbone during the first 19 epochs only, whatever
  `n_epoch` is), the reverse proportion of the original paper.
- One forward pass per batch: DeepAL+ runs the task model twice on every batch
  (`out, feature = clf(x)` then `out, e1 = clf(x)`), which updates the BatchNorm
  running statistics twice and doubles the cost.
- The module is rebuilt with every task model (DalMax retrains from scratch each
  round); DeepAL+ keeps one module instance across rounds while it rebuilds the
  backbone. Its initial weights come from a per-run generator (`derive_seed(seed,
  LOSS_PREDICTION_SEED_TAG)`), built in `dalmax.seeding.isolated_torch_seed`, so the
  run's global torch stream (task-model init, data order) is not consumed.
- Global average pooling is adaptive (any input size); DeepAL+ hardcodes the pooling
  kernel per dataset. The stage maps are ResNet-50's (256/512/1024/2048 channels,
  `DaninhasModelResNet50.forward_with_stages`); DeepAL+'s ResNet-18 models skip the
  max-pool in their loss-prediction variant, ours is the same network every other
  strategy trains.
- The ranking loss needs an even batch: with an odd batch the last sample is left
  out of the ranking loss (it still counts in the cross-entropy); a batch of one
  has no pair and contributes 0. DeepAL+ asserts an even batch.

The module optimizer is Adam with `module_lr` (DeepAL+; the original paper uses the
backbone's SGD). The trained `LossNet` is returned to the strategy and is not part
of `saved_model.pth`, which holds the task model only (inference is unaffected).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dalmax.config.schema import LossPredictionArgs
from dalmax.seeding import isolated_torch_seed, torch_generator

# `dalmax.seeding.derive_seed` tag of the per-run module-initialisation generator.
LOSS_PREDICTION_SEED_TAG = "loss_prediction"


class LossNet(nn.Module):
    """Predicted loss from the backbone's stage maps (GAP -> FC -> ReLU per stage)."""

    def __init__(self, stage_channels: Sequence[int], interm_dim: int = 128) -> None:
        super().__init__()
        self.fcs = nn.ModuleList(nn.Linear(c, interm_dim) for c in stage_channels)
        self.linear = nn.Linear(len(stage_channels) * interm_dim, 1)

    def forward(self, stages: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled = [
            F.relu(fc(F.adaptive_avg_pool2d(s, 1).flatten(1)))
            for fc, s in zip(self.fcs, stages, strict=True)
        ]
        return self.linear(torch.cat(pooled, dim=1)).view(-1)


def loss_prediction_ranking_loss(
    pred_loss: torch.Tensor, target_loss: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Pairwise margin ranking loss of Yoo & Kweon (DeepAL+ `LossPredLoss`, reduction mean).

    The batch (even length `2B`) is split into the pairs `(i, 2B-1-i)`, `i < B`; with
    `d_pred = pred_i - pred_j`, `d_true = true_i - true_j` (detached) and
    `s = +1 if d_true > 0 else -1`, the loss is `sum_i max(0, margin - s * d_pred) / B`.
    """
    if len(pred_loss) % 2:
        raise ValueError(f"the ranking loss needs an even batch, got {len(pred_loss)}")
    half = len(pred_loss) // 2
    d_pred = (pred_loss - pred_loss.flip(0))[:half]
    d_true = (target_loss - target_loss.flip(0))[:half].detach()
    sign = 2 * torch.sign(torch.clamp(d_true, min=0)) - 1
    return torch.clamp(margin - sign * d_pred, min=0).sum() / half


def attached_epochs(detach_fraction: float, n_epoch: int) -> int:
    """Number of leading epochs during which the module's gradient reaches the backbone."""
    return int(math.floor(detach_fraction * n_epoch + 1e-9))


class LossPredictionTrainer:
    """Joint task-model + `LossNet` training (see the module docstring)."""

    def __init__(self, args: LossPredictionArgs, rng: np.random.Generator) -> None:
        self.args = args
        self.rng = rng

    def _next_seed(self) -> int:
        return int(self.rng.integers(0, 2**32 - 1))

    def build_module(self, stage_channels: Sequence[int], device: torch.device) -> LossNet:
        """A fresh `LossNet`, initialised from the next seed of the per-run generator."""
        with isolated_torch_seed(self._next_seed()):
            module = LossNet(stage_channels, self.args.interm_dim)
        return module.to(device)

    def fit(self, net, data) -> LossNet:
        """Train `net.clf` (rebuilt) and a new `LossNet` on `data`; return the module."""
        params = net.params
        n_epoch = params["n_epoch"]
        clf = net.build_model()
        if not hasattr(clf, "forward_with_stages"):
            raise TypeError(
                f"LossPredictionLoss needs a model exposing forward_with_stages(); "
                f"{type(clf).__name__} does not"
            )
        module = self.build_module(clf.stage_channels(), net.device)
        net.clf = clf
        clf.train()
        module.train()
        optimizer = optim.SGD(clf.parameters(), **params["optimizer_args"])
        optimizer_module = optim.Adam(module.parameters(), lr=self.args.module_lr)
        n_attached = attached_epochs(self.args.detach_fraction, n_epoch)

        loader = DataLoader(data, shuffle=True, **params["train_args"])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            losses: list[float] = []
            for x, y, _idxs in loader:
                x, y = x.to(net.device), y.to(net.device)
                optimizer.zero_grad()
                optimizer_module.zero_grad()
                logits, _features, stages = clf.forward_with_stages(x)
                target_loss = F.cross_entropy(logits, y, reduction="none")
                if epoch > n_attached:
                    stages = [s.detach() for s in stages]
                backbone_loss = target_loss.mean()
                even = len(target_loss) - len(target_loss) % 2
                if even:
                    pred_loss = module(stages)
                    module_loss = loss_prediction_ranking_loss(
                        pred_loss[:even], target_loss[:even], self.args.margin
                    )
                    loss = backbone_loss + self.args.weight * module_loss
                else:
                    loss = backbone_loss
                loss.backward()
                optimizer.step()
                optimizer_module.step()
                losses.append(loss.item())
            print(f" - Epoch {epoch}/{n_epoch} Loss: {np.mean(losses):.4f}")
        return module

    def predict_losses(
        self, net, module: LossNet, data, *, with_true_loss: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predicted loss of every sample of `data` (eval mode, CPU tensor).

        With `with_true_loss`, also the true per-sample cross-entropy (uses the labels of
        `data`: only for the labeled set, as a reporting-only diagnostic). The loader gets
        an explicit generator (an unshuffled DataLoader otherwise draws its worker base
        seed from the global torch stream).
        """
        clf = net.clf
        clf.eval()
        module.eval()
        scores = torch.zeros(len(data))
        true_loss = torch.zeros(len(data))
        loader = DataLoader(
            data,
            shuffle=False,
            generator=torch_generator(self._next_seed()),
            **net.params["test_args"],
        )
        with torch.no_grad():
            for x, y, idxs in loader:
                logits, _features, stages = clf.forward_with_stages(x.to(net.device))
                scores[idxs] = module(stages).cpu()
                if with_true_loss:
                    true_loss[idxs] = F.cross_entropy(
                        logits, torch.as_tensor(y).to(net.device), reduction="none"
                    ).cpu()
        return (scores, true_loss) if with_true_loss else scores
