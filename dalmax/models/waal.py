"""WAAL training (Shui et al., "Deep Active Learning: Unified and Principled Method
for Query and Training", AISTATS 2020), port of DeepAL+ `nets_waal.py::Net_WAAL`
(ADR 0015).

`WAALTrainer.fit` replaces `DeepLearning.train` for the `WAAL` strategy. Feature
extractor = the task model's backbone (`DaninhasModelResNet50.feature_extractor`,
whose pooled 2048-d output is the `e1` of `forward`), classifier = its final linear
layer; one `model(x)` call gives both `z` and the logits, so the evaluated model is
the ordinary task model (`net.clf`): `predict`, `saved_model.pth` and the inference
tools work unchanged. The Wasserstein critic (`Critic`, three linear layers with a
sigmoid output, DeepAL+'s `Discriminator`) is trained alongside, returned to the
strategy for the query, and is NOT saved in the checkpoint.

Per batch of the joint loader (`dalmax.models.joint_dataset`, labeled set cycled to
the unlabeled pool's length, `n_epoch` epochs):

1. task step (critic frozen): `CE(labeled) + alpha * W + alpha * gp_weight_task * gp`,
   `W = critic(z_u).mean() - gamma_ratio * critic(z_l).mean()`; the gradient penalty
   is computed on features recomputed without gradient, so it carries no gradient
   into the task model (as in DeepAL+ and the original code; the recomputation is
   kept because, in train mode, it also updates the BatchNorm running statistics);
2. critic step (task model frozen, features recomputed without gradient):
   `-alpha * W - alpha * gp_weight_critic * gp`.

Both steps use SGD with the task's `optimizer_args` (DeepAL+ and the original), one
optimizer for the whole task model (equivalent to DeepAL+'s separate feature and
classifier SGD optimizers: SGD state is per parameter).

Deviations from DeepAL+, deliberate:

- The critic's initial weights come from a per-run generator (`derive_seed(seed,
  WAAL_SEED_TAG)`), built in `dalmax.seeding.isolated_torch_seed`; the interpolation
  coefficients of the gradient penalty and the joint loader's order use explicit
  generators from the same per-run generator. The task model itself is built with
  `DeepLearning.build_model` from the run's torch stream, like every strategy.
- No hardcoded `.cuda()`: everything runs on `net.device`.
- The query's `log p` is `log_softmax` of the logits (float64) instead of
  `log(softmax)` in float32: equal wherever the latter is finite, but a saturated
  softmax no longer gives `-inf` (an infinite score, ties at `inf`).

Observability (ADR 0015, reporting only): `fit` returns a `WAALFitResult` with the
number of optimizer steps, the mean Wasserstein estimate over the last epoch and a
frozen copy of the critic at initialisation; `score_components` returns, per pool
sample, the uncertainty term, the trained critic's score and the initial critic's
score, from which the strategy reports critic statistics, the correlation of the
query-time critic with its initialisation, and the spread of both score terms.

Kept as in DeepAL+ and the original WAAL code, and flagged in ADR 0015: the critic
step subtracts the gradient penalty (`- alpha * gp * 2`), i.e. the critic is pushed
to *increase* it, the opposite sign of WGAN-GP; and DeepAL+ fixes `gamma_ratio = 1`
where the original uses `|L| / |U|`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dalmax.config.schema import WAALArgs
from dalmax.models.joint_dataset import joint_loader
from dalmax.seeding import isolated_torch_seed, torch_generator

# `dalmax.seeding.derive_seed` tag of the per-run WAAL generator.
WAAL_SEED_TAG = "waal"

_MAX_SEED = 2**32 - 1


def kaiming_init(module: nn.Module) -> None:
    """DeepAL+/VAAL `kaiming_init`: Kaiming-normal weights and zero bias for Linear and
    Conv2d, unit weight and zero bias for BatchNorm; other modules untouched."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.weight.data.fill_(1)
        if module.bias is not None:
            module.bias.data.fill_(0)


class Critic(nn.Module):
    """DeepAL+ `Discriminator`: `dim -> 512 -> 512 -> 1`, ReLU, sigmoid output."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        for module in self.net:
            kaiming_init(module)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(-1)


def gradient_penalty(
    critic: nn.Module, h_s: torch.Tensor, h_t: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """DeepAL+/WAAL gradient penalty: `mean((||grad critic(u)||_2 - 1)^2)` over
    `u` = random interpolates of `h_s`/`h_t` (coefficients from `generator`), plus
    `h_s` and `h_t` themselves."""
    alpha = torch.rand(h_s.size(0), 1, generator=generator).to(h_s.device)
    interpolates = h_s + alpha * (h_t - h_s)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    preds = critic(interpolates)
    gradients = torch.autograd.grad(
        preds,
        interpolates,
        grad_outputs=torch.ones_like(preds),
        retain_graph=True,
        create_graph=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def waal_scores(log_probs: torch.Tensor, critic_scores: torch.Tensor, selection: float) -> torch.Tensor:
    """WAAL query score, lowest first: `0.5 ||log p||_2 + 0.5 ||log p||_1 - selection * critic`.

    `||log p||_1 = sum_c -log p_c` is smallest for the most uncertain predictions,
    and a high critic score means "looks unlabeled"; both push a sample up the ranking.
    """
    l2_upper = torch.norm(log_probs, dim=1)
    l1_upper = torch.sum(-log_probs, dim=1)
    return 0.5 * l2_upper + 0.5 * l1_upper - selection * critic_scores


@dataclass
class WAALFitResult:
    """What `WAALTrainer.fit` hands to the strategy: the trained critic plus diagnostics."""

    critic: Critic
    critic_init: Critic
    n_steps: int
    wasserstein_last_epoch: float


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


class WAALTrainer:
    """WAAL min-max training of the task model and the critic (module docstring)."""

    def __init__(self, args: WAALArgs, rng: np.random.Generator) -> None:
        self.args = args
        self.rng = rng

    def _next_seed(self) -> int:
        return int(self.rng.integers(0, _MAX_SEED))

    def fit(self, net, labeled, unlabeled) -> WAALFitResult:
        """Train `net.clf` (rebuilt) and a new critic; return it with diagnostics."""
        params = net.params
        device = net.device
        args = self.args
        clf = net.build_model()
        with isolated_torch_seed(self._next_seed()):
            critic = Critic(clf.get_embedding_dim())
        critic = critic.to(device)
        critic_init = copy.deepcopy(critic).eval()
        _set_requires_grad(critic_init, False)
        net.clf = clf
        gp_generator = torch_generator(self._next_seed())
        loader = joint_loader(
            labeled, unlabeled, params["train_args"], torch_generator(self._next_seed())
        )
        opt_task = optim.SGD(clf.parameters(), **params["optimizer_args"])
        opt_critic = optim.SGD(critic.parameters(), **params["optimizer_args"])

        n_epoch = params["n_epoch"]
        n_steps = 0
        wasserstein_epoch: list[float] = []
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            clf.train()
            critic.train()
            pred_losses: list[float] = []
            wasserstein_epoch = []
            for _index, label_x, label_y, unlabel_x, _ in loader:
                label_x, label_y = label_x.to(device), label_y.to(device)
                unlabel_x = unlabel_x.to(device)

                # Task step: feature extractor + classifier, critic frozen.
                _set_requires_grad(clf, True)
                _set_requires_grad(critic, False)
                lb_out, lb_z = clf(label_x)
                _unlb_out, unlb_z = clf(unlabel_x)
                opt_task.zero_grad()
                pred_loss = torch.mean(F.cross_entropy(lb_out, label_y))
                wasserstein = critic(unlb_z).mean() - args.gamma_ratio * critic(lb_z).mean()
                with torch.no_grad():
                    _, lb_z = clf(label_x)
                    _, unlb_z = clf(unlabel_x)
                gp = gradient_penalty(critic, unlb_z, lb_z, gp_generator)
                loss = pred_loss + args.alpha * wasserstein + args.alpha * gp * args.gp_weight_task
                loss.backward()
                opt_task.step()
                pred_losses.append(pred_loss.item())

                # Critic step: task model frozen, features recomputed after the update.
                _set_requires_grad(clf, False)
                _set_requires_grad(critic, True)
                with torch.no_grad():
                    _, lb_z = clf(label_x)
                    _, unlb_z = clf(unlabel_x)
                gp = gradient_penalty(critic, unlb_z, lb_z, gp_generator)
                wasserstein = critic(unlb_z).mean() - args.gamma_ratio * critic(lb_z).mean()
                critic_loss = -args.alpha * wasserstein - args.alpha * gp * args.gp_weight_critic
                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()
                n_steps += 1
                wasserstein_epoch.append(wasserstein.item())
            print(f" - Epoch {epoch}/{n_epoch} task CE: {np.mean(pred_losses):.4f}")
        _set_requires_grad(clf, True)
        _set_requires_grad(critic, True)
        return WAALFitResult(
            critic=critic,
            critic_init=critic_init,
            n_steps=n_steps,
            wasserstein_last_epoch=float(np.mean(wasserstein_epoch)) if wasserstein_epoch else float("nan"),
        )

    def score_components(
        self, net, fit: WAALFitResult, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per sample of `data` (eval mode, float64): the uncertainty term
        `0.5 ||log p||_2 + 0.5 ||log p||_1` (`log_softmax` of the logits), the trained
        critic's score and the initial critic's score. The query score is
        `uncertainty - selection * critic` (`waal_scores`)."""
        clf = net.clf
        clf.eval()
        fit.critic.eval()
        uncertainty = torch.zeros(len(data), dtype=torch.float64)
        critic_scores = torch.zeros(len(data), dtype=torch.float64)
        init_scores = torch.zeros(len(data), dtype=torch.float64)
        loader = DataLoader(
            data,
            shuffle=False,
            generator=torch_generator(self._next_seed()),
            **net.params["test_args"],
        )
        with torch.no_grad():
            for x, _y, idxs in loader:
                logits, z = clf(x.to(net.device))
                log_probs = F.log_softmax(logits.to(torch.float64), dim=1)
                zeros = torch.zeros(len(idxs), dtype=torch.float64, device=log_probs.device)
                uncertainty[idxs] = waal_scores(log_probs, zeros, 0.0).cpu()
                critic_scores[idxs] = fit.critic(z).to(torch.float64).cpu()
                init_scores[idxs] = fit.critic_init(z).to(torch.float64).cpu()
        return uncertainty, critic_scores, init_scores

