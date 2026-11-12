"""VAAL networks and their per-query training (Sinha, Ebrahimi & Darrell, "Variational
Adversarial Active Learning", ICCV 2019; original code github.com/sinhasam/vaal),
port of DeepAL+ `query_strategies/vaal.py::train_vaal` and `nets.py::VAE_*` (ADR 0015).

The task model of a VAAL run is trained exactly as for every other strategy
(`DeepLearning.train`). At every query, a fresh VAE and a fresh latent discriminator
are trained on the labeled set and the unlabeled pool (joint loader, labeled set
cycled to the pool's length, `epochs` epochs), then the discriminator scores the
unlabeled pool through the VAE means and the samples with the LOWEST score (most
likely unlabeled) are queried.

Per batch: `num_vae_steps` VAE steps on
`MSE(x_l) + beta * KLD(x_l) + MSE(x_u) + beta * KLD(x_u) + adversary_param * BCE`
where the BCE term asks the discriminator to call BOTH sets labeled (target 1, the
VAE tries to fool it); then `num_adv_steps` discriminator steps on the VAE means
(computed without gradient) with targets labeled = 1, unlabeled = 0
(`discriminator_targets`).

Network (`VAE`): the DeepAL+/original architecture for `image_size` x `image_size`
RGB inputs, `image_size` a multiple of 16: four stride-2 4x4 convolutions
(128-256-512-1024 channels, BatchNorm, ReLU) down to `image_size / 16`, linear layers
to the mean and log-variance (`latent_dim` = 32), and a decoder from a
`1024 x (image_size / 8)^2` linear projection through three stride-2 transposed
convolutions and a final 1x1 one. DANINHAS images are 128 x 128; they are resized to
`image_size` = 64 (area interpolation) before entering the VAE, because the
`VAE_ImageNet`-style network is the 64 x 64 one (its layer sizes `1024*4*4` /
`1024*8*8`, also the shapes the original VAAL `model.py` comments describe). The
discriminator scores the VAE means; it is DeepAL+'s `Discriminator`
(`latent_dim -> 512 -> 512 -> 1`, sigmoid). Weight initialisation as in both codes:
`kaiming_init` on the encoder/decoder Conv2d/Linear/BatchNorm layers and on the
discriminator, PyTorch defaults for `fc_mu`/`fc_logvar` and the transposed
convolutions (the original's `weight_init` silently skips them).

Deviations from DeepAL+, deliberate:

- Discriminator targets: DeepAL+ uses target 1 for BOTH the labeled and the
  unlabeled batch in the discriminator step, so the discriminator learns nothing
  about labeled-vs-unlabeled; here labeled = 1, unlabeled = 0 (original VAAL).
- Query order: the LOWEST discriminator outputs (original VAAL's sampler); DeepAL+
  sorts descending.
- `num_adv_steps` discriminator steps per batch (default 1, original VAAL); DeepAL+'s
  loop runs `num_vae_steps` (2) discriminator steps.
- Optimizer: Adam with `lr` = 5e-4 for both networks (original VAAL); DeepAL+ reuses
  the task model's optimizer and arguments (SGD lr 0.05 / momentum 0.3 here).
- `VAE_ImageNet` in DeepAL+ cannot run as written: its first convolution has stride 1
  (4x4, padding 2), so a 64 x 64 input leaves an 8 x 8 map and `View((-1, 1024*4*4))`
  silently turns every image into four rows (the reconstruction then does not match
  the input's batch). Here the first convolution has stride 2 like the others, which
  is the shape the `1024*4*4` / `1024*8*8` sizes imply.
- Randomness: weights from a per-run generator (`derive_seed(seed, VAAL_SEED_TAG)`)
  inside `dalmax.seeding.isolated_torch_seed`; the reparameterisation noise and the
  joint loader's order from explicit generators drawn from it. Nothing consumes the
  run's global torch stream, so the task model's training is untouched by the query.
- Networks in eval mode for the scoring (BatchNorm running statistics); the original
  sampler leaves them in train mode. No hardcoded `.cuda()`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dalmax.config.schema import VAALArgs
from dalmax.models.joint_dataset import joint_loader
from dalmax.models.waal import Critic, kaiming_init
from dalmax.seeding import isolated_torch_seed, torch_generator

# `dalmax.seeding.derive_seed` tag of the per-run VAAL generator.
VAAL_SEED_TAG = "vaal"

_MAX_SEED = 2**32 - 1
_CHANNELS = (128, 256, 512, 1024)


class VAE(nn.Module):
    """DeepAL+/VAAL VAE for `image_size` x `image_size` RGB inputs (module docstring)."""

    def __init__(self, image_size: int = 64, latent_dim: int = 32, nc: int = 3) -> None:
        super().__init__()
        if image_size % 16:
            raise ValueError(f"image_size must be a multiple of 16, got {image_size}")
        self.image_size = image_size
        self.latent_dim = latent_dim
        enc_side, dec_side = image_size // 16, image_size // 8
        c1, c2, c3, c4 = _CHANNELS
        self._dec_shape = (c4, dec_side, dec_side)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, c1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(True),
            nn.Conv2d(c3, c4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(c4 * enc_side * enc_side, latent_dim)
        self.fc_logvar = nn.Linear(c4 * enc_side * enc_side, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, c4 * dec_side * dec_side)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(True),
            nn.ConvTranspose2d(c3, c2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.ConvTranspose2d(c2, c1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.ConvTranspose2d(c1, nc, 1),
        )
        for module in [*self.encoder, self.decoder_input, *self.decoder]:
            kaiming_init(module)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.decoder_input(z).view(-1, *self._dec_shape))

    def forward(
        self, x: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(reconstruction, z, mu, logvar)`; the noise of `z` comes from `generator`."""
        mu, logvar = self.encode(x)
        epsilon = torch.randn(mu.shape, generator=generator).to(mu.device)
        z = mu + epsilon * (0.5 * logvar).exp()
        return self.decode(z), z, mu, logvar


def resize_for_vae(x: torch.Tensor, image_size: int) -> torch.Tensor:
    """Area-resize a batch to `image_size` x `image_size` (no-op if already that size)."""
    if x.shape[-2:] == (image_size, image_size):
        return x
    return F.interpolate(x, size=(image_size, image_size), mode="area")


def vae_loss(
    x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float
) -> torch.Tensor:
    """`MSE(recon, x)` (mean) `+ beta * KLD` (summed over batch and latent, as both codes)."""
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return F.mse_loss(recon, x) + beta * kld


def discriminator_targets(
    n_labeled: int, n_unlabeled: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets of the discriminator step: labeled = 1, unlabeled = 0 (original VAAL)."""
    return torch.ones(n_labeled, device=device), torch.zeros(n_unlabeled, device=device)


def discriminator_loss(labeled_preds: torch.Tensor, unlabeled_preds: torch.Tensor) -> torch.Tensor:
    """Discriminator step loss: `BCE(labeled, 1) + BCE(unlabeled, 0)`."""
    lab_target, unlab_target = discriminator_targets(
        len(labeled_preds), len(unlabeled_preds), labeled_preds.device
    )
    return F.binary_cross_entropy(labeled_preds, lab_target) + F.binary_cross_entropy(
        unlabeled_preds, unlab_target
    )


def adversarial_loss(labeled_preds: torch.Tensor, unlabeled_preds: torch.Tensor) -> torch.Tensor:
    """VAE step's adversarial term: both sets labeled, `BCE(labeled, 1) + BCE(unlabeled, 1)`."""
    return F.binary_cross_entropy(
        labeled_preds, torch.ones_like(labeled_preds)
    ) + F.binary_cross_entropy(unlabeled_preds, torch.ones_like(unlabeled_preds))


class VAALSampler:
    """Trains a VAE + discriminator per query and ranks the pool by the discriminator."""

    def __init__(
        self,
        args: VAALArgs,
        rng: np.random.Generator,
        device: torch.device,
        train_args: dict,
        test_args: dict,
    ) -> None:
        self.args = args
        self.rng = rng
        self.device = device
        self.train_args = train_args
        self.test_args = test_args

    def _next_seed(self) -> int:
        return int(self.rng.integers(0, _MAX_SEED))

    def build_networks(self) -> tuple[VAE, Critic]:
        with isolated_torch_seed(self._next_seed()):
            vae = VAE(self.args.image_size, self.args.latent_dim)
            discriminator = Critic(self.args.latent_dim)
        return vae.to(self.device), discriminator.to(self.device)

    def fit(self, labeled: Dataset, unlabeled: Dataset) -> tuple[VAE, Critic]:
        """Train a fresh VAE and discriminator on the joint labeled/unlabeled loader."""
        args = self.args
        vae, discriminator = self.build_networks()
        noise = torch_generator(self._next_seed())
        loader = joint_loader(labeled, unlabeled, self.train_args, torch_generator(self._next_seed()))
        opt_vae = optim.Adam(vae.parameters(), lr=args.lr)
        opt_dis = optim.Adam(discriminator.parameters(), lr=args.lr)
        size = args.image_size

        for epoch in tqdm(range(1, args.epochs + 1), ncols=100):
            vae.train()
            discriminator.train()
            vae_losses: list[float] = []
            dis_losses: list[float] = []
            for _index, label_x, _label_y, unlabel_x, _ in loader:
                label_x = resize_for_vae(label_x.to(self.device), size)
                unlabel_x = resize_for_vae(unlabel_x.to(self.device), size)

                for _ in range(args.num_vae_steps):
                    recon, _z, mu, logvar = vae(label_x, noise)
                    unsup_loss = vae_loss(label_x, recon, mu, logvar, args.beta)
                    u_recon, _u_z, u_mu, u_logvar = vae(unlabel_x, noise)
                    transductive_loss = vae_loss(unlabel_x, u_recon, u_mu, u_logvar, args.beta)
                    adv = adversarial_loss(discriminator(mu), discriminator(u_mu))
                    total_vae_loss = unsup_loss + transductive_loss + args.adversary_param * adv
                    opt_vae.zero_grad()
                    total_vae_loss.backward()
                    opt_vae.step()
                    vae_losses.append(total_vae_loss.item())

                for _ in range(args.num_adv_steps):
                    with torch.no_grad():
                        mu, _ = vae.encode(label_x)
                        u_mu, _ = vae.encode(unlabel_x)
                    dsc_loss = discriminator_loss(discriminator(mu), discriminator(u_mu))
                    opt_dis.zero_grad()
                    dsc_loss.backward()
                    opt_dis.step()
                    dis_losses.append(dsc_loss.item())
            print(
                f" - VAAL epoch {epoch}/{args.epochs} VAE loss: {np.mean(vae_losses):.4f} "
                f"discriminator loss: {np.mean(dis_losses):.4f}"
            )
        return vae, discriminator

    def score(self, vae: VAE, discriminator: Critic, data: Dataset) -> torch.Tensor:
        """Discriminator output on the VAE mean of every sample of `data` (eval mode)."""
        vae.eval()
        discriminator.eval()
        scores = torch.zeros(len(data))
        # An explicit generator: even an unshuffled DataLoader draws its worker base
        # seed from the global torch stream otherwise.
        loader = DataLoader(
            data, shuffle=False, generator=torch_generator(self._next_seed()), **self.test_args
        )
        with torch.no_grad():
            for x, _y, idxs in loader:
                mu, _ = vae.encode(resize_for_vae(x.to(self.device), self.args.image_size))
                scores[idxs] = discriminator(mu).cpu()
        return scores
