"""VAAL: variational adversarial active learning (Sinha et al., ICCV 2019), port of
DeepAL+ `query_strategies/vaal.py` with the original method's discriminator targets
(ADR 0015).

The task model is trained exactly as for every other strategy (`Strategy.train`).
Each query trains a fresh VAE + latent discriminator on the labeled set and the
unlabeled pool (`dalmax.models.vaal.VAALSampler`, injected by `build_strategy`) and
queries the `n` unlabeled samples with the LOWEST discriminator output, i.e. those
the discriminator considers most likely unlabeled; exact ties are broken at random
with the generator injected by `build_strategy` (`derive_seed(config.seed,
VAAL_TIES_SEED_TAG)`, `ranking.rank_ascending`), so a collapsed discriminator never
degenerates into pool order (KI-45). See `dalmax/models/vaal.py` for the network, the
image resize and every deviation from DeepAL+.

Reporting only (`diagnostics()`): per query, the discriminator's ROC AUC labeled vs
unlabeled (0.5 = it separates nothing), the std and number of distinct pool scores,
the mean KL divergence per pool sample (posterior collapse if ~0), the mean
per-dimension std of the pool's VAE means, and the KI-45 pool-order fraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dalmax.models.vaal import VAALSampler, kl_per_sample

from .base import Strategy
from .diagnostics import RoundDiagnostics, binary_auc, pool_order_fraction, score_spread
from .ranking import rank_ascending

# `dalmax.seeding.derive_seed` tag of the query tie-break generator.
VAAL_TIES_SEED_TAG = "vaal_ties"


class VAAL(Strategy):
    """Query the unlabeled samples the VAAL discriminator rates least "labeled"."""

    def __init__(
        self, dataset, net, logger, *, sampler: VAALSampler, rng: np.random.Generator
    ) -> None:
        super().__init__(dataset, net, logger)
        self.sampler = sampler
        self.rng = rng
        self._diag = RoundDiagnostics()

    def query(self, n: int) -> np.ndarray:
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        self.logger.warning(
            f"VAAL: training VAE + discriminator on {len(labeled_data)} labeled and "
            f"{len(unlabeled_data)} unlabeled samples ({self.sampler.args.epochs} epochs)"
        )
        vae, discriminator = self.sampler.fit(labeled_data, unlabeled_data)
        scores, mu, logvar = self.sampler.encode_and_score(vae, discriminator, unlabeled_data)
        scores_np = scores.numpy()
        picked = unlabeled_idxs[rank_ascending(scores_np, self.rng)[:n]]

        labeled_scores = self.sampler.score(vae, discriminator, labeled_data).numpy()
        spread = score_spread(scores_np)
        self._diag.add(
            vaal_discriminator_auc=binary_auc(labeled_scores, scores_np),
            vaal_score_std=spread["std"],
            vaal_score_n_unique=spread["n_unique"],
            vaal_mean_kl=float(kl_per_sample(mu, logvar).mean()),
            vaal_mu_dim_std=float(mu.std(dim=0, unbiased=False).mean()),
            pool_order_fraction=pool_order_fraction(picked, unlabeled_idxs, n),
        )
        return picked

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()
