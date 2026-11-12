"""VAAL: variational adversarial active learning (Sinha et al., ICCV 2019), port of
DeepAL+ `query_strategies/vaal.py` with the original method's discriminator targets
(ADR 0015).

The task model is trained exactly as for every other strategy (`Strategy.train`).
Each query trains a fresh VAE + latent discriminator on the labeled set and the
unlabeled pool (`dalmax.models.vaal.VAALSampler`, injected by `build_strategy`) and
queries the `n` unlabeled samples with the LOWEST discriminator output, i.e. those
the discriminator considers most likely unlabeled (stable sort: exact ties keep pool
order). See `dalmax/models/vaal.py` for the network, the 128 -> 64 resize and every
deviation from DeepAL+.
"""

from __future__ import annotations

import numpy as np

from dalmax.models.vaal import VAALSampler

from .base import Strategy


class VAAL(Strategy):
    """Query the unlabeled samples the VAAL discriminator rates least "labeled"."""

    def __init__(self, dataset, net, logger, *, sampler: VAALSampler) -> None:
        super().__init__(dataset, net, logger)
        self.sampler = sampler

    def query(self, n: int) -> np.ndarray:
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        self.logger.warning(
            f"VAAL: training VAE + discriminator on {len(labeled_data)} labeled and "
            f"{len(unlabeled_data)} unlabeled samples ({self.sampler.args.epochs} epochs)"
        )
        vae, discriminator = self.sampler.fit(labeled_data, unlabeled_data)
        scores = self.sampler.score(vae, discriminator, unlabeled_data).numpy()
        return unlabeled_idxs[np.argsort(scores, kind="stable")[:n]]
