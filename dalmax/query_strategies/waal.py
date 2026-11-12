"""WAAL: Wasserstein adversarial active learning (Shui et al., AISTATS 2020), port of
DeepAL+ `query_strategies/waal.py` (ADR 0015).

Training (every round, round 0 included) is the WAAL min-max training of the injected
`dalmax.models.waal.WAALTrainer` on the labeled set and the unlabeled pool, instead of
plain cross-entropy training; the evaluated and saved model is the ordinary task model
(feature extractor + classifier), so `saved_model.pth` and the inference tools work as
for every strategy. The critic is kept by the strategy for the query only.

Query: `0.5 * ||log p||_2 + 0.5 * ||log p||_1 - selection * critic(z)` over the
unlabeled pool, the `n` LOWEST first (DeepAL+ and the original); exact ties broken at
random with the generator injected by `build_strategy`
(`derive_seed(config.seed, WAAL_TIES_SEED_TAG)`, `ranking.rank_ascending`), so the
query never falls back to pool order (KI-45).

Reporting only (`diagnostics()`, ADR 0015): per training, the number of optimizer
steps (the joint loader iterates over the pool, so WAAL takes far more steps per round
than the plain strategies) and the mean Wasserstein estimate of the last epoch; per
query, the critic's mean/std on the labeled set and on the pool, the Pearson
correlation between the trained critic's and the initial critic's pool scores (close
to 1 = the critic barely moved), the std of the uncertainty term and of the
`selection * critic` term, and the KI-45 pool-order fraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dalmax.models.waal import WAALFitResult, WAALTrainer

from .base import Strategy
from .diagnostics import RoundDiagnostics, pearson, pool_order_fraction
from .ranking import rank_ascending

# `dalmax.seeding.derive_seed` tag of the query tie-break generator.
WAAL_TIES_SEED_TAG = "waal_ties"


class WAAL(Strategy):
    """WAAL training plus the uncertainty-minus-critic query."""

    def __init__(
        self,
        dataset,
        net,
        logger,
        *,
        trainer: WAALTrainer,
        selection: float,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(dataset, net, logger)
        self.trainer = trainer
        self.selection = selection
        self.rng = rng
        self.fit_result: WAALFitResult | None = None
        self._diag = RoundDiagnostics()

    def train(self) -> None:
        self.logger.warning("Training the DAL strategy (WAAL min-max on labeled + unlabeled)")
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        _unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        self.fit_result = self.trainer.fit(self.net, labeled_data, unlabeled_data)
        self._diag.add(
            waal_n_steps=self.fit_result.n_steps,
            waal_wasserstein_last_epoch=self.fit_result.wasserstein_last_epoch,
        )

    def query(self, n: int) -> np.ndarray:
        if self.fit_result is None:
            raise RuntimeError("WAAL.query() called before train()")
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        uncertainty, critic, critic_init = self.trainer.score_components(
            self.net, self.fit_result, unlabeled_data
        )
        scores = (uncertainty - self.selection * critic).numpy()
        picked = unlabeled_idxs[rank_ascending(scores, self.rng)[:n]]
        self._record(critic.numpy(), critic_init.numpy(), uncertainty.numpy(), picked, unlabeled_idxs, n)
        return picked

    def _record(
        self,
        critic: np.ndarray,
        critic_init: np.ndarray,
        uncertainty: np.ndarray,
        picked: np.ndarray,
        unlabeled_idxs: np.ndarray,
        n: int,
    ) -> None:
        """Reporting only: critic statistics and the spread of both score terms."""
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        _, labeled_critic, _ = self.trainer.score_components(self.net, self.fit_result, labeled_data)
        labeled_critic = labeled_critic.numpy()
        self._diag.add(
            waal_critic_labeled_mean=labeled_critic.mean(),
            waal_critic_labeled_std=labeled_critic.std(),
            waal_critic_unlabeled_mean=critic.mean(),
            waal_critic_unlabeled_std=critic.std(),
            waal_critic_init_corr=pearson(critic, critic_init),
            waal_uncertainty_std=uncertainty.std(),
            waal_critic_term_std=(self.selection * critic).std(),
            pool_order_fraction=pool_order_fraction(picked, unlabeled_idxs, n),
        )

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()
