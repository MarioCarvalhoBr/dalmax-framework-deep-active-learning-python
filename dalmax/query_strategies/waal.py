"""WAAL: Wasserstein adversarial active learning (Shui et al., AISTATS 2020), port of
DeepAL+ `query_strategies/waal.py` (ADR 0015), and its training-budget control
`WAALRandomQuery` (ADR 0015 amendment of 2026-11-12).

Training (every round, round 0 included) is the WAAL min-max training of the injected
`dalmax.models.waal.WAALTrainer` on the labeled set and the unlabeled pool, instead of
plain cross-entropy training; the evaluated and saved model is the ordinary task model
(feature extractor + classifier), so `saved_model.pth` and the inference tools work as
for every strategy. The critic is kept by the strategy for the query only.

`WAAL` query: `0.5 * ||log p||_2 + 0.5 * ||log p||_1 - selection * critic(z)` over the
unlabeled pool, the `n` LOWEST first (DeepAL+ and the original); exact ties broken at
random with the generator injected by `build_strategy`
(`derive_seed(config.seed, WAAL_TIES_SEED_TAG)`, `ranking.rank_ascending`), so the
query never falls back to pool order (KI-45).

`WAALRandomQuery` query: a uniform random draw of `n` unlabeled samples (without
replacement) from its own injected generator
(`derive_seed(config.seed, WAAL_RANDOM_QUERY_SEED_TAG)`). It exists because WAAL takes
about 280 SGD steps per round against 10 to 40 for every other strategy (ADR 0015
amendment, finding (b)): a gain of WAAL over `RandomSampling` could come from that
training, not from which images WAAL queries. The control keeps WAAL's training and
replaces only the query.

Shared between `WAAL` and `WAALRandomQuery` (`WAALTrainedStrategy`, one code path):
the training (`train()`: the same `WAALTrainer.fit`, so the same task-model
construction from the run's torch stream via `DeepLearning.build_model`, the same
critic built inside `isolated_torch_seed`, the same joint loader and optimizers), the
hyperparameters (both read the params JSON's `strategy_args.WAAL` block; the control
has no block of its own, so the two cannot drift), the trainer generator's tag
(`dalmax.models.waal.WAAL_SEED_TAG`), and the reporting-only critic evaluation at
query time (`score_components` on the pool, then on the labeled set, in that order).
That evaluation also advances the trainer generator (each call draws one loader seed),
so with the same seed and the same labeled set the two strategies hand the trainer an
identical generator state at every round: round 0 (training and metrics) is identical,
and later rounds differ only through the labeled set. Not shared: the query rule and
its generator (`waal_ties` vs `waal_random_query`), and the query diagnostics.

Reporting only (`diagnostics()`, ADR 0015): per training, the number of optimizer
steps (the joint loader iterates over the pool, so WAAL takes far more steps per round
than the plain strategies) and the mean Wasserstein estimate of the last epoch; per
query, the critic's mean/std on the labeled set and on the pool, the Pearson
correlation between the trained critic's and the initial critic's pool scores (close
to 1 = the critic barely moved), and the KI-45 pool-order fraction. `WAAL` adds the
std of the uncertainty term and of the `selection * critic` term; `WAALRandomQuery`
adds `query_rule = "uniform_random"` (the critic and uncertainty terms are not used by
its query).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dalmax.models.waal import WAALFitResult, WAALTrainer

from .base import Strategy
from .diagnostics import RoundDiagnostics, pearson, pool_order_fraction
from .ranking import rank_ascending

# `dalmax.seeding.derive_seed` tag of WAAL's query tie-break generator.
WAAL_TIES_SEED_TAG = "waal_ties"
# `dalmax.seeding.derive_seed` tag of the control's uniform random query generator.
WAAL_RANDOM_QUERY_SEED_TAG = "waal_random_query"
# Value of the control's `query_rule` diagnostic.
UNIFORM_RANDOM_QUERY_RULE = "uniform_random"


class WAALTrainedStrategy(Strategy):
    """WAAL min-max training plus the reporting-only critic evaluation; the query rule
    is the subclass's (`WAAL` or `WAALRandomQuery`). Not registered itself."""

    def __init__(self, dataset, net, logger, *, trainer: WAALTrainer) -> None:
        super().__init__(dataset, net, logger)
        self.trainer = trainer
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

    def _score_pool(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unlabeled ids and, per id, the uncertainty term, the trained critic's score
        and the initial critic's score (first trainer-generator draw of a query)."""
        if self.fit_result is None:
            raise RuntimeError(f"{type(self).__name__}.query() called before train()")
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        uncertainty, critic, critic_init = self.trainer.score_components(
            self.net, self.fit_result, unlabeled_data
        )
        return unlabeled_idxs, uncertainty.numpy(), critic.numpy(), critic_init.numpy()

    def _record_critic(
        self,
        critic: np.ndarray,
        critic_init: np.ndarray,
        picked: np.ndarray,
        unlabeled_idxs: np.ndarray,
        n: int,
        **extra: Any,
    ) -> None:
        """Reporting only: critic statistics (second trainer-generator draw of a query:
        the labeled set), the KI-45 pool-order fraction and the subclass's `extra`."""
        _labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        _, labeled_critic, _ = self.trainer.score_components(self.net, self.fit_result, labeled_data)
        labeled_critic = labeled_critic.numpy()
        self._diag.add(
            waal_critic_labeled_mean=labeled_critic.mean(),
            waal_critic_labeled_std=labeled_critic.std(),
            waal_critic_unlabeled_mean=critic.mean(),
            waal_critic_unlabeled_std=critic.std(),
            waal_critic_init_corr=pearson(critic, critic_init),
            **extra,
            pool_order_fraction=pool_order_fraction(picked, unlabeled_idxs, n),
        )

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()


class WAAL(WAALTrainedStrategy):
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
        super().__init__(dataset, net, logger, trainer=trainer)
        self.selection = selection
        self.rng = rng

    def query(self, n: int) -> np.ndarray:
        unlabeled_idxs, uncertainty, critic, critic_init = self._score_pool()
        scores = uncertainty - self.selection * critic
        picked = unlabeled_idxs[rank_ascending(scores, self.rng)[:n]]
        self._record_critic(
            critic,
            critic_init,
            picked,
            unlabeled_idxs,
            n,
            waal_uncertainty_std=uncertainty.std(),
            waal_critic_term_std=(self.selection * critic).std(),
        )
        return picked


class WAALRandomQuery(WAALTrainedStrategy):
    """Control of WAAL: WAAL's training unchanged, query = uniform random draw from the
    unlabeled pool (module docstring for exactly what is shared)."""

    def __init__(
        self,
        dataset,
        net,
        logger,
        *,
        trainer: WAALTrainer,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(dataset, net, logger, trainer=trainer)
        self.rng = rng

    def query(self, n: int) -> np.ndarray:
        # The pool is scored for the critic diagnostics only (and so that the trainer
        # generator advances exactly as in WAAL); the scores never reach the pick.
        unlabeled_idxs, _uncertainty, critic, critic_init = self._score_pool()
        if n > len(unlabeled_idxs):
            raise ValueError(f"cannot query {n} samples from {len(unlabeled_idxs)} unlabeled")
        picked = self.rng.choice(unlabeled_idxs, size=n, replace=False)
        self._record_critic(
            critic, critic_init, picked, unlabeled_idxs, n, query_rule=UNIFORM_RANDOM_QUERY_RULE
        )
        return picked
