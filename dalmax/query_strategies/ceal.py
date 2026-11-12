"""CEALSampling: cost-effective active learning (Wang et al., IEEE TCSVT 2017),
port of DeepAL+ `query_strategies/ceal.py` (ADR 0015).

Every query round `t` (1-based):

1. the base strategy (default `EntropySampling`; one of
   `dalmax.config.schema.CEAL_BASE_STRATEGIES`, built through the strategy registry)
   picks the `n` samples sent to the oracle;
2. among the remaining unlabeled samples, those whose predictive entropy under the
   current model is below `delta_t = delta_0 - delta_decay * t` get their predicted
   class as pseudo-label;
3. the NEXT training (the one right after this query) uses the labeled set plus
   these pseudo-labelled samples. Pseudo-labels are never written into the labeled
   set (`dataset.labeled_idxs`); they are recomputed at every round and discarded
   after that training. Round 0 trains on the labeled set only.

The number of pseudo-labelled samples per round and their accuracy against the
true labels (available in this simulation) are logged and reported in
`results.json` (`strategy_diagnostics`, via `diagnostics()`); they are reporting
only and never feed selection or training.

Deviations from DeepAL+, deliberate:

- The threshold decays linearly, `delta_0 - dr * t` (the paper's schedule, and the
  "decreases by dr per round" of DeepAL+'s own description). DeepAL+'s code does
  `self.delta -= dr * t` at every round, i.e. `delta_0 - dr * t (t + 1) / 2`.
- Entropy uses `torch.xlogy`, so a probability of exactly 0 contributes 0 instead of
  NaN (with NaN the comparison `< delta` was silently False in DeepAL+).
- The base strategy is chosen by name through the registry and restricted to the
  plain uncertainty strategies (DeepAL+'s name list is misaligned with its strategy
  list from index 11 on, e.g. `CEALSampling+BadgeSampling` runs MeanSTD there).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import Strategy


def ceal_threshold(delta: float, delta_decay: float, round_idx: int) -> float:
    """Entropy threshold of query round `round_idx` (1-based): `delta - delta_decay * t`."""
    return delta - delta_decay * round_idx


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """`-sum_c p_c log p_c` per row (nats), 0 * log 0 = 0."""
    return -torch.xlogy(probs, probs).sum(dim=1)


class CEALSampling(Strategy):
    """Base-strategy query plus high-confidence pseudo-labels for the next training."""

    def __init__(
        self,
        dataset,
        net,
        logger,
        *,
        base_strategy: Strategy,
        delta: float,
        delta_decay: float,
    ) -> None:
        super().__init__(dataset, net, logger)
        self.base_strategy = base_strategy
        self.delta = delta
        self.delta_decay = delta_decay
        self._round = 0
        self._pseudo_idxs = np.zeros(0, dtype=np.int64)
        self._pseudo_labels = np.zeros(0, dtype=np.int64)
        self._history: dict[str, list[Any]] = {
            "ceal_round": [],
            "ceal_delta": [],
            "ceal_n_pseudo_labels": [],
            "ceal_pseudo_label_accuracy": [],
        }

    def query(self, n: int) -> np.ndarray:
        self._round += 1
        q_idxs = np.asarray(self.base_strategy.query(n))

        delta = ceal_threshold(self.delta, self.delta_decay, self._round)
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        remaining = ~np.isin(unlabeled_idxs, q_idxs)
        probs = self.predict_prob(unlabeled_data)
        confident = (predictive_entropy(probs) < delta).numpy() & remaining
        self._pseudo_idxs = unlabeled_idxs[confident]
        self._pseudo_labels = probs.argmax(dim=1).numpy()[confident].astype(np.int64)
        self._record(delta)
        return q_idxs

    def _record(self, delta: float) -> None:
        """Log and keep the round's pseudo-label count and accuracy (reporting only)."""
        n_pseudo = len(self._pseudo_idxs)
        true_labels = np.asarray(self.dataset.Y_train)[self._pseudo_idxs]
        accuracy = float((true_labels == self._pseudo_labels).mean()) if n_pseudo else None
        self._history["ceal_round"].append(self._round)
        self._history["ceal_delta"].append(delta)
        self._history["ceal_n_pseudo_labels"].append(n_pseudo)
        self._history["ceal_pseudo_label_accuracy"].append(accuracy)
        self.logger.warning(
            f"CEAL round {self._round}: delta={delta:.6g}, {n_pseudo} pseudo-labelled samples "
            f"for the next training, pseudo-label accuracy={accuracy}"
        )

    def train(self) -> None:
        self.logger.warning("Training the DAL strategy (CEAL: labeled + pseudo-labelled)")
        labeled_idxs = np.arange(self.dataset.n_pool)[self.dataset.labeled_idxs]
        pseudo = self._pseudo_idxs[~self.dataset.labeled_idxs[self._pseudo_idxs]]
        pseudo_labels = self._pseudo_labels[~self.dataset.labeled_idxs[self._pseudo_idxs]]
        if len(pseudo) == 0:
            super().train()
            return
        x = np.concatenate([self.dataset.X_train[labeled_idxs], self.dataset.X_train[pseudo]])
        y_labeled = np.asarray(self.dataset.Y_train)[labeled_idxs]
        y = np.concatenate([y_labeled, pseudo_labels.astype(y_labeled.dtype)])
        self.net.train(self.dataset.handler(x, y))

    def diagnostics(self) -> dict[str, list[Any]]:
        return {key: list(values) for key, values in self._history.items()}
