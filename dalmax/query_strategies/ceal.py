"""CEALSampling: cost-effective active learning (Wang et al., IEEE TCSVT 2017,
arXiv 1701.03551), port of DeepAL+ `query_strategies/ceal.py` (ADR 0015).

Every query round `t` (1-based):

1. the base strategy (default `EntropySampling`; one of
   `dalmax.config.schema.CEAL_BASE_STRATEGIES`, built through the strategy registry)
   picks the `n` samples sent to the oracle;
2. among the remaining unlabeled samples, those whose predictive entropy under the
   current model is below the round's threshold `delta_t` (clamped at 0, see
   `ceal_threshold`) get their predicted class as pseudo-label;
3. the NEXT training (the one right after this query) uses the labeled set plus
   these pseudo-labelled samples. Pseudo-labels are never written into the labeled
   set (`dataset.labeled_idxs`); they are recomputed at every round and discarded
   after that training. Round 0 trains on the labeled set only.

Training cost grows with the pseudo-labelled set: with the same `n_epoch`, a CEAL
round takes `ceil((|L| + |P|) / batch)` SGD steps per epoch instead of
`ceil(|L| / batch)`, so it also takes more optimizer steps than the plain strategies
(reported per training as `ceal_train_size`).

Reporting only (`diagnostics()`, never used by selection or training): per query
round the threshold, the number of pseudo-labelled samples and their accuracy against
the true labels (available in this simulation), and the KI-45 pool-order fraction of
the query; per training the size of the training set.

Deviations from DeepAL+, deliberate:

- Defaults `delta_0 = 0.05`, `dr = 0.0033` (the paper's CACD setting). DeepAL+'s
  `5e-5` / `0.033e-5` pseudo-label nothing on DANINHAS (minimum entropy over 1,000
  pool images of a trained nq100 model: 5.4e-5), which turns CEAL into a copy of its
  base strategy.
- The schedule is a params key (`delta_schedule`, default `"linear"`:
  `delta_0 - dr * t`); `"cumulative"` (`delta_0 - dr * t (t + 1) / 2`) is DeepAL+'s
  code and the literal reading of the paper's Eq. 9. Clamped at 0 (no sample has a
  negative entropy, so a negative threshold just means "no pseudo-labels").
- Entropy uses `torch.xlogy` (0 log 0 = 0; DeepAL+'s NaN was silently excluded), and
  the probabilities are sized by `n_classes` from the params (one eval-mode pass via
  `DeepLearning.predict_logits_and_embeddings`), not by the classes present among the
  unlabeled samples' true labels.
- The base strategy is chosen by name through the registry and restricted to the
  plain uncertainty strategies (DeepAL+'s name list is misaligned with its strategy
  list from index 11 on, e.g. `CEALSampling+BadgeSampling` runs MeanSTD there).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import Strategy
from .diagnostics import RoundDiagnostics, pool_order_fraction


def ceal_threshold(
    delta: float, delta_decay: float, round_idx: int, schedule: str = "linear"
) -> float:
    """Entropy threshold of query round `round_idx` (1-based), clamped at 0.

    `"linear"`: `delta - delta_decay * t`; `"cumulative"`: `delta - delta_decay * t (t + 1) / 2`.
    """
    if schedule == "linear":
        steps = round_idx
    elif schedule == "cumulative":
        steps = round_idx * (round_idx + 1) / 2
    else:
        raise ValueError(f"unknown CEAL delta schedule {schedule!r}")
    return max(0.0, delta - delta_decay * steps)


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
        delta_schedule: str = "linear",
    ) -> None:
        super().__init__(dataset, net, logger)
        self.base_strategy = base_strategy
        self.delta = delta
        self.delta_decay = delta_decay
        self.delta_schedule = delta_schedule
        self._round = 0
        self._pseudo_idxs = np.zeros(0, dtype=np.int64)
        self._pseudo_labels = np.zeros(0, dtype=np.int64)
        self._diag = RoundDiagnostics()

    def _probabilities(self, data) -> torch.Tensor:
        logits, _ = self.net.predict_logits_and_embeddings(data)
        return torch.softmax(logits.to(torch.float64), dim=1)

    def query(self, n: int) -> np.ndarray:
        self._round += 1
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        q_idxs = np.asarray(self.base_strategy.query(n))

        delta = ceal_threshold(self.delta, self.delta_decay, self._round, self.delta_schedule)
        remaining = ~np.isin(unlabeled_idxs, q_idxs)
        probs = self._probabilities(unlabeled_data)
        confident = (predictive_entropy(probs) < delta).numpy() & remaining
        self._pseudo_idxs = unlabeled_idxs[confident]
        self._pseudo_labels = probs.argmax(dim=1).numpy()[confident].astype(np.int64)
        self._record(delta, q_idxs, unlabeled_idxs, n)
        return q_idxs

    def _record(
        self, delta: float, q_idxs: np.ndarray, unlabeled_idxs: np.ndarray, n: int
    ) -> None:
        """Log and keep the round's pseudo-label count and accuracy (reporting only)."""
        n_pseudo = len(self._pseudo_idxs)
        true_labels = np.asarray(self.dataset.Y_train)[self._pseudo_idxs]
        accuracy = float((true_labels == self._pseudo_labels).mean()) if n_pseudo else None
        self._diag.add(
            ceal_round=self._round,
            ceal_delta=delta,
            ceal_n_pseudo_labels=n_pseudo,
            ceal_pseudo_label_accuracy=accuracy,
            pool_order_fraction=pool_order_fraction(q_idxs, unlabeled_idxs, n),
        )
        self.logger.warning(
            f"CEAL round {self._round}: delta={delta:.6g}, {n_pseudo} pseudo-labelled samples "
            f"for the next training, pseudo-label accuracy={accuracy}"
        )

    def train(self) -> None:
        self.logger.warning("Training the DAL strategy (CEAL: labeled + pseudo-labelled)")
        labeled_idxs = np.arange(self.dataset.n_pool)[self.dataset.labeled_idxs]
        keep = ~self.dataset.labeled_idxs[self._pseudo_idxs]
        pseudo, pseudo_labels = self._pseudo_idxs[keep], self._pseudo_labels[keep]
        self._diag.add(ceal_train_size=len(labeled_idxs) + len(pseudo))
        if len(pseudo) == 0:
            super().train()
            return
        x = np.concatenate([self.dataset.X_train[labeled_idxs], self.dataset.X_train[pseudo]])
        y_labeled = np.asarray(self.dataset.Y_train)[labeled_idxs]
        y = np.concatenate([y_labeled, pseudo_labels.astype(y_labeled.dtype)])
        self.net.train(self.dataset.handler(x, y))

    def diagnostics(self) -> dict[str, list[Any]]:
        return self._diag.as_dict()
