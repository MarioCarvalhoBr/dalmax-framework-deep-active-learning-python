"""Score rankings with a seeded random tie-break, shared by every strategy whose
scores take few discrete values.

`rank_ascending` is the rule introduced for `AdversarialBIM` (KI-45, ADR 0014,
historically `adversarial_base.rank_by_distance`, still exported there under that
name); `rank_descending` is the same rule for "highest score first" strategies such
as `VarRatio` (ADR 0015), whose scores take at most `n_drop + 1` values.

The generator is always injected by the caller (created once per run by
`dalmax.query_strategies.registry.build_strategy` from
`dalmax.seeding.derive_seed(config.seed, <tag>)`) and advances by one
`permutation` call per ranking.
"""

from __future__ import annotations

import numpy as np


def rank_ascending(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of `scores` in ascending order, exact ties in a uniformly random order.

    Primary key `scores`, secondary key a random permutation drawn from `rng` (one
    `permutation(len(scores))` call per ranking). `inf` ranks after every finite
    value; without ties the result equals `scores.argsort(kind="stable")`.
    """
    scores = np.asarray(scores)
    return np.lexsort((rng.permutation(len(scores)), scores))


def rank_descending(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of `scores` in descending order, exact ties in a uniformly random order.

    Same contract as `rank_ascending` on `-scores` (one `permutation` draw per call);
    without ties the result equals `(-scores).argsort(kind="stable")`.
    """
    return rank_ascending(-np.asarray(scores, dtype=np.float64), rng)
