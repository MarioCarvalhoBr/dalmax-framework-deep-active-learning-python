"""`FullSupervised`: the paper-1 upper bound -- NOT an active-learning method.

Trains the network once on the ENTIRE pool (every training image labeled at
initialization) and evaluates on the fixed test set: the "full training
session using the entire pool of images" of the paper-1 protocol
(`.specs/experiments/experimental-protocol.md`). There is deliberately no
validation split -- the project's protocol is pool/test only, and this
strategy does not invent one.

Contract (enforced elsewhere, see `dalmax.experiment.runner`):

- `ExperimentConfig` rejects `n_round != 0` for this strategy
  (`dalmax.config.schema`), and the runner overrides `n_init_labeled` with
  the real pool size (logging a warning if the user passed another value) so
  the results directory reads `NIL_<pool>` and `run_metadata.json` records
  the truth;
- `n_query` plays no role for this strategy (nothing is ever queried); the
  campaign still passes the paper-1 primary budget only so the results
  directory keeps the uniform `NQ_<n>_...` layout;
- `query()` must never be called -- it raises.
"""

from __future__ import annotations

from dalmax.query_strategies.base import Strategy


class FullSupervised(Strategy):
    """Upper bound: all pool labels, one training round, no query."""

    def query(self, n: int):  # `n` is intentionally ignored: see the module docstring
        raise RuntimeError(
            "FullSupervised is the full-pool upper bound, not an active-learning strategy: "
            "query() must never be called (n_round must be 0)."
        )
