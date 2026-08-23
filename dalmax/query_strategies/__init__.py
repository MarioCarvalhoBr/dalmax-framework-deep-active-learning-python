"""Query strategies that plug into the legacy `core.query_strategies.strategy.Strategy`
interface (`train`/`predict`/`update`/`query`), so `dalmax` strategies and the 12
untouched legacy strategies (`RandomSampling`, ..., `AdversarialDeepFool`) are
interchangeable from `demo.py`/`dalmax.cli`'s point of view.

See `dalmax.query_strategies.representation.RepresentationStrategy` (the single
class backing all four legacy `*Kmeans*Sampling` names plus the new generic
`RepresentationStrategy` CLI choice) and `dalmax.query_strategies.registry`
(`STRATEGY_REGISTRY` / `build_strategy`).

Importing this package must never have side effects (no dataset/model
construction, no file I/O) — see `.claude/rules/code-quality.md`.
"""

from __future__ import annotations
