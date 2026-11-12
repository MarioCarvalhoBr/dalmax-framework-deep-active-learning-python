"""Query strategies: the 12 legacy baseline classes (`RandomSampling`, ...,
`AdversarialDeepFool`, moved here unchanged in refactor Phase 4 from
`core/query_strategies/`), the 8 DeepAL+ ports of ADR 0015 (`VarRatio`, `MeanSTD`,
`KCenterGreedyPCA`, `BadgeSampling`, `CEALSampling`, `LossPredictionLoss`, `VAAL`,
`WAAL`), the shared `Strategy` base class, and the generic
`RepresentationStrategy` (the single class backing all four legacy
`*Kmeans*Sampling` CLI names plus the new generic `RepresentationStrategy`
choice).

See `dalmax.query_strategies.registry` (`STRATEGY_REGISTRY` / `build_strategy`)
for how a `--strategy_name` string resolves to one of these classes.

Importing this package must never have side effects (no dataset/model
construction, no file I/O) — see `.claude/rules/code-quality.md`.
"""

from __future__ import annotations

from dalmax.query_strategies.adversarial_bim import AdversarialBIM
from dalmax.query_strategies.adversarial_deepfool import AdversarialDeepFool
from dalmax.query_strategies.badge_sampling import BadgeSampling
from dalmax.query_strategies.base import Strategy
from dalmax.query_strategies.bayesian_active_learning_disagreement_dropout import BALDDropout
from dalmax.query_strategies.ceal import CEALSampling
from dalmax.query_strategies.entropy_sampling import EntropySampling
from dalmax.query_strategies.entropy_sampling_dropout import EntropySamplingDropout
from dalmax.query_strategies.kcenter_greedy import KCenterGreedy
from dalmax.query_strategies.kcenter_greedy_pca import KCenterGreedyPCA
from dalmax.query_strategies.kmeans_sampling import KMeansSampling
from dalmax.query_strategies.least_confidence import LeastConfidence
from dalmax.query_strategies.least_confidence_dropout import LeastConfidenceDropout
from dalmax.query_strategies.loss_prediction import LossPredictionLoss
from dalmax.query_strategies.margin_sampling import MarginSampling
from dalmax.query_strategies.margin_sampling_dropout import MarginSamplingDropout
from dalmax.query_strategies.mean_std import MeanSTD
from dalmax.query_strategies.random_sampling import RandomSampling
from dalmax.query_strategies.registry import (
    DEEPALPLUS_STRATEGY_REGISTRY,
    GENERIC_REPRESENTATION_NAME,
    LEGACY_STRATEGY_REGISTRY,
    REPRESENTATION_PRESETS,
    STRATEGY_REGISTRY,
    build_strategy,
)
from dalmax.query_strategies.representation import RepresentationStrategy
from dalmax.query_strategies.vaal import VAAL
from dalmax.query_strategies.var_ratio import VarRatio
from dalmax.query_strategies.waal import WAAL

__all__ = [
    "DEEPALPLUS_STRATEGY_REGISTRY",
    "GENERIC_REPRESENTATION_NAME",
    "LEGACY_STRATEGY_REGISTRY",
    "REPRESENTATION_PRESETS",
    "STRATEGY_REGISTRY",
    "AdversarialBIM",
    "AdversarialDeepFool",
    "BALDDropout",
    "BadgeSampling",
    "CEALSampling",
    "EntropySampling",
    "EntropySamplingDropout",
    "KCenterGreedy",
    "KCenterGreedyPCA",
    "KMeansSampling",
    "LeastConfidence",
    "LeastConfidenceDropout",
    "LossPredictionLoss",
    "MarginSampling",
    "MarginSamplingDropout",
    "MeanSTD",
    "RandomSampling",
    "RepresentationStrategy",
    "Strategy",
    "VAAL",
    "VarRatio",
    "WAAL",
    "build_strategy",
]
