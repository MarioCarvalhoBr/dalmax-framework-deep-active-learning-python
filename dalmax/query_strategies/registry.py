"""`STRATEGY_REGISTRY` / `build_strategy`: the single place that maps a
`--strategy_name` string to a constructed `Strategy` instance.

Replaces `utils/orchestrator.py::get_strategy`'s 16-branch `if/elif` chain
(`.claude/rules/code-quality.md`, "registry pattern over if/elif chains") for
the `dalmax`-routed CLI (`dalmax.cli`); `utils/orchestrator.py` itself is
left untouched and keeps working for any legacy caller.

Two kinds of names are registered:

- The 12 legacy strategies (`RandomSampling` ... `AdversarialDeepFool`) are
  constructed exactly as `demo.py` always has:
  `LegacyClass(dataset, net, logger)`, no `dalmax` involvement.
- The 4 legacy `*Kmeans*Sampling` names (`SSRAEKmeansSampling`,
  `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`)
  become **presets**: fixed `(extractor, selection method)` pairs that
  override `config.dataset.embedding`/`selection` and construct a
  `dalmax.query_strategies.representation.RepresentationStrategy`. This is
  the fixed-behavior compatibility path — see
  `.specs/architecture/target-architecture.md` §6 for the mapping table.
- The generic `"RepresentationStrategy"` name uses `config.dataset.embedding`/
  `selection` verbatim, with no preset override — this is what Phase 3's
  ablations (`.specs/experiments/ablation-study.md`) drive via the params
  JSON's `"embedding"`/`"selection"` keys (e.g. `resnet_imagenet` +
  `hierarchical`, `ssrae` + `flat_proportional`, an SSRAE `spatial`/`spectral`
  variant).

`SelectionStrategy` construction always derives its RNG seed from
`config.seed` via `dalmax.seeding.derive_seed(config.seed, "selection")` —
never a hardcoded literal (the `KMeans(random_state=3)` violation this
refactor fixes, see `.claude/rules/reproducibility.md`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.query_strategies import (
    AdversarialBIM,
    AdversarialDeepFool,
    BALDDropout,
    EntropySampling,
    EntropySamplingDropout,
    KCenterGreedy,
    KMeansSampling,
    LeastConfidence,
    LeastConfidenceDropout,
    MarginSampling,
    MarginSamplingDropout,
    RandomSampling,
)
from core.query_strategies.strategy import Strategy
from dalmax.config.schema import ConfigError, ExperimentConfig
from dalmax.embeddings.registry import get_embedding_provider
from dalmax.query_strategies.representation import RepresentationStrategy
from dalmax.seeding import derive_seed
from dalmax.selection.registry import get_selection_strategy

LEGACY_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "RandomSampling": RandomSampling,
    "LeastConfidence": LeastConfidence,
    "MarginSampling": MarginSampling,
    "EntropySampling": EntropySampling,
    "LeastConfidenceDropout": LeastConfidenceDropout,
    "MarginSamplingDropout": MarginSamplingDropout,
    "EntropySamplingDropout": EntropySamplingDropout,
    "KMeansSampling": KMeansSampling,
    "KCenterGreedy": KCenterGreedy,
    "BALDDropout": BALDDropout,
    "AdversarialBIM": AdversarialBIM,
    "AdversarialDeepFool": AdversarialDeepFool,
}

# name -> (embedding extractor, selection method)
REPRESENTATION_PRESETS: dict[str, tuple[str, str]] = {
    "SSRAEKmeansSampling": ("ssrae", "flat_closest"),
    "VCTexKmeansSampling": ("vctex", "flat_closest"),
    "SSRAEKmeansHCSampling": ("ssrae", "hierarchical"),
    "VCTexKmeansHCSampling": ("vctex", "hierarchical"),
}

GENERIC_REPRESENTATION_NAME = "RepresentationStrategy"

# Fallback `q` for a preset/generic build when `config.dataset.embedding.q`
# is `None` (mirrors the historical literals `Q=13`/`Q=(5, 17)` this refactor
# is threading through config instead of hardcoding at the call site).
_DEFAULT_Q: dict[str, Any] = {"ssrae": 13, "vctex": (5, 17)}

STRATEGY_REGISTRY: dict[str, Any] = {
    **LEGACY_STRATEGY_REGISTRY,
    **dict.fromkeys(REPRESENTATION_PRESETS, RepresentationStrategy),
    GENERIC_REPRESENTATION_NAME: RepresentationStrategy,
}


def _resolve_q(extractor: str, q: Any) -> Any:
    if q is None and extractor in _DEFAULT_Q:
        return _DEFAULT_Q[extractor]
    return q


def _build_representation_strategy(
    dataset,
    net,
    config: ExperimentConfig,
    logger,
    *,
    extractor: str,
    q: Any,
    selection_method: str,
    hierarchy: Any,
) -> RepresentationStrategy:
    provider_cls = get_embedding_provider(extractor)
    provider = provider_cls(q=_resolve_q(extractor, q), device=config.device)

    selection_cls = get_selection_strategy(selection_method)
    if selection_method == "hierarchical":
        if hierarchy is None:
            raise ConfigError(
                f"selection method 'hierarchical' requires config.dataset.selection.hierarchy "
                f"to be set (strategy_name={config.strategy_name!r})"
            )
        selection = selection_cls(hierarchy=hierarchy, device=config.device)
    else:
        selection = selection_cls()

    selection_rng = np.random.default_rng(derive_seed(config.seed, "selection"))
    return RepresentationStrategy(
        dataset,
        net,
        config,
        logger,
        embedding_provider=provider,
        selection=selection,
        rng=selection_rng,
    )


def build_strategy(
    name: str,
    dataset,
    net,
    config: ExperimentConfig,
    logger,
    rng: np.random.Generator,
) -> Strategy:
    """Construct the `Strategy` instance for `--strategy_name` `name`.

    `rng` is accepted for every name (uniform signature across the
    registry) but only consumed indirectly, for representation strategies,
    via `dalmax.seeding.derive_seed(config.seed, "selection")` — legacy
    strategies never touch it, matching their unchanged
    `LegacyClass(dataset, net, logger)` construction.

    Raises
    ------
    KeyError
        If `name` is not a registered strategy name.
    ConfigError
        If `name` resolves to the hierarchical selection method but
        `config.dataset.selection.hierarchy` (or, for a preset, an
        equivalent) is missing.
    """
    if name in LEGACY_STRATEGY_REGISTRY:
        return LEGACY_STRATEGY_REGISTRY[name](dataset, net, logger)

    if name in REPRESENTATION_PRESETS:
        extractor, selection_method = REPRESENTATION_PRESETS[name]
        hierarchy = config.dataset.selection.hierarchy
        return _build_representation_strategy(
            dataset,
            net,
            config,
            logger,
            extractor=extractor,
            q=config.dataset.embedding.q,
            selection_method=selection_method,
            hierarchy=hierarchy,
        )

    if name == GENERIC_REPRESENTATION_NAME:
        return _build_representation_strategy(
            dataset,
            net,
            config,
            logger,
            extractor=config.dataset.embedding.extractor,
            q=config.dataset.embedding.q,
            selection_method=config.dataset.selection.method,
            hierarchy=config.dataset.selection.hierarchy,
        )

    valid = ", ".join(sorted(STRATEGY_REGISTRY))
    raise KeyError(f"Unknown strategy {name!r}. Valid names: {valid}")
