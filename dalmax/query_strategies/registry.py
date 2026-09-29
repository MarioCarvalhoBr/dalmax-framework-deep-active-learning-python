"""`STRATEGY_REGISTRY` / `build_strategy`: the single place that maps a
`--strategy_name` string to a constructed `Strategy` instance.

Replaces the deleted `utils/orchestrator.py::get_strategy`'s 16-branch
`if/elif` chain (`.claude/rules/code-quality.md`, "registry pattern over
if/elif chains") for the `dalmax`-routed CLI (`dalmax.cli`) — that module was
removed as dead code in refactor Phase 4
(`.specs/architecture/refactor-plan.md`).

Three kinds of names are registered (the third, `FullSupervised`, is the
paper-1 upper bound -- not active learning, see `full_supervised.py`):

- The 12 legacy strategies (`RandomSampling` ... `AdversarialDeepFool`) are
  constructed exactly as the historical `demo.py` always has:
  `LegacyClass(dataset, net, logger)`, no `dalmax` involvement.
- The 4 legacy `*Kmeans*Sampling` names (`SSRAEKmeansSampling`,
  `VCTexKmeansSampling`, `SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`)
  become **presets**: fixed `(extractor, selection method)` pairs that
  override `config.dataset.embedding`/`selection` and construct a
  `dalmax.query_strategies.representation.RepresentationStrategy`. This is
  the fixed-behavior compatibility path — see
  `.specs/architecture/target-architecture.md` §6 for the mapping table.
  Presets also pin their embedding provider's `q` to the legacy value in
  `REPRESENTATION_PRESET_Q` (`ssrae` → `13`, `vctex` → `(5, 17)`) and
  **ignore** `config.dataset.embedding.q` entirely — a preset name means
  "reproduce the exact legacy behavior", so it must not silently pick up
  whatever `q` a params JSON's generic `"embedding"` block happens to set.
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

from dalmax.config.schema import FULL_SUPERVISED_STRATEGY, ConfigError, ExperimentConfig
from dalmax.embeddings.registry import get_embedding_provider
from dalmax.query_strategies.adversarial_bim import AdversarialBIM
from dalmax.query_strategies.adversarial_deepfool import AdversarialDeepFool
from dalmax.query_strategies.base import Strategy
from dalmax.query_strategies.bayesian_active_learning_disagreement_dropout import BALDDropout
from dalmax.query_strategies.entropy_sampling import EntropySampling
from dalmax.query_strategies.entropy_sampling_dropout import EntropySamplingDropout
from dalmax.query_strategies.full_supervised import FullSupervised
from dalmax.query_strategies.kcenter_greedy import KCenterGreedy
from dalmax.query_strategies.kmeans_sampling import KMeansSampling
from dalmax.query_strategies.least_confidence import LeastConfidence
from dalmax.query_strategies.least_confidence_dropout import LeastConfidenceDropout
from dalmax.query_strategies.margin_sampling import MarginSampling
from dalmax.query_strategies.margin_sampling_dropout import MarginSamplingDropout
from dalmax.query_strategies.random_sampling import RandomSampling
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

# Not active learning: the paper-1 upper bound (train once on the whole pool).
NON_ACTIVE_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    FULL_SUPERVISED_STRATEGY: FullSupervised,
}

# name -> (embedding extractor, selection method)
REPRESENTATION_PRESETS: dict[str, tuple[str, str]] = {
    "SSRAEKmeansSampling": ("ssrae", "flat_closest"),
    "VCTexKmeansSampling": ("vctex", "flat_closest"),
    "SSRAEKmeansHCSampling": ("ssrae", "hierarchical"),
    "VCTexKmeansHCSampling": ("vctex", "hierarchical"),
}

GENERIC_REPRESENTATION_NAME = "RepresentationStrategy"

# Legacy `Q` per extractor for the 4 compatibility presets in
# `REPRESENTATION_PRESETS`, above. A preset name (e.g. `VCTexKmeansSampling`)
# means "reproduce the exact legacy SSRAE/VCTex behavior" — so its `q` is
# always this pinned legacy value, **never** `config.dataset.embedding.q`,
# even if the params JSON's "embedding" block sets a different `q` for the
# generic `RepresentationStrategy` path. This is deliberate and documented:
# using `config.dataset.embedding.q` here was the HIGH-severity bug where
# `VCTexKmeansSampling`/`VCTexKmeansHCSampling` silently built
# `VCTexProvider(q=13)` (SSRAE's default) instead of the legacy `Q=(5, 17)`
# whenever the params JSON had no "embedding" key (see
# `.claude/rules/reproducibility.md`). `dalmax.config.loader` also now
# defaults `config.dataset.embedding.q` per-extractor for the *generic*
# `RepresentationStrategy` path, but presets intentionally bypass that value
# entirely rather than depend on it staying correct.
REPRESENTATION_PRESET_Q: dict[str, Any] = {"ssrae": 13, "vctex": (5, 17)}

STRATEGY_REGISTRY: dict[str, Any] = {
    **LEGACY_STRATEGY_REGISTRY,
    **NON_ACTIVE_STRATEGY_REGISTRY,
    **dict.fromkeys(REPRESENTATION_PRESETS, RepresentationStrategy),
    GENERIC_REPRESENTATION_NAME: RepresentationStrategy,
}


def _resolve_q(extractor: str, q: Any) -> Any:
    # Defensive fallback for the generic `RepresentationStrategy` path only:
    # `dalmax.config.loader` already defaults `config.dataset.embedding.q`
    # per-extractor, but a hand-built `ExperimentConfig` (e.g. in a test, or
    # a future caller that skips the loader) could still pass `q=None` for
    # an extractor that needs one. Presets never reach this fallback — they
    # pass their pinned `REPRESENTATION_PRESET_Q` value directly, see
    # `build_strategy`.
    if q is None and extractor in REPRESENTATION_PRESET_Q:
        return REPRESENTATION_PRESET_Q[extractor]
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

    if name in NON_ACTIVE_STRATEGY_REGISTRY:
        return NON_ACTIVE_STRATEGY_REGISTRY[name](dataset, net, logger)

    if name in REPRESENTATION_PRESETS:
        extractor, selection_method = REPRESENTATION_PRESETS[name]
        hierarchy = config.dataset.selection.hierarchy
        return _build_representation_strategy(
            dataset,
            net,
            config,
            logger,
            extractor=extractor,
            # Presets pin the legacy Q for `extractor` and deliberately
            # ignore `config.dataset.embedding.q` — see
            # `REPRESENTATION_PRESET_Q`'s docstring, above, for why.
            q=REPRESENTATION_PRESET_Q[extractor],
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
