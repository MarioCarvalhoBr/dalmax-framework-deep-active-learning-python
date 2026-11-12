"""`STRATEGY_REGISTRY` / `build_strategy`: the single place that maps a
`--strategy_name` string to a constructed `Strategy` instance.

Replaces the deleted `utils/orchestrator.py::get_strategy`'s 16-branch
`if/elif` chain (`.claude/rules/code-quality.md`, "registry pattern over
if/elif chains") for the `dalmax`-routed CLI (`dalmax.cli`) — that module was
removed as dead code in refactor Phase 4
(`.specs/architecture/refactor-plan.md`).

Four kinds of names are registered (the third, `FullSupervised`, is the
paper-1 upper bound -- not active learning, see `full_supervised.py`; the fourth,
the DeepAL+ ports of ADR 0015, is listed at the end):

- The 12 legacy strategies (`RandomSampling` ... `AdversarialDeepFool`) are
  constructed as the historical `demo.py` always has,
  `LegacyClass(dataset, net, logger)`, plus the extra keyword arguments listed
  in `LEGACY_STRATEGY_KWARGS` (today only `AdversarialBIM`, which receives its
  tie-break generator `rng`, KI-45, ADR 0014; `AdversarialDeepFool` keeps the
  pool-order tie-break of its executed runs, see `adversarial_base`).
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

- The 8 DeepAL+ ports (`VarRatio`, `MeanSTD`, `KCenterGreedyPCA`,
  `BadgeSampling`, `CEALSampling`, `LossPredictionLoss`, `VAAL`, `WAAL`; ADR 0015)
  are built as `Class(dataset, net, logger, **kwargs)` with the keyword arguments
  of `DEEPALPLUS_STRATEGY_KWARGS[name](config, dataset, net, logger)`: their
  seeded generators (one per run, `derive_seed(config.seed, <tag>)`), their
  `config.dataset.strategy_args` block, and, for `CEALSampling`, the base strategy
  resolved through `LEGACY_STRATEGY_REGISTRY` (restricted to
  `dalmax.config.schema.CEAL_BASE_STRATEGIES`). None of them advances the run's
  main generator or the global torch stream at construction.

`SelectionStrategy` construction always derives its RNG seed from
`config.seed` via `dalmax.seeding.derive_seed(config.seed, "selection")` —
never a hardcoded literal (the `KMeans(random_state=3)` violation this
refactor fixes, see `.claude/rules/reproducibility.md`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from dalmax.config.schema import (
    FULL_SUPERVISED_STRATEGY,
    ConfigError,
    ExperimentConfig,
    TrainArgs,
)
from dalmax.embeddings.registry import get_embedding_provider
from dalmax.models.loss_prediction import LOSS_PREDICTION_SEED_TAG, LossPredictionTrainer
from dalmax.models.vaal import VAAL_SEED_TAG, VAALSampler
from dalmax.models.waal import WAAL_SEED_TAG, WAALTrainer
from dalmax.query_strategies.adversarial_base import TIE_BREAK_SEED_TAG
from dalmax.query_strategies.adversarial_bim import AdversarialBIM
from dalmax.query_strategies.adversarial_deepfool import AdversarialDeepFool
from dalmax.query_strategies.badge_sampling import BADGE_SEED_TAG, BadgeSampling
from dalmax.query_strategies.base import Strategy
from dalmax.query_strategies.bayesian_active_learning_disagreement_dropout import BALDDropout
from dalmax.query_strategies.ceal import CEALSampling
from dalmax.query_strategies.entropy_sampling import EntropySampling
from dalmax.query_strategies.entropy_sampling_dropout import EntropySamplingDropout
from dalmax.query_strategies.full_supervised import FullSupervised
from dalmax.query_strategies.kcenter_greedy import KCenterGreedy
from dalmax.query_strategies.kcenter_greedy_pca import KCENTER_PCA_SEED_TAG, KCenterGreedyPCA
from dalmax.query_strategies.kmeans_sampling import KMeansSampling
from dalmax.query_strategies.least_confidence import LeastConfidence
from dalmax.query_strategies.least_confidence_dropout import LeastConfidenceDropout
from dalmax.query_strategies.loss_prediction import LossPredictionLoss
from dalmax.query_strategies.margin_sampling import MarginSampling
from dalmax.query_strategies.margin_sampling_dropout import MarginSamplingDropout
from dalmax.query_strategies.mean_std import MeanSTD
from dalmax.query_strategies.random_sampling import RandomSampling
from dalmax.query_strategies.representation import RepresentationStrategy
from dalmax.query_strategies.vaal import VAAL
from dalmax.query_strategies.var_ratio import VAR_RATIO_SEED_TAG, VarRatio
from dalmax.query_strategies.waal import WAAL
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


def _tie_break_rng_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    """`AdversarialBIM`'s tie-break generator (KI-45): one per run, derived
    from the experiment seed, advancing across query rounds."""
    return {"rng": np.random.default_rng(derive_seed(config.seed, TIE_BREAK_SEED_TAG))}


# Extra constructor keyword arguments of a legacy strategy, built from the run's
# config (explicit injection, no `setattr`). Absent name = no extra arguments.
LEGACY_STRATEGY_KWARGS: dict[str, Callable[[ExperimentConfig], dict[str, Any]]] = {
    "AdversarialBIM": _tie_break_rng_kwargs,
}

def _seeded_rng(config: ExperimentConfig, tag: str) -> np.random.Generator:
    """One generator per run for a strategy's own randomness (never the main `rng`)."""
    return np.random.default_rng(derive_seed(config.seed, tag))


def _loader_args(args: TrainArgs) -> dict[str, int]:
    return {"batch_size": args.batch_size, "num_workers": args.num_workers}


def _var_ratio_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    return {"rng": _seeded_rng(config, VAR_RATIO_SEED_TAG)}


def _no_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    return {}


def _kcenter_pca_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    return {
        "n_components": config.dataset.strategy_args.kcenter_greedy_pca.n_components,
        "random_state": derive_seed(config.seed, KCENTER_PCA_SEED_TAG),
    }


def _badge_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    return {"rng": _seeded_rng(config, BADGE_SEED_TAG)}


def _ceal_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    # `CEALArgs` restricts `base_strategy` to `CEAL_BASE_STRATEGIES` (plain uncertainty).
    args = config.dataset.strategy_args.ceal
    base = LEGACY_STRATEGY_REGISTRY[args.base_strategy](dataset, net, logger)
    return {"base_strategy": base, "delta": args.delta, "delta_decay": args.delta_decay}


def _loss_prediction_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    args = config.dataset.strategy_args.loss_prediction
    return {"trainer": LossPredictionTrainer(args, _seeded_rng(config, LOSS_PREDICTION_SEED_TAG))}


def _vaal_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    sampler = VAALSampler(
        config.dataset.strategy_args.vaal,
        _seeded_rng(config, VAAL_SEED_TAG),
        torch.device(config.device),
        train_args=_loader_args(config.dataset.train_args),
        test_args=_loader_args(config.dataset.test_args),
    )
    return {"sampler": sampler}


def _waal_kwargs(config: ExperimentConfig, dataset, net, logger) -> dict[str, Any]:
    args = config.dataset.strategy_args.waal
    return {
        "trainer": WAALTrainer(args, _seeded_rng(config, WAAL_SEED_TAG)),
        "selection": args.selection,
    }


# DeepAL+ ports (ADR 0015): name -> class, and name -> constructor keyword arguments
# built from the run's config (explicit injection, no `setattr`).
DEEPALPLUS_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "VarRatio": VarRatio,
    "MeanSTD": MeanSTD,
    "KCenterGreedyPCA": KCenterGreedyPCA,
    "BadgeSampling": BadgeSampling,
    "CEALSampling": CEALSampling,
    "LossPredictionLoss": LossPredictionLoss,
    "VAAL": VAAL,
    "WAAL": WAAL,
}

StrategyKwargsFactory = Callable[[ExperimentConfig, Any, Any, Any], dict[str, Any]]

DEEPALPLUS_STRATEGY_KWARGS: dict[str, StrategyKwargsFactory] = {
    "VarRatio": _var_ratio_kwargs,
    "MeanSTD": _no_kwargs,
    "KCenterGreedyPCA": _kcenter_pca_kwargs,
    "BadgeSampling": _badge_kwargs,
    "CEALSampling": _ceal_kwargs,
    "LossPredictionLoss": _loss_prediction_kwargs,
    "VAAL": _vaal_kwargs,
    "WAAL": _waal_kwargs,
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
    **DEEPALPLUS_STRATEGY_REGISTRY,
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
    registry) but never consumed: representation strategies derive their own
    generator from `dalmax.seeding.derive_seed(config.seed, "selection")` and
    `AdversarialBIM` from `derive_seed(config.seed, "adversarial_ties")`
    (`LEGACY_STRATEGY_KWARGS`) and the DeepAL+ ports from their own tags
    (`DEEPALPLUS_STRATEGY_KWARGS`), so building a strategy never advances the
    run's main generator.

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
        extra_kwargs = LEGACY_STRATEGY_KWARGS[name](config) if name in LEGACY_STRATEGY_KWARGS else {}
        return LEGACY_STRATEGY_REGISTRY[name](dataset, net, logger, **extra_kwargs)

    if name in DEEPALPLUS_STRATEGY_REGISTRY:
        kwargs = DEEPALPLUS_STRATEGY_KWARGS[name](config, dataset, net, logger)
        return DEEPALPLUS_STRATEGY_REGISTRY[name](dataset, net, logger, **kwargs)

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
