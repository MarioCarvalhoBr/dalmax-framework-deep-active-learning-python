"""Typed configuration dataclasses for DalMax experiments.

These dataclasses replace raw ``params[dataset_name]`` dict indexing
(the historical ``demo.py``, ``core/query_strategies/ssl_ssrae_sampling.py``) with a
validated, typed structure. See ``.specs/architecture/target-architecture.md``
§2 and ``dalmax.config.loader`` for how instances are built from the existing
params JSON files.

All dataclasses are frozen (immutable) — an ``ExperimentConfig`` fully
determines a run together with the CLI args and git commit
(``.claude/rules/reproducibility.md``), so it must not be mutated after
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_EXTRACTORS = ("ssrae", "vctex", "resnet_imagenet")
VALID_EMBEDDING_VARIANTS = ("full", "spatial", "spectral")
VALID_SELECTION_METHODS = ("flat_closest", "flat_proportional", "hierarchical")
VALID_DEVICES = ("cuda", "cpu")

# The upper-bound "strategy" (paper 1): train once on the ENTIRE pool, no
# active learning. Lives here (not in `dalmax.query_strategies`) so the config
# layer can validate `n_round == 0` without importing torch-heavy modules.
FULL_SUPERVISED_STRATEGY = "FullSupervised"


class ConfigError(ValueError):
    """Raised when a params JSON / experiment configuration is invalid.

    Always carries a human-readable message naming the offending key(s) —
    never a bare ``KeyError`` raised deep inside application logic (see
    ``.claude/rules/code-quality.md`` "fail-fast errors over silent
    fallbacks").
    """


@dataclass(frozen=True)
class TrainArgs:
    """DataLoader-style arguments shared by train/test loaders."""

    batch_size: int
    num_workers: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"TrainArgs.batch_size must be > 0, got {self.batch_size}")
        if self.num_workers < 0:
            raise ConfigError(f"TrainArgs.num_workers must be >= 0, got {self.num_workers}")


@dataclass(frozen=True)
class OptimizerArgs:
    """SGD-style optimizer hyperparameters."""

    lr: float
    momentum: float

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ConfigError(f"OptimizerArgs.lr must be > 0, got {self.lr}")
        if self.momentum < 0:
            raise ConfigError(f"OptimizerArgs.momentum must be >= 0, got {self.momentum}")


HIERARCHY_CHOICES: dict[str, tuple[str, ...]] = {
    "init_method": ("kmeans++", "random"),
    "resample_strategy": ("closest", "random"),
    "leaf_sampling": ("random", "closest"),
}


@dataclass(frozen=True)
class HierarchyConfig:
    """Hierarchical k-means configuration (formerly ``config_kmh``).

    ``n_clusters[i]`` and ``sample_sizes[i]`` describe level ``i`` of the
    hierarchy; both sequences must have exactly ``n_levels`` entries.

    Optional knobs (2026-11-07; defaults reproduce the previous hard-wired behaviour):

    - ``n_resamples``: centroid-refinement resampling iterations per level
      (int >= 0; ``0`` disables resampling).
    - ``init_method``: k-means centroid initialisation, ``"kmeans++"`` or ``"random"``.
    - ``resample_strategy``: how points are drawn from each cluster during
      resampling, ``"closest"`` (to the centroid) or ``"random"``.
    - ``leaf_sampling``: how the final selection picks points inside a leaf
      cluster, ``"random"`` or ``"closest"`` (to the centroid); maps to the vendored
      ``sampling_strategy`` ``"r"`` / ``"c"``.
    """

    n_clusters: tuple[int, ...]
    n_levels: int
    sample_sizes: tuple[int, ...]
    n_resamples: int = 10
    init_method: str = "kmeans++"
    resample_strategy: str = "closest"
    leaf_sampling: str = "random"

    def __post_init__(self) -> None:
        if self.n_levels <= 0:
            raise ConfigError(f"HierarchyConfig.n_levels must be > 0, got {self.n_levels}")
        if len(self.n_clusters) != self.n_levels:
            raise ConfigError(
                "HierarchyConfig.n_levels "
                f"({self.n_levels}) must match len(n_clusters) "
                f"({len(self.n_clusters)}: {self.n_clusters})"
            )
        if len(self.sample_sizes) != self.n_levels:
            raise ConfigError(
                "HierarchyConfig.n_levels "
                f"({self.n_levels}) must match len(sample_sizes) "
                f"({len(self.sample_sizes)}: {self.sample_sizes})"
            )
        if any(n <= 0 for n in self.n_clusters):
            raise ConfigError(f"HierarchyConfig.n_clusters must all be > 0, got {self.n_clusters}")
        if any(n <= 0 for n in self.sample_sizes):
            raise ConfigError(
                f"HierarchyConfig.sample_sizes must all be > 0, got {self.sample_sizes}"
            )
        if isinstance(self.n_resamples, bool) or self.n_resamples < 0:
            raise ConfigError(
                f"HierarchyConfig.n_resamples must be an int >= 0, got {self.n_resamples!r}"
            )
        for name, valid in HIERARCHY_CHOICES.items():
            value = getattr(self, name)
            if value not in valid:
                raise ConfigError(
                    f"HierarchyConfig.{name} must be one of {list(valid)}, got {value!r}"
                )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Which embedding provider to use and how to slice its output.

    ``q`` is the extractor hyperparameter: an ``int`` for SSRAE (``13``), a
    tuple for VCTex (``(5, 17)``), or ``None`` for ResNet-ImageNet (fixed
    2048-d penultimate layer, no ``Q``).
    """

    extractor: str = "ssrae"
    q: int | tuple[int, ...] | None = 13
    variant: str = "full"

    def __post_init__(self) -> None:
        if self.extractor not in VALID_EXTRACTORS:
            raise ConfigError(
                f"EmbeddingConfig.extractor must be one of {VALID_EXTRACTORS}, got {self.extractor!r}"
            )
        if self.variant not in VALID_EMBEDDING_VARIANTS:
            raise ConfigError(
                "EmbeddingConfig.variant must be one of "
                f"{VALID_EMBEDDING_VARIANTS}, got {self.variant!r}"
            )


@dataclass(frozen=True)
class SelectionConfig:
    """Which query-batch selection strategy to use.

    ``hierarchy`` is required iff ``method == "hierarchical"``; it must be
    ``None`` for every other method (a hierarchy config silently ignored by
    a non-hierarchical method would be a reproducibility hazard).
    """

    method: str = "hierarchical"
    hierarchy: HierarchyConfig | None = None

    def __post_init__(self) -> None:
        if self.method not in VALID_SELECTION_METHODS:
            raise ConfigError(
                "SelectionConfig.method must be one of "
                f"{VALID_SELECTION_METHODS}, got {self.method!r}"
            )
        if self.method == "hierarchical" and self.hierarchy is None:
            raise ConfigError(
                "SelectionConfig.method is 'hierarchical' but 'hierarchy' is missing "
                "(missing key: 'hierarchy')"
            )
        if self.method != "hierarchical" and self.hierarchy is not None:
            raise ConfigError(
                f"SelectionConfig.hierarchy must be None when method={self.method!r}, "
                "not just unused (avoids silently ignored config)"
            )


@dataclass(frozen=True)
class ModelArgs:
    """Optional constructor arguments of the dataset's model class (params JSON
    ``"model_args"`` block, ADR 0014).

    ``None`` means "not set": the argument is not passed and the model class
    uses its own default, so a params JSON without ``"model_args"`` builds the
    model exactly as before this block existed.

    - ``dropout_p``: probability of the dropout layer applied before the final
      classifier, in ``[0, 1)``. Default of the DANINHAS ResNet-50: ``0.0`` (no
      dropout layer); the CIFAR10 CNN has its built-in DeepAL dropout ``0.5``.
      ``files_config/campaign/params_paper1_mc_dropout.json`` sets ``0.5`` for the
      four MC-dropout strategies (KI-41).
    """

    dropout_p: float | None = None

    def __post_init__(self) -> None:
        if self.dropout_p is not None and not 0.0 <= self.dropout_p < 1.0:
            raise ConfigError(f"ModelArgs.dropout_p must be in [0, 1), got {self.dropout_p!r}")

    def as_kwargs(self) -> dict[str, float]:
        """The arguments that are set, as keyword arguments for the model class."""
        return {} if self.dropout_p is None else {"dropout_p": self.dropout_p}


# Base strategies `CEALSampling` may query with (ADR 0015): the plain, single-pass
# uncertainty strategies, resolved through the strategy registry.
CEAL_BASE_STRATEGIES = ("LeastConfidence", "MarginSampling", "EntropySampling")


def _check_positive(owner: str, **values: float) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ConfigError(f"{owner}.{name} must be > 0, got {value!r}")


def _check_non_negative(owner: str, **values: float) -> None:
    for name, value in values.items():
        if not value >= 0:
            raise ConfigError(f"{owner}.{name} must be >= 0, got {value!r}")


CEAL_DELTA_SCHEDULES = ("linear", "cumulative")


@dataclass(frozen=True)
class CEALArgs:
    """``CEALSampling`` (Wang et al., IEEE TCSVT 2017, arXiv 1701.03551).

    - ``base_strategy``: strategy that picks the queried samples, one of
      ``CEAL_BASE_STRATEGIES``.
    - ``delta``: initial entropy threshold ``delta_0`` (nats) below which an
      unlabeled sample is pseudo-labelled for the next training. Default 0.05 and
      ``delta_decay`` 0.0033: the paper's CACD setting (its Caltech-256 setting is
      0.005 / 0.00033; DeepAL+ uses 5e-5 / 0.033e-5, which pseudo-labels nothing on
      DANINHAS, ADR 0015).
    - ``delta_schedule``: threshold of query round ``t`` (1-based), clamped at 0:
      ``"linear"`` = ``delta_0 - dr * t``; ``"cumulative"`` = ``delta_0 - dr * t (t + 1) / 2``
      (the literal recursive reading of the paper's Eq. 9, ``delta = delta - dr * t``,
      and DeepAL+'s code).
    """

    base_strategy: str = "EntropySampling"
    delta: float = 0.05
    delta_decay: float = 0.0033
    delta_schedule: str = "linear"

    def __post_init__(self) -> None:
        if self.base_strategy not in CEAL_BASE_STRATEGIES:
            raise ConfigError(
                f"CEALArgs.base_strategy must be one of {list(CEAL_BASE_STRATEGIES)}, "
                f"got {self.base_strategy!r}"
            )
        if self.delta_schedule not in CEAL_DELTA_SCHEDULES:
            raise ConfigError(
                f"CEALArgs.delta_schedule must be one of {list(CEAL_DELTA_SCHEDULES)}, "
                f"got {self.delta_schedule!r}"
            )
        _check_positive("CEALArgs", delta=self.delta)
        _check_non_negative("CEALArgs", delta_decay=self.delta_decay)


@dataclass(frozen=True)
class KCenterGreedyPCAArgs:
    """``KCenterGreedyPCA``: embeddings wider than ``n_components`` are reduced by PCA."""

    n_components: int = 50

    def __post_init__(self) -> None:
        _check_positive("KCenterGreedyPCAArgs", n_components=self.n_components)


@dataclass(frozen=True)
class LossPredictionArgs:
    """``LossPredictionLoss`` (Yoo & Kweon, 2019).

    - ``margin``/``weight``: margin of the pairwise ranking loss and its weight in
      ``CE + weight * ranking``.
    - ``detach_fraction``: the module's gradient reaches the backbone during the
      first ``floor(detach_fraction * n_epoch)`` epochs only (120/200 = 0.6 in the
      original paper).
    - ``interm_dim``: width of each per-stage FC of the module.
    - ``module_lr``: learning rate of the module's Adam optimizer (DeepAL+).
    """

    margin: float = 1.0
    weight: float = 1.0
    detach_fraction: float = 0.6
    interm_dim: int = 128
    module_lr: float = 0.01

    def __post_init__(self) -> None:
        _check_positive("LossPredictionArgs", interm_dim=self.interm_dim, module_lr=self.module_lr)
        _check_non_negative("LossPredictionArgs", margin=self.margin, weight=self.weight)
        if not 0.0 <= self.detach_fraction <= 1.0:
            raise ConfigError(
                f"LossPredictionArgs.detach_fraction must be in [0, 1], got {self.detach_fraction!r}"
            )


@dataclass(frozen=True)
class VAALArgs:
    """``VAAL`` (Sinha et al., 2019): VAE + latent discriminator trained at every query.

    ``epochs``/``latent_dim``/``beta``/``adversary_param``/``num_vae_steps``/
    ``num_adv_steps`` are the DeepAL+/original defaults; ``lr`` is the Adam learning
    rate of both networks (original VAAL; DeepAL+ reuses the task optimizer);
    ``image_size`` is the side the images are resized to for the VAE (a multiple of
    16; 64 = the ``VAE_ImageNet``-style network).
    """

    epochs: int = 30
    latent_dim: int = 32
    beta: float = 1.0
    adversary_param: float = 1.0
    num_vae_steps: int = 2
    num_adv_steps: int = 1
    lr: float = 5e-4
    image_size: int = 64

    def __post_init__(self) -> None:
        _check_positive(
            "VAALArgs",
            epochs=self.epochs,
            latent_dim=self.latent_dim,
            num_vae_steps=self.num_vae_steps,
            num_adv_steps=self.num_adv_steps,
            lr=self.lr,
            image_size=self.image_size,
        )
        _check_non_negative("VAALArgs", beta=self.beta, adversary_param=self.adversary_param)
        if self.image_size % 16:
            raise ConfigError(f"VAALArgs.image_size must be a multiple of 16, got {self.image_size}")


@dataclass(frozen=True)
class WAALArgs:
    """``WAAL`` (Shui et al., 2020; DeepAL+ defaults).

    - ``alpha``: weight of the Wasserstein terms in both min-max steps.
    - ``gp_weight_task``/``gp_weight_critic``: gradient-penalty weights of the task
      step (``+ alpha * gp * 5``) and of the critic step (``- alpha * gp * 2``, sign
      as in the original code, see ADR 0015).
    - ``gamma_ratio``: weight of the labelled critic mean in the Wasserstein term
      (DeepAL+ fixes 1; the original WAAL uses ``|L| / |U|``).
    - ``selection``: weight of the critic score in the query score.
    """

    alpha: float = 1e-3
    gp_weight_task: float = 5.0
    gp_weight_critic: float = 2.0
    gamma_ratio: float = 1.0
    selection: float = 10.0

    def __post_init__(self) -> None:
        _check_positive("WAALArgs", alpha=self.alpha)
        _check_non_negative(
            "WAALArgs",
            gp_weight_task=self.gp_weight_task,
            gp_weight_critic=self.gp_weight_critic,
            gamma_ratio=self.gamma_ratio,
            selection=self.selection,
        )


@dataclass(frozen=True)
class StrategyArgs:
    """Optional per-strategy hyperparameters (params JSON ``"strategy_args"`` block,
    ADR 0015), one sub-block per strategy name; absent = the defaults above. Each
    block is read only by its own strategy."""

    ceal: CEALArgs = field(default_factory=CEALArgs)
    kcenter_greedy_pca: KCenterGreedyPCAArgs = field(default_factory=KCenterGreedyPCAArgs)
    loss_prediction: LossPredictionArgs = field(default_factory=LossPredictionArgs)
    vaal: VAALArgs = field(default_factory=VAALArgs)
    waal: WAALArgs = field(default_factory=WAALArgs)


# params JSON key (= the strategy's registry name) -> `StrategyArgs` field.
STRATEGY_ARGS_KEYS: dict[str, str] = {
    "CEALSampling": "ceal",
    "KCenterGreedyPCA": "kcenter_greedy_pca",
    "LossPredictionLoss": "loss_prediction",
    "VAAL": "vaal",
    "WAAL": "waal",
}

# Strategy name -> the `StrategyArgs` field the strategy READS. Equal to
# `STRATEGY_ARGS_KEYS` plus the strategies without a block of their own that read
# another strategy's block: the WAAL control `WAALRandomQuery` reads the `WAAL` block
# (one block for both, so the control's training cannot drift from WAAL's; ADR 0015
# amendment of 2026-11-12). A params JSON has no `"WAALRandomQuery"` key.
STRATEGY_ARGS_READ_BY: dict[str, str] = {
    **STRATEGY_ARGS_KEYS,
    "WAALRandomQuery": STRATEGY_ARGS_KEYS["WAAL"],
}


@dataclass(frozen=True)
class DatasetConfig:
    """Everything the existing params JSON specifies for one dataset."""

    name: str
    data_dir: str
    n_classes: int
    n_epoch: int
    n_drop: int
    train_args: TrainArgs
    test_args: TrainArgs
    optimizer_args: OptimizerArgs
    embedding: EmbeddingConfig
    selection: SelectionConfig
    model_args: ModelArgs = field(default_factory=ModelArgs)
    strategy_args: StrategyArgs = field(default_factory=StrategyArgs)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("DatasetConfig.name must be non-empty")
        if not self.data_dir:
            raise ConfigError("DatasetConfig.data_dir must be non-empty")
        if self.n_classes <= 0:
            raise ConfigError(f"DatasetConfig.n_classes must be > 0, got {self.n_classes}")
        if self.n_epoch <= 0:
            raise ConfigError(f"DatasetConfig.n_epoch must be > 0, got {self.n_epoch}")
        if self.n_drop < 0:
            raise ConfigError(f"DatasetConfig.n_drop must be >= 0, got {self.n_drop}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully-resolved configuration for one experiment run.

    Together with the git commit hash (``dalmax.experiment.run_metadata``),
    this is exactly the "params JSON + CLI args + seed + git commit" tuple
    that ``.claude/rules/reproducibility.md`` requires to reproduce a run.
    """

    dataset: DatasetConfig
    strategy_name: str
    seed: int
    n_init_labeled: int
    n_query: int
    n_round: int
    dir_results: str
    device: str
    params_json_path: str

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ConfigError("ExperimentConfig.strategy_name must be non-empty")
        if self.device not in VALID_DEVICES:
            raise ConfigError(
                f"ExperimentConfig.device must be resolved to one of {VALID_DEVICES} "
                f"before construction (use dalmax.config.loader, got {self.device!r})"
            )
        if self.n_init_labeled <= 0:
            raise ConfigError(
                f"ExperimentConfig.n_init_labeled must be > 0, got {self.n_init_labeled}"
            )
        if self.n_query <= 0:
            raise ConfigError(f"ExperimentConfig.n_query must be > 0, got {self.n_query}")
        if self.n_round < 0:
            raise ConfigError(f"ExperimentConfig.n_round must be >= 0, got {self.n_round}")
        if self.strategy_name == FULL_SUPERVISED_STRATEGY and self.n_round != 0:
            raise ConfigError(
                f"strategy {FULL_SUPERVISED_STRATEGY!r} is the full-pool upper bound (no active "
                f"learning rounds): --n_round must be 0, got {self.n_round}"
            )
        if not self.dir_results:
            raise ConfigError("ExperimentConfig.dir_results must be non-empty")
