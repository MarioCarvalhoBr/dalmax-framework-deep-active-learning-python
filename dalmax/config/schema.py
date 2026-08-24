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

from dataclasses import dataclass

VALID_EXTRACTORS = ("ssrae", "vctex", "resnet_imagenet")
VALID_EMBEDDING_VARIANTS = ("full", "spatial", "spectral")
VALID_SELECTION_METHODS = ("flat_closest", "flat_proportional", "hierarchical")
VALID_DEVICES = ("cuda", "cpu")


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


@dataclass(frozen=True)
class HierarchyConfig:
    """Hierarchical k-means configuration (formerly ``config_kmh``).

    ``n_clusters[i]`` and ``sample_sizes[i]`` describe level ``i`` of the
    hierarchy; both sequences must have exactly ``n_levels`` entries.
    """

    n_clusters: tuple[int, ...]
    n_levels: int
    sample_sizes: tuple[int, ...]

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
        if not self.dir_results:
            raise ConfigError("ExperimentConfig.dir_results must be non-empty")
