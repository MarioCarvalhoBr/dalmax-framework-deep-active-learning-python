"""Validate every ablation params JSON under `files_config/ablations/` (and its
`micro/` mirror) loads via `dalmax.config.loader.load_experiment_config` and
resolves to the extractor/variant/method/hierarchy the filename implies.

This is the Phase 3 acceptance check requested by
`.specs/architecture/refactor-plan.md`: every ablation cell must be a
config-only change, i.e. it must actually parse into a valid `ExperimentConfig`
before any lab-machine run is attempted. Parametrized over the folder so a new
ablation file added later is validated automatically without a new test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dalmax.config.loader import load_experiment_config
from dalmax.config.schema import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = REPO_ROOT / "files_config" / "ablations"
MICRO_DIR = ABLATIONS_DIR / "micro"

# name -> (extractor, embedding_variant, selection_method, hierarchy_n_clusters | None)
EXPECTED: dict[str, tuple[str, str, str, tuple[int, ...] | None]] = {
    "rep_full": ("ssrae", "full", "hierarchical", (600, 200, 100)),
    "rep_spatial": ("ssrae", "spatial", "hierarchical", (600, 200, 100)),
    "rep_spectral": ("ssrae", "spectral", "hierarchical", (600, 200, 100)),
    "hier_L1": ("ssrae", "full", "hierarchical", (50,)),
    "hier_L2a": ("ssrae", "full", "hierarchical", (300, 100)),
    "hier_L2b": ("ssrae", "full", "hierarchical", (100, 50)),
    "hier_L3": ("ssrae", "full", "hierarchical", (300, 100, 50)),
    "hier_L4": ("ssrae", "full", "hierarchical", (300, 100, 50, 25)),
    "stage_full": ("ssrae", "full", "hierarchical", (600, 200, 100)),
    "stage_no_representation": ("resnet_imagenet", "full", "hierarchical", (600, 200, 100)),
    "stage_no_hierarchy": ("ssrae", "full", "flat_proportional", None),
}

# Micro mirrors reuse smaller hierarchies (see files_config/ablations/README.md);
# only n_levels/shape need re-deriving here, not the exact counts.
EXPECTED_MICRO_N_CLUSTERS: dict[str, tuple[int, ...]] = {
    "rep_full": (60, 20, 10),
    "rep_spatial": (60, 20, 10),
    "rep_spectral": (60, 20, 10),
    "hier_L1": (5,),
    "hier_L2a": (30, 10),
    "hier_L2b": (10, 5),
    "hier_L3": (30, 10, 5),
    "hier_L4": (30, 10, 5, 2),
    "stage_full": (60, 20, 10),
    "stage_no_representation": (60, 20, 10),
    "stage_no_hierarchy": None,
}

ABLATION_NAMES = sorted(EXPECTED)


def _assert_full_scale_dataset_fields(config: ExperimentConfig) -> None:
    assert config.dataset.n_classes == 5
    assert config.dataset.data_dir == "DATA/daninhas_full/"
    assert config.dataset.n_epoch == 10
    assert config.dataset.train_args.batch_size == 256
    assert config.dataset.train_args.num_workers == 4
    assert config.dataset.test_args.batch_size == 256
    assert config.dataset.test_args.num_workers == 4


def _assert_micro_dataset_fields(config: ExperimentConfig) -> None:
    assert config.dataset.n_classes == 5
    assert config.dataset.data_dir == "DATA/daninhas_micro/"
    assert config.dataset.n_epoch == 1
    assert config.dataset.train_args.batch_size == 16
    assert config.dataset.train_args.num_workers == 0
    assert config.dataset.test_args.batch_size == 16
    assert config.dataset.test_args.num_workers == 0


def _load(params_json: Path) -> ExperimentConfig:
    return load_experiment_config(
        params_json,
        "DANINHAS",
        strategy_name="RepresentationStrategy",
        seed=1,
        n_init_labeled=100,
        n_query=100,
        n_round=8,
        dir_results="results/ablations/_test/",
        device="cpu",
    )


@pytest.mark.parametrize("name", ABLATION_NAMES)
def test_full_scale_ablation_config_loads_and_matches_spec(name: str) -> None:
    extractor, variant, method, n_clusters = EXPECTED[name]
    config = _load(ABLATIONS_DIR / f"{name}.json")

    _assert_full_scale_dataset_fields(config)
    assert config.dataset.embedding.extractor == extractor
    assert config.dataset.embedding.variant == variant
    assert config.dataset.selection.method == method
    if n_clusters is None:
        assert config.dataset.selection.hierarchy is None
    else:
        assert config.dataset.selection.hierarchy is not None
        assert config.dataset.selection.hierarchy.n_clusters == n_clusters
        assert config.dataset.selection.hierarchy.n_levels == len(n_clusters)


@pytest.mark.parametrize("name", ABLATION_NAMES)
def test_micro_ablation_config_loads_and_matches_spec(name: str) -> None:
    extractor, variant, method, _ = EXPECTED[name]
    n_clusters = EXPECTED_MICRO_N_CLUSTERS[name]
    config = _load(MICRO_DIR / f"{name}.json")

    _assert_micro_dataset_fields(config)
    assert config.dataset.embedding.extractor == extractor
    assert config.dataset.embedding.variant == variant
    assert config.dataset.selection.method == method
    if n_clusters is None:
        assert config.dataset.selection.hierarchy is None
    else:
        assert config.dataset.selection.hierarchy is not None
        assert config.dataset.selection.hierarchy.n_clusters == n_clusters
        assert config.dataset.selection.hierarchy.n_levels == len(n_clusters)


def test_ablation_folder_has_exactly_the_eleven_run_table_rows() -> None:
    on_disk = {p.stem for p in ABLATIONS_DIR.glob("*.json")}
    assert on_disk == set(ABLATION_NAMES)


def test_micro_folder_mirrors_every_full_scale_config() -> None:
    on_disk = {p.stem for p in MICRO_DIR.glob("*.json")}
    assert on_disk == set(ABLATION_NAMES)
