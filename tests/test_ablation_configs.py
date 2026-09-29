"""Validate every ablation params JSON under `files_config/ablations/{rnhal,texhal}/`
(and each method's `micro/` mirror) loads via `dalmax.config.loader.load_experiment_config`
and resolves to the extractor/q/variant/method/hierarchy the filename implies.

This is the Phase 3 acceptance check requested by
`.specs/architecture/refactor-plan.md`: every ablation cell must be a
config-only change, i.e. it must actually parse into a valid `ExperimentConfig`
before any lab-machine run is attempted. Parametrized over method (`rnhal`,
`texhal`) and folder, so a new ablation file added later is validated
automatically without a new test.

Since the 2026-08-30 per-method reorganization (see
`files_config/ablations/README.md` and `.specs/experiments/papers-roadmap.md`),
`rnhal/` (paper 3, SSRAE) carries 11 full-scale + 11 micro configs, and
`texhal/` (paper 2, VCTex) carries 12 full-scale + 12 micro configs (§6.1 has
a 4th row, `rep_q13`, added 2026-08-30 per the VCTex method authors — see
`.specs/experiments/ablation-study-texhal.md`) — 46 files total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dalmax.config.loader import load_experiment_config
from dalmax.config.schema import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = REPO_ROOT / "files_config" / "ablations"

# name -> (extractor, q, embedding_variant, selection_method, hierarchy_n_clusters | None)
# Note: SSRAE's `q` is a plain int in the params JSON (`"q": 13`), so it loads
# as `int`, not a tuple -- unlike VCTex's `"q": [5, 17]`-style list, which the
# loader always converts to a tuple (`dalmax/config/loader.py::_build_embedding_config`).
EXPECTED_RNHAL: dict[str, tuple[str, int | None, str, str, tuple[int, ...] | None]] = {
    "rep_full": ("ssrae", 13, "full", "hierarchical", (600, 200, 100)),
    "rep_spatial": ("ssrae", 13, "spatial", "hierarchical", (600, 200, 100)),
    "rep_spectral": ("ssrae", 13, "spectral", "hierarchical", (600, 200, 100)),
    "hier_L1": ("ssrae", 13, "full", "hierarchical", (50,)),
    "hier_L2a": ("ssrae", 13, "full", "hierarchical", (300, 100)),
    "hier_L2b": ("ssrae", 13, "full", "hierarchical", (100, 50)),
    "hier_L3": ("ssrae", 13, "full", "hierarchical", (300, 100, 50)),
    "hier_L4": ("ssrae", 13, "full", "hierarchical", (300, 100, 50, 25)),
    # 5 advisor-requested hierarchy rows (2026-09-29, paper 3 only) -- see
    # files_config/ablations/README.md and .specs/experiments/campaign.md.
    "hier_L1_k100": ("ssrae", 13, "full", "hierarchical", (100,)),
    "hier_L1_k200": ("ssrae", 13, "full", "hierarchical", (200,)),
    "hier_L1_k600": ("ssrae", 13, "full", "hierarchical", (600,)),
    "hier_L2_k200_100": ("ssrae", 13, "full", "hierarchical", (200, 100)),
    "hier_L4_k800_600_200_100": ("ssrae", 13, "full", "hierarchical", (800, 600, 200, 100)),
    "stage_full": ("ssrae", 13, "full", "hierarchical", (600, 200, 100)),
    "stage_no_hierarchy": ("ssrae", 13, "full", "flat_proportional", None),
}

EXPECTED_TEXHAL: dict[str, tuple[str, tuple[int, ...] | None, str, str, tuple[int, ...] | None]] = {
    # §6.1 has 4 rows (not 3, unlike rnhal's spatial/spectral split): per the
    # VCTex method authors (Fares & Ribas, email 2026-08-26, and the original
    # manuscript phd_files/artigo-original-tecnica-vctex-manuscript.pdf),
    # Q=13 and Q=17 alone are both good smaller-vector single-scale settings,
    # and Q=[5,17] (concatenated) is the paper's best-parameters multi-scale
    # setting -- see files_config/ablations/README.md and
    # .specs/experiments/ablation-study-texhal.md for the full rationale.
    "rep_q5": ("vctex", (5,), "full", "hierarchical", (600, 200, 100)),
    "rep_q13": ("vctex", (13,), "full", "hierarchical", (600, 200, 100)),
    "rep_q17": ("vctex", (17,), "full", "hierarchical", (600, 200, 100)),
    "rep_full": ("vctex", (5, 17), "full", "hierarchical", (600, 200, 100)),
    "hier_L1": ("vctex", (5, 17), "full", "hierarchical", (50,)),
    "hier_L2a": ("vctex", (5, 17), "full", "hierarchical", (300, 100)),
    "hier_L2b": ("vctex", (5, 17), "full", "hierarchical", (100, 50)),
    "hier_L3": ("vctex", (5, 17), "full", "hierarchical", (300, 100, 50)),
    "hier_L4": ("vctex", (5, 17), "full", "hierarchical", (300, 100, 50, 25)),
    "stage_full": ("vctex", (5, 17), "full", "hierarchical", (600, 200, 100)),
    # No texhal `stage_no_representation`: one shared run, files_config/ablations/rnhal/ (ADR 0009).
    "stage_no_hierarchy": ("vctex", (5, 17), "full", "flat_proportional", None),
}

# Micro mirrors reuse smaller hierarchies (see files_config/ablations/README.md);
# only n_levels/shape need re-deriving here, not the exact counts. Identical
# hierarchy grid for both methods -- only the embedding block differs.
EXPECTED_MICRO_N_CLUSTERS_RNHAL: dict[str, tuple[int, ...] | None] = {
    "rep_full": (60, 20, 10),
    "rep_spatial": (60, 20, 10),
    "rep_spectral": (60, 20, 10),
    "hier_L1": (5,),
    "hier_L2a": (30, 10),
    "hier_L2b": (10, 5),
    "hier_L3": (30, 10, 5),
    "hier_L4": (30, 10, 5, 2),
    "hier_L1_k100": (10,),
    "hier_L1_k200": (20,),
    "hier_L1_k600": (60,),
    "hier_L2_k200_100": (20, 10),
    "hier_L4_k800_600_200_100": (80, 60, 20, 10),
    "stage_full": (60, 20, 10),
    "stage_no_hierarchy": None,
}

EXPECTED_MICRO_N_CLUSTERS_TEXHAL: dict[str, tuple[int, ...] | None] = {
    "rep_q5": (60, 20, 10),
    "rep_q13": (60, 20, 10),
    "rep_q17": (60, 20, 10),
    "rep_full": (60, 20, 10),
    "hier_L1": (5,),
    "hier_L2a": (30, 10),
    "hier_L2b": (10, 5),
    "hier_L3": (30, 10, 5),
    "hier_L4": (30, 10, 5, 2),
    "stage_full": (60, 20, 10),
    "stage_no_hierarchy": None,
}

METHODS: dict[str, dict[str, tuple]] = {
    "rnhal": EXPECTED_RNHAL,
    "texhal": EXPECTED_TEXHAL,
}
MICRO_EXPECTED: dict[str, dict[str, tuple[int, ...] | None]] = {
    "rnhal": EXPECTED_MICRO_N_CLUSTERS_RNHAL,
    "texhal": EXPECTED_MICRO_N_CLUSTERS_TEXHAL,
}

# (method, name) pairs for parametrization -- 15 rnhal + 11 texhal.
METHOD_NAME_PAIRS = [(method, name) for method, rows in METHODS.items() for name in sorted(rows)]


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


def _assert_embedding_and_selection(
    config: ExperimentConfig,
    extractor: str,
    q: tuple[int, ...] | None,
    variant: str,
    method: str,
    n_clusters: tuple[int, ...] | None,
) -> None:
    assert config.dataset.embedding.extractor == extractor
    if q is None:
        assert config.dataset.embedding.q is None
    else:
        assert config.dataset.embedding.q == q
    assert config.dataset.embedding.variant == variant
    assert config.dataset.selection.method == method
    if n_clusters is None:
        assert config.dataset.selection.hierarchy is None
    else:
        assert config.dataset.selection.hierarchy is not None
        assert config.dataset.selection.hierarchy.n_clusters == n_clusters
        assert config.dataset.selection.hierarchy.n_levels == len(n_clusters)


@pytest.mark.parametrize("method,name", METHOD_NAME_PAIRS, ids=[f"{m}-{n}" for m, n in METHOD_NAME_PAIRS])
def test_full_scale_ablation_config_loads_and_matches_spec(method: str, name: str) -> None:
    extractor, q, variant, selection_method, n_clusters = METHODS[method][name]
    config = _load(ABLATIONS_DIR / method / f"{name}.json")

    _assert_full_scale_dataset_fields(config)
    _assert_embedding_and_selection(config, extractor, q, variant, selection_method, n_clusters)


@pytest.mark.parametrize("method,name", METHOD_NAME_PAIRS, ids=[f"{m}-{n}" for m, n in METHOD_NAME_PAIRS])
def test_micro_ablation_config_loads_and_matches_spec(method: str, name: str) -> None:
    extractor, q, variant, selection_method, _ = METHODS[method][name]
    n_clusters = MICRO_EXPECTED[method][name]
    config = _load(ABLATIONS_DIR / method / "micro" / f"{name}.json")

    _assert_micro_dataset_fields(config)
    _assert_embedding_and_selection(config, extractor, q, variant, selection_method, n_clusters)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_ablation_folder_has_exactly_the_run_table_rows(method: str) -> None:
    on_disk = {p.stem for p in (ABLATIONS_DIR / method).glob("*.json")}
    assert on_disk == set(METHODS[method])


@pytest.mark.parametrize("method", sorted(METHODS))
def test_micro_folder_mirrors_every_full_scale_config(method: str) -> None:
    on_disk = {p.stem for p in (ABLATIONS_DIR / method / "micro").glob("*.json")}
    assert on_disk == set(METHODS[method])


def test_texhal_variant_is_always_full() -> None:
    """slice_embedding (full/spatial/spectral) is SSRAE-only; every texhal
    config must use variant="full" -- see files_config/ablations/README.md.
    """
    for name in EXPECTED_TEXHAL:
        for subdir in (ABLATIONS_DIR / "texhal", ABLATIONS_DIR / "texhal" / "micro"):
            config = _load(subdir / f"{name}.json")
            assert config.dataset.embedding.variant == "full"


def test_no_ablation_folder_carries_stage_no_representation() -> None:
    """ADR 0009 (2026-09-29): the ResNet-ImageNet + hierarchical row is ONE run shared by
    papers 1 (KMH@100), 2 and 3; its only config lives in files_config/campaign/."""
    for method in ("rnhal", "texhal"):
        assert not (ABLATIONS_DIR / method / "stage_no_representation.json").exists()
        assert not (ABLATIONS_DIR / method / "micro" / "stage_no_representation.json").exists()
    for name in ("params_kmh.json", "params_kmh_micro.json"):
        config = _load(ABLATIONS_DIR.parent / "campaign" / name)
        assert config.dataset.embedding.extractor == "resnet_imagenet"
        assert config.dataset.embedding.q is None
        assert config.dataset.selection.method == "hierarchical"
    full = _load(ABLATIONS_DIR.parent / "campaign" / "params_kmh.json")
    assert full.dataset.selection.hierarchy.n_clusters == (600, 200, 100)
    assert full.dataset.selection.hierarchy.sample_sizes == (30, 15, 2)
