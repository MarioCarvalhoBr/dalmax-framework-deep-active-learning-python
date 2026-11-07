"""Tests for `dalmax.selection.hierarchical_kmeans.HierarchicalKMeansSelection`.

The vendored pipeline it wraps (`core.tools.SSL.src.hierarchical_kmeans_gpu`,
`hierarchical_sampling`, `clusters`) has no hardcoded `.cuda()` call and runs
on CPU tensors (verified manually with a small `make_blobs` dataset before
writing this module -- see `dalmax/selection/hierarchical_kmeans.py`'s module
docstring for the exact finding and the file:line of the one hardcoded-device
default that does exist in that source tree, which this pipeline never
calls). So these tests run unconditionally on CPU, no `gpu` marker needed.

Most fixtures use *imbalanced* cluster sizes (historical reason: KI-30, the
vendored `create_clusters_from_cluster_assignment` collapsed equal-size
clusters into a 2-D `dtype=object` array that broke torch indexing). KI-30
was fixed on 2026-11-07 (`dalmax/tools/SSL/src/utils.py::to_object_array`),
and `test_perfectly_balanced_clusters_do_not_crash` below covers the
balanced case explicitly.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from dalmax.selection.hierarchical_kmeans import HierarchicalKMeansSelection, _read_hierarchy


def _synthetic_pool(seed: int = 0):
    """200 points in 8 imbalanced blobs (avoids the equal-cluster-size
    vendored quirk described in the module docstring)."""
    embeddings, _ = make_blobs(
        n_samples=[30, 18, 25, 40, 12, 33, 22, 20],
        centers=None,
        n_features=6,
        cluster_std=2.5,
        random_state=seed,
    )
    embeddings = embeddings.astype(np.float32)
    ids = np.arange(5000, 5000 + len(embeddings))
    return embeddings, ids


HIERARCHY_DICT = {"n_clusters": [8, 4], "n_levels": 2, "sample_sizes": [4, 2]}


class _HierarchyConfigLike:
    """Minimal duck-typed stand-in for `dalmax.config.schema.HierarchyConfig`,
    used to prove the class does not import/depend on that module."""

    def __init__(self, n_clusters, n_levels, sample_sizes):
        self.n_clusters = n_clusters
        self.n_levels = n_levels
        self.sample_sizes = sample_sizes


def test_read_hierarchy_accepts_dict():
    n_clusters, n_levels, sample_sizes = _read_hierarchy(HIERARCHY_DICT)
    assert n_clusters == [8, 4]
    assert n_levels == 2
    assert sample_sizes == [4, 2]


def test_read_hierarchy_accepts_duck_typed_object():
    obj = _HierarchyConfigLike(n_clusters=(8, 4), n_levels=2, sample_sizes=(4, 2))
    n_clusters, n_levels, sample_sizes = _read_hierarchy(obj)
    assert n_clusters == [8, 4]
    assert sample_sizes == [4, 2]


def test_read_hierarchy_rejects_inconsistent_lengths():
    with pytest.raises(ValueError):
        _read_hierarchy({"n_clusters": [8, 4], "n_levels": 3, "sample_sizes": [4, 2]})


def test_read_hierarchy_rejects_missing_attrs():
    with pytest.raises(ValueError):
        _read_hierarchy(types.SimpleNamespace(n_clusters=[8, 4]))


@pytest.mark.parametrize("budget", [10, 20])
def test_selection_on_cpu_is_unique_subset_with_expected_size(budget):
    embeddings, ids = _synthetic_pool()
    strategy = HierarchicalKMeansSelection(HIERARCHY_DICT, device="cpu")

    selected = strategy.select(embeddings, ids, budget=budget, rng=np.random.default_rng(1))

    assert len(selected) == budget
    assert len(set(selected.tolist())) == len(selected)
    assert set(selected.tolist()) <= set(ids.tolist())


def test_same_seed_is_deterministic():
    embeddings, ids = _synthetic_pool()
    strategy = HierarchicalKMeansSelection(HIERARCHY_DICT, device="cpu")

    selected_a = strategy.select(embeddings, ids, budget=15, rng=np.random.default_rng(123))
    selected_b = strategy.select(embeddings, ids, budget=15, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(np.sort(selected_a), np.sort(selected_b))


def test_budget_at_least_pool_size_returns_all_ids():
    embeddings, ids = _synthetic_pool()
    strategy = HierarchicalKMeansSelection(HIERARCHY_DICT, device="cpu")

    selected = strategy.select(embeddings, ids, budget=10_000, rng=np.random.default_rng(1))

    assert set(selected.tolist()) == set(ids.tolist())


def test_budget_zero_returns_empty():
    embeddings, ids = _synthetic_pool()
    strategy = HierarchicalKMeansSelection(HIERARCHY_DICT, device="cpu")

    selected = strategy.select(embeddings, ids, budget=0, rng=np.random.default_rng(1))

    assert len(selected) == 0


def test_works_with_duck_typed_hierarchy_object():
    embeddings, ids = _synthetic_pool()
    hierarchy = _HierarchyConfigLike(n_clusters=[8, 4], n_levels=2, sample_sizes=[4, 2])
    strategy = HierarchicalKMeansSelection(hierarchy, device="cpu")

    selected = strategy.select(embeddings, ids, budget=12, rng=np.random.default_rng(4))

    assert len(selected) == 12


@pytest.mark.gpu
def test_selection_on_cuda_matches_cpu_contract():
    """Same contract, exercised on an actual CUDA device when available."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    embeddings, ids = _synthetic_pool()
    strategy = HierarchicalKMeansSelection(HIERARCHY_DICT, device="cuda")

    selected = strategy.select(embeddings, ids, budget=15, rng=np.random.default_rng(1))

    assert len(selected) == 15
    assert len(set(selected.tolist())) == 15
    assert set(selected.tolist()) <= set(ids.tolist())


# --- optional knobs (2026-11-07) -------------------------------------------------

KNOB_CASES = [
    {"init_method": "random"},
    {"resample_strategy": "random"},
    {"n_resamples": 0},
    {"leaf_sampling": "closest"},
]


@pytest.mark.parametrize("knobs", KNOB_CASES, ids=lambda k: next(iter(k)))
def test_every_knob_keeps_the_selection_contract_and_is_deterministic(knobs):
    embeddings, ids = _synthetic_pool()
    hierarchy = {**HIERARCHY_DICT, **knobs}
    runs = [
        HierarchicalKMeansSelection(hierarchy, device="cpu").select(
            embeddings, ids, budget=20, rng=np.random.default_rng(7)
        )
        for _ in range(2)
    ]
    for selected in runs:
        assert len(selected) == 20
        assert len(set(selected.tolist())) == 20
        assert set(selected.tolist()) <= set(ids.tolist())
    np.testing.assert_array_equal(runs[0], runs[1])


def test_leaf_sampling_closest_differs_from_random_and_is_valid():
    embeddings, ids = _synthetic_pool()
    selections = {}
    for leaf in ("random", "closest"):
        hierarchy = {**HIERARCHY_DICT, "leaf_sampling": leaf}
        selections[leaf] = HierarchicalKMeansSelection(hierarchy, device="cpu").select(
            embeddings, ids, budget=20, rng=np.random.default_rng(7)
        )
        assert len(set(selections[leaf].tolist())) == 20
    assert set(selections["random"].tolist()) != set(selections["closest"].tolist())


def test_knob_defaults_when_absent_and_hierarchy_value_beats_constructor_default():
    default = HierarchicalKMeansSelection(HIERARCHY_DICT)
    assert (default.n_resamples, default.init_method) == (10, "kmeans++")
    assert (default.resample_strategy, default.leaf_sampling) == ("closest", "random")
    assert HierarchicalKMeansSelection(HIERARCHY_DICT, n_resamples=4).n_resamples == 4
    in_hierarchy = {**HIERARCHY_DICT, "n_resamples": 2}
    assert HierarchicalKMeansSelection(in_hierarchy, n_resamples=4).n_resamples == 2


def test_knobs_are_read_from_duck_typed_object_and_invalid_values_fail_fast():
    obj = _HierarchyConfigLike([8, 4], 2, [4, 2])
    obj.leaf_sampling = "closest"
    assert HierarchicalKMeansSelection(obj).leaf_sampling == "closest"
    with pytest.raises(ValueError, match="init_method"):
        HierarchicalKMeansSelection({**HIERARCHY_DICT, "init_method": "bogus"})


@pytest.mark.parametrize("leaf_sampling", ["random", "closest"])
def test_perfectly_balanced_clusters_do_not_crash(leaf_sampling):
    """KI-30 regression: 6 identical-size blobs clustered into [6, 2, 2] gives
    equal-size clusters at every level (3 points per cluster in level 1 when
    the blobs are well separated, 3 clusters per cluster in level 2, ...);
    before the 2026-11-07 fix this raised a vendored ``TypeError``."""
    embeddings, _ = make_blobs(
        n_samples=[3] * 6, centers=None, n_features=4, cluster_std=0.05, random_state=3
    )
    embeddings = embeddings.astype(np.float32)
    ids = np.arange(100, 100 + len(embeddings))
    hierarchy = {
        "n_clusters": [6, 2, 2],
        "n_levels": 3,
        "sample_sizes": [2, 2, 2],
        "leaf_sampling": leaf_sampling,
    }
    sel = HierarchicalKMeansSelection(hierarchy=hierarchy, device="cpu")
    out = sel.select(embeddings, ids, budget=4, rng=np.random.default_rng(0))
    assert len(out) == 4 and len(set(out.tolist())) == 4
    assert set(out.tolist()) <= set(ids.tolist())
