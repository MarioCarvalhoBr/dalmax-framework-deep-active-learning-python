"""Tests for `dalmax.selection.hierarchical_kmeans.HierarchicalKMeansSelection`.

The vendored pipeline it wraps (`core.tools.SSL.src.hierarchical_kmeans_gpu`,
`hierarchical_sampling`, `clusters`) has no hardcoded `.cuda()` call and runs
on CPU tensors (verified manually with a small `make_blobs` dataset before
writing this module -- see `dalmax/selection/hierarchical_kmeans.py`'s module
docstring for the exact finding and the file:line of the one hardcoded-device
default that does exist in that source tree, which this pipeline never
calls). So these tests run unconditionally on CPU, no `gpu` marker needed.

Test fixtures deliberately use *imbalanced* cluster sizes: when every
k-means cluster has exactly the same size, a separate, unrelated bug in
`core/tools/SSL/src/utils.py:28` (`create_clusters_from_cluster_assignment`)
makes NumPy collapse the ragged per-cluster index arrays into a single
`dtype=object` 2-D array, which then fails torch tensor indexing --
independent of device. See the module docstring for detail; it is a
pre-existing vendored issue, not something this wrapper fixes.
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
