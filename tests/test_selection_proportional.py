"""Tests for `dalmax.selection.flat_kmeans_proportional.FlatKMeansProportionalRandom`.

Acceptance criteria (per Phase 2 selection contract):
- per-cluster quotas sum exactly to the requested budget;
- every selected sample belongs to the cluster it was quota-assigned to;
- selections are a unique subset of `ids` with the expected size.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from dalmax.selection.flat_kmeans_proportional import (
    FlatKMeansProportionalRandom,
    _largest_remainder_quotas,
)

N_TRUE_CLUSTERS = 4


def _synthetic_pool(n_samples: int = 200, seed: int = 0):
    """Well-separated, tight blobs so k-means with k=N_TRUE_CLUSTERS reliably
    recovers the ground-truth grouping, letting us check cluster membership
    against the known blob label without reaching into the strategy's
    internals."""
    embeddings, true_labels, centers = make_blobs(
        n_samples=n_samples,
        centers=N_TRUE_CLUSTERS,
        n_features=8,
        cluster_std=0.4,
        center_box=(-20.0, 20.0),
        random_state=seed,
        return_centers=True,
    )
    embeddings = embeddings.astype(np.float32)
    ids = np.arange(2000, 2000 + n_samples)
    return embeddings, ids, true_labels, centers


def test_quotas_sum_to_budget_helper():
    sizes = np.array([50, 30, 15, 5])
    for budget in (1, 10, 37, 99, 100):
        quotas = _largest_remainder_quotas(sizes, budget)
        assert quotas.sum() == budget
        assert np.all(quotas <= sizes)
        assert np.all(quotas >= 0)


@pytest.mark.parametrize("budget", [8, 20, 50])
def test_each_pick_belongs_to_its_geometric_cluster(budget):
    embeddings, ids, true_labels, centers = _synthetic_pool()
    strategy = FlatKMeansProportionalRandom(n_clusters=N_TRUE_CLUSTERS)

    selected = strategy.select(embeddings, ids, budget=budget, rng=np.random.default_rng(11))

    assert len(selected) == budget
    assert len(set(selected.tolist())) == len(selected)
    assert set(selected.tolist()) <= set(ids.tolist())

    # With tight, well-separated blobs, k-means(k=N_TRUE_CLUSTERS) recovers
    # the ground-truth grouping, so every selected point's nearest true
    # center must match its own generative blob label -- i.e. it was drawn
    # from (and belongs to) a single coherent cluster, not scattered noise.
    positions = np.searchsorted(ids, selected)
    for pos in positions:
        point = embeddings[pos]
        nearest_center = int(np.argmin(np.linalg.norm(centers - point, axis=1)))
        assert nearest_center == true_labels[pos]


def test_quotas_are_proportional_to_cluster_size_and_sum_to_budget():
    embeddings, ids, true_labels, _ = _synthetic_pool()
    budget = 40
    strategy = FlatKMeansProportionalRandom(n_clusters=N_TRUE_CLUSTERS)

    selected = strategy.select(embeddings, ids, budget=budget, rng=np.random.default_rng(5))
    positions = np.searchsorted(ids, selected)
    picked_labels = true_labels[positions]

    assert len(selected) == budget  # quotas sum to budget

    true_sizes = np.bincount(true_labels, minlength=N_TRUE_CLUSTERS)
    picked_counts = np.bincount(picked_labels, minlength=N_TRUE_CLUSTERS)
    ideal = true_sizes / true_sizes.sum() * budget
    # Largest-remainder rounding keeps every cluster's count within 1 unit
    # of its exact proportional share.
    assert np.all(np.abs(picked_counts - ideal) <= 1.0 + 1e-9)


def test_default_n_clusters_falls_back_to_budget():
    embeddings, ids, _, _ = _synthetic_pool(n_samples=60)
    strategy = FlatKMeansProportionalRandom()  # n_clusters=None -> budget

    selected = strategy.select(embeddings, ids, budget=12, rng=np.random.default_rng(2))

    assert len(selected) == 12
    assert len(set(selected.tolist())) == 12


def test_same_seed_is_deterministic():
    embeddings, ids, _, _ = _synthetic_pool()
    strategy = FlatKMeansProportionalRandom(n_clusters=N_TRUE_CLUSTERS)

    selected_a = strategy.select(embeddings, ids, budget=20, rng=np.random.default_rng(99))
    selected_b = strategy.select(embeddings, ids, budget=20, rng=np.random.default_rng(99))

    np.testing.assert_array_equal(selected_a, selected_b)


def test_budget_at_least_pool_size_returns_all_ids():
    embeddings, ids, _, _ = _synthetic_pool(n_samples=15)
    strategy = FlatKMeansProportionalRandom(n_clusters=3)

    selected = strategy.select(embeddings, ids, budget=1000, rng=np.random.default_rng(3))

    assert set(selected.tolist()) == set(ids.tolist())
