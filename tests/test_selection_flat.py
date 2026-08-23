"""Tests for `dalmax.selection.flat_kmeans_closest.FlatKMeansClosest`.

Key acceptance test: different `rng` seeds must produce different
selections on the same data — this is the regression test for the
hardcoded `random_state=3` bug in
`core/query_strategies/ssrae_kmeans_sampling.py:23` that this class fixes
(see `.specs/architecture/target-architecture.md` §6).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from dalmax.selection.flat_kmeans_closest import FlatKMeansClosest


def _synthetic_pool(n_samples: int = 150, centers: int = 6, seed: int = 0):
    embeddings, _ = make_blobs(
        n_samples=n_samples, centers=centers, n_features=8, cluster_std=1.5, random_state=seed
    )
    embeddings = embeddings.astype(np.float32)
    ids = np.arange(1000, 1000 + n_samples)
    return embeddings, ids


def test_different_seeds_give_different_selections():
    """Regression test for the `random_state=3` fix: seeds 1 vs 2 on the
    same data must not select the same batch."""
    embeddings, ids = _synthetic_pool()
    strategy = FlatKMeansClosest()

    selected_1 = strategy.select(embeddings, ids, budget=10, rng=np.random.default_rng(1))
    selected_2 = strategy.select(embeddings, ids, budget=10, rng=np.random.default_rng(2))

    assert set(selected_1.tolist()) != set(selected_2.tolist())


def test_same_seed_is_deterministic():
    embeddings, ids = _synthetic_pool()
    strategy = FlatKMeansClosest()

    selected_a = strategy.select(embeddings, ids, budget=10, rng=np.random.default_rng(42))
    selected_b = strategy.select(embeddings, ids, budget=10, rng=np.random.default_rng(42))

    np.testing.assert_array_equal(selected_a, selected_b)


@pytest.mark.parametrize("budget", [1, 5, 10, 40])
def test_selection_is_unique_subset_with_expected_size(budget):
    embeddings, ids = _synthetic_pool()
    strategy = FlatKMeansClosest()

    selected = strategy.select(embeddings, ids, budget=budget, rng=np.random.default_rng(7))

    assert len(selected) == min(budget, len(ids))
    assert len(set(selected.tolist())) == len(selected)  # no duplicates
    assert set(selected.tolist()) <= set(ids.tolist())


def test_budget_at_least_pool_size_returns_all_ids():
    embeddings, ids = _synthetic_pool(n_samples=20, centers=3)
    strategy = FlatKMeansClosest()

    selected = strategy.select(embeddings, ids, budget=1000, rng=np.random.default_rng(3))

    assert len(selected) == len(ids)
    assert set(selected.tolist()) == set(ids.tolist())


def test_budget_zero_returns_empty():
    embeddings, ids = _synthetic_pool(n_samples=20, centers=3)
    strategy = FlatKMeansClosest()

    selected = strategy.select(embeddings, ids, budget=0, rng=np.random.default_rng(3))

    assert len(selected) == 0
