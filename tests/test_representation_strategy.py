"""Tests for `dalmax.query_strategies.representation.RepresentationStrategy`.

Uses a minimal stub dataset (random 8x8 uint8 images, no real DANINHAS/CIFAR10
data needed) with a real `SSRAEProvider(q=2)` (cheap at this image size) and
`FlatKMeansClosest` selection, so this exercises the actual embedding
extraction + caching + selection pipeline end to end on CPU, fast.

Covers: selected ids are a subset of the currently-unlabeled ids and unique
(the `SelectionStrategy` contract, `dalmax/selection/base.py`), embeddings
are cached across `query()` calls (no mutation of `dataset.features_dict`,
per the module docstring), and the `embedding_variant="spatial"` override
changes the embedding dimensionality actually handed to the selection
strategy (`dalmax/embeddings/variants.py`).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from dalmax.embeddings.cache import EmbeddingCache
from dalmax.embeddings.ssrae_provider import SSRAEProvider
from dalmax.query_strategies.representation import RepresentationStrategy
from dalmax.selection.base import SelectionStrategy
from dalmax.selection.flat_kmeans_closest import FlatKMeansClosest

Q = 2  # small SSRAE hidden-layer size, keeps the 8x8 test images fast to extract
N_POOL = 20
N_LABELED_INITIALLY = 5


class _FakeDataset:
    """Duck-typed stand-in for `utils.data.Data`: only the attributes
    `RepresentationStrategy.query` actually reads."""

    def __init__(self, n_pool: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.X_train = rng.integers(0, 256, size=(n_pool, 8, 8, 3), dtype=np.uint8)
        self.labeled_idxs = np.zeros(n_pool, dtype=bool)
        self.dataset_folder = "fake_micro_dataset"


class _RecordingSelection(SelectionStrategy):
    """Wraps `FlatKMeansClosest` but records the `embeddings` array shape it
    was called with, so tests can assert on the dimensionality actually
    reaching the selection stage without depending on its internals."""

    def __init__(self) -> None:
        self._inner = FlatKMeansClosest()
        self.last_embeddings_shape: tuple[int, ...] | None = None

    def select(self, embeddings, ids, budget, rng):
        self.last_embeddings_shape = embeddings.shape
        return self._inner.select(embeddings, ids, budget, rng)


def _make_config(variant: str = "full"):
    """Minimal duck-typed `ExperimentConfig`: only
    `config.dataset.embedding.variant` is read by `RepresentationStrategy.query`."""
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(embedding=types.SimpleNamespace(variant=variant))
    )


def _make_strategy(tmp_path, *, variant: str = "full", selection=None, dataset=None):
    dataset = dataset if dataset is not None else _FakeDataset(N_POOL)
    dataset.labeled_idxs[:N_LABELED_INITIALLY] = True
    provider = SSRAEProvider(q=Q, device="cpu")
    selection = selection if selection is not None else FlatKMeansClosest()
    cache = EmbeddingCache(root=tmp_path / "embeddings_cache")
    rng = np.random.default_rng(1)
    strategy = RepresentationStrategy(
        dataset,
        net=None,
        config=_make_config(variant=variant),
        logger=None,
        embedding_provider=provider,
        selection=selection,
        rng=rng,
        cache=cache,
    )
    return strategy, dataset


def test_query_returns_unlabeled_unique_ids(tmp_path):
    strategy, dataset = _make_strategy(tmp_path)

    budget = 3
    selected = strategy.query(budget)

    assert isinstance(selected, np.ndarray)
    assert len(selected) == budget
    assert len(set(selected.tolist())) == budget, "selected ids must be unique"

    unlabeled_ids = set(np.where(~dataset.labeled_idxs)[0].tolist())
    assert set(selected.tolist()).issubset(unlabeled_ids)


def test_query_never_mutates_dataset_features_dict(tmp_path):
    """`RepresentationStrategy` must never touch `dataset.features_dict`
    (unlike the legacy `SSRAEKmeansSampling`/`SSLStrategy`, which mutate it
    in place) — the fake dataset here does not even define the attribute,
    so touching it would raise."""
    strategy, dataset = _make_strategy(tmp_path)
    assert not hasattr(dataset, "features_dict")

    strategy.query(3)

    assert not hasattr(dataset, "features_dict")


def test_query_across_rounds_shrinks_available_pool(tmp_path):
    strategy, dataset = _make_strategy(tmp_path)

    first_batch = strategy.query(3)
    dataset.labeled_idxs[first_batch] = True

    second_batch = strategy.query(3)

    # No id can be selected twice across rounds (it became labeled).
    assert set(first_batch.tolist()).isdisjoint(set(second_batch.tolist()))
    unlabeled_ids = set(np.where(~dataset.labeled_idxs)[0].tolist())
    assert set(second_batch.tolist()).issubset(unlabeled_ids)


def test_query_budget_exceeding_pool_returns_all_unlabeled(tmp_path):
    strategy, dataset = _make_strategy(tmp_path)
    unlabeled_ids = set(np.where(~dataset.labeled_idxs)[0].tolist())

    selected = strategy.query(len(unlabeled_ids) + 10)

    assert set(selected.tolist()) == unlabeled_ids


def test_embedding_variant_spatial_reduces_dimensionality(tmp_path):
    recording_full = _RecordingSelection()
    strategy_full, _ = _make_strategy(tmp_path, variant="full", selection=recording_full)
    strategy_full.query(3)

    recording_spatial = _RecordingSelection()
    strategy_spatial, _ = _make_strategy(
        tmp_path, variant="spatial", selection=recording_spatial
    )
    strategy_spatial.query(3)

    full_dim = recording_full.last_embeddings_shape[1]
    spatial_dim = recording_spatial.last_embeddings_shape[1]

    assert full_dim == 54 * (Q + 1)
    assert spatial_dim == 3 * (Q + 1) * 9
    assert spatial_dim < full_dim


def test_non_ssrae_provider_ignores_variant_override(tmp_path):
    """`slice_embedding` is only ever applied when `embedding_provider.name
    == "ssrae"` (see the class docstring: VCTex/ResNet-ImageNet embeddings
    have no such column-group structure). This is exercised indirectly by
    confirming the SSRAE-only guard exists: a non-"full" variant on a
    non-SSRAE provider must not raise or attempt to slice."""
    from dalmax.embeddings.base import EmbeddingProvider

    class _ConstantProvider(EmbeddingProvider):
        name = "vctex"

        def embed(self, images: np.ndarray) -> np.ndarray:
            return np.ones((len(images), 16), dtype=np.float32)

    dataset = _FakeDataset(N_POOL)
    dataset.labeled_idxs[:N_LABELED_INITIALLY] = True
    recording = _RecordingSelection()
    strategy = RepresentationStrategy(
        dataset,
        net=None,
        config=_make_config(variant="spatial"),
        logger=None,
        embedding_provider=_ConstantProvider(q=None, device="cpu"),
        selection=recording,
        rng=np.random.default_rng(1),
        cache=EmbeddingCache(root=tmp_path / "embeddings_cache"),
    )

    strategy.query(3)

    assert recording.last_embeddings_shape[1] == 16


@pytest.mark.parametrize("budget", [0, -1])
def test_query_non_positive_budget_returns_empty(tmp_path, budget):
    strategy, _ = _make_strategy(tmp_path)
    selected = strategy.query(budget)
    assert len(selected) == 0
