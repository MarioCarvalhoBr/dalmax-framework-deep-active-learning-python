"""Tests for `dalmax.selection.registry`."""

from __future__ import annotations

import pytest

from dalmax.selection.base import SelectionStrategy
from dalmax.selection.flat_kmeans_closest import FlatKMeansClosest
from dalmax.selection.flat_kmeans_proportional import FlatKMeansProportionalRandom
from dalmax.selection.hierarchical_kmeans import HierarchicalKMeansSelection
from dalmax.selection.registry import SELECTION_REGISTRY, get_selection_strategy

EXPECTED = {
    "flat_closest": FlatKMeansClosest,
    "flat_proportional": FlatKMeansProportionalRandom,
    "hierarchical": HierarchicalKMeansSelection,
}


def test_registry_has_exactly_the_expected_names():
    assert set(SELECTION_REGISTRY) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_get_selection_strategy_resolves_expected_class(name):
    strategy_cls = get_selection_strategy(name)
    assert strategy_cls is EXPECTED[name]
    assert issubclass(strategy_cls, SelectionStrategy)


def test_get_selection_strategy_rejects_unknown_name_with_helpful_message():
    with pytest.raises(KeyError) as exc_info:
        get_selection_strategy("__not_a_real_strategy__")

    message = str(exc_info.value)
    for name in EXPECTED:
        assert name in message
