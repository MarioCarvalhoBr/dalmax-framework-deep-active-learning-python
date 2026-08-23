"""Selection strategies: decide which pool ids to query next given embeddings.

See `SelectionStrategy` (`base.py`) for the interface and
`.specs/architecture/target-architecture.md` §6 for the design rationale.
"""

from dalmax.selection.base import SelectionStrategy
from dalmax.selection.flat_kmeans_closest import FlatKMeansClosest
from dalmax.selection.flat_kmeans_proportional import FlatKMeansProportionalRandom
from dalmax.selection.hierarchical_kmeans import HierarchicalKMeansSelection
from dalmax.selection.registry import SELECTION_REGISTRY, get_selection_strategy

__all__ = [
    "SelectionStrategy",
    "FlatKMeansClosest",
    "FlatKMeansProportionalRandom",
    "HierarchicalKMeansSelection",
    "SELECTION_REGISTRY",
    "get_selection_strategy",
]
