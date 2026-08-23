"""Registry mapping selection strategy names to classes.

Dict-based registry, per `.claude/rules/code-quality.md` ("registry pattern
over if/elif chains") — mirrors the pattern documented for
`dalmax/embeddings/registry.py` in
`.specs/architecture/target-architecture.md`.
"""

from __future__ import annotations

from dalmax.selection.base import SelectionStrategy
from dalmax.selection.flat_kmeans_closest import FlatKMeansClosest
from dalmax.selection.flat_kmeans_proportional import FlatKMeansProportionalRandom
from dalmax.selection.hierarchical_kmeans import HierarchicalKMeansSelection

SELECTION_REGISTRY: dict[str, type[SelectionStrategy]] = {
    "flat_closest": FlatKMeansClosest,
    "flat_proportional": FlatKMeansProportionalRandom,
    "hierarchical": HierarchicalKMeansSelection,
}


def get_selection_strategy(name: str) -> type[SelectionStrategy]:
    """Look up a `SelectionStrategy` class by its registry name.

    Raises
    ------
    KeyError
        If `name` is not a registered selection strategy; the message lists
        the valid names.
    """
    try:
        return SELECTION_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(SELECTION_REGISTRY))
        raise KeyError(f"Unknown selection strategy {name!r}. Valid names: {valid}") from exc
