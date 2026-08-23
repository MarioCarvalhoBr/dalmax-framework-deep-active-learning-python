"""`EMBEDDING_REGISTRY`: name -> `EmbeddingProvider` class.

Replaces the pattern in `.claude/rules/code-quality.md` ("registry pattern
over if/elif chains") for embedding providers — the counterpart of
`dalmax/selection/registry.py`'s `SELECTION_REGISTRY`.
"""

from __future__ import annotations

from .base import EmbeddingProvider
from .resnet_imagenet_provider import ResNetImageNetProvider
from .ssrae_provider import SSRAEProvider
from .vctex_provider import VCTexProvider

EMBEDDING_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "ssrae": SSRAEProvider,
    "vctex": VCTexProvider,
    "resnet_imagenet": ResNetImageNetProvider,
}


def get_embedding_provider(name: str) -> type[EmbeddingProvider]:
    """Look up an `EmbeddingProvider` class by its registry name.

    Raises:
        KeyError: if `name` is not registered, listing the valid names.
    """
    try:
        return EMBEDDING_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(EMBEDDING_REGISTRY))
        raise KeyError(f"Unknown embedding provider {name!r}. Valid names: {valid}") from exc
