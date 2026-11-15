"""`EMBEDDING_REGISTRY`: name -> `EmbeddingProvider` class, and
`POSTPROCESS_REGISTRY`: method -> `FeaturePostprocessor` class (ADR 0019).

Replaces the pattern in `.claude/rules/code-quality.md` ("registry pattern
over if/elif chains") for embedding providers — the counterpart of
`dalmax/selection/registry.py`'s `SELECTION_REGISTRY`.
"""

from __future__ import annotations

from typing import Any

from dalmax.config.schema import ConfigError

from .base import EmbeddingProvider
from .postprocess import (
    FeaturePostprocessor,
    IdentityPostprocessor,
    L2Postprocessor,
    PCAL2Postprocessor,
    ZScorePostprocessor,
)
from .resnet_imagenet_provider import ResNetImageNetProvider
from .ssrae_provider import SSRAEProvider
from .vctex_provider import VCTexProvider

EMBEDDING_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "ssrae": SSRAEProvider,
    "vctex": VCTexProvider,
    "resnet_imagenet": ResNetImageNetProvider,
}

# Keys = `dalmax.config.schema.VALID_POSTPROCESS_METHODS` (checked by
# `tests/test_embedding_postprocess.py`).
POSTPROCESS_REGISTRY: dict[str, type[FeaturePostprocessor]] = {
    IdentityPostprocessor.method: IdentityPostprocessor,
    L2Postprocessor.method: L2Postprocessor,
    ZScorePostprocessor.method: ZScorePostprocessor,
    PCAL2Postprocessor.method: PCAL2Postprocessor,
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


def build_postprocessor(config: Any) -> FeaturePostprocessor:
    """A new, unfitted `FeaturePostprocessor` for a `dalmax.config.schema.PostprocessConfig`
    (`config.method`, `config.pca_dim`).

    Raises:
        ConfigError: if `config.method` is not registered, listing the valid methods.
    """
    try:
        cls = POSTPROCESS_REGISTRY[config.method]
    except KeyError as exc:
        valid = ", ".join(sorted(POSTPROCESS_REGISTRY))
        raise ConfigError(f"Unknown embedding post-processing method {config.method!r}. Valid: {valid}") from exc
    return cls.from_config(config)
