"""Common interface every embedding provider implements.

See `.specs/architecture/target-architecture.md` §2 (`embeddings/base.py`)
and §4 (embedding cache contract). `EmbeddingKey` is the tuple that keys the
on-disk cache in `dalmax/embeddings/cache.py`; `EmbeddingProvider` is the
abstract base every concrete provider (`ssrae_provider.py`,
`vctex_provider.py`, `resnet_imagenet_provider.py`) subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


def format_q(q: Any) -> str:
    """Render a provider's `q` hyperparameter as a filename-safe string.

    Mirrors the (still-live) `dalmax.data.datasets.cache_file_path`'s `q_str` convention so cache
    keys stay human-readable and stable: a `list`/`tuple` (e.g. VCTex's
    `[5, 17]`) is rendered as `"5-17"`; anything else (an `int`, or `None`
    for the ResNet-ImageNet provider, which has no `Q` hyperparameter) is
    rendered via `str(...)`.
    """
    if isinstance(q, (list, tuple)):
        return "-".join(str(v) for v in q)
    return str(q)


@dataclass(frozen=True)
class EmbeddingKey:
    """Identifies one cached embedding artifact.

    `q` is already formatted (see `format_q`) so the key is directly usable
    to build a cache filename without callers needing to know each
    provider's `q` type.
    """

    dataset: str
    extractor: str
    q: str
    variant: str
    split: str
    pool_hash: str


class EmbeddingProvider(ABC):
    """Abstract base for an image -> embedding extractor.

    Subclasses wrap a vendored feature extractor (SSRAE, VCTex) or a
    torchvision model (ResNet-ImageNet) behind a uniform `embed` method, so
    callers (selection strategies, the embedding cache) never need to know
    which concrete extractor produced a given embedding matrix.
    """

    name: str

    def __init__(self, q: Any, device: str = "cpu") -> None:
        """
        Args:
            q: The extractor's hyperparameter — an `int` for SSRAE (`Q=13`),
                a `tuple[int, ...]`/`list[int]` for VCTex (`Q=(5, 17)`), or
                `None` for ResNet-ImageNet (fixed 2048-d penultimate layer,
                no `Q` hyperparameter).
            device: `"cpu"` or `"cuda"` — where the extractor's torch
                computation runs. Providers whose underlying implementation
                is CPU-only (SSRAE) accept and store this for interface
                uniformity but ignore it; see each provider's docstring.
        """
        self.q = q
        self.device = device

    @abstractmethod
    def embed(self, images: np.ndarray) -> np.ndarray:
        """Compute embeddings for a batch of images.

        Args:
            images: `(N, H, W, 3)` `uint8` ndarray.

        Returns:
            `(N, D)` `float32` ndarray, one embedding row per input image,
            in the same order as `images`.
        """
        raise NotImplementedError

    def key(self, dataset: str, split: str, pool_hash: str, variant: str = "full") -> EmbeddingKey:
        """Build the `EmbeddingKey` this provider's output should be cached
        under, for the given `dataset`/`split`/unlabeled-pool identity
        (`pool_hash`, see `dalmax/embeddings/cache.py::pool_hash`)."""
        return EmbeddingKey(
            dataset=dataset,
            extractor=self.name,
            q=format_q(self.q),
            variant=variant,
            split=split,
            pool_hash=pool_hash,
        )
