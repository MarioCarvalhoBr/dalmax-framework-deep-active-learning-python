"""Explicitly-keyed on-disk cache for embedding dictionaries.

Replaces the two fixed, unkeyed pickle paths
(`results/features_dict_ssrae.pkl`, `results/features_dict_vctex.pkl`) with
one file per `(dataset, extractor, Q, variant, split, pool_hash)` key — see
`.specs/architecture/target-architecture.md` §4 and
`.claude/rules/reproducibility.md` ("Embedding cache discipline"). Only the
`"full"` variant is ever written; `spatial`/`spectral` are always derived
from a loaded `"full"` cache entry via `dalmax/embeddings/variants.py`, so
running all three ablation-6.1 variants back-to-back extracts SSRAE features
exactly once.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import EmbeddingKey, EmbeddingProvider

DEFAULT_CACHE_ROOT = "results/cache/embeddings"


def pool_hash(unlabeled_ids: np.ndarray) -> str:
    """Hash the identity of the current unlabeled pool.

    Must equal `utils/data.py`'s inline pool-hash computation
    (`hashlib.sha256(unlabeled_ids.astype(np.int64).tobytes()).hexdigest()[:12]`,
    see `create_feature_maps_ssrae`/`create_feature_maps_vctex`) so that an
    identical unlabeled pool produced by the legacy code path and by the new
    `dalmax` code path hashes to the same value.
    """
    ids = np.asarray(unlabeled_ids).astype(np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest()[:12]


class EmbeddingCache:
    """One pickle file per `EmbeddingKey`, under `root`.

    Filename: `{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{pool_hash}.pkl`.
    Always store/load the `"full"` variant — callers wanting `"spatial"`/
    `"spectral"` should load the `"full"` entry and slice it via
    `dalmax.embeddings.variants.slice_embedding`, never save a
    non-`"full"` key.
    """

    def __init__(self, root: str | Path = DEFAULT_CACHE_ROOT) -> None:
        self.root = Path(root)

    def path(self, key: EmbeddingKey) -> Path:
        filename = (
            f"{key.dataset}__{key.extractor}__Q{key.q}__{key.variant}"
            f"__{key.split}__pool{key.pool_hash}.pkl"
        )
        return self.root / filename

    def exists(self, key: EmbeddingKey) -> bool:
        return self.path(key).exists()

    def load(self, key: EmbeddingKey) -> dict[int, np.ndarray]:
        with open(self.path(key), "rb") as f:
            return pickle.load(f)

    def save(self, key: EmbeddingKey, features: dict[int, np.ndarray]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(key)
        with open(path, "wb") as f:
            pickle.dump(features, f)
        return path

    def invalidate(self, key: EmbeddingKey) -> None:
        path = self.path(key)
        if path.exists():
            path.unlink()


def compute_pool_embeddings(
    provider: EmbeddingProvider,
    images: np.ndarray,
    ids: np.ndarray,
    cache: EmbeddingCache,
    key: EmbeddingKey,
) -> dict[int, np.ndarray]:
    """Return `{id: embedding}` for `key`, loading from `cache` if present,
    otherwise computing via `provider.embed(images)` (aligned with `ids`)
    and saving the result before returning it."""
    if cache.exists(key):
        return cache.load(key)

    vectors = provider.embed(images)
    features = {int(img_id): vectors[idx] for idx, img_id in enumerate(ids)}
    cache.save(key, features)
    return features
