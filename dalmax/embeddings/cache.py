"""Explicitly-keyed on-disk cache for pool embeddings.

Replaces the two fixed, unkeyed pickle paths
(`results/features_dict_ssrae.pkl`, `results/features_dict_vctex.pkl`) with
one file per `EmbeddingKey` -- see `.specs/architecture/target-architecture.md`
§4 and `.claude/rules/reproducibility.md` ("Embedding cache discipline"). Only
the `"full"` variant is ever written; `spatial`/`spectral` are always derived
from a loaded `"full"` cache entry via `dalmax/embeddings/variants.py`, so
running all three ablation-6.1 variants back-to-back extracts SSRAE features
exactly once.

Image identity (KI-47, 2026-11-12). Pool ids are positions in the loader's
image list, so a cache keyed on ids alone was hit by later sessions whose
(then unsorted) listing put other images at the same ids: the features at id `i`
belonged to another image, silently. Two independent guards now make a stale or
foreign cache unusable:

1. the key carries `order_hash` = `dalmax.data_order.ordered_paths_hash` of the
   split's ordered image list, so any other order (or any file written before
   this fix, whose name has no `__order...` segment) is a different file name;
2. the payload stores the relative paths of the cached ids
   (`dalmax.data_order.relative_image_path`) together with the key, the ids and
   a format version, and `EmbeddingCache.load` refuses (`EmbeddingCacheError`)
   anything that does not match the caller's ids and paths exactly.

`compute_pool_embeddings` treats a refused or unreadable (e.g. truncated) file
as a miss: it logs a warning and recomputes. Writes are atomic (temporary file
in the same directory + `os.replace`), so an interrupted write never leaves a
half file under the final name.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pickle
from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import numpy as np

from dalmax.data_order import relative_image_paths

if TYPE_CHECKING:
    from .base import EmbeddingKey, EmbeddingProvider

DEFAULT_CACHE_ROOT = "results/cache/embeddings"
CACHE_FORMAT = "dalmax-embedding-cache"
# Version 2 (2026-11-12, KI-47): ids + relative paths + vectors + key. Version 1 was a bare
# `{id: vector}` pickle without any image identity; such files are never loaded.
CACHE_VERSION = 2

_MODULE_LOGGER = logging.getLogger(__name__)


class EmbeddingCacheError(ValueError):
    """A cache file that must not be used: unreadable, another format/version, or features of other images."""


def pool_hash(unlabeled_ids: np.ndarray) -> str:
    """Hash the identity of the current unlabeled pool (its ids, in order).

    Must equal the now-deleted `Data.create_feature_maps_ssrae`/`_vctex`'s inline pool-hash computation
    (`hashlib.sha256(unlabeled_ids.astype(np.int64).tobytes()).hexdigest()[:12]`)
    so that an identical unlabeled pool produced by the legacy code path and by the new
    `dalmax` code path hashes to the same value. Ids alone do not identify images: see
    the module docstring and `EmbeddingKey.order_hash`.
    """
    ids = np.asarray(unlabeled_ids).astype(np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest()[:12]


class EmbeddingCache:
    """One pickle file per `EmbeddingKey`, under `root`.

    Filename: `{dataset}__{extractor}__Q{q}__{variant}__{split}__order{order_hash}__pool{pool_hash}.pkl`.
    Always store/load the `"full"` variant -- callers wanting `"spatial"`/
    `"spectral"` should load the `"full"` entry and slice it via
    `dalmax.embeddings.variants.slice_embedding`, never save a
    non-`"full"` key.
    """

    def __init__(self, root: str | Path = DEFAULT_CACHE_ROOT) -> None:
        self.root = Path(root)

    def path(self, key: EmbeddingKey) -> Path:
        filename = (
            f"{key.dataset}__{key.extractor}__Q{key.q}__{key.variant}"
            f"__{key.split}__order{key.order_hash}__pool{key.pool_hash}.pkl"
        )
        return self.root / filename

    def exists(self, key: EmbeddingKey) -> bool:
        return self.path(key).exists()

    def load(
        self, key: EmbeddingKey, ids: np.ndarray, paths: Sequence[str | PurePath]
    ) -> dict[int, np.ndarray]:
        """`{id: embedding}` of a file written for exactly `key`, `ids` and the images at `paths`.

        Raises `EmbeddingCacheError` if the file cannot be read or does not match
        (format, version, key, ids, relative paths, row count); never returns
        features of other images.
        """
        path = self.path(key)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 - any unpickling failure means "unusable"
            raise EmbeddingCacheError(f"unreadable embedding cache {path}: {type(exc).__name__}: {exc}") from exc
        _validate(payload, key, np.asarray(ids, dtype=np.int64), relative_image_paths(paths), path)
        vectors = payload["vectors"]
        return {int(img_id): vectors[row] for row, img_id in enumerate(payload["ids"])}

    def save(
        self, key: EmbeddingKey, ids: np.ndarray, paths: Sequence[str | PurePath], vectors: np.ndarray
    ) -> Path:
        """Write the embeddings of the images at `paths` (rows of `vectors`, aligned with `ids`) atomically."""
        ids = np.asarray(ids, dtype=np.int64)
        rel_paths = relative_image_paths(paths)
        if not (len(ids) == len(rel_paths) == len(vectors)):
            raise ValueError(
                f"EmbeddingCache.save: {len(ids)} ids, {len(rel_paths)} paths and {len(vectors)} vectors must align"
            )
        payload = {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "key": dataclasses.asdict(key),
            "ids": ids,
            "paths": rel_paths,
            "vectors": np.asarray(vectors),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(key)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def invalidate(self, key: EmbeddingKey) -> None:
        path = self.path(key)
        if path.exists():
            path.unlink()


def _validate(payload: Any, key: EmbeddingKey, ids: np.ndarray, rel_paths: list[str], path: Path) -> None:
    """Raise `EmbeddingCacheError` unless `payload` is a current-format entry for exactly these images."""
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise EmbeddingCacheError(f"{path}: not a {CACHE_FORMAT} payload (a pre-KI-47 cache file?)")
    if payload.get("version") != CACHE_VERSION:
        raise EmbeddingCacheError(f"{path}: version {payload.get('version')!r} != {CACHE_VERSION}")
    if payload.get("key") != dataclasses.asdict(key):
        raise EmbeddingCacheError(f"{path}: written for key {payload.get('key')!r}, not {dataclasses.asdict(key)!r}")
    stored_ids = np.asarray(payload.get("ids"), dtype=np.int64)
    if stored_ids.shape != ids.shape or not np.array_equal(stored_ids, ids):
        raise EmbeddingCacheError(f"{path}: cached ids differ from the requested pool")
    if list(payload.get("paths", ())) != rel_paths:
        raise EmbeddingCacheError(f"{path}: cached images differ from the images at the requested ids")
    vectors = payload.get("vectors")
    if not isinstance(vectors, np.ndarray) or vectors.shape[0] != len(ids):
        raise EmbeddingCacheError(f"{path}: vectors missing or not one row per id")


def compute_pool_embeddings(
    provider: EmbeddingProvider,
    images: np.ndarray,
    ids: np.ndarray,
    paths: Sequence[str | PurePath],
    cache: EmbeddingCache,
    key: EmbeddingKey,
    logger: logging.Logger | None = None,
) -> dict[int, np.ndarray]:
    """Return `{id: embedding}` for the images at `paths` (aligned with `ids` and `images`).

    Loads from `cache` when a matching entry exists; otherwise (no file, or a file
    `EmbeddingCache.load` refuses: unreadable, truncated, foreign) computes via
    `provider.embed(images)`, logs why when a file was refused, and saves the result
    atomically before returning it.
    """
    log = logger if logger is not None else _MODULE_LOGGER
    if cache.exists(key):
        try:
            return cache.load(key, ids, paths)
        except EmbeddingCacheError as exc:
            log.warning(f"Embedding cache not usable, recomputing: {exc}")

    vectors = provider.embed(images)
    cache.save(key, ids, paths, vectors)
    return {int(img_id): vectors[row] for row, img_id in enumerate(np.asarray(ids))}
