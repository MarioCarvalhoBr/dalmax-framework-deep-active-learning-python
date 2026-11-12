"""`dalmax.embeddings.cache.EmbeddingCache`, `pool_hash` and `compute_pool_embeddings`.

`pool_hash` must equal the inline computation in `utils/data.py`
(`create_feature_maps_ssrae`/`create_feature_maps_vctex`):
`hashlib.sha256(unlabeled_ids.astype(np.int64).tobytes()).hexdigest()[:12]`,
so an identical unlabeled pool hashes identically whether it flows through
the legacy code path or the new `dalmax` code path.

KI-47 (2026-11-12): pool ids name images only under one data order, so the key
carries the split's order hash and the payload stores the image paths of the
cached ids. These tests pin down that a cache written for another order, a
pre-KI-47 file, or a truncated file is never used (recompute with a warning),
and that writes are atomic.
"""

from __future__ import annotations

import hashlib
import logging
import pickle

import numpy as np
import pytest

from dalmax.data_order import ordered_paths_hash
from dalmax.embeddings.base import EmbeddingKey
from dalmax.embeddings.cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    EmbeddingCache,
    EmbeddingCacheError,
    compute_pool_embeddings,
    pool_hash,
)

PATHS = ["DATA/x/train/c0/a.png", "DATA/x/train/c1/b.png"]


def _legacy_pool_hash(unlabeled_ids: np.ndarray) -> str:
    """Faithful copy of the inline computation in `utils/data.py`."""
    return hashlib.sha256(unlabeled_ids.astype(np.int64).tobytes()).hexdigest()[:12]


def test_pool_hash_matches_legacy_inline_computation():
    ids = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    assert pool_hash(ids) == _legacy_pool_hash(ids)


def test_pool_hash_is_deterministic():
    ids = np.arange(50)
    assert pool_hash(ids) == pool_hash(ids.copy())


def test_pool_hash_differs_for_different_pools():
    assert pool_hash(np.array([1, 2, 3])) != pool_hash(np.array([1, 2, 4]))


def test_pool_hash_is_order_sensitive_like_the_legacy_bytes_hash():
    # np.where(...)[0] always yields sorted ascending ids in the legacy code
    # path, but the hash itself is over raw bytes, so a differently-ordered
    # array of the same ids must NOT collide (pins down that this is a byte
    # hash of the array as given, not a hash of the id *set*).
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])
    assert pool_hash(a) != pool_hash(b)


def _make_key(**overrides) -> EmbeddingKey:
    defaults = dict(
        dataset="daninhas_micro",
        extractor="ssrae",
        q="13",
        variant="full",
        split="train",
        order_hash="0123456789ab",
        pool_hash="abc123def456",
    )
    defaults.update(overrides)
    return EmbeddingKey(**defaults)


class _StubProvider:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, imgs: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.arange(len(imgs), dtype=np.float32).reshape(-1, 1) + 1.0


def test_filename_follows_the_contract_template(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()

    path = cache.path(key)

    assert path.name == "daninhas_micro__ssrae__Q13__full__train__order0123456789ab__poolabc123def456.pkl"
    assert path.parent == tmp_path


def test_save_then_load_roundtrips(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    ids = np.array([0, 1])
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    assert not cache.exists(key)
    cache.save(key, ids, PATHS, vectors)
    assert cache.exists(key)

    loaded = cache.load(key, ids, PATHS)
    assert set(loaded) == {0, 1}
    np.testing.assert_array_equal(loaded[0], vectors[0])
    np.testing.assert_array_equal(loaded[1], vectors[1])


def test_paths_are_stored_relative_so_the_dataset_location_does_not_matter(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0, 1]), PATHS, np.ones((2, 1)))
    elsewhere = ["/content/dalmax/DATA/x/train/c0/a.png", "/abs/other/train/c1/b.png"]
    assert set(cache.load(key, np.array([0, 1]), elsewhere)) == {0, 1}


def test_load_refuses_a_payload_of_other_images_at_the_same_ids(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0, 1]), PATHS, np.ones((2, 1)))
    with pytest.raises(EmbeddingCacheError, match="images differ"):
        cache.load(key, np.array([0, 1]), list(reversed(PATHS)))
    with pytest.raises(EmbeddingCacheError, match="ids differ"):
        cache.load(key, np.array([0, 2]), PATHS)


def test_load_refuses_a_pre_ki47_payload_and_a_wrong_key(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.root.mkdir(parents=True, exist_ok=True)
    with open(cache.path(key), "wb") as f:  # the version-1 format: a bare {id: vector} dict
        pickle.dump({0: np.zeros(1), 1: np.zeros(1)}, f)
    with pytest.raises(EmbeddingCacheError, match="not a dalmax-embedding-cache payload"):
        cache.load(key, np.array([0, 1]), PATHS)

    other = _make_key(extractor="vctex")
    cache.save(other, np.array([0, 1]), PATHS, np.ones((2, 1)))
    cache.path(other).replace(cache.path(key))  # a file renamed by hand into another key
    with pytest.raises(EmbeddingCacheError, match="written for key"):
        cache.load(key, np.array([0, 1]), PATHS)


def test_payload_carries_format_version_key_ids_and_relative_paths(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0, 1]), PATHS, np.ones((2, 3)))
    with open(cache.path(key), "rb") as f:
        payload = pickle.load(f)
    assert payload["format"] == CACHE_FORMAT and payload["version"] == CACHE_VERSION
    assert payload["key"]["order_hash"] == "0123456789ab"
    assert payload["paths"] == ["c0/a.png", "c1/b.png"]
    assert payload["vectors"].shape == (2, 3)


def test_save_rejects_misaligned_inputs(tmp_path):
    with pytest.raises(ValueError, match="must align"):
        EmbeddingCache(root=tmp_path).save(_make_key(), np.array([0, 1]), PATHS[:1], np.ones((2, 1)))


def test_invalidate_removes_the_file(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0]), PATHS[:1], np.zeros((1, 3)))

    cache.invalidate(key)

    assert not cache.exists(key)


def test_invalidate_is_a_no_op_when_file_missing(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.invalidate(key)  # must not raise


def test_different_keys_never_collide_on_disk(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    ssrae_key = _make_key(extractor="ssrae")
    vctex_key = _make_key(extractor="vctex", q="5-17")

    assert cache.path(ssrae_key) != cache.path(vctex_key)


def test_another_image_order_is_another_file(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    order_a = ordered_paths_hash(PATHS)
    order_b = ordered_paths_hash(list(reversed(PATHS)))
    assert order_a != order_b
    assert cache.path(_make_key(order_hash=order_a)) != cache.path(_make_key(order_hash=order_b))


def test_a_pre_ki47_file_name_is_never_consulted(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    legacy = tmp_path / f"{key.dataset}__{key.extractor}__Q{key.q}__{key.variant}__{key.split}__pool{key.pool_hash}.pkl"
    with open(legacy, "wb") as f:
        pickle.dump({0: np.full(1, 99.0), 1: np.full(1, 99.0)}, f)
    provider = _StubProvider()

    features = compute_pool_embeddings(provider, np.zeros((2, 2, 2, 3)), np.array([0, 1]), PATHS, cache, key)

    assert provider.calls == 1 and features[0][0] == 1.0  # recomputed, the legacy 99.0 never read
    assert legacy.exists()  # left untouched


def test_compute_pool_embeddings_computes_and_saves_on_first_call(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    ids = np.array([10, 20])
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    provider = _StubProvider()

    features = compute_pool_embeddings(provider, images, ids, PATHS, cache, key)

    assert provider.calls == 1
    assert cache.exists(key)
    np.testing.assert_array_equal(features[10], np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(features[20], np.array([2.0], dtype=np.float32))


def test_compute_pool_embeddings_loads_from_cache_on_second_call(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    ids = np.array([10, 20])
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    provider = _StubProvider()

    first = compute_pool_embeddings(provider, images, ids, PATHS, cache, key)
    second = compute_pool_embeddings(provider, images, ids, PATHS, cache, key)

    assert provider.calls == 1
    for i in (10, 20):
        np.testing.assert_array_equal(first[i], second[i])


def test_foreign_payload_is_recomputed_with_a_warning(tmp_path, caplog):
    """A file under the right name but holding the features of other images (another order) is never used."""
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0, 1]), list(reversed(PATHS)), np.full((2, 1), 99.0))
    provider = _StubProvider()

    with caplog.at_level(logging.WARNING):
        features = compute_pool_embeddings(provider, np.zeros((2, 2, 2, 3)), np.array([0, 1]), PATHS, cache, key)

    assert provider.calls == 1 and features[0][0] == 1.0
    assert "not usable, recomputing" in caplog.text and "images differ" in caplog.text
    assert set(cache.load(key, np.array([0, 1]), PATHS)) == {0, 1}  # rewritten for the right images


def test_truncated_file_is_recomputed_with_a_warning(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0, 1]), PATHS, np.ones((2, 64)))
    raw = cache.path(key).read_bytes()
    cache.path(key).write_bytes(raw[: len(raw) // 2])  # e.g. a session killed mid-write before the fix
    messages: list[str] = []

    class _Logger:
        def warning(self, msg: str) -> None:
            messages.append(msg)

    provider = _StubProvider()
    features = compute_pool_embeddings(
        provider, np.zeros((2, 2, 2, 3)), np.array([0, 1]), PATHS, cache, key, logger=_Logger()
    )

    assert provider.calls == 1 and set(features) == {0, 1}
    assert len(messages) == 1 and "unreadable embedding cache" in messages[0]
    assert set(cache.load(key, np.array([0, 1]), PATHS)) == {0, 1}


def test_write_is_atomic_on_failure(tmp_path, monkeypatch):
    """A write that dies midway leaves neither a partial final file nor a temporary file, and keeps
    a previous complete file intact."""
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, np.array([0, 1]), PATHS, np.ones((2, 1)))
    before = cache.path(key).read_bytes()

    def failing_dump(obj, f, protocol=None):
        f.write(b"partial bytes")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("dalmax.embeddings.cache.pickle.dump", failing_dump)
    with pytest.raises(OSError):
        cache.save(key, np.array([0, 1]), PATHS, np.zeros((2, 1)))

    assert cache.path(key).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [cache.path(key).name]  # no leftover temporary file


def test_default_root_is_under_results_cache_embeddings():
    cache = EmbeddingCache()
    assert str(cache.root) == "results/cache/embeddings"
