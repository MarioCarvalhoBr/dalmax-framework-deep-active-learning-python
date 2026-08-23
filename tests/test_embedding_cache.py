"""`dalmax.embeddings.cache.EmbeddingCache` and `pool_hash`.

`pool_hash` must equal the inline computation in `utils/data.py`
(`create_feature_maps_ssrae`/`create_feature_maps_vctex`):
`hashlib.sha256(unlabeled_ids.astype(np.int64).tobytes()).hexdigest()[:12]`,
so an identical unlabeled pool hashes identically whether it flows through
the legacy code path or the new `dalmax` code path.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dalmax.embeddings.base import EmbeddingKey
from dalmax.embeddings.cache import EmbeddingCache, compute_pool_embeddings, pool_hash


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
        pool_hash="abc123def456",
    )
    defaults.update(overrides)
    return EmbeddingKey(**defaults)


def test_filename_follows_the_contract_template(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()

    path = cache.path(key)

    assert path.name == "daninhas_micro__ssrae__Q13__full__train__poolabc123def456.pkl"
    assert path.parent == tmp_path


def test_save_then_load_roundtrips(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    features = {0: np.array([1.0, 2.0], dtype=np.float32), 1: np.array([3.0, 4.0], dtype=np.float32)}

    assert not cache.exists(key)
    cache.save(key, features)
    assert cache.exists(key)

    loaded = cache.load(key)
    assert set(loaded) == {0, 1}
    np.testing.assert_array_equal(loaded[0], features[0])
    np.testing.assert_array_equal(loaded[1], features[1])


def test_invalidate_removes_the_file(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    cache.save(key, {0: np.zeros(3)})

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


def test_compute_pool_embeddings_computes_and_saves_on_first_call(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    ids = np.array([10, 20])
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    calls = []

    class _StubProvider:
        def embed(self, imgs):
            calls.append(imgs)
            return np.array([[1.0], [2.0]], dtype=np.float32)

    features = compute_pool_embeddings(_StubProvider(), images, ids, cache, key)

    assert len(calls) == 1
    assert cache.exists(key)
    np.testing.assert_array_equal(features[10], np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(features[20], np.array([2.0], dtype=np.float32))


def test_compute_pool_embeddings_loads_from_cache_on_second_call(tmp_path):
    cache = EmbeddingCache(root=tmp_path)
    key = _make_key()
    ids = np.array([10, 20])
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    call_count = {"n": 0}

    class _StubProvider:
        def embed(self, imgs):
            call_count["n"] += 1
            return np.array([[1.0], [2.0]], dtype=np.float32)

    compute_pool_embeddings(_StubProvider(), images, ids, cache, key)
    compute_pool_embeddings(_StubProvider(), images, ids, cache, key)

    assert call_count["n"] == 1


def test_default_root_is_under_results_cache_embeddings():
    cache = EmbeddingCache()
    assert str(cache.root) == "results/cache/embeddings"
