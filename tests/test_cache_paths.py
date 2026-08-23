"""`dalmax.data.datasets.cache_file_path` must key cache files on dataset + Q + pool.

See `.claude/rules/reproducibility.md` and `.specs/quality/known-issues.md`
KI-3: before this helper existed, `utils/data.py` cached SSRAE/VCTex features
and `Y_train` at fixed, unkeyed paths (`results/features_dict_ssrae.pkl`,
`results/features_dict_vctex.pkl`, `results/Y_train.pkl`), so a cache computed
for one dataset/Q was silently reused for a different one. A later gap in the
same helper is that features are extracted only for the *unlabeled pool*,
which depends on `--seed`/`--n_init_labeled`: two runs with the same
`(dataset_folder, q)` but a different pool could otherwise silently share a
cache file. `pool_hash` closes that gap. These tests pin down that different
`(dataset_folder, q, pool_hash)` combinations never collide, that the same
combination is stable/idempotent, that `Y_train` (which has no pool concept)
never carries a pool segment, and that the cache directory is created as a
side effect (mirroring the pre-existing behavior of writing under `results/`,
which is gitignored — see `.claude/rules/data-safety.md`).

Note: every `cache_file_path(...)` call below has the side effect of calling
`os.makedirs("results/cache/", exist_ok=True)` — these tests create that real
directory under the repo's `results/` (gitignored), they do not mock the
filesystem.
"""

from __future__ import annotations

import os

from dalmax.data.datasets import cache_file_path


def test_same_inputs_produce_the_same_path():
    a = cache_file_path("features_ssrae", "daninhas_micro", 13)
    b = cache_file_path("features_ssrae", "daninhas_micro", 13)
    assert a == b


def test_different_dataset_folders_never_collide():
    micro = cache_file_path("features_ssrae", "daninhas_micro", 13)
    full = cache_file_path("features_ssrae", "daninhas_full", 13)
    assert micro != full


def test_different_q_values_never_collide():
    q13 = cache_file_path("features_ssrae", "daninhas_micro", 13)
    q17 = cache_file_path("features_ssrae", "daninhas_micro", 17)
    assert q13 != q17


def test_list_q_is_rendered_deterministically():
    a = cache_file_path("features_vctex", "daninhas_micro", [5, 17])
    b = cache_file_path("features_vctex", "daninhas_micro", [5, 17])
    assert a == b
    assert "5-17" in a


def test_q_none_omits_q_segment_and_still_keys_on_dataset():
    y_micro = cache_file_path("Y_train", "daninhas_micro")
    y_full = cache_file_path("Y_train", "daninhas_full")
    assert "Q" not in os.path.basename(y_micro)
    assert y_micro != y_full


def test_cache_dir_is_created():
    path = cache_file_path("features_ssrae", "daninhas_micro", 13)
    assert os.path.isdir(os.path.dirname(path))


def test_different_extractor_names_never_collide():
    ssrae = cache_file_path("features_ssrae", "daninhas_micro", 13)
    vctex = cache_file_path("features_vctex", "daninhas_micro", 13)
    assert ssrae != vctex


def test_different_pool_hashes_never_collide():
    pool_a = cache_file_path("features_ssrae", "daninhas_micro", 13, pool_hash="aaaaaaaaaaaa")
    pool_b = cache_file_path("features_ssrae", "daninhas_micro", 13, pool_hash="bbbbbbbbbbbb")
    assert pool_a != pool_b


def test_pool_hash_is_deterministic_for_same_inputs():
    a = cache_file_path("features_ssrae", "daninhas_micro", 13, pool_hash="aaaaaaaaaaaa")
    b = cache_file_path("features_ssrae", "daninhas_micro", 13, pool_hash="aaaaaaaaaaaa")
    assert a == b


def test_pool_hash_none_omits_pool_segment():
    path = cache_file_path("features_ssrae", "daninhas_micro", 13)
    assert "pool" not in os.path.basename(path)


def test_y_train_never_carries_a_pool_segment():
    # Y_train has no notion of an unlabeled pool (it covers the whole training
    # split), so it is always called with q=None, and cache_file_path must
    # ignore any pool_hash passed alongside q=None.
    without_pool = cache_file_path("Y_train", "daninhas_micro")
    with_pool_ignored = cache_file_path("Y_train", "daninhas_micro", pool_hash="aaaaaaaaaaaa")
    assert without_pool == with_pool_ignored
    assert "pool" not in os.path.basename(without_pool)
