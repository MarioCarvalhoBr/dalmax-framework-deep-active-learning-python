"""`utils.data.cache_file_path` must key cache files on dataset + Q.

See `.claude/rules/reproducibility.md` and `.specs/quality/known-issues.md`
KI-3: before this helper existed, `utils/data.py` cached SSRAE/VCTex features
and `Y_train` at fixed, unkeyed paths (`results/features_dict_ssrae.pkl`,
`results/features_dict_vctex.pkl`, `results/Y_train.pkl`), so a cache computed
for one dataset/Q was silently reused for a different one. These tests pin
down that different `(dataset_folder, q)` combinations never collide, that the
same combination is stable/idempotent, and that the cache directory is
created as a side effect (mirroring the pre-existing behavior of writing
under `results/`, which is gitignored — see `.claude/rules/data-safety.md`).
"""

from __future__ import annotations

import os

from utils.data import cache_file_path


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
