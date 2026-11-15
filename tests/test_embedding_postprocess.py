"""Unit tests of the feature post-processing of pool embeddings (ADR 0019):
`dalmax/embeddings/postprocess.py`, its registry (`dalmax/embeddings/registry.py`)
and the `embedding.postprocess` params block (`dalmax/config/schema.py`,
`dalmax/config/loader.py`).

The identity guarantee for the runs executed before the block existed is tested
separately, in `tests/test_postprocess_identity.py`.
"""

from __future__ import annotations

import json
import random
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.config.schema import (
    VALID_POSTPROCESS_METHODS,
    ConfigError,
    EmbeddingConfig,
    PostprocessConfig,
)
from dalmax.embeddings.postprocess import (
    IdentityPostprocessor,
    L2Postprocessor,
    PCAL2Postprocessor,
    PostprocessError,
    ZScorePostprocessor,
)
from dalmax.embeddings.registry import POSTPROCESS_REGISTRY, build_postprocessor

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_KMH = REPO_ROOT / "files_config/campaign/params_kmh.json"


def _pool(n: int = 120, d: int = 32, seed: int = 0) -> np.ndarray:
    """A float32 pool with dimensions of very different scales and offsets (like pooled CNN features)."""
    rng = np.random.default_rng(seed)
    scales = np.geomspace(0.01, 10.0, d)
    return (rng.normal(size=(n, d)) * scales + rng.uniform(0, 5, size=d)).astype(np.float32)


def _fitted(method: str, pool: np.ndarray, pca_dim: int | None = None):
    post = build_postprocessor(PostprocessConfig(method=method, pca_dim=pca_dim))
    post.fit(pool)
    return post


# --- registry and config -------------------------------------------------------------------------


def test_registry_keys_are_the_valid_methods() -> None:
    assert tuple(POSTPROCESS_REGISTRY) == VALID_POSTPROCESS_METHODS == ("none", "l2", "zscore", "pca_l2")
    assert {m: cls.method for m, cls in POSTPROCESS_REGISTRY.items()} == {m: m for m in POSTPROCESS_REGISTRY}


@pytest.mark.parametrize(
    ("config", "cls"),
    [
        (PostprocessConfig(), IdentityPostprocessor),
        (PostprocessConfig(method="l2"), L2Postprocessor),
        (PostprocessConfig(method="zscore"), ZScorePostprocessor),
        (PostprocessConfig(method="pca_l2", pca_dim=8), PCAL2Postprocessor),
    ],
)
def test_build_postprocessor_returns_a_new_unfitted_instance(config: PostprocessConfig, cls: type) -> None:
    first, second = build_postprocessor(config), build_postprocessor(config)
    assert type(first) is cls and first is not second
    assert first.is_identity == (cls is IdentityPostprocessor)
    if cls is PCAL2Postprocessor:
        assert first.pca_dim == 8


def test_build_postprocessor_rejects_an_unknown_method_with_config_error() -> None:
    with pytest.raises(ConfigError, match="Valid: l2, none, pca_l2, zscore"):
        build_postprocessor(types.SimpleNamespace(method="whiten", pca_dim=None))


def test_postprocess_config_default_is_the_identity() -> None:
    assert PostprocessConfig() == PostprocessConfig(method="none", pca_dim=None)
    assert EmbeddingConfig().postprocess == PostprocessConfig()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "whiten"}, "must be one of"),
        ({"method": "pca_l2"}, "missing key: 'pca_dim'"),
        ({"method": "pca_l2", "pca_dim": 0}, "int > 0"),
        ({"method": "pca_l2", "pca_dim": -3}, "int > 0"),
        ({"method": "pca_l2", "pca_dim": True}, "int > 0"),
        ({"method": "l2", "pca_dim": 64}, "must be None"),
        ({"method": "none", "pca_dim": 64}, "must be None"),
        ({"method": "zscore", "pca_dim": 64}, "must be None"),
    ],
)
def test_postprocess_config_fails_fast(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        PostprocessConfig(**kwargs)


def _params_with(tmp_path: Path, postprocess: object, *, drop: bool = False) -> Path:
    params = json.loads(PARAMS_KMH.read_text())
    if drop:
        params["DANINHAS"]["embedding"].pop("postprocess", None)
    else:
        params["DANINHAS"]["embedding"]["postprocess"] = postprocess
    path = tmp_path / "params.json"
    path.write_text(json.dumps(params))
    return path


def _load(path: Path):
    return load_experiment_config(path, "DANINHAS", strategy_name="RepresentationStrategy", seed=1,
                                  n_init_labeled=100, n_query=100, n_round=8, dir_results="results/_unused/",
                                  device="cpu")


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ({"method": "none"}, PostprocessConfig()),
        ({"method": "l2"}, PostprocessConfig(method="l2")),
        ({"method": "zscore", "_comment": "ignored"}, PostprocessConfig(method="zscore")),
        ({"method": "pca_l2", "pca_dim": 64}, PostprocessConfig(method="pca_l2", pca_dim=64)),
        ({"method": "pca_l2", "pca_dim": 64.0}, PostprocessConfig(method="pca_l2", pca_dim=64)),
        ({"method": "l2", "pca_dim": None}, PostprocessConfig(method="l2")),
    ],
)
def test_loader_parses_the_postprocess_block(tmp_path: Path, block: dict, expected: PostprocessConfig) -> None:
    config = _load(_params_with(tmp_path, block))
    assert config.dataset.embedding.postprocess == expected
    assert to_dict(config)["dataset"]["embedding"]["postprocess"] == {
        "method": expected.method, "pca_dim": expected.pca_dim}


def test_loader_absent_block_resolves_to_the_identity(tmp_path: Path) -> None:
    config = _load(_params_with(tmp_path, None, drop=True))
    assert config.dataset.embedding.postprocess == PostprocessConfig()
    assert to_dict(config)["dataset"]["embedding"] == {
        "extractor": "resnet_imagenet", "q": None, "variant": "full",
        "postprocess": {"method": "none", "pca_dim": None}}


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({}, "missing required key"),
        ({"pca_dim": 64}, "missing required key"),
        ({"method": "pca_l2"}, "missing key: 'pca_dim'"),
        ({"method": "l2", "dims": 3}, "unknown key"),
        ({"method": 2}, "must be a string"),
        ({"method": "pca_l2", "pca_dim": "64"}, "must be an integer"),
        ({"method": "pca_l2", "pca_dim": 6.5}, "must be an integer"),
        ({"method": "whiten"}, "must be one of"),
        ("l2", "must be a JSON object"),
        (["l2"], "must be a JSON object"),
    ],
)
def test_loader_rejects_a_malformed_block_with_config_error(tmp_path: Path, block: object, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _load(_params_with(tmp_path, block))


# --- the transforms --------------------------------------------------------------------------------


def test_identity_returns_the_array_itself() -> None:
    pool = _pool()
    post = IdentityPostprocessor()
    summary = post.fit(pool)
    assert post.transform(pool) is pool
    subset = pool[5:9]
    assert post.transform(subset) is subset
    assert (summary.method, summary.n_fit, summary.input_dim, summary.output_dim) == ("none", 120, 32, 32)
    assert summary.explained_variance is None


def test_l2_gives_unit_norms_and_keeps_zero_vectors_zero() -> None:
    pool = _pool()
    pool[7] = 0.0
    post = L2Postprocessor()
    summary = post.fit(pool)
    out = post.transform(pool)
    assert out.shape == pool.shape and out.dtype == np.float32
    norms = np.linalg.norm(out.astype(np.float64), axis=1)
    np.testing.assert_allclose(np.delete(norms, 7), 1.0, atol=1e-6)
    assert np.all(out[7] == 0.0)
    assert (summary.guarded, summary.input_dim, summary.output_dim) == (1, 32, 32)
    # Direction preserved: the same vector up to a positive factor.
    np.testing.assert_allclose(out[3] * np.linalg.norm(pool[3].astype(np.float64)), pool[3], rtol=1e-5)


def test_zscore_gives_zero_mean_unit_variance_on_the_pool_and_zero_for_constant_dims() -> None:
    pool = _pool()
    pool[:, 4] = 3.25  # constant dimension
    pool[:, 9] = 0.0  # constant (dead) dimension
    post = ZScorePostprocessor()
    summary = post.fit(pool)
    out = post.transform(pool).astype(np.float64)
    assert out.shape == pool.shape
    varying = [j for j in range(pool.shape[1]) if j not in (4, 9)]
    np.testing.assert_allclose(out[:, varying].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out[:, varying].std(axis=0), 1.0, atol=1e-5)
    assert np.all(out[:, [4, 9]] == 0.0)
    assert summary.guarded == 2


def test_pca_l2_projects_on_the_leading_components_and_normalises() -> None:
    pool = _pool(n=150, d=40)
    post = PCAL2Postprocessor(pca_dim=8)
    summary = post.fit(pool)
    out = post.transform(pool)
    assert out.shape == (150, 8) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out.astype(np.float64), axis=1), 1.0, atol=1e-6)
    assert (summary.input_dim, summary.output_dim, summary.n_fit) == (40, 8, 150)
    ratios = np.asarray(summary.explained_variance_ratio)
    assert len(ratios) == 8 and np.all(np.diff(ratios) <= 0) and 0 < ratios.sum() <= 1
    # Cross-check against scikit-learn's full-SVD PCA (same subspace; axes equal up to sign).
    reference = PCA(n_components=8, svd_solver="full").fit(pool.astype(np.float64))
    np.testing.assert_allclose(ratios, reference.explained_variance_ratio_, rtol=1e-6)
    projected = reference.transform(pool.astype(np.float64))
    projected /= np.linalg.norm(projected, axis=1, keepdims=True)
    np.testing.assert_allclose(np.abs(out), np.abs(projected), atol=1e-5)
    assert "explained variance of the 8 kept components" in summary.describe()


@pytest.mark.parametrize(("method", "pca_dim"), [("l2", None), ("zscore", None), ("pca_l2", 6)])
def test_transforms_are_deterministic_across_calls_and_instances(method: str, pca_dim: int | None) -> None:
    pool = _pool()
    first = _fitted(method, pool, pca_dim)
    second = _fitted(method, pool, pca_dim)
    np.testing.assert_array_equal(first.transform(pool), first.transform(pool))
    np.testing.assert_array_equal(first.transform(pool), second.transform(pool))


@pytest.mark.parametrize(("method", "pca_dim"), [("l2", None), ("zscore", None), ("pca_l2", 6)])
def test_statistics_of_the_initial_pool_are_reused_on_a_shrunk_pool(method: str, pca_dim: int | None) -> None:
    """Later rounds transform a subset of the fitted pool with the initial statistics: each row is
    transformed as it was in the full pool (no refit on the subset)."""
    pool = _pool()
    post = _fitted(method, pool, pca_dim)
    full = post.transform(pool)
    remaining = np.arange(len(pool))[10::3]
    np.testing.assert_allclose(post.transform(pool[remaining]), full[remaining], rtol=1e-6, atol=1e-7)
    refit = _fitted(method, pool[remaining], pca_dim)
    if method != "l2":  # l2 has no statistics: a refit gives the same rows
        assert not np.allclose(refit.transform(pool[remaining]), full[remaining])


def test_pca_dim_larger_than_the_feature_dimension_or_the_pool_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match=r"min\(120, 32\)"):
        PCAL2Postprocessor(pca_dim=33).fit(_pool(n=120, d=32))
    with pytest.raises(ConfigError, match=r"min\(20, 32\)"):
        PCAL2Postprocessor(pca_dim=21).fit(_pool(n=20, d=32))
    assert PCAL2Postprocessor(pca_dim=20).fit(_pool(n=20, d=32)).output_dim == 20
    with pytest.raises(ConfigError, match="int > 0"):
        PCAL2Postprocessor(pca_dim=0)


@pytest.mark.parametrize(("method", "pca_dim"), [("l2", None), ("zscore", None), ("pca_l2", 4)])
def test_malformed_input_fails_fast(method: str, pca_dim: int | None) -> None:
    pool = _pool()
    unfitted = build_postprocessor(PostprocessConfig(method=method, pca_dim=pca_dim))
    with pytest.raises(PostprocessError, match="fit\\(\\) was not called"):
        unfitted.transform(pool)
    post = _fitted(method, pool, pca_dim)
    with pytest.raises(PostprocessError, match="dimension 31 != fitted 32"):
        post.transform(pool[:, :31])
    with pytest.raises(PostprocessError, match="matrix"):
        post.transform(pool[0])
    bad = pool.copy()
    bad[2, 3] = np.nan
    with pytest.raises(PostprocessError, match="NaN"):
        build_postprocessor(PostprocessConfig(method=method, pca_dim=pca_dim)).fit(bad)
    with pytest.raises(PostprocessError, match="NaN"):
        post.transform(bad)


@pytest.mark.parametrize(("method", "pca_dim"), [("none", None), ("l2", None), ("zscore", None), ("pca_l2", 6)])
def test_fit_and_transform_draw_no_random_number(method: str, pca_dim: int | None) -> None:
    """Round 0 is trained before the first query; the post-processing must not move any global stream
    (NumPy legacy, Python `random`, torch) so that everything after it draws what an identity run draws."""
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)
    before = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone())
    post = build_postprocessor(PostprocessConfig(method=method, pca_dim=pca_dim))
    pool = _pool()
    post.fit(pool)
    post.transform(pool)
    post.transform(pool[::2])
    np.testing.assert_array_equal(np.random.get_state()[1], before[0])
    assert random.getstate() == before[1]
    assert torch.equal(torch.get_rng_state(), before[2])
