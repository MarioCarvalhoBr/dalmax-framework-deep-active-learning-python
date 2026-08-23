"""`ResNetImageNetProvider` (ablation 6.3's "without representation module"
row, `.specs/experiments/ablation-study.md` §6.3).

Marked `slow`: constructing the provider loads (and, on a machine without a
warm cache, downloads ~98 MB of) `ResNet50_Weights.IMAGENET1K_V1` — see
`dalmax/embeddings/resnet_imagenet_provider.py`'s docstring. On this
development machine the weights are already cached under
`~/.cache/torch/hub/checkpoints/` from Phase 1, so this test runs offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from dalmax.embeddings.resnet_imagenet_provider import ResNetImageNetProvider


@pytest.mark.slow
def test_embed_a_random_batch_on_cpu_returns_2048_d_features():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(4, 64, 64, 3), dtype=np.uint8)

    provider = ResNetImageNetProvider(device="cpu")
    embedded = provider.embed(images)

    assert embedded.shape == (4, 2048)
    assert embedded.dtype == np.float32


@pytest.mark.slow
def test_name_and_embedding_dim():
    assert ResNetImageNetProvider.name == "resnet_imagenet"
    assert ResNetImageNetProvider.embedding_dim == 2048


@pytest.mark.slow
def test_embed_is_deterministic_across_calls_in_eval_mode():
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)

    provider = ResNetImageNetProvider(device="cpu")
    first = provider.embed(images)
    second = provider.embed(images)

    np.testing.assert_allclose(first, second)
