"""`SSRAEProvider` must be numerically identical to the legacy per-image
`ColorFeatureExtractor.extract` loop (`utils/data.py::create_feature_maps_ssrae`).

`ColorFeatureExtractor.extract` is deterministic given the same image and Q
(see `tests/test_ssrae_embedding_layout.py`'s determinism note: the RNN's
random weight matrix is a pure LCG seeded only by `Q`/`P`, no global RNG
involved), so calling it twice on the same image must produce the exact
same tensor — this test relies on that property to assert bit-identical
output between the provider and a direct extractor call.
"""

from __future__ import annotations

import numpy as np

from core.tools.SSRAE.extractor import ColorFeatureExtractor
from dalmax.embeddings.ssrae_provider import SSRAEProvider


def test_embed_matches_direct_extractor_call_for_a_single_image(tiny_rgb_image):
    q = 3
    provider = SSRAEProvider(q=q)
    images = np.stack([tiny_rgb_image], axis=0)

    embedded = provider.embed(images)

    direct = ColorFeatureExtractor(Q=q).extract(tiny_rgb_image).detach().cpu().numpy()
    assert embedded.shape == (1, direct.shape[0])
    np.testing.assert_allclose(embedded[0], direct, rtol=1e-6, atol=1e-6)


def test_embed_output_shape_is_54_times_q_plus_1(tiny_rgb_image):
    q = 5
    provider = SSRAEProvider(q=q)
    rng = np.random.default_rng(1)
    other_image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    images = np.stack([tiny_rgb_image, other_image], axis=0)

    embedded = provider.embed(images)

    assert embedded.shape == (2, 54 * (q + 1))
    assert embedded.dtype == np.float32


def test_embed_matches_direct_extractor_call_per_image_in_a_batch(tiny_rgb_image):
    q = 2
    rng = np.random.default_rng(7)
    image_b = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    images = np.stack([tiny_rgb_image, image_b], axis=0)

    provider = SSRAEProvider(q=q)
    embedded = provider.embed(images)

    extractor = ColorFeatureExtractor(Q=q)
    for i, image in enumerate(images):
        direct = extractor.extract(image).detach().cpu().numpy()
        np.testing.assert_allclose(embedded[i], direct, rtol=1e-6, atol=1e-6)


def test_name_is_ssrae():
    assert SSRAEProvider(q=13).name == "ssrae"
