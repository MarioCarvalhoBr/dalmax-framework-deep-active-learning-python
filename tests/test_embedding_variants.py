"""`dalmax.embeddings.variants.slice_embedding` against the verified SSRAE
layout pinned down in `tests/test_ssrae_embedding_layout.py`: the flat
`54*(Q+1)`-length embedding reshapes to `(9, 6*(Q+1))`, and "spatial"/
"spectral" are column-groups of that matrix (`[:3*(Q+1)]` /
`[3*(Q+1):]`), NOT a naive contiguous half-split.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dalmax.embeddings.variants import slice_embedding
from dalmax.tools.SSRAE.extractor import ColorFeatureExtractor
from dalmax.tools.SSRAE.rnn import RNN
from dalmax.tools.SSRAE.splitter import WindowSplitter


def _prep_channel(channel: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    windows = WindowSplitter().split(channel, window_size=3, padding=True)
    X = torch.from_numpy(windows.T).float()
    XX = X.clone()
    X = X.t_()
    X = torch.divide(torch.subtract(X, torch.mean(X, dim=0)), torch.std(X, dim=0) + 1e-3)
    X = X.t_()
    return X, XX


def _recompute_blocks(image: np.ndarray, q: int) -> list[torch.Tensor]:
    X_R, XX_R = _prep_channel(image[:, :, 0])
    X_G, XX_G = _prep_channel(image[:, :, 1])
    X_B, XX_B = _prep_channel(image[:, :, 2])

    rnn = RNN(Q=q, P=X_R.shape[0], tikhonov=True, lambda_=1e-3)

    rnn.fit(X_R, XX_R)
    beta_R = rnn.beta
    rnn.fit(X_G, XX_G)
    beta_G = rnn.beta
    rnn.fit(X_B, XX_B)
    beta_B = rnn.beta
    rnn.fit(X_R, XX_G)
    beta_S_R = rnn.beta
    rnn.fit(X_G, XX_B)
    beta_S_G = rnn.beta
    rnn.fit(X_B, XX_R)
    beta_S_B = rnn.beta

    return [beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]


def test_full_variant_is_identity(tiny_rgb_image):
    q = 2
    emb = ColorFeatureExtractor(Q=q).extract(tiny_rgb_image).detach().cpu().numpy()

    sliced = slice_embedding(emb, "full", q)

    np.testing.assert_array_equal(sliced, emb)


def test_spatial_and_spectral_recover_the_verified_column_groups(tiny_rgb_image):
    q = 2
    emb = ColorFeatureExtractor(Q=q).extract(tiny_rgb_image).detach().cpu().numpy()

    beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B = (
        b.numpy() for b in _recompute_blocks(tiny_rgb_image, q)
    )
    n_rows, block_width = beta_R.shape

    spatial = slice_embedding(emb, "spatial", q)
    spectral = slice_embedding(emb, "spectral", q)

    expected_spatial = np.hstack([beta_R, beta_G, beta_B])
    expected_spectral = np.hstack([beta_S_R, beta_S_G, beta_S_B])

    np.testing.assert_allclose(
        spatial.reshape(n_rows, 3 * block_width), expected_spatial, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        spectral.reshape(n_rows, 3 * block_width), expected_spectral, rtol=1e-6, atol=1e-6
    )
    assert spatial.size + spectral.size == emb.size


def test_naive_half_split_does_not_match_spatial_or_spectral(tiny_rgb_image):
    q = 2
    emb = ColorFeatureExtractor(Q=q).extract(tiny_rgb_image).detach().cpu().numpy()

    half = emb.size // 2
    spatial = slice_embedding(emb, "spatial", q)
    spectral = slice_embedding(emb, "spectral", q)

    assert not np.allclose(emb[:half], spatial)
    assert not np.allclose(emb[half:], spectral)


def test_batch_of_embeddings_is_sliced_per_row(tiny_rgb_image):
    q = 2
    rng = np.random.default_rng(3)
    image_b = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

    extractor = ColorFeatureExtractor(Q=q)
    emb_a = extractor.extract(tiny_rgb_image).detach().cpu().numpy()
    emb_b = extractor.extract(image_b).detach().cpu().numpy()
    batch = np.stack([emb_a, emb_b], axis=0)

    spatial_batch = slice_embedding(batch, "spatial", q)
    spectral_batch = slice_embedding(batch, "spectral", q)

    assert spatial_batch.shape[0] == 2
    assert spectral_batch.shape[0] == 2
    np.testing.assert_allclose(spatial_batch[0], slice_embedding(emb_a, "spatial", q))
    np.testing.assert_allclose(spatial_batch[1], slice_embedding(emb_b, "spatial", q))
    np.testing.assert_allclose(spectral_batch[0], slice_embedding(emb_a, "spectral", q))
    np.testing.assert_allclose(spectral_batch[1], slice_embedding(emb_b, "spectral", q))


def test_unknown_variant_raises_value_error(tiny_rgb_image):
    q = 2
    emb = ColorFeatureExtractor(Q=q).extract(tiny_rgb_image).detach().cpu().numpy()

    with pytest.raises(ValueError, match="unknown embedding_variant"):
        slice_embedding(emb, "bogus", q)


def test_wrong_dimension_raises_value_error():
    wrong = np.zeros(10, dtype=np.float32)

    with pytest.raises(ValueError, match="54"):
        slice_embedding(wrong, "spatial", q=13)
