"""SSRAE embedding layout: what `ColorFeatureExtractor.extract` actually produces.

Background
----------
`core/tools/SSRAE/extractor.py::ColorFeatureExtractor.extract` computes six
"beta" matrices (one randomized-network fit per spatial channel R, G, B, and
per cross-channel/spectral pair R->G, G->B, B->R), each of shape
``(r, Q+1)`` where ``r`` is the window size squared (9 for the hardcoded
``window_size=3``). It then does::

    beta = torch.hstack([beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]).reshape((1, -1))

`prompt-master.md` (Deliverable A/§6.1) and the shared scaffolding context
describe this as a block-contiguous concatenation: "first half = spatial
signatures (R,G,B), second half = spectral signatures (RG,GB,BR)", i.e. the
documented expectation is that ``emb[:len//2] == concat(beta_R, beta_G, beta_B)``.

That is NOT what the code does. `torch.hstack` on 2-D tensors concatenates
along axis=1 (columns), producing a single ``(r, 6*(Q+1))`` matrix; the
subsequent ``.reshape((1, -1))`` flattens it row-major, so the six
sub-blocks are interleaved *per row* rather than laid out as six
contiguous chunks. This was verified empirically below by faithfully
reproducing the extractor's internal computation.

Determinism note
-----------------
`core/tools/SSRAE/rnn.py::RNN._setup_weight_matrix` builds the random
weight matrix with a pure Linear Congruent Generator seeded only by
``Q``/``P`` (no call into `torch`'s or `numpy`'s global RNG state), and
`RNN.fit` is a deterministic linear-algebra computation. So a freshly
constructed `RNN(Q=..., P=...)` reproduces bit-identical weights to the one
used internally by `extract`, with no `torch.manual_seed` needed — this
test relies on that property to independently recompute each beta block
and compare it against slices of the real embedding.
"""

from __future__ import annotations

import numpy as np
import torch

from dalmax.tools.SSRAE.extractor import ColorFeatureExtractor
from dalmax.tools.SSRAE.rnn import RNN
from dalmax.tools.SSRAE.splitter import WindowSplitter


def _prep_channel(channel: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror extractor.py's per-channel window extraction + z-score exactly."""
    windows = WindowSplitter().split(channel, window_size=3, padding=True)
    X = torch.from_numpy(windows.T).float()
    XX = X.clone()
    X = X.t_()
    X = torch.divide(torch.subtract(X, torch.mean(X, dim=0)), torch.std(X, dim=0) + 1e-3)
    X = X.t_()
    return X, XX


def _recompute_blocks(image: np.ndarray, Q: int) -> list[torch.Tensor]:
    """Independently recompute the six beta blocks using the public RNN/
    WindowSplitter API, faithfully mirroring `ColorFeatureExtractor.extract`."""
    X_R, XX_R = _prep_channel(image[:, :, 0])
    X_G, XX_G = _prep_channel(image[:, :, 1])
    X_B, XX_B = _prep_channel(image[:, :, 2])

    rnn = RNN(Q=Q, P=X_R.shape[0], tikhonov=True, lambda_=1e-3)

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


def test_extract_returns_1d_tensor_with_length_divisible_by_six(tiny_rgb_image):
    Q = 2
    extractor = ColorFeatureExtractor(Q=Q)
    emb = extractor.extract(tiny_rgb_image)

    assert emb.dim() == 1
    assert emb.numel() % 6 == 0


def test_extract_matches_faithful_reconstruction_of_the_six_beta_blocks(tiny_rgb_image):
    """Ground truth: independently recomputing the six beta blocks (same
    windowing, same z-score, same RNN weights) and re-doing the exact
    `torch.hstack(...).reshape((1, -1))` the extractor performs must exactly
    reproduce the extractor's own output. This pins down what the code
    *actually* does, decoupled from any documentation claim about the
    layout's semantics."""
    Q = 2
    extractor = ColorFeatureExtractor(Q=Q)
    emb = extractor.extract(tiny_rgb_image)

    blocks = _recompute_blocks(tiny_rgb_image, Q)
    block_len = blocks[0].numel()
    assert emb.numel() == 6 * block_len

    reconstructed = torch.hstack(blocks).reshape((1, -1)).squeeze()
    assert torch.allclose(emb, reconstructed)


def test_naive_half_split_does_not_match_documented_spatial_spectral_blocks(tiny_rgb_image):
    """Negative check pinning down the bug: a naive contiguous-halves split
    (`emb[:len//2]` / `emb[len//2:]`) does NOT recover the spatial-only
    (R,G,B) / spectral-only (RG,GB,BR) signatures, because the layout is
    row-interleaved (see the module docstring and KI-26 in
    `.specs/quality/known-issues.md`)."""
    Q = 2
    extractor = ColorFeatureExtractor(Q=Q)
    emb = extractor.extract(tiny_rgb_image)

    beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B = _recompute_blocks(tiny_rgb_image, Q)

    half = emb.numel() // 2
    spatial_half = emb[:half]
    spectral_half = emb[half:]

    documented_spatial = torch.cat([beta_R.reshape(-1), beta_G.reshape(-1), beta_B.reshape(-1)])
    documented_spectral = torch.cat(
        [beta_S_R.reshape(-1), beta_S_G.reshape(-1), beta_S_B.reshape(-1)]
    )

    assert not torch.allclose(spatial_half, documented_spatial)
    assert not torch.allclose(spectral_half, documented_spectral)


def test_embedding_layout_is_row_interleaved_column_groups(tiny_rgb_image):
    """Verified layout: reshaping the flat embedding to (9, 6*(Q+1)) recovers
    six column-groups, in order [R, G, B, S_R, S_G, S_B], each equal to the
    corresponding recomputed beta block. This is the correct decomposition
    of the SSRAE embedding (see KI-26 / ablation-study.md §6.1 Layout
    caveat)."""
    Q = 2
    extractor = ColorFeatureExtractor(Q=Q)
    emb = extractor.extract(tiny_rgb_image)

    blocks = _recompute_blocks(tiny_rgb_image, Q)
    beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B = blocks
    n_rows, block_width = beta_R.shape  # (9, Q+1)

    M = emb.reshape(n_rows, 6 * block_width)

    assert torch.allclose(M[:, 0:block_width], beta_R)
    assert torch.allclose(M[:, block_width : 2 * block_width], beta_G)
    assert torch.allclose(M[:, 2 * block_width : 3 * block_width], beta_B)
    assert torch.allclose(M[:, 3 * block_width : 4 * block_width], beta_S_R)
    assert torch.allclose(M[:, 4 * block_width : 5 * block_width], beta_S_G)
    assert torch.allclose(M[:, 5 * block_width : 6 * block_width], beta_S_B)


def test_corrected_slicing_helper_recovers_spatial_and_spectral_column_groups(tiny_rgb_image):
    """The corrected slicing helper (inline here, mirrors the spec in
    `.specs/architecture/target-architecture.md` §5 and
    `.specs/experiments/ablation-study.md` §6.1): reshape to
    (9, 6*(Q+1)) and take column-groups, NOT `emb[:len//2]`. This must
    yield spatial = [beta_R, beta_G, beta_B] and spectral =
    [beta_S_R, beta_S_G, beta_S_B] as column-concatenations (`torch.hstack`)
    of the recomputed beta matrices — i.e. `emb_spatial` reshaped back to
    (9, 3*(Q+1)) equals `hstack([beta_R, beta_G, beta_B])` exactly, matching
    the per-row column-group layout the extractor actually produces (NOT a
    block-contiguous flatten-then-concatenate of the three matrices, which
    would silently reorder elements relative to the true layout)."""
    Q = 2
    extractor = ColorFeatureExtractor(Q=Q)
    emb = extractor.extract(tiny_rgb_image)

    beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B = _recompute_blocks(tiny_rgb_image, Q)
    n_rows, block_width = beta_R.shape  # (9, Q+1)

    def slice_embedding(vector: torch.Tensor, variant: str, q_plus_1: int) -> torch.Tensor:
        matrix = vector.reshape(n_rows, 6 * q_plus_1)
        if variant == "full":
            return vector
        if variant == "spatial":
            return matrix[:, : 3 * q_plus_1].reshape(-1)
        if variant == "spectral":
            return matrix[:, 3 * q_plus_1 :].reshape(-1)
        raise ValueError(f"unknown embedding_variant: {variant}")

    emb_spatial = slice_embedding(emb, "spatial", block_width)
    emb_spectral = slice_embedding(emb, "spectral", block_width)

    expected_spatial = torch.hstack([beta_R, beta_G, beta_B])
    expected_spectral = torch.hstack([beta_S_R, beta_S_G, beta_S_B])

    assert torch.allclose(emb_spatial.reshape(n_rows, 3 * block_width), expected_spatial)
    assert torch.allclose(emb_spectral.reshape(n_rows, 3 * block_width), expected_spectral)
    assert emb_spatial.numel() + emb_spectral.numel() == emb.numel()
