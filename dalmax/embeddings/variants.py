"""`embedding_variant` slicing: `full | spatial | spectral`.

Only meaningful for the SSRAE extractor. Implements the verified
row-interleaved layout documented in
`.specs/architecture/target-architecture.md` §5 and
`.specs/experiments/ablation-study.md` §6.1, and pinned down empirically by
`tests/test_ssrae_embedding_layout.py`:

`ColorFeatureExtractor.extract` computes six `(9, Q+1)` "beta" blocks
(`[R, G, B, S_R, S_G, S_B]`), `torch.hstack`s them into a `(9, 6*(Q+1))`
matrix, then flattens row-major into a length-`54*(Q+1)` vector. The six
blocks are therefore interleaved *by column-group within each row*, not
laid out as six contiguous chunks — `vector[:len//2]` is NOT "spatial-only"
and must never be used as a shortcut for this slicing.

The cache always stores the "full" embedding; `spatial`/`spectral` are
always derived here from the cached full vector, never recomputed (see
`dalmax/embeddings/cache.py` and `.specs/architecture/target-architecture.md`
§4). Callers must only apply this to SSRAE embeddings — VCTex and
ResNet-ImageNet embeddings have no such column-group structure and always
use `variant="full"`.
"""

from __future__ import annotations

import numpy as np

_N_ROWS = 9  # 3x3 patch positions, fixed by ColorFeatureExtractor's window_size=3.
_N_BLOCKS = 6  # [R, G, B, S_R, S_G, S_B]
_N_SPATIAL_BLOCKS = 3  # [R, G, B]

_VALID_VARIANTS = ("full", "spatial", "spectral")


def slice_embedding(vectors: np.ndarray, variant: str, q: int) -> np.ndarray:
    """Slice a full SSRAE embedding (or batch of embeddings) into a variant.

    Args:
        vectors: `(D,)` for a single embedding or `(N, D)` for a batch,
            where `D` must equal `54*(q+1)`.
        variant: `"full"` (identity), `"spatial"` (R, G, B column-groups),
            or `"spectral"` (S_R, S_G, S_B column-groups).
        q: The SSRAE `Q` hyperparameter used to compute `vectors` — required
            to recover the `(9, 6*(q+1))` column-group boundaries.

    Returns:
        `vectors` unchanged for `"full"`; otherwise the corresponding
        column-group slice, with the same number of leading dimensions
        (`(D',)` or `(N, D')`) as the input.

    Raises:
        ValueError: if `variant` is not one of `"full"`/`"spatial"`/
            `"spectral"`, or if the last dimension of `vectors` does not
            equal `54*(q+1)` for the given `q`.
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"unknown embedding_variant: {variant!r}. Valid: {_VALID_VARIANTS}")

    block_width = q + 1
    expected_d = _N_ROWS * _N_BLOCKS * block_width
    d = vectors.shape[-1]
    if d != expected_d:
        raise ValueError(
            f"embedding dimension {d} does not match 54*(q+1)={expected_d} for q={q}"
        )

    if variant == "full":
        return vectors

    is_batch = vectors.ndim == 2
    n = vectors.shape[0] if is_batch else 1
    matrix = vectors.reshape(n, _N_ROWS, _N_BLOCKS * block_width)

    boundary = _N_SPATIAL_BLOCKS * block_width
    sliced = matrix[:, :, :boundary] if variant == "spatial" else matrix[:, :, boundary:]
    sliced = sliced.reshape(n, -1)
    return sliced if is_batch else sliced.reshape(-1)
