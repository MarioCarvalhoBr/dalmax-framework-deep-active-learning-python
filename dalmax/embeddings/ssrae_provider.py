"""SSRAE embedding provider.

Wraps `dalmax.tools.SSRAE.extractor.ColorFeatureExtractor` (vendored,
untouched per `.specs/architecture/refactor-plan.md` Phase 2) behind the
`EmbeddingProvider` interface. Replicates the now-deleted
`Data.create_feature_maps_ssrae` (`.specs/architecture/refactor-plan.md`
Phase 4) exactly: each image is a `(H, W, 3)` `uint8` ndarray (as produced by
`dalmax.data.loaders.get_DANINHAS`'s `np.array(PIL.Image.convert("RGB")...)`),
passed unchanged — no dtype cast,
no external normalization — to `ColorFeatureExtractor.extract`, which
performs its own internal per-channel z-score normalization. Output is
`(N, 54*(Q+1))` per image (`54 = 9 patch positions * 6 beta blocks`).
"""

from __future__ import annotations

import numpy as np

from dalmax.tools.SSRAE.extractor import ColorFeatureExtractor

from .base import EmbeddingProvider


class SSRAEProvider(EmbeddingProvider):
    """SSRAE (spatio-spectral randomized autoencoder) embedding provider.

    Note: `ColorFeatureExtractor`/its underlying `core.tools.SSRAE.rnn.RNN`
    is a CPU-only implementation (no `device`/`.to(...)` support at all) —
    the `device` constructor argument is accepted for interface uniformity
    with the other providers but has no effect here.
    """

    name = "ssrae"

    def __init__(self, q: int = 13, device: str = "cpu") -> None:
        super().__init__(q=q, device=device)
        self._extractor = ColorFeatureExtractor(Q=q)

    def embed(self, images: np.ndarray) -> np.ndarray:
        features = [
            self._extractor.extract(image).detach().cpu().numpy().astype(np.float32)
            for image in images
        ]
        return np.stack(features, axis=0).astype(np.float32)
