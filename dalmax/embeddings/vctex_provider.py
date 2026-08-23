"""VCTex embedding provider.

Wraps `core.tools.VCTex.VCTexMethod` (vendored, untouched) behind the
`EmbeddingProvider` interface. Mirrors
`utils/data.py::Data.create_feature_maps_vctex` exactly: each image is a
`(H, W, 3)` `uint8` ndarray passed unchanged to `VCTexMethod.__call__`,
which returns a flat `list[float]` per image (concatenated across all `Q`
values when `Q` is a list, e.g. the paper's best-parameters
`Q=(5, 17)`).

CPU support: unlike SSRAE, `VCTexMethod` -> `core.tools.VCTex.extractor.VCTex`
-> `core.tools.VCTex.rnn.RNN` is NOT CUDA-only. `RNN` only ever moves
tensors with generic `.to(self._device)` calls (`core/tools/VCTex/rnn.py`);
it never hardcodes `.cuda()`. Passing `device="cpu"` works correctly — this
was verified by running the extractor end-to-end on a CPU tensor. The
legacy call site (`utils/data.py:117`) hardcodes `torch.device("cuda:0")`;
this provider instead takes `device` from its constructor (fixes that
coupling point, see `.specs/architecture/current-state.md`).
"""

from __future__ import annotations

import numpy as np
import torch

from core.tools.VCTex.VCTexMethod import VCTexMethod

from .base import EmbeddingProvider


class VCTexProvider(EmbeddingProvider):
    """VCTex (color-texture) embedding provider."""

    name = "vctex"

    def __init__(self, q: int | tuple[int, ...] | list[int] = (5, 17), device: str = "cpu") -> None:
        super().__init__(q=q, device=device)
        q_list = list(q) if isinstance(q, (list, tuple)) else [q]
        self._extractor = VCTexMethod(Q=q_list, device=torch.device(device))

    def embed(self, images: np.ndarray) -> np.ndarray:
        features = [np.asarray(self._extractor(image), dtype=np.float32) for image in images]
        return np.stack(features, axis=0).astype(np.float32)
