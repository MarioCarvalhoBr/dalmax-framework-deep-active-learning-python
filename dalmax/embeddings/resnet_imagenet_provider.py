"""ResNet-ImageNet embedding provider.

NEW provider (ablation 6.3, "without representation module" —
`.specs/experiments/ablation-study.md` §6.3): penultimate-layer (2048-d)
features from a `torchvision` ResNet50 pretrained on ImageNet, used as a
generic-representation stand-in for SSRAE while keeping the hierarchical
selection stage (`dalmax/selection/hierarchical_kmeans.py`) unchanged. This
is the same backbone `dalmax/models/daninhas_resnet50.py::DaninhasModelResNet50` uses,
but here the raw penultimate (pre-`fc`) 2048-d vector is returned directly,
with no learned embedding/classification head, since it is meant to be fed
to a `SelectionStrategy`, not trained.

Model weights (`ResNet50_Weights.IMAGENET1K_V1`, ~98 MB) are loaded inside
`__init__` — never at import time (see `.claude/rules/code-quality.md`,
"every module importable without side effects") — so constructing this
provider is the point at which the one-time download happens (already
cached under `~/.cache/torch/hub/checkpoints/` from Phase 1 on this
machine).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .base import EmbeddingProvider


class ResNetImageNetProvider(EmbeddingProvider):
    """ImageNet-pretrained ResNet50 penultimate-layer embedding provider."""

    name = "resnet_imagenet"
    embedding_dim = 2048

    def __init__(self, q: None = None, device: str = "cpu", batch_size: int = 32) -> None:
        super().__init__(q=q, device=device)
        self._batch_size = batch_size
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        # Replace the classification head with identity so the model's
        # forward pass returns the pooled, flattened 2048-d penultimate
        # feature vector directly (no learned embedding/classifier layer).
        model.fc = nn.Identity()
        model.eval()
        self._model = model.to(device)
        self._transform = weights.transforms()

    @torch.no_grad()
    def embed(self, images: np.ndarray) -> np.ndarray:
        # (N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]; the weights'
        # own transform (resize/center-crop to 224 + ImageNet normalization)
        # is applied per batch below.
        tensor = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2).float() / 255.0

        outputs = []
        for start in range(0, tensor.shape[0], self._batch_size):
            batch = self._transform(tensor[start : start + self._batch_size]).to(self.device)
            features = self._model(batch)
            outputs.append(features.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)
