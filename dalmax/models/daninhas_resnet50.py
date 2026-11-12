"""ResNet-50 classifier for the DANINHAS dataset (ImageNet-pretrained backbone).

`forward` returns `(logits, e1)` like every DalMax model; `e1` is the
embedding consumed by `DeepLearning.get_embeddings` (`KMeansSampling`,
`KCenterGreedy`). Two fixes of 2026-11-11 (ADR 0014):

- KI-42: `e1` is the 2048-d pooled feature vector that feeds the classifier
  (the trained penultimate representation, as in DeepAL), no longer a ReLU of
  the untrained `embedding_layer` projection.
- KI-41: an optional dropout layer (`dropout_p > 0`) on those pooled features,
  right before the classifier, so the MC-dropout strategies have dropout to
  sample from. The default `dropout_p = 0.0` builds exactly the pre-fix model
  (same parameters, same `state_dict` keys, same RNG draws at construction,
  same forward pass).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# Width of the legacy `embedding_layer` (see `DaninhasModelResNet50.__init__`).
_LEGACY_EMBEDDING_WIDTH = 50

# Positions of `layer1`..`layer4` in `feature_extractor` (conv1, bn1, relu, maxpool,
# layer1, layer2, layer3, layer4, avgpool), see `forward_with_stages`.
_STAGE_INDICES = (4, 5, 6, 7)


class DaninhasModelResNet50(nn.Module):
    def __init__(self, n_classes: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p!r}")
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        # Load the ImageNet-pretrained ResNet-50.
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)

        # Every layer up to and including the global average pool (drops `fc`).
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Width of the pooled features (2048 for ResNet-50).
        num_features = resnet.fc.in_features
        self.embedding_dim = num_features

        # KI-42: kept but unused by `forward`. Its construction consumes torch RNG draws
        # that determine the classifier's initial weights below, and its keys are in every
        # saved checkpoint; removing it would change the training of every strategy.
        self.embedding_layer = nn.Linear(num_features, _LEGACY_EMBEDDING_WIDTH)

        # Final layer: n_classes logits from the pooled features.
        self.classifier = nn.Linear(num_features, self.n_classes)

        # KI-41: dropout before the classifier. `nn.Dropout` has no parameters and draws
        # no random numbers at construction; `nn.Identity` keeps p = 0 a no-op.
        self.dropout: nn.Module = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone features: [batch_size, num_features, 1, 1] -> [batch_size, num_features].
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        logits = self.classifier(self.dropout(features))
        return logits, features

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def forward_with_stages(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """`forward`, plus the output maps of the four residual stages (`layer1`..`layer4`).

        Runs the same modules in the same order as `forward` (an `nn.Sequential` is
        exactly this loop), so logits and features are identical; used by the
        loss-prediction module of `LossPredictionLoss` (ADR 0015). Adds no
        parameter, state or random draw to the model.
        """
        stages: list[torch.Tensor] = []
        h = x
        for index, module in enumerate(self.feature_extractor):
            h = module(h)
            if index in _STAGE_INDICES:
                stages.append(h)
        features = h.view(h.size(0), -1)
        logits = self.classifier(self.dropout(features))
        return logits, features, stages

    def stage_channels(self) -> tuple[int, ...]:
        """Channel count of each map returned by `forward_with_stages`."""
        return tuple(self.feature_extractor[i][-1].bn3.num_features for i in _STAGE_INDICES)
