"""Small CNN for CIFAR10 (the DeepAL `Net` architecture).

Its dropout (p=0.5 between `fc1` and `fc2`) is part of the original
architecture. Since 2026-11-11 (ADR 0014, KI-41) it is an `nn.Dropout` module
instead of a functional `F.dropout(training=self.training)` call: the forward
pass and the random draws are identical, but MC-dropout prediction
(`DeepLearning.predict_prob_dropout*`) can now enable dropout alone, with the
rest of the model in eval mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10Model(nn.Module):
    def __init__(self, n_classes: int, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, n_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = self.dropout(e1)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self) -> int:
        return 50
