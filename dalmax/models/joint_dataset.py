"""Joint labeled/unlabeled dataset for the adversarial strategies VAAL and WAAL
(DeepAL+ `handlers.py::*_Handler_joint`, ADR 0015).

`JointLabeledUnlabeled(labeled, unlabeled)` has length `max(len(labeled),
len(unlabeled))`; item `i` is `(i, x_l, y_l, x_u, y_u)` with the labeled item
`i mod len(labeled)` and the unlabeled item `i mod len(unlabeled)`, i.e. the
shorter set is cycled, exactly as DeepAL+ pairs them. Both inputs are DalMax
handler datasets returning `(x, y, index)` (the preprocessing is theirs).

`joint_loader` wraps it in a shuffled `DataLoader` whose order comes from an
explicit generator, never from the run's global torch stream.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class JointLabeledUnlabeled(Dataset):
    """Pairs every index with one labeled and one unlabeled sample (shorter set cycled)."""

    def __init__(self, labeled: Dataset, unlabeled: Dataset) -> None:
        if len(labeled) == 0 or len(unlabeled) == 0:
            raise ValueError(
                f"both sets must be non-empty (labeled={len(labeled)}, unlabeled={len(unlabeled)})"
            )
        self.labeled = labeled
        self.unlabeled = unlabeled

    def __len__(self) -> int:
        return max(len(self.labeled), len(self.unlabeled))

    def __getitem__(self, index: int) -> tuple[int, Any, Any, Any, Any]:
        x_l, y_l, _ = self.labeled[index % len(self.labeled)]
        x_u, y_u, _ = self.unlabeled[index % len(self.unlabeled)]
        return index, x_l, y_l, x_u, y_u


def joint_loader(
    labeled: Dataset,
    unlabeled: Dataset,
    train_args: dict[str, Any],
    generator: torch.Generator,
) -> DataLoader:
    """Shuffled loader over `JointLabeledUnlabeled(labeled, unlabeled)`; `generator`
    (explicit, seeded by the caller) fixes the order and the worker base seed."""
    return DataLoader(
        JointLabeledUnlabeled(labeled, unlabeled),
        shuffle=True,
        generator=generator,
        **train_args,
    )
