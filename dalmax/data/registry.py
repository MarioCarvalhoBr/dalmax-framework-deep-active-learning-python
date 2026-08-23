"""`DATASET_REGISTRY` / `get_dataset`: replaces `utils/orchestrator.py`'s
(deleted, Phase 4) `get_handler`/`get_dataset` `if/elif` chains for the
`dalmax`-routed CLI.

Adding a new dataset means adding one entry to `DATASET_REGISTRY` here —
never a new hardcoded `if name == "..."` branch
(`.claude/rules/code-quality.md`).
"""

from __future__ import annotations

from collections.abc import Callable

from dalmax.config.schema import ExperimentConfig
from dalmax.data.datasets import Data
from dalmax.data.handlers import CIFAR10_Handler, DANINHAS_Hander
from dalmax.data.loaders import get_CIFAR10, get_DANINHAS

_HANDLER_REGISTRY: dict[str, type] = {
    "DANINHAS": DANINHAS_Hander,
    "CIFAR10": CIFAR10_Handler,
}

DATASET_REGISTRY: dict[str, Callable[..., Data]] = {
    "DANINHAS": get_DANINHAS,
    "CIFAR10": get_CIFAR10,
}


def get_dataset(config: ExperimentConfig) -> Data:
    """Build the `Data` instance for `config.dataset.name`/`data_dir`.

    Raises
    ------
    KeyError
        If `config.dataset.name` is not a registered dataset name.
    """
    name = config.dataset.name
    if name not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}. Valid names: {valid}")
    loader = DATASET_REGISTRY[name]
    handler = _HANDLER_REGISTRY[name]
    return loader(handler=handler, data_dir=config.dataset.data_dir)
