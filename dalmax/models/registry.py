"""`MODEL_REGISTRY` / `get_network`: replaces `utils/orchestrator.py`'s
(deleted, Phase 4) `get_network_deep_learning` `if/elif` chain for the
`dalmax`-routed CLI.

`DeepLearning` (`dalmax/models/base.py`, vendored, untouched) still expects a
plain dict of the historical shape
(`n_epoch`/`n_classes`/`n_drop`/`train_args`/`test_args`/`optimizer_args`) —
`_legacy_params_dict` is the one place that reconstructs that dict from the
typed `dalmax.config.schema.DatasetConfig`, so no other new code has to know
about that legacy shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from dalmax.config.schema import DatasetConfig, ExperimentConfig
from dalmax.models.base import DeepLearning
from dalmax.models.cifar10_cnn import CIFAR10Model
from dalmax.models.daninhas_resnet50 import DaninhasModelResNet50

MODEL_REGISTRY: dict[str, Callable] = {
    "DANINHAS": DaninhasModelResNet50,
    "CIFAR10": CIFAR10Model,
}


def _legacy_params_dict(dataset_config: DatasetConfig) -> dict[str, Any]:
    """Rebuild the historical `params[dataset_name]` dict shape that
    `dalmax.models.base.DeepLearning` indexes into (`self.params['n_epoch']`,
    `self.params['train_args']`, ...)."""
    return {
        "n_epoch": dataset_config.n_epoch,
        "n_classes": dataset_config.n_classes,
        "n_drop": dataset_config.n_drop,
        "train_args": {
            "batch_size": dataset_config.train_args.batch_size,
            "num_workers": dataset_config.train_args.num_workers,
        },
        "test_args": {
            "batch_size": dataset_config.test_args.batch_size,
            "num_workers": dataset_config.test_args.num_workers,
        },
        "optimizer_args": {
            "lr": dataset_config.optimizer_args.lr,
            "momentum": dataset_config.optimizer_args.momentum,
        },
    }


def get_network(config: ExperimentConfig, device: str) -> DeepLearning:
    """Build the `DeepLearning` wrapper for `config.dataset.name`, on `device`.

    Raises
    ------
    KeyError
        If `config.dataset.name` is not a registered model name.
    """
    name = config.dataset.name
    model_cls = get_model_class(name)
    params = _legacy_params_dict(config.dataset)
    return DeepLearning(model_cls, params, torch.device(device))


def get_model_class(model_name: str) -> Callable:
    """Look up a model class by name, e.g. for
    `dalmax.models.checkpoint.load_checkpoint` rebuilding an architecture
    from a checkpoint's stored `model_name` — the same registry `get_network`
    uses, keyed the same way (dataset name doubles as model name; there is
    exactly one model architecture per dataset today).

    Raises
    ------
    KeyError
        If `model_name` is not a registered model name.
    """
    if model_name not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model_name {model_name!r}. Valid names: {valid}")
    return MODEL_REGISTRY[model_name]
