"""Checkpoint format for trained DalMax models.

Fixes the historical save/load bug confirmed 2026-08-23
(`.specs/quality/known-issues.md`): `dalmax.models.base.DeepLearning.save_model`
used to do `torch.save(self.net, path)`, but `self.net` is the *model class*
passed at construction time (e.g. `DaninhasModelResNet50`), never the trained
instance `self.clf` built in `.train()`. Every `saved_model.pth` produced
before this fix (~900 bytes) is a pickled reference to the class object only
— it contains zero weights and cannot be used for inference. `load_model`
was also unimplemented (a `# TODO: FIX THIS` that never rebuilt a usable
model).

This module is the single place that knows the on-disk checkpoint format: a
plain dict, saved with `torch.save`, carrying the trained `state_dict` plus
enough metadata to rebuild the exact architecture and label space without
any side channel (`.claude/rules/code-quality.md` "explicit dependency
injection — no setattr magic"). `dalmax.models.base.DeepLearning.save_model`/
`load_model` and the inference tools (`dalmax/inference/predictor.py`,
`loader.py`, `predict.py`, `gui.py`) all go through `save_checkpoint`/
`load_checkpoint`/`describe_checkpoint` here — never `torch.save`/`torch.load`
directly.
"""

from __future__ import annotations

import os
from typing import Any

import torch

CHECKPOINT_FORMAT = "dalmax-checkpoint"
CHECKPOINT_VERSION = 1

_LEGACY_BUG_MESSAGE = (
    "{path} was produced by the pre-2026-08-23 save bug "
    "(DeepLearning.save_model saved the model CLASS via torch.save(self.net, path) "
    "instead of the trained instance self.clf) and contains no weights. "
    "Re-train to regenerate a real checkpoint; see .specs/quality/known-issues.md."
)


class CheckpointError(Exception):
    """Raised when a file is not a valid `dalmax-checkpoint`.

    Covers both "not one of ours at all" and the specific, previously-common
    case of a legacy pre-fix class-pickle file (see module docstring), whose
    message names the bug explicitly rather than surfacing a generic
    unpickling error.
    """


def save_checkpoint(
    path: str,
    clf: torch.nn.Module,
    *,
    model_name: str,
    n_classes: int,
    class_names: list[str],
    img_size: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a trained model's weights + reconstruction metadata to `path`.

    `clf` must be the trained `nn.Module` *instance* (`DeepLearning.clf`),
    never the model *class* (`DeepLearning.net`) — that mismatch was exactly
    the historical bug this module fixes.

    Parameters
    ----------
    model_name:
        Key into `dalmax.models.registry.MODEL_REGISTRY` (e.g. `"DANINHAS"`,
        `"CIFAR10"`) — `load_checkpoint` uses it to rebuild the same
        architecture.
    n_classes, class_names, img_size:
        Enough to reconstruct the model's output layer and the exact
        preprocessing (`Predictor`, `dalmax/inference/predictor.py`) without
        access to the original dataset/config objects.
    extra:
        Free-form provenance the caller wants to keep alongside the
        checkpoint (strategy name, seed, dataset name, git commit, torch
        version, save timestamp, ...). Not interpreted by this module.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "state_dict": clf.state_dict(),
        "model_name": model_name,
        "n_classes": int(n_classes),
        "class_names": list(class_names),
        "img_size": int(img_size),
        "extra": dict(extra or {}),
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)


def _rebuild_model(model_name: str, n_classes: int) -> torch.nn.Module:
    # Local import: dalmax.models.registry imports dalmax.config.schema, and
    # keeping this import inside the function avoids any import-order cycle
    # between dalmax.models.checkpoint and dalmax.models.registry.
    from dalmax.models.registry import get_model_class

    try:
        model_cls = get_model_class(model_name)
    except KeyError as exc:
        raise CheckpointError(str(exc)) from exc
    return model_cls(n_classes)


def _load_payload(path: str, device: str) -> dict[str, Any]:
    """Load the raw checkpoint dict from `path`, detecting the legacy bug.

    Tries `weights_only=True` first (safe: only succeeds for the new
    dict-of-tensors/plain-Python format). If that fails — which is exactly
    what happens for a legacy file, since it unpickles a bare class object —
    falls back to `weights_only=False` *only* to positively identify the
    legacy bug (a `type` instance) and raise `CheckpointError` with a clear
    message; a `weights_only=False` result is never otherwise trusted as a
    live payload.
    """
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = None

    if isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT:
        return payload

    try:
        legacy = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"{path} is not a valid dalmax checkpoint: {exc}") from exc

    if isinstance(legacy, type):
        raise CheckpointError(_LEGACY_BUG_MESSAGE.format(path=path))
    if not isinstance(legacy, dict) or legacy.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path} is not a dalmax checkpoint (unrecognized format)")
    return legacy


def load_checkpoint(path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpoint saved by `save_checkpoint`.

    Returns
    -------
    (clf, meta):
        `clf` is a `.eval()`'d model instance on `device`. `meta` is
        `{"model_name", "n_classes", "class_names", "img_size", "extra",
        "version"}`.

    Raises
    ------
    CheckpointError
        If `path` is not a dalmax checkpoint, including the historical
        pre-2026-08-23 class-pickle bug (see module docstring) — the message
        names that bug explicitly and instructs to re-train.
    """
    payload = _load_payload(path, device)

    meta = {
        "model_name": payload["model_name"],
        "n_classes": payload["n_classes"],
        "class_names": payload["class_names"],
        "img_size": payload["img_size"],
        "extra": payload.get("extra", {}),
        "version": payload.get("version"),
    }

    clf = _rebuild_model(meta["model_name"], meta["n_classes"])
    clf.load_state_dict(payload["state_dict"])
    clf.eval()
    clf.to(device)
    return clf, meta


def describe_checkpoint(path: str) -> dict[str, Any]:
    """Inspect a checkpoint's metadata and parameter counts without
    constructing a model (no `dalmax.models.registry` lookup, no `nn.Module`
    instantiation) — used by `loader.py` to report on a checkpoint even if
    its `model_name` were unknown.

    Raises
    ------
    CheckpointError
        Same conditions as `load_checkpoint` (not a checkpoint at all, or
        the historical legacy class-pickle bug).
    """
    file_size_bytes = os.path.getsize(path)
    payload = _load_payload(path, "cpu")

    state_dict = payload["state_dict"]
    layers = []
    total_params = 0
    for name, tensor in state_dict.items():
        numel = int(tensor.numel())
        total_params += numel
        layers.append({"name": name, "shape": list(tensor.shape), "numel": numel})

    return {
        "format": payload.get("format"),
        "version": payload.get("version"),
        "model_name": payload["model_name"],
        "n_classes": payload["n_classes"],
        "class_names": payload["class_names"],
        "img_size": payload["img_size"],
        "extra": payload.get("extra", {}),
        "total_params": total_params,
        "layers": layers,
        "file_size_bytes": file_size_bytes,
    }
