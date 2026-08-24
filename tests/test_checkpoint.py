"""Unit tests for `dalmax.models.checkpoint` — the fix for the historical
save/load bug confirmed 2026-08-23 (`dalmax.models.base.DeepLearning.save_model`
used to `torch.save(self.net, path)`, saving the model *class* instead of the
trained instance, producing ~900-byte weight-less files; see the module
docstring of `dalmax/models/checkpoint.py` and
`.specs/quality/known-issues.md`).

Uses `DaninhasModelResNet50` with a random-initialized (untrained) instance —
no training loop needed to exercise save/load correctness. The pretrained
ResNet-50 ImageNet backbone weights are loaded from the local torch hub cache
(`~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`), not the network, so
this test runs offline.
"""

from __future__ import annotations

import pickle

import pytest
import torch

from dalmax.models.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    CheckpointError,
    describe_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from dalmax.models.daninhas_resnet50 import DaninhasModelResNet50

CLASS_NAMES = ["DATASET_A", "DATASET_B"]
IMG_SIZE = 128


def _fixed_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, IMG_SIZE, IMG_SIZE, generator=generator)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    clf = DaninhasModelResNet50(n_classes=len(CLASS_NAMES))
    clf.eval()

    path = str(tmp_path / "saved_model.pth")
    save_checkpoint(
        path,
        clf,
        model_name="DANINHAS",
        n_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        img_size=IMG_SIZE,
        extra={"strategy_name": "RandomSampling", "seed": 1},
    )

    loaded_clf, meta = load_checkpoint(path, device="cpu")

    assert meta["model_name"] == "DANINHAS"
    assert meta["n_classes"] == len(CLASS_NAMES)
    assert meta["class_names"] == CLASS_NAMES
    assert meta["img_size"] == IMG_SIZE
    assert meta["extra"] == {"strategy_name": "RandomSampling", "seed": 1}
    assert meta["version"] == CHECKPOINT_VERSION

    # Identical state_dict tensors.
    original_state = clf.state_dict()
    loaded_state = loaded_clf.state_dict()
    assert set(original_state) == set(loaded_state)
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key]), f"tensor mismatch at {key}"

    # Identical outputs on a fixed random input.
    x = _fixed_input()
    with torch.no_grad():
        original_out, original_embed = clf(x)
        loaded_out, loaded_embed = loaded_clf(x)
    assert torch.equal(original_out, loaded_out)
    assert torch.equal(original_embed, loaded_embed)

    # load_checkpoint leaves the model in eval mode on the requested device.
    assert not loaded_clf.training
    assert next(loaded_clf.parameters()).device == torch.device("cpu")


def test_load_checkpoint_rejects_legacy_class_pickle_file(tmp_path):
    """Reproduces the historical bug's on-disk artifact: `torch.save(self.net,
    path)` where `self.net` is the model *class*, not an instance."""
    path = str(tmp_path / "legacy_saved_model.pth")
    torch.save(DaninhasModelResNet50, path)

    with pytest.raises(CheckpointError) as exc_info:
        load_checkpoint(path, device="cpu")

    message = str(exc_info.value)
    assert "pre-2026-08-23" in message
    assert "no weights" in message
    assert "re-train" in message.lower()


def test_describe_checkpoint_rejects_legacy_class_pickle_file(tmp_path):
    path = str(tmp_path / "legacy_saved_model.pth")
    torch.save(DaninhasModelResNet50, path)

    with pytest.raises(CheckpointError) as exc_info:
        describe_checkpoint(path)
    assert "pre-2026-08-23" in str(exc_info.value)


def test_load_checkpoint_rejects_non_checkpoint_file(tmp_path):
    path = tmp_path / "not_a_checkpoint.pth"
    with open(path, "wb") as f:
        pickle.dump({"some": "unrelated dict"}, f)

    with pytest.raises(CheckpointError):
        load_checkpoint(str(path), device="cpu")


def test_load_checkpoint_unknown_model_name(tmp_path):
    clf = DaninhasModelResNet50(n_classes=2)
    path = str(tmp_path / "saved_model.pth")
    save_checkpoint(
        path,
        clf,
        model_name="NOT_A_REGISTERED_MODEL",
        n_classes=2,
        class_names=CLASS_NAMES,
        img_size=IMG_SIZE,
    )
    with pytest.raises(CheckpointError, match="NOT_A_REGISTERED_MODEL"):
        load_checkpoint(path, device="cpu")


def test_describe_checkpoint_sanity(tmp_path):
    clf = DaninhasModelResNet50(n_classes=len(CLASS_NAMES))
    path = str(tmp_path / "saved_model.pth")
    save_checkpoint(
        path,
        clf,
        model_name="DANINHAS",
        n_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        img_size=IMG_SIZE,
        extra={"seed": 1},
    )

    info = describe_checkpoint(path)

    assert info["format"] == CHECKPOINT_FORMAT
    assert info["version"] == CHECKPOINT_VERSION
    assert info["model_name"] == "DANINHAS"
    assert info["n_classes"] == len(CLASS_NAMES)
    assert info["class_names"] == CLASS_NAMES
    assert info["img_size"] == IMG_SIZE
    assert info["extra"] == {"seed": 1}
    assert info["file_size_bytes"] > 0
    # total_params counts every state_dict entry (parameters AND buffers,
    # e.g. BatchNorm running_mean/running_var/num_batches_tracked) — not
    # just clf.parameters(), which excludes buffers.
    assert info["total_params"] == sum(t.numel() for t in clf.state_dict().values())
    assert info["total_params"] > sum(p.numel() for p in clf.parameters())
    assert len(info["layers"]) == len(list(clf.state_dict()))
    for layer in info["layers"]:
        assert layer["numel"] > 0
        assert isinstance(layer["shape"], list)

    # describe_checkpoint never constructs a model — sanity-check it doesn't
    # even require a registered model_name to succeed on an otherwise-valid
    # checkpoint (unlike load_checkpoint).
    save_checkpoint(
        path,
        clf,
        model_name="NOT_A_REGISTERED_MODEL",
        n_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        img_size=IMG_SIZE,
    )
    info2 = describe_checkpoint(path)
    assert info2["model_name"] == "NOT_A_REGISTERED_MODEL"
