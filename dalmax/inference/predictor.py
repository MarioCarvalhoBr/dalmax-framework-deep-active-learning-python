"""`Predictor`: run a trained `dalmax-checkpoint` (`dalmax.models.checkpoint`)
on arbitrary image files outside the active-learning loop.

Preprocessing MUST replicate exactly what training/testing used, or a
checkpoint's predictions here would silently disagree with what
`dalmax.experiment.reporter._write_predictions_csv` recorded for the same
image during a run:

1. Open the image, convert to RGB, resize to the checkpoint's `img_size`
   (`dalmax/data/loaders.py::get_DANINHAS`/`get_CIFAR10` resize every image
   to `img_size` *before* it ever reaches a handler).
2. Apply the exact `torchvision.transforms` pipeline the dataset's handler
   class builds in `__init__` (`dalmax/data/handlers.py`) — `ToTensor()` +
   `Normalize(mean, std)` with that dataset's specific statistics. This
   module never re-derives those statistics: it looks the handler class up
   via `dalmax.data.registry.get_handler(meta["model_name"])` and reads its
   `.transform` attribute, so it can never drift from what a real
   training/test run applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from dalmax.data.registry import get_handler
from dalmax.models.checkpoint import load_checkpoint

_DEFAULT_BATCH_SIZE = 32


def resolve_device(device: str) -> torch.device:
    """`"auto"` -> cuda iff available else cpu; `"cpu"`/`"cuda"` passed through."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class PredictionRow:
    """One image's prediction: `path`, the argmax class name, its
    confidence (the argmax probability), and the full per-class probability
    distribution (`{class_name: probability}`, summing to ~1.0)."""

    path: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class Predictor:
    """Loads a `dalmax-checkpoint` once, then predicts on any number of
    image files via `predict_paths`.

    Parameters
    ----------
    checkpoint_path:
        Path to a `saved_model.pth` written by
        `dalmax.query_strategies.base.Strategy.save_model` (via
        `dalmax.models.checkpoint.save_checkpoint`).
    device:
        `"auto"` (default), `"cpu"`, or `"cuda"`.

    Raises
    ------
    dalmax.models.checkpoint.CheckpointError
        If `checkpoint_path` is not a `dalmax-checkpoint` — in particular,
        the historical pre-2026-08-23 class-pickle bug (see
        `dalmax.models.checkpoint` module docstring).
    """

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.clf, self.meta = load_checkpoint(checkpoint_path, device=self.device)
        handler_cls = get_handler(self.meta["model_name"])
        # The handler's __init__ only stores X/Y and builds self.transform
        # from fixed, hardcoded statistics -- it never touches X/Y at
        # construction time, so empty arrays are enough to get `.transform`
        # without loading any real data.
        self._transform = handler_cls([], []).transform
        self.class_names: list[str] = list(self.meta["class_names"])
        self.img_size: int = int(self.meta["img_size"])
        self.dataset_name: str = self.meta["model_name"]
        self.extra: dict[str, Any] = dict(self.meta.get("extra", {}))

    def _load_image_tensor(self, path: str | Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        return self._transform(image)

    def predict_paths(
        self, paths: list[str | Path], *, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> list[PredictionRow]:
        """Predict on every path in `paths`, in the order given.

        Runs in `batch_size`-sized chunks (default 32) rather than one
        `DataLoader` over the whole list, since inference input here is an
        arbitrary user-supplied file list, not a `dalmax.data.datasets.Data`
        pool.
        """
        rows: list[PredictionRow] = []
        self.clf.eval()
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            batch = torch.stack([self._load_image_tensor(p) for p in chunk]).to(self.device)
            with torch.no_grad():
                logits, _ = self.clf(batch)
                probs = torch.softmax(logits, dim=1).cpu()
            for path, prob in zip(chunk, probs, strict=True):
                prob_list = [float(v) for v in prob.tolist()]
                pred_idx = int(prob.argmax())
                rows.append(
                    PredictionRow(
                        path=str(path),
                        predicted_class=self.class_names[pred_idx],
                        confidence=prob_list[pred_idx],
                        probabilities=dict(zip(self.class_names, prob_list, strict=True)),
                    )
                )
        return rows
