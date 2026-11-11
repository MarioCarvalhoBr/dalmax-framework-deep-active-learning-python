"""Monte Carlo dropout inference mode (KI-41 fix, ADR 0014).

MC dropout (Gal & Ghahramani, 2016; the DeepAL `predict_prob_dropout*`) samples
predictions with dropout active and everything else in inference mode. The
pre-fix `DeepLearning.predict_prob_dropout*` called `clf.train()`, which also
switched BatchNorm to batch statistics and updated its running statistics during
"prediction"; with no dropout layer in the DANINHAS ResNet-50 the `n_drop` passes
were then identical (`.specs/quality/known-issues.md`, KI-41).

`mc_dropout_mode(model)` puts `model` in eval mode, switches only its dropout
modules to train mode, and restores every module's previous mode on exit. It
fails fast if the model has no dropout module with `p > 0`, so an MC-dropout
strategy can never run without dropout again. Importing this module has no
side effects.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch.nn as nn

# The public dropout module family (all subclasses of torch's private `_DropoutNd`).
DROPOUT_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


class MCDropoutError(RuntimeError):
    """Raised when MC-dropout inference is requested on a model without active dropout."""


def active_dropout_modules(model: nn.Module) -> list[nn.Module]:
    """The dropout modules of `model` with a drop probability `p > 0`."""
    return [m for m in model.modules() if isinstance(m, DROPOUT_MODULE_TYPES) and m.p > 0]


@contextmanager
def mc_dropout_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Eval mode for every module except the dropout ones, restored on exit.

    Raises
    ------
    MCDropoutError
        If `model` contains no dropout module with `p > 0` (KI-41): the
        MC-dropout passes would all be identical and the strategy degenerate.
        For the DANINHAS ResNet-50, set `"model_args": {"dropout_p": ...}` in
        the params JSON (e.g. `files_config/campaign/params_paper1_mc_dropout.json`).
    """
    dropout_modules = active_dropout_modules(model)
    if not dropout_modules:
        raise MCDropoutError(
            f"{type(model).__name__} has no dropout module with p > 0, so MC-dropout passes would "
            "be identical (KI-41, .specs/quality/known-issues.md). Configure a dropout layer, e.g. "
            '"model_args": {"dropout_p": 0.5} in the dataset block of the params JSON.'
        )
    previous_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for module in dropout_modules:
            module.train()
        yield model
    finally:
        for module, mode in previous_modes:
            module.training = mode
