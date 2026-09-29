"""The single place that touches global RNG state (`.claude/rules/reproducibility.md`).

`seed_everything` replaces the scattered `np.random.seed(args.seed)` /
`torch.manual_seed(args.seed)` calls in the historical `demo.py` (renamed
`trainer.py` 2026-08-23) and the blanket
`torch.backends.cudnn.enabled = False` determinism shortcut (`demo.py:59`, historical)
with the cheaper `deterministic=True` / `benchmark=False` pair recommended in
`.specs/architecture/target-architecture.md` §7. It must be called exactly
once, before any dataset/model/strategy object is constructed
(`ExperimentRunner.run`, Phase 2 item 7 — not implemented in this module).

`derive_seed` lets a `SelectionStrategy` (e.g. `HierarchicalKMeansSelection`,
which reseeds Python's `random` module internally) or any other stochastic
component obtain a sub-seed that is deterministic given the experiment seed
and a component-specific tag, without ever hardcoding a literal
`random_state` (the `KMeans(random_state=3)` violation this refactor fixes,
see `.claude/rules/reproducibility.md`).
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_MAX_SUB_SEED = 2**32 - 1  # inclusive upper bound accepted by numpy/sklearn/torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed every global RNG source from a single experiment `seed`.

    Seeds Python's `random`, NumPy's legacy global RNG (`np.random.seed`,
    still used transitively by some vendored code), and PyTorch (CPU + all
    CUDA devices, no-op if CUDA is unavailable). Also sets
    `torch.backends.cudnn.deterministic = True` and `benchmark = False`
    (never `cudnn.enabled = False`, which globally disables cuDNN and costs
    performance for no additional determinism benefit).

    Returns
    -------
    np.random.Generator
        A `np.random.default_rng(seed)` instance for new code to use instead
        of the global `np.random` state (e.g. passed into
        `SelectionStrategy.select(..., rng)`).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cuBLAS needs a fixed workspace config to be deterministic on CUDA >= 10.2;
    # it must be set before the first cuBLAS call (setdefault: never clobber a
    # value the operator set deliberately). `warn_only=True`: an op without a
    # deterministic kernel warns instead of crashing a long run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Without this, torch.empty & co. are filled with garbage on some builds,
    # which would make "uninitialized" reads a hidden source of run-to-run drift.
    if hasattr(torch.utils, "deterministic"):
        torch.utils.deterministic.fill_uninitialized_memory = False
    return np.random.default_rng(seed)


def _deterministic_algorithms_mode() -> str:
    """`"off"`, `"warn_only"` or `"strict"` -- what torch is currently set to."""
    if not torch.are_deterministic_algorithms_enabled():
        return "off"
    return "warn_only" if torch.is_deterministic_algorithms_warn_only_enabled() else "strict"


def determinism_state(seed: int | None = None) -> dict:
    """The determinism-relevant global state, for `run_metadata.json`.

    `deterministic_algorithms` is `"warn_only"` on purpose: an op without a
    deterministic kernel warns instead of aborting a long run, so a run is
    only *best-effort* deterministic on GPU. The ops that actually warned are
    appended (as `nondeterministic_op_warnings`) when the run ends -- see
    `record_nondeterministic_ops`.
    """
    fill = getattr(getattr(torch.utils, "deterministic", None), "fill_uninitialized_memory", None)
    return {
        "seed": seed,
        "deterministic_algorithms": _deterministic_algorithms_mode(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "fill_uninitialized_memory": fill,
        "nondeterministic_op_warnings": [],
    }


_NONDETERMINISTIC_WARNING_RE = re.compile(r"^(?P<op>.+?) does not have a deterministic implementation")


@contextmanager
def record_nondeterministic_ops() -> Iterator[list[str]]:
    """Collect the ops torch warns about under `warn_only` deterministic mode.

    Yields a list that is filled (unique, in first-seen order) with the op
    names of every "... does not have a deterministic implementation" warning
    raised inside the block. Other warnings are forwarded untouched to the
    previous `warnings.showwarning`.
    """
    seen: list[str] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("always", message=r".*does not have a deterministic implementation")
        previous = warnings.showwarning

        def _hook(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
            match = _NONDETERMINISTIC_WARNING_RE.match(str(message))
            if match is None:
                previous(message, category, filename, lineno, file, line)
                return
            op = match.group("op")
            if op not in seen:
                seen.append(op)

        warnings.showwarning = _hook
        yield seen


def derive_seed(seed: int, tag: str) -> int:
    """Deterministically derive an integer sub-seed from `seed` and `tag`.

    Same `(seed, tag)` always yields the same sub-seed; different `tag`s
    (almost certainly) yield different sub-seeds for the same `seed`. Use
    this instead of a hardcoded literal wherever a component needs its own
    independent-looking but reproducible seed (e.g.
    `random.seed(derive_seed(seed, "hierarchical_kmeans"))`).
    """
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (_MAX_SUB_SEED + 1)
