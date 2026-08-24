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
import random

import numpy as np
import torch

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
    return np.random.default_rng(seed)


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
