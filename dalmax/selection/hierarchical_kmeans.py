"""Hierarchical k-means selection: wraps the vendored `dalmax/tools/SSL` pipeline.

Replaces today's `SSLStrategy` (`core/query_strategies/ssl_ssrae_sampling.py`,
backing `SSRAEKmeansHCSampling` / `VCTexKmeansHCSampling`), fixing two known
couplings (`.specs/architecture/target-architecture.md` §6):

- no hardcoded `self.params['DANINHAS']['config_kmh']` — the hierarchy
  (`n_clusters`, `n_levels`, `sample_sizes`) is injected via the
  constructor, duck-typed so this module does not import `dalmax.config`;
- no hardcoded `device="cuda"` — the device is an explicit constructor
  argument.

## Vendored pipeline

This wraps (the only edit to the vendored tree is the KI-30 fix below):
- `dalmax.tools.SSL.src.hierarchical_kmeans_gpu.hierarchical_kmeans_with_resampling`
- `dalmax.tools.SSL.src.clusters.HierarchicalCluster.from_dict`
- `dalmax.tools.SSL.src.hierarchical_sampling.hierarchical_sampling`

## CPU compatibility (verified)

Despite the "GPU" naming, `hierarchical_kmeans_with_resampling` and the
`kmeans_gpu` helpers it calls (`kmeans`, `assign_clusters`,
`create_clusters_from_cluster_assignment`) contain no hardcoded `.cuda()`
call and run correctly on CPU tensors — verified empirically with a 200-point
`make_blobs` dataset, `n_clusters=[8, 4]`, `n_levels=2`,
`sample_sizes=[4, 2]`, `device="cpu"`. The only hardcoded-device default in
that source tree is `kmeans_gpu.sort_cluster_by_distance`'s
`device="cuda"` keyword argument default
(`dalmax/tools/SSL/src/kmeans_gpu.py:407`), a function that this pipeline
(`hierarchical_kmeans_with_resampling` + `hierarchical_sampling`) never
calls — the hardcoded `device="cuda"` actually seen in production
(`core/query_strategies/ssl_ssrae_sampling.py:85`) is a choice made by the
*caller*, not a constraint of the vendored library. So this class's tests
run on CPU unconditionally (no `skipif`/`gpu` marker needed for the
"does it run" property); a real `device="cuda"` run is exercised only when
CUDA is actually available (see `tests/test_selection_hierarchical.py`).

## Equal-size clusters (KI-30, fixed 2026-11-07)

`dalmax.tools.SSL.src.utils.create_clusters_from_cluster_assignment` used to
build `np.array(clusters, dtype=object)` from a list of per-cluster index
arrays; when every cluster had exactly the same size NumPy produced a 2-D
object array and `torch.cdist`/tensor indexing raised
`TypeError: can't convert np.ndarray of type numpy.object_ ...`. The KMH sweep's
budget-matched hierarchies (small cluster counts at the upper levels) made the
case likely, so the vendored builder was fixed at the source:
`dalmax/tools/SSL/src/utils.py::to_object_array` pre-allocates the 1-D object
array and replaces the four `np.array(..., dtype=object)` sites (each marked
`DalMax fix (KI-30)`). Regression test:
`tests/test_selection_hierarchical.py::test_perfectly_balanced_clusters_do_not_crash`.

## `sample_sizes` semantics (resolves the `ablation-study.md` §6.2 TBD)

`sample_sizes[level]` and the final selection budget (`n_query`) are
**independent knobs** that do not compose the way one might guess:

- `sample_sizes[level]` only affects the *centroid-refinement resampling*
  performed **inside** `hierarchical_kmeans_with_resampling` for that
  level: at each of `n_resamples` iterations, up to `sample_sizes[level]`
  points are drawn from every cluster at that level (closest-to-centroid,
  by default) and used to recompute that level's centroids. It is a
  clustering-quality/runtime knob — larger values use more points per
  resampling step (closer to the full level population), smaller values
  are cheaper and noisier. If `sample_sizes[level] <= 1` no resampling is
  performed for that level (the initial k-means fit is kept as-is).
- The **number of ids finally returned** is controlled entirely by
  `target_size` (here, `budget`) passed to
  `hierarchical_sampling.hierarchical_sampling`, which recursively splits
  `budget` across the top-level clusters (and their subclusters) by a
  balanced "water-filling" allocation (`find_subcluster_target_size`,
  `_find_best_cut_left`): every cluster gets `min(x, size)` points for the
  largest integer cut `x` whose total does not exceed the target, and the
  leftover points are given one each (`x + 1`) to randomly chosen clusters
  larger than `x`. Small clusters are therefore taken whole and large ones
  are capped at about the same quota (NOT proportionally to size). This
  does **not** consult `sample_sizes` at all.

So changing `sample_sizes` in `ablation-study.md` §6.2's grid changes
*clustering quality*, not how many samples come out per level — `budget`
(`n_query`) alone determines the output size. A reasonable derivation rule
for the TBD'd `sample_sizes` per row of that table (not settled here, but
consistent with the two params files on disk) is to scale
`sample_sizes[level]` with `n_clusters[level]` (e.g.
`sample_sizes[level] ~= max(2, round(n_clusters[level] * ratio))` for a
fixed `ratio` taken from an existing run, such as
`files_config/benchmark/params_df_gpu_0.json`'s `30/600 = 0.05`), not with `n_query`.

## Optional knobs (2026-11-07)

The `hierarchy` object/dict may also carry four optional keys (defaults reproduce the
previous hard-wired behaviour exactly):

- `n_resamples` (int >= 0, default 10): resampling iterations per level; 0 = no resampling.
- `init_method` (`"kmeans++"` | `"random"`, default `"kmeans++"`): k-means centroid init.
- `resample_strategy` (`"closest"` | `"random"`, default `"closest"`): how points are drawn
  from each cluster during resampling.
- `leaf_sampling` (`"random"` | `"closest"`, default `"random"`): how the final selection
  picks points inside a leaf cluster (vendored `sampling_strategy` `"r"` / `"c"`).

The vendored `"c"` branch takes `current_cluster[:k]` and assumes leaf clusters are sorted by
increasing distance to their centroid, but `hierarchical_kmeans_with_resampling` returns them
unsorted. This wrapper therefore sorts the level-1 (leaf) clusters by distance to the level-1
centroids (stable sort, so deterministic) before `HierarchicalCluster.from_dict`, only when
`leaf_sampling == "closest"`; the vendored code is untouched.

## RNG isolation

`hierarchical_sampling`'s core randomness (`find_subcluster_target_size`,
`recursive_hierarchical_sampling`'s default `sampling_strategy="r"`) is
implemented with **global NumPy legacy random** (`np.random.choice`), and
`kmeans_gpu.kmeans`/`kmeans_plusplus` resolve their `random_state=None`
argument to that same global legacy state
(`sklearn.utils.check_random_state(None)` returns `np.random.mtrand._rand`).
Neither vendored module exposes a way to inject an explicit generator, so
this class seeds NumPy's global legacy state (`np.random.seed`) — and,
defensively, Python's `random` module too, since
`hierarchical_sampling.random_selection` (unused on the default
`sampling_strategy="r"` path, but present in the same module) uses it —
from a value derived from the caller-supplied `rng` immediately before
calling into the vendored code. This trades a small amount of global-state
coupling (documented here, not hidden) for determinism: the same `rng`
seed always reseeds the same global state before selection.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from dalmax.selection.base import SelectionStrategy
from dalmax.tools.SSL.src import hierarchical_kmeans_gpu as hkmg
from dalmax.tools.SSL.src import hierarchical_sampling as hs
from dalmax.tools.SSL.src.clusters import HierarchicalCluster

_MAX_SEED = 2**32 - 1


_KNOB_DEFAULTS: dict[str, Any] = {
    "n_resamples": 10,
    "init_method": "kmeans++",
    "resample_strategy": "closest",
    "leaf_sampling": "random",
}
_VALID_CHOICES: dict[str, tuple[str, ...]] = {
    "init_method": ("kmeans++", "random"),
    "resample_strategy": ("closest", "random"),
    "leaf_sampling": ("random", "closest"),
}


def _read_hierarchy(hierarchy: Any) -> tuple[list[int], int, list[int]]:
    """Duck-type a hierarchy config: a `dalmax.config.schema.HierarchyConfig`-like
    object with `n_clusters`/`n_levels`/`sample_sizes` attributes, or an
    equivalent plain dict with the same three keys.
    """
    if isinstance(hierarchy, dict):
        try:
            n_clusters = hierarchy["n_clusters"]
            n_levels = hierarchy["n_levels"]
            sample_sizes = hierarchy["sample_sizes"]
        except KeyError as exc:
            raise ValueError(
                "hierarchy dict must have keys 'n_clusters', 'n_levels', 'sample_sizes' "
                f"(missing {exc})"
            ) from exc
    else:
        missing = [
            attr
            for attr in ("n_clusters", "n_levels", "sample_sizes")
            if not hasattr(hierarchy, attr)
        ]
        if missing:
            raise ValueError(
                "hierarchy object must have attributes 'n_clusters', 'n_levels', "
                f"'sample_sizes' (missing {missing})"
            )
        n_clusters = hierarchy.n_clusters
        n_levels = hierarchy.n_levels
        sample_sizes = hierarchy.sample_sizes

    n_clusters = list(n_clusters)
    sample_sizes = list(sample_sizes)
    if not (len(n_clusters) == n_levels == len(sample_sizes)):
        raise ValueError(
            "hierarchy config is inconsistent: "
            f"len(n_clusters)={len(n_clusters)}, n_levels={n_levels}, "
            f"len(sample_sizes)={len(sample_sizes)} must all be equal"
        )
    if n_levels <= 0 or any(c <= 0 for c in n_clusters) or any(s <= 0 for s in sample_sizes):
        raise ValueError("hierarchy config must have n_levels > 0 and all values > 0")
    return n_clusters, n_levels, sample_sizes


def _read_knobs(hierarchy: Any) -> dict[str, Any]:
    """Read the optional knobs (`n_resamples`, `init_method`, `resample_strategy`,
    `leaf_sampling`) from a dict or duck-typed object; absent keys take the defaults.
    Invalid values raise `ValueError` naming the key and the valid values.
    """
    knobs: dict[str, Any] = {}
    for key, default in _KNOB_DEFAULTS.items():
        if isinstance(hierarchy, dict):
            knobs[key] = hierarchy.get(key, default)
        else:
            knobs[key] = getattr(hierarchy, key, default)
    if isinstance(knobs["n_resamples"], bool) or not isinstance(knobs["n_resamples"], int):
        raise ValueError(f"hierarchy n_resamples must be an int >= 0, got {knobs['n_resamples']!r}")
    if knobs["n_resamples"] < 0:
        raise ValueError(f"hierarchy n_resamples must be an int >= 0, got {knobs['n_resamples']!r}")
    for key, valid in _VALID_CHOICES.items():
        if knobs[key] not in valid:
            raise ValueError(f"hierarchy {key} must be one of {list(valid)}, got {knobs[key]!r}")
    return knobs


def _sort_leaf_clusters_by_distance(level: dict[str, Any], data: torch.Tensor) -> dict[str, Any]:
    """Return a copy of the level-1 result with each cluster's ids sorted by increasing
    L2 distance to its centroid (stable sort), as the vendored `"c"` sampling expects.
    """
    centroids = level["centroids"]
    sorted_clusters = np.empty(len(level["clusters"]), dtype=object)
    for i, members in enumerate(level["clusters"]):
        members = np.asarray(members, dtype=np.int64)
        if members.size == 0:
            sorted_clusters[i] = members
            continue
        dist = torch.cdist(data[members], centroids[i, None].to(data.dtype)).flatten()
        order = torch.argsort(dist, stable=True).cpu().numpy()
        sorted_clusters[i] = members[order]
    return {**level, "clusters": sorted_clusters}


class HierarchicalKMeansSelection(SelectionStrategy):
    """Hierarchical k-means with resampling, then balanced (water-filling) hierarchical sampling.

    Parameters
    ----------
    hierarchy:
        Object with `n_clusters: Sequence[int]`, `n_levels: int`,
        `sample_sizes: Sequence[int]` attributes (e.g.
        `dalmax.config.schema.HierarchyConfig`), or an equivalent dict.
        See the module docstring for `sample_sizes` semantics.
    device:
        Torch device string (`"cpu"` or `"cuda"`) the intermediate data
        tensor is placed on. CPU is fully supported (see module docstring).
    n_resamples:
        Number of resampling iterations per level. ``None`` (default) means "take
        it from `hierarchy` (key/attribute `n_resamples`), else 10". An explicit
        value overrides nothing present in `hierarchy`: a value in `hierarchy`
        always takes precedence, the argument is only the fallback.

    The hierarchy may also carry `init_method`, `resample_strategy` and
    `leaf_sampling` (see the module docstring, "Optional knobs").
    """

    def __init__(self, hierarchy: Any, device: str = "cpu", n_resamples: int | None = None):
        self.n_clusters, self.n_levels, self.sample_sizes = _read_hierarchy(hierarchy)
        knobs = _read_knobs(hierarchy)
        has_n_resamples = (
            "n_resamples" in hierarchy
            if isinstance(hierarchy, dict)
            else hasattr(hierarchy, "n_resamples")
        )
        if not has_n_resamples and n_resamples is not None:
            knobs["n_resamples"] = n_resamples
        self.device = device
        self.n_resamples: int = knobs["n_resamples"]
        self.init_method: str = knobs["init_method"]
        self.resample_strategy: str = knobs["resample_strategy"]
        self.leaf_sampling: str = knobs["leaf_sampling"]

    def select(
        self,
        embeddings: np.ndarray,
        ids: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = len(ids)
        budget = min(budget, n)
        if budget <= 0:
            return np.array([], dtype=ids.dtype)
        if budget == n:
            return np.array(ids, copy=True)

        # See module docstring "RNG isolation": the vendored pipeline consults
        # global random state, not an injectable generator, so we reseed it
        # deterministically from the caller's rng right before using it.
        seed = int(rng.integers(0, _MAX_SEED))
        random.seed(seed)
        np.random.seed(seed)

        data = torch.tensor(
            np.asarray(embeddings, dtype=np.float32), device=self.device, dtype=torch.float32
        )
        clusters_by_level = hkmg.hierarchical_kmeans_with_resampling(
            data=data,
            n_clusters=list(self.n_clusters),
            n_levels=self.n_levels,
            sample_sizes=list(self.sample_sizes),
            n_resamples=self.n_resamples,
            init_method=self.init_method,
            sample_strategy=self.resample_strategy,
            verbose=False,
        )
        sampling_strategy = "r" if self.leaf_sampling == "random" else "c"
        if sampling_strategy == "c":
            clusters_by_level = [
                _sort_leaf_clusters_by_distance(clusters_by_level[0], data),
                *clusters_by_level[1:],
            ]
        cl = HierarchicalCluster.from_dict(clusters_by_level)
        positions = np.asarray(
            hs.hierarchical_sampling(cl, target_size=budget, sampling_strategy=sampling_strategy),
            dtype=np.int64,
        )
        positions = np.unique(positions)

        # Defensive top-up/trim: `hierarchical_sampling` is expected to return
        # exactly `budget` unique positions, but its leaf-cluster branch can
        # return fewer when a leaf cluster is smaller than its allocated
        # sub-quota (see `hierarchical_sampling.py::recursive_hierarchical_sampling`).
        # We guarantee the `SelectionStrategy` contract (unique subset of
        # `ids`, length == min(budget, N)) regardless of that edge case.
        if positions.size < budget:
            remaining = np.setdiff1d(np.arange(n), positions, assume_unique=True)
            extra = rng.choice(remaining, size=budget - positions.size, replace=False)
            positions = np.concatenate([positions, extra])
        elif positions.size > budget:
            positions = rng.choice(positions, size=budget, replace=False)

        return ids[positions]
