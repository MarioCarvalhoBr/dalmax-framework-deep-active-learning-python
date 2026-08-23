"""Load and validate `ExperimentConfig` from the existing params JSON format.

This is a drop-in reader for the params JSON files already on disk
(`params_df_gpu_0.json`, `params_df_gpu_1.json`,
`files_config/params_micro.json`) — the on-disk schema is not changed by this
module. It only adds two backward-compatible, optional keys per dataset:
`"embedding"` and `"selection"` (see `.specs/architecture/target-architecture.md`
§2 and `.specs/architecture/refactor-plan.md` Phase 2, item 1).

Backward-compatibility rule
----------------------------
If a dataset entry has a legacy `"config_kmh"` block and no `"selection"` key,
`selection` is built as `SelectionConfig(method="hierarchical", hierarchy=<config_kmh>)`
— this is exactly what every strategy that read `params[dataset]["config_kmh"]`
today assumes implicitly.

If a dataset entry has neither `"selection"` nor `"config_kmh"` (e.g. `CIFAR10`
in `params_df_gpu_0.json` today), `selection` defaults to
`SelectionConfig(method="flat_closest", hierarchy=None)` rather than the
`SelectionConfig` dataclass's own `method="hierarchical"` default — a
hierarchical method with `hierarchy=None` is invalid
(`SelectionConfig.__post_init__` rejects it), so silently defaulting a
dataset with no hierarchy config to "hierarchical" would always raise. A
selection-free dataset entry means "run whatever flat/non-hierarchical
strategy is requested at the CLI"; `flat_closest` is the safe, always-valid
default for that case.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from dalmax.config.schema import (
    ConfigError,
    DatasetConfig,
    EmbeddingConfig,
    ExperimentConfig,
    HierarchyConfig,
    OptimizerArgs,
    SelectionConfig,
    TrainArgs,
)

_REQUIRED_DATASET_KEYS = (
    "data_dir",
    "n_epoch",
    "n_drop",
    "n_classes",
    "train_args",
    "test_args",
    "optimizer_args",
)
_REQUIRED_TRAIN_ARGS_KEYS = ("batch_size", "num_workers")
_REQUIRED_OPTIMIZER_ARGS_KEYS = ("lr", "momentum")
_REQUIRED_HIERARCHY_KEYS = ("n_clusters", "n_levels", "sample_sizes")


def _require_keys(d: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in d]
    if missing:
        raise ConfigError(f"{context}: missing required key(s) {missing}")


def _build_train_args(d: dict[str, Any], context: str) -> TrainArgs:
    _require_keys(d, _REQUIRED_TRAIN_ARGS_KEYS, context)
    return TrainArgs(batch_size=int(d["batch_size"]), num_workers=int(d["num_workers"]))


def _build_optimizer_args(d: dict[str, Any], context: str) -> OptimizerArgs:
    _require_keys(d, _REQUIRED_OPTIMIZER_ARGS_KEYS, context)
    return OptimizerArgs(lr=float(d["lr"]), momentum=float(d["momentum"]))


def _build_hierarchy_config(d: dict[str, Any], context: str) -> HierarchyConfig:
    _require_keys(d, _REQUIRED_HIERARCHY_KEYS, context)
    return HierarchyConfig(
        n_clusters=tuple(int(n) for n in d["n_clusters"]),
        n_levels=int(d["n_levels"]),
        sample_sizes=tuple(int(n) for n in d["sample_sizes"]),
    )


# Legacy `Q` per extractor, applied only when the "embedding" block (or the
# whole "embedding" key) omits "q" explicitly. Mirrors the historical
# literals `utils/data.py` always used: `Q=13` for SSRAE,
# `Q=[5, 17]` for VCTex; ResNet-ImageNet has no `Q` (fixed 2048-d penultimate
# layer). A flat `d.get("q", 13)` here would silently force VCTex's Q down
# to SSRAE's 13 whenever "q" is absent — this table is the fix (see
# `.claude/rules/reproducibility.md`).
_EXTRACTOR_DEFAULT_Q: dict[str, Any] = {
    "ssrae": 13,
    "vctex": (5, 17),
    "resnet_imagenet": None,
}


def _build_embedding_config(d: dict[str, Any] | None) -> EmbeddingConfig:
    d = d or {}
    extractor = d.get("extractor", "ssrae")
    if "q" in d:
        q = d["q"]
        if isinstance(q, list):
            q = tuple(int(v) for v in q)
    else:
        q = _EXTRACTOR_DEFAULT_Q.get(extractor)
    return EmbeddingConfig(
        extractor=extractor,
        q=q,
        variant=d.get("variant", "full"),
    )


def _build_selection_config(dataset_dict: dict[str, Any], context: str) -> SelectionConfig:
    selection_dict = dataset_dict.get("selection")
    config_kmh = dataset_dict.get("config_kmh")

    if selection_dict is not None:
        method = selection_dict.get("method", "hierarchical")
        hierarchy_dict = selection_dict.get("hierarchy")
        if method == "hierarchical":
            if hierarchy_dict is None:
                raise ConfigError(
                    f"{context}.selection: method is 'hierarchical' but 'hierarchy' is "
                    "missing (missing key: 'hierarchy')"
                )
            hierarchy = _build_hierarchy_config(hierarchy_dict, f"{context}.selection.hierarchy")
        else:
            hierarchy = None
        return SelectionConfig(method=method, hierarchy=hierarchy)

    if config_kmh is not None:
        # Backward compat: a legacy `config_kmh` block with no `selection` key
        # implies hierarchical selection using that hierarchy (see module docstring).
        hierarchy = _build_hierarchy_config(config_kmh, f"{context}.config_kmh")
        return SelectionConfig(method="hierarchical", hierarchy=hierarchy)

    # Neither key present: default to a method that is valid without a
    # hierarchy (see module docstring) instead of the SelectionConfig
    # dataclass's own "hierarchical" default.
    return SelectionConfig(method="flat_closest", hierarchy=None)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cuda", "cpu"):
        raise ConfigError(f"device must be 'auto', 'cuda', or 'cpu', got {device!r}")
    return device


def load_experiment_config(
    params_json: str | Path,
    dataset_name: str,
    *,
    strategy_name: str,
    seed: int,
    n_init_labeled: int,
    n_query: int,
    n_round: int,
    dir_results: str,
    device: str = "auto",
) -> ExperimentConfig:
    """Build a validated `ExperimentConfig` from an existing params JSON file.

    Parameters
    ----------
    params_json:
        Path to a params JSON file in the existing on-disk format (see module
        docstring), e.g. `params_df_gpu_0.json` or `files_config/params_micro.json`.
    dataset_name:
        Top-level key to read from `params_json` (e.g. `"DANINHAS"`, `"CIFAR10"`).
    strategy_name, seed, n_init_labeled, n_query, n_round, dir_results:
        The CLI-supplied arguments that, together with `params_json` and the
        git commit, fully determine a run (`.claude/rules/reproducibility.md`).
    device:
        `"cuda"`, `"cpu"`, or `"auto"` (default) to resolve to `"cuda"` iff
        `torch.cuda.is_available()`, else `"cpu"`.

    Raises
    ------
    ConfigError
        If `params_json` cannot be read/parsed, `dataset_name` is not present,
        a required key is missing, or the resulting configuration is invalid
        (e.g. `selection.method == "hierarchical"` without a hierarchy, or a
        hierarchy whose `n_levels` does not match its list lengths). Never a
        bare `KeyError`.
    """
    path = Path(params_json)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"params JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"params JSON at {path} is not valid JSON: {exc}") from exc

    if dataset_name not in raw:
        available = sorted(k for k in raw if not str(k).startswith("_"))
        raise ConfigError(
            f"dataset {dataset_name!r} not found in {path}; available datasets: {available}"
        )

    dataset_dict = raw[dataset_name]
    context = f"{path}:{dataset_name}"
    _require_keys(dataset_dict, _REQUIRED_DATASET_KEYS, context)

    train_args = _build_train_args(dataset_dict["train_args"], f"{context}.train_args")
    test_args = _build_train_args(dataset_dict["test_args"], f"{context}.test_args")
    optimizer_args = _build_optimizer_args(
        dataset_dict["optimizer_args"], f"{context}.optimizer_args"
    )
    embedding = _build_embedding_config(dataset_dict.get("embedding"))
    selection = _build_selection_config(dataset_dict, context)

    dataset_config = DatasetConfig(
        name=dataset_name,
        data_dir=dataset_dict["data_dir"],
        n_classes=int(dataset_dict["n_classes"]),
        n_epoch=int(dataset_dict["n_epoch"]),
        n_drop=int(dataset_dict["n_drop"]),
        train_args=train_args,
        test_args=test_args,
        optimizer_args=optimizer_args,
        embedding=embedding,
        selection=selection,
    )

    return ExperimentConfig(
        dataset=dataset_config,
        strategy_name=strategy_name,
        seed=seed,
        n_init_labeled=n_init_labeled,
        n_query=n_query,
        n_round=n_round,
        dir_results=dir_results,
        device=_resolve_device(device),
        params_json_path=str(path),
    )


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert tuples (and dataclass-derived containers) to lists.

    `dataclasses.asdict` preserves tuple fields as tuples, which round-trip
    through `json.dumps`/`json.loads` as lists — so a dict compared before vs.
    after a JSON round-trip would spuriously differ on every `tuple`-typed
    field (`HierarchyConfig.n_clusters`, `.sample_sizes`, `EmbeddingConfig.q`).
    Converting eagerly makes `to_dict`'s output already fully JSON-native.
    """
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert `config` into a plain, JSON-serializable, round-trippable dict."""
    return _to_jsonable(dataclasses.asdict(config))
