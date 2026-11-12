"""Load and validate `ExperimentConfig` from the existing params JSON format.

This is a drop-in reader for the params JSON files already on disk
(`files_config/benchmark/params_df_gpu_0.json`, `files_config/benchmark/params_df_gpu_1.json`,
`files_config/params_micro.json`) — the on-disk schema is not changed by this
module. It only adds backward-compatible, optional keys per dataset:
`"embedding"` and `"selection"` (see `.specs/architecture/target-architecture.md`
§2 and `.specs/architecture/refactor-plan.md` Phase 2, item 1), and
`"model_args"` (ADR 0014: model constructor arguments, today only
`{"dropout_p": <float in [0, 1)>}`; absent = the model class's own defaults), and
`"strategy_args"` (ADR 0015: per-strategy hyperparameters, one optional sub-block per
strategy name, e.g. `{"VAAL": {"epochs": 30}}`; absent = `StrategyArgs()` defaults).

Backward-compatibility rule
----------------------------
If a dataset entry has a legacy `"config_kmh"` block and no `"selection"` key,
`selection` is built as `SelectionConfig(method="hierarchical", hierarchy=<config_kmh>)`
— this is exactly what every strategy that read `params[dataset]["config_kmh"]`
today assumes implicitly.

If a dataset entry has neither `"selection"` nor `"config_kmh"` (e.g. `CIFAR10`
in `files_config/benchmark/params_df_gpu_0.json` today), `selection` defaults to
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
    STRATEGY_ARGS_KEYS,
    ConfigError,
    DatasetConfig,
    EmbeddingConfig,
    ExperimentConfig,
    HierarchyConfig,
    ModelArgs,
    OptimizerArgs,
    SelectionConfig,
    StrategyArgs,
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


_HIERARCHY_DEFAULTS = HierarchyConfig(n_clusters=(1,), n_levels=1, sample_sizes=(1,))


def _as_int(value: Any, key: str) -> int:
    """Parse an int-valued key, raising ConfigError (naming the key) on garbage."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ConfigError(f"{key} must be an integer, got {value!r}")
    return int(value)


def _build_hierarchy_config(d: dict[str, Any], context: str) -> HierarchyConfig:
    _require_keys(d, _REQUIRED_HIERARCHY_KEYS, context)
    return HierarchyConfig(
        n_clusters=tuple(int(n) for n in d["n_clusters"]),
        n_levels=int(d["n_levels"]),
        sample_sizes=tuple(int(n) for n in d["sample_sizes"]),
        n_resamples=_HIERARCHY_DEFAULTS.n_resamples
        if "n_resamples" not in d
        else _as_int(d["n_resamples"], f"{context}.n_resamples"),
        init_method=d.get("init_method", _HIERARCHY_DEFAULTS.init_method),
        resample_strategy=d.get("resample_strategy", _HIERARCHY_DEFAULTS.resample_strategy),
        leaf_sampling=d.get("leaf_sampling", _HIERARCHY_DEFAULTS.leaf_sampling),
    )


# Legacy `Q` per extractor, applied only when the "embedding" block (or the
# whole "embedding" key) omits "q" explicitly. Mirrors the historical
# literals the now-deleted `Data.create_feature_maps_ssrae`/`_vctex` always used: `Q=13` for SSRAE,
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


_MODEL_ARGS_KEYS = tuple(field.name for field in dataclasses.fields(ModelArgs))


def _build_model_args(d: Any, context: str) -> ModelArgs:
    """Parse the optional `"model_args"` block (absent -> every argument unset).

    Fails fast on a non-object block, an unknown key (`"_comment"`-style keys
    excepted) or a non-numeric `dropout_p`; the range check is `ModelArgs`'s.
    """
    if d is None:
        return ModelArgs()
    if not isinstance(d, dict):
        raise ConfigError(f"{context} must be a JSON object, got {d!r}")
    unknown = sorted(k for k in d if k not in _MODEL_ARGS_KEYS and not str(k).startswith("_"))
    if unknown:
        raise ConfigError(f"{context}: unknown key(s) {unknown}; valid keys: {list(_MODEL_ARGS_KEYS)}")
    dropout_p = d.get("dropout_p")
    if dropout_p is not None and (isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float))):
        raise ConfigError(f"{context}.dropout_p must be a number, got {dropout_p!r}")
    return ModelArgs(dropout_p=None if dropout_p is None else float(dropout_p))


def _parse_typed_value(value: Any, default: Any, key: str) -> Any:
    """Coerce a JSON value to the type of the dataclass field's default, failing fast.

    `int` fields take integers only, `float` fields take any number, `str` fields
    take strings only; booleans are never numbers here.
    """
    if isinstance(default, bool) or isinstance(value, bool):
        raise ConfigError(f"{key} must be a {type(default).__name__}, got {value!r}")
    if isinstance(default, int):
        return _as_int(value, key)
    if isinstance(default, float):
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{key} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{key} must be a string, got {value!r}")
        return value
    raise ConfigError(f"{key}: unsupported field type {type(default).__name__}")


def _build_args_block(block_cls: type, d: Any, context: str) -> Any:
    """Build one `StrategyArgs` sub-dataclass from its JSON object (absent keys keep
    their defaults; unknown keys, `_comment`-style keys excepted, are rejected)."""
    if not isinstance(d, dict):
        raise ConfigError(f"{context} must be a JSON object, got {d!r}")
    defaults = block_cls()
    valid = [f.name for f in dataclasses.fields(block_cls)]
    unknown = sorted(k for k in d if k not in valid and not str(k).startswith("_"))
    if unknown:
        raise ConfigError(f"{context}: unknown key(s) {unknown}; valid keys: {valid}")
    values = {
        name: _parse_typed_value(d[name], getattr(defaults, name), f"{context}.{name}")
        for name in valid
        if name in d
    }
    return block_cls(**values)


def _build_strategy_args(d: Any, context: str) -> StrategyArgs:
    """Parse the optional `"strategy_args"` block (ADR 0015), keyed by strategy name."""
    if d is None:
        return StrategyArgs()
    if not isinstance(d, dict):
        raise ConfigError(f"{context} must be a JSON object, got {d!r}")
    unknown = sorted(k for k in d if k not in STRATEGY_ARGS_KEYS and not str(k).startswith("_"))
    if unknown:
        raise ConfigError(
            f"{context}: unknown strategy name(s) {unknown}; valid: {sorted(STRATEGY_ARGS_KEYS)}"
        )
    field_types = {f.name: f.default_factory for f in dataclasses.fields(StrategyArgs)}
    blocks = {
        STRATEGY_ARGS_KEYS[name]: _build_args_block(
            field_types[STRATEGY_ARGS_KEYS[name]], d[name], f"{context}.{name}"
        )
        for name in STRATEGY_ARGS_KEYS
        if name in d
    }
    return StrategyArgs(**blocks)


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
        docstring), e.g. `files_config/benchmark/params_df_gpu_0.json` or `files_config/params_micro.json`.
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
    model_args = _build_model_args(dataset_dict.get("model_args"), f"{context}.model_args")
    strategy_args = _build_strategy_args(
        dataset_dict.get("strategy_args"), f"{context}.strategy_args"
    )

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
        model_args=model_args,
        strategy_args=strategy_args,
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
