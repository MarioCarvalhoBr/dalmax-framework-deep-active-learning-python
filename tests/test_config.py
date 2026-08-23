"""Tests for `dalmax.config.schema` and `dalmax.config.loader`.

Covers: parsing the two existing on-disk params JSON files
(`params_df_gpu_0.json`, `files_config/params_micro.json`) unmodified,
backward-compat `config_kmh` -> hierarchical `selection`, the
no-`config_kmh`-no-`selection` -> `flat_closest` default (so a dataset like
`CIFAR10` never gets an invalid `hierarchical`/`hierarchy=None` combination),
`ConfigError` on a missing-hierarchy `selection.method == "hierarchical"`,
`ConfigError` on an `n_levels` mismatch, and JSON round-trippability of
`to_dict`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dalmax.config.loader import load_experiment_config, to_dict
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

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_GPU_0 = REPO_ROOT / "params_df_gpu_0.json"
PARAMS_MICRO = REPO_ROOT / "files_config" / "params_micro.json"


def _load(params_json: Path, dataset_name: str, **overrides) -> ExperimentConfig:
    kwargs = dict(
        strategy_name="SSRAEKmeansHCSampling",
        seed=1,
        n_init_labeled=100,
        n_query=100,
        n_round=8,
        dir_results="results/dalmax1/",
        device="cpu",
    )
    kwargs.update(overrides)
    return load_experiment_config(params_json, dataset_name, **kwargs)


# --- Parsing existing on-disk params files, unmodified -----------------------


def test_loads_daninhas_from_params_df_gpu_0_without_modifying_the_file():
    original_text = PARAMS_GPU_0.read_text()

    config = _load(PARAMS_GPU_0, "DANINHAS")

    assert PARAMS_GPU_0.read_text() == original_text
    assert config.dataset.name == "DANINHAS"
    assert config.dataset.data_dir == "DATA/daninhas_full/"
    assert config.dataset.n_classes == 5
    assert config.dataset.n_epoch == 10
    assert config.dataset.n_drop == 10
    assert config.dataset.train_args == TrainArgs(batch_size=256, num_workers=4)
    assert config.dataset.test_args == TrainArgs(batch_size=256, num_workers=4)
    assert config.dataset.optimizer_args == OptimizerArgs(lr=0.05, momentum=0.3)
    # config_kmh present, no explicit "selection" -> backward-compat hierarchical.
    assert config.dataset.selection.method == "hierarchical"
    assert config.dataset.selection.hierarchy == HierarchyConfig(
        n_clusters=(600, 200, 100), n_levels=3, sample_sizes=(30, 15, 2)
    )
    # No "embedding" key in the file -> defaults.
    assert config.dataset.embedding == EmbeddingConfig(extractor="ssrae", q=13, variant="full")
    assert config.device == "cpu"
    assert config.params_json_path == str(PARAMS_GPU_0)


def test_loads_cifar10_from_params_df_gpu_0_defaults_selection_to_flat_closest():
    original_text = PARAMS_GPU_0.read_text()

    config = _load(PARAMS_GPU_0, "CIFAR10")

    assert PARAMS_GPU_0.read_text() == original_text
    assert config.dataset.name == "CIFAR10"
    assert config.dataset.data_dir == "DATA/DATA_CIFAR10/"
    assert config.dataset.n_classes == 10
    assert config.dataset.n_epoch == 20
    # No config_kmh in the CIFAR10 block: a hierarchical default (the
    # SelectionConfig dataclass's own default) would be invalid because
    # hierarchy=None -- must default to flat_closest instead.
    assert config.dataset.selection == SelectionConfig(method="flat_closest", hierarchy=None)
    assert config.dataset.embedding == EmbeddingConfig()


def test_loads_daninhas_from_params_micro_without_modifying_the_file():
    original_text = PARAMS_MICRO.read_text()

    config = _load(
        PARAMS_MICRO,
        "DANINHAS",
        n_init_labeled=10,
        n_query=4,
        n_round=1,
        dir_results="results/smoke/",
    )

    assert PARAMS_MICRO.read_text() == original_text
    assert config.dataset.data_dir == "DATA/daninhas_micro/"
    assert config.dataset.n_epoch == 1
    assert config.dataset.selection.method == "hierarchical"
    assert config.dataset.selection.hierarchy == HierarchyConfig(
        n_clusters=(8, 4), n_levels=2, sample_sizes=(4, 2)
    )


# --- device resolution --------------------------------------------------------


def test_device_auto_resolves_to_cpu_or_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _load(PARAMS_GPU_0, "CIFAR10", device="auto")
    assert config.device == "cpu"


def test_device_invalid_value_raises_config_error():
    with pytest.raises(ConfigError):
        _load(PARAMS_GPU_0, "CIFAR10", device="tpu")


# --- Missing dataset / missing keys -> ConfigError, never bare KeyError ------


def test_unknown_dataset_name_raises_config_error_listing_available_datasets():
    with pytest.raises(ConfigError, match="CIFAR10"):
        _load(PARAMS_GPU_0, "NOT_A_DATASET")


def test_missing_required_dataset_key_raises_config_error(tmp_path):
    payload = {
        "DANINHAS": {
            # "data_dir" missing on purpose.
            "n_epoch": 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 256, "num_workers": 4},
            "test_args": {"batch_size": 256, "num_workers": 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        }
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    with pytest.raises(ConfigError, match="data_dir"):
        _load(params_path, "DANINHAS")


def test_malformed_json_raises_config_error(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text("{not valid json")

    with pytest.raises(ConfigError):
        _load(params_path, "DANINHAS")


# --- selection.method == "hierarchical" without hierarchy -> ConfigError ----


def test_selection_hierarchical_without_hierarchy_raises_config_error_naming_key(tmp_path):
    payload = {
        "DANINHAS": {
            "data_dir": "DATA/daninhas_full/",
            "n_epoch": 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 256, "num_workers": 4},
            "test_args": {"batch_size": 256, "num_workers": 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
            "selection": {"method": "hierarchical"},
        }
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    with pytest.raises(ConfigError, match="hierarchy"):
        _load(params_path, "DANINHAS")


def test_hierarchy_n_levels_mismatch_raises_config_error(tmp_path):
    payload = {
        "DANINHAS": {
            "data_dir": "DATA/daninhas_full/",
            "n_epoch": 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 256, "num_workers": 4},
            "test_args": {"batch_size": 256, "num_workers": 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
            "config_kmh": {
                "n_clusters": [600, 200, 100],
                "n_levels": 2,  # mismatch: 3 entries in n_clusters
                "sample_sizes": [30, 15, 2],
            },
        }
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    with pytest.raises(ConfigError, match="n_levels"):
        _load(params_path, "DANINHAS")


def test_hierarchy_config_rejects_mismatched_lengths_directly():
    with pytest.raises(ConfigError, match="n_levels"):
        HierarchyConfig(n_clusters=(600, 200, 100), n_levels=3, sample_sizes=(30, 15))


def test_selection_config_rejects_hierarchical_with_no_hierarchy_directly():
    with pytest.raises(ConfigError, match="hierarchy"):
        SelectionConfig(method="hierarchical", hierarchy=None)


def test_selection_config_rejects_non_hierarchical_with_a_hierarchy():
    hierarchy = HierarchyConfig(n_clusters=(50,), n_levels=1, sample_sizes=(10,))
    with pytest.raises(ConfigError):
        SelectionConfig(method="flat_closest", hierarchy=hierarchy)


def test_embedding_config_rejects_unknown_extractor():
    with pytest.raises(ConfigError):
        EmbeddingConfig(extractor="not_a_real_extractor")


# --- explicit "selection"/"embedding" keys are read ---------------------------


def test_explicit_selection_flat_proportional_is_read(tmp_path):
    payload = {
        "DANINHAS": {
            "data_dir": "DATA/daninhas_full/",
            "n_epoch": 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 256, "num_workers": 4},
            "test_args": {"batch_size": 256, "num_workers": 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
            "selection": {"method": "flat_proportional"},
            "embedding": {"extractor": "resnet_imagenet", "q": None, "variant": "full"},
        }
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    config = _load(params_path, "DANINHAS")

    assert config.dataset.selection == SelectionConfig(method="flat_proportional", hierarchy=None)
    assert config.dataset.embedding == EmbeddingConfig(
        extractor="resnet_imagenet", q=None, variant="full"
    )


# --- per-extractor `q` defaults (HIGH finding: a flat `d.get("q", 13)` -------
# --- silently forced VCTex's legacy Q=(5, 17) down to SSRAE's Q=13) ----------


def _payload_with_embedding(embedding: dict) -> dict:
    return {
        "DANINHAS": {
            "data_dir": "DATA/daninhas_full/",
            "n_epoch": 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 256, "num_workers": 4},
            "test_args": {"batch_size": 256, "num_workers": 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
            "embedding": embedding,
        }
    }


def test_embedding_q_defaults_to_13_for_ssrae_when_q_key_absent(tmp_path):
    payload = _payload_with_embedding({"extractor": "ssrae"})
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    config = _load(params_path, "DANINHAS")

    assert config.dataset.embedding.extractor == "ssrae"
    assert config.dataset.embedding.q == 13


def test_embedding_q_defaults_to_5_17_for_vctex_when_q_key_absent(tmp_path):
    payload = _payload_with_embedding({"extractor": "vctex"})
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    config = _load(params_path, "DANINHAS")

    assert config.dataset.embedding.extractor == "vctex"
    assert config.dataset.embedding.q == (5, 17)


def test_embedding_q_defaults_to_none_for_resnet_imagenet_when_q_key_absent(tmp_path):
    payload = _payload_with_embedding({"extractor": "resnet_imagenet"})
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    config = _load(params_path, "DANINHAS")

    assert config.dataset.embedding.extractor == "resnet_imagenet"
    assert config.dataset.embedding.q is None


def test_embedding_q_explicit_value_overrides_the_per_extractor_default(tmp_path):
    payload = _payload_with_embedding({"extractor": "vctex", "q": [3, 9]})
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    config = _load(params_path, "DANINHAS")

    assert config.dataset.embedding.q == (3, 9)


def test_embedding_defaults_to_ssrae_q_13_when_embedding_key_entirely_absent(tmp_path):
    payload = {
        "DANINHAS": {
            "data_dir": "DATA/daninhas_full/",
            "n_epoch": 10,
            "n_drop": 10,
            "n_classes": 5,
            "train_args": {"batch_size": 256, "num_workers": 4},
            "test_args": {"batch_size": 256, "num_workers": 4},
            "optimizer_args": {"lr": 0.05, "momentum": 0.3},
        }
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(payload))

    config = _load(params_path, "DANINHAS")

    assert config.dataset.embedding == EmbeddingConfig(extractor="ssrae", q=13, variant="full")


# --- to_dict is JSON round-trippable ------------------------------------------


def test_to_dict_is_json_round_trippable_for_daninhas():
    config = _load(PARAMS_GPU_0, "DANINHAS")
    as_dict = to_dict(config)

    round_tripped = json.loads(json.dumps(as_dict))

    assert round_tripped == as_dict
    assert as_dict["dataset"]["name"] == "DANINHAS"
    assert as_dict["dataset"]["selection"]["hierarchy"]["n_clusters"] == [600, 200, 100]
    assert isinstance(as_dict["dataset"]["selection"]["hierarchy"]["n_clusters"], list)


def test_to_dict_is_json_round_trippable_for_cifar10():
    config = _load(PARAMS_GPU_0, "CIFAR10")
    as_dict = to_dict(config)

    round_tripped = json.loads(json.dumps(as_dict))

    assert round_tripped == as_dict
    assert as_dict["dataset"]["selection"]["method"] == "flat_closest"
    assert as_dict["dataset"]["selection"]["hierarchy"] is None


# --- DatasetConfig / ExperimentConfig direct validation ----------------------


def test_dataset_config_rejects_non_positive_n_classes():
    with pytest.raises(ConfigError):
        DatasetConfig(
            name="X",
            data_dir="DATA/x/",
            n_classes=0,
            n_epoch=1,
            n_drop=1,
            train_args=TrainArgs(batch_size=1, num_workers=0),
            test_args=TrainArgs(batch_size=1, num_workers=0),
            optimizer_args=OptimizerArgs(lr=0.1, momentum=0.1),
            embedding=EmbeddingConfig(),
            selection=SelectionConfig(method="flat_closest", hierarchy=None),
        )


def test_experiment_config_rejects_unresolved_auto_device():
    dataset = DatasetConfig(
        name="X",
        data_dir="DATA/x/",
        n_classes=2,
        n_epoch=1,
        n_drop=1,
        train_args=TrainArgs(batch_size=1, num_workers=0),
        test_args=TrainArgs(batch_size=1, num_workers=0),
        optimizer_args=OptimizerArgs(lr=0.1, momentum=0.1),
        embedding=EmbeddingConfig(),
        selection=SelectionConfig(method="flat_closest", hierarchy=None),
    )
    with pytest.raises(ConfigError):
        ExperimentConfig(
            dataset=dataset,
            strategy_name="RandomSampling",
            seed=1,
            n_init_labeled=10,
            n_query=10,
            n_round=1,
            dir_results="results/x/",
            device="auto",
            params_json_path="params.json",
        )


# --- n_round >= 0 is allowed (legacy demo.py allowed `--n_round 0`: train/ ---
# --- evaluate once, no query rounds); n_round < 0 still rejected; n_query ---
# --- must still be > 0 ---------------------------------------------------


def _minimal_dataset_config() -> DatasetConfig:
    return DatasetConfig(
        name="X",
        data_dir="DATA/x/",
        n_classes=2,
        n_epoch=1,
        n_drop=1,
        train_args=TrainArgs(batch_size=1, num_workers=0),
        test_args=TrainArgs(batch_size=1, num_workers=0),
        optimizer_args=OptimizerArgs(lr=0.1, momentum=0.1),
        embedding=EmbeddingConfig(),
        selection=SelectionConfig(method="flat_closest", hierarchy=None),
    )


def test_experiment_config_allows_n_round_zero():
    config = ExperimentConfig(
        dataset=_minimal_dataset_config(),
        strategy_name="RandomSampling",
        seed=1,
        n_init_labeled=10,
        n_query=10,
        n_round=0,
        dir_results="results/x/",
        device="cpu",
        params_json_path="params.json",
    )
    assert config.n_round == 0


def test_experiment_config_rejects_negative_n_round():
    with pytest.raises(ConfigError, match="n_round"):
        ExperimentConfig(
            dataset=_minimal_dataset_config(),
            strategy_name="RandomSampling",
            seed=1,
            n_init_labeled=10,
            n_query=10,
            n_round=-1,
            dir_results="results/x/",
            device="cpu",
            params_json_path="params.json",
        )


def test_experiment_config_still_rejects_non_positive_n_query():
    with pytest.raises(ConfigError, match="n_query"):
        ExperimentConfig(
            dataset=_minimal_dataset_config(),
            strategy_name="RandomSampling",
            seed=1,
            n_init_labeled=10,
            n_query=0,
            n_round=0,
            dir_results="results/x/",
            device="cpu",
            params_json_path="params.json",
        )


def test_load_experiment_config_allows_n_round_zero():
    config = _load(PARAMS_GPU_0, "CIFAR10", n_round=0)
    assert config.n_round == 0
