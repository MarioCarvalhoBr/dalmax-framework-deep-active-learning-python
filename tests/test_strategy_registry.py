"""Tests for `dalmax.query_strategies.registry` (`STRATEGY_REGISTRY` /
`build_strategy`).

Building every one of the 12 legacy strategies + 4 presets +
`RepresentationStrategy` on the real DANINHAS/CIFAR10 dataset would be far
too heavy for a fast unit test (real image loading, real ResNet50 download).
Instead this uses a minimal stub dataset/net (module-level, shared with
`tests/test_representation_strategy.py`'s style) so `build_strategy` itself
— provider/selection construction, preset overrides, `ConfigError` on a
missing hierarchy, seed derivation — is exercised without any real training.

`STRATEGY_REGISTRY`'s keys are checked against `dalmax/cli.py`'s
`--strategy_name` choices in `tests/test_registry.py` (via `ast`, without
importing `dalmax.cli`); this file only tests `build_strategy`'s behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from dalmax.config.loader import load_experiment_config
from dalmax.config.schema import ConfigError
from dalmax.query_strategies.registry import (
    GENERIC_REPRESENTATION_NAME,
    LEGACY_STRATEGY_REGISTRY,
    REPRESENTATION_PRESETS,
    STRATEGY_REGISTRY,
    build_strategy,
)
from dalmax.query_strategies.representation import RepresentationStrategy
from dalmax.selection.flat_kmeans_closest import FlatKMeansClosest
from dalmax.selection.hierarchical_kmeans import HierarchicalKMeansSelection
from tests.test_representation_strategy import _FakeDataset

PARAMS_MICRO = "files_config/params_micro.json"


def _config(strategy_name: str, **overrides):
    kwargs = dict(
        strategy_name=strategy_name,
        seed=1,
        n_init_labeled=10,
        n_query=5,
        n_round=1,
        dir_results="results/unused/",
        device="cpu",
    )
    kwargs.update(overrides)
    return load_experiment_config(PARAMS_MICRO, "DANINHAS", **kwargs)


class _FakeNet:
    """Minimal stand-in for `core.deep_learning.DeepLearning`: some legacy
    strategies (`*Dropout`) read `net.params['n_drop']` at construction
    time (`core/query_strategies/least_confidence_dropout.py` and
    friends)."""

    def __init__(self) -> None:
        self.params = {"n_drop": 10}


def _dataset_and_net():
    dataset = _FakeDataset(20)
    dataset.labeled_idxs[:5] = True
    return dataset, _FakeNet()


def test_strategy_registry_covers_all_legacy_names():
    assert set(LEGACY_STRATEGY_REGISTRY) <= set(STRATEGY_REGISTRY)


def test_strategy_registry_covers_presets_and_generic_name():
    for name in REPRESENTATION_PRESETS:
        assert STRATEGY_REGISTRY[name] is RepresentationStrategy
    assert STRATEGY_REGISTRY[GENERIC_REPRESENTATION_NAME] is RepresentationStrategy


@pytest.mark.parametrize("name", sorted(LEGACY_STRATEGY_REGISTRY))
def test_build_strategy_legacy_names_construct_exact_legacy_class(name):
    dataset, net = _dataset_and_net()
    config = _config(name)
    rng = np.random.default_rng(0)
    strategy = build_strategy(name, dataset, net, config, logger=None, rng=rng)
    assert type(strategy) is LEGACY_STRATEGY_REGISTRY[name]
    assert strategy.dataset is dataset
    assert strategy.net is net


def test_build_strategy_flat_presets_use_flat_closest_selection():
    for name, (extractor, _method) in (
        ("SSRAEKmeansSampling", ("ssrae", "flat_closest")),
        ("VCTexKmeansSampling", ("vctex", "flat_closest")),
    ):
        dataset, net = _dataset_and_net()
        config = _config(name)
        rng = np.random.default_rng(0)
        strategy = build_strategy(name, dataset, net, config, logger=None, rng=rng)
        assert isinstance(strategy, RepresentationStrategy)
        assert strategy.embedding_provider.name == extractor
        assert isinstance(strategy.selection, FlatKMeansClosest)


def test_build_strategy_flat_presets_pin_legacy_q_ignoring_config_embedding():
    # params_micro.json's DANINHAS entry has no "embedding" key at all, so
    # config.dataset.embedding defaults to EmbeddingConfig(extractor="ssrae",
    # q=13, ...) via dalmax.config.loader (per-extractor default). The
    # VCTex presets must still build a VCTexProvider with the legacy
    # Q=(5, 17), never SSRAE's default Q=13 -- this is exactly the HIGH
    # finding this test guards against.
    dataset, net = _dataset_and_net()
    config = _config("VCTexKmeansSampling")
    assert "embedding" not in _raw_daninhas_entry()  # no "embedding" key present at all
    rng = np.random.default_rng(0)
    strategy = build_strategy(
        "VCTexKmeansSampling", dataset, net, config, logger=None, rng=rng
    )
    assert strategy.embedding_provider.name == "vctex"
    assert strategy.embedding_provider.q == (5, 17)

    dataset, net = _dataset_and_net()
    config = _config("SSRAEKmeansSampling")
    rng = np.random.default_rng(0)
    strategy = build_strategy(
        "SSRAEKmeansSampling", dataset, net, config, logger=None, rng=rng
    )
    assert strategy.embedding_provider.name == "ssrae"
    assert strategy.embedding_provider.q == 13


def test_build_strategy_hc_presets_pin_legacy_q_ignoring_config_embedding():
    dataset, net = _dataset_and_net()
    config = _config("VCTexKmeansHCSampling")
    rng = np.random.default_rng(0)
    strategy = build_strategy(
        "VCTexKmeansHCSampling", dataset, net, config, logger=None, rng=rng
    )
    assert strategy.embedding_provider.name == "vctex"
    assert strategy.embedding_provider.q == (5, 17)

    dataset, net = _dataset_and_net()
    config = _config("SSRAEKmeansHCSampling")
    rng = np.random.default_rng(0)
    strategy = build_strategy(
        "SSRAEKmeansHCSampling", dataset, net, config, logger=None, rng=rng
    )
    assert strategy.embedding_provider.name == "ssrae"
    assert strategy.embedding_provider.q == 13


def _raw_daninhas_entry() -> dict:
    import json

    with open(PARAMS_MICRO) as f:
        raw = json.load(f)
    return raw["DANINHAS"]


def test_build_strategy_hc_presets_use_hierarchical_selection():
    for name, extractor in (
        ("SSRAEKmeansHCSampling", "ssrae"),
        ("VCTexKmeansHCSampling", "vctex"),
    ):
        dataset, net = _dataset_and_net()
        config = _config(name)
        rng = np.random.default_rng(0)
        strategy = build_strategy(name, dataset, net, config, logger=None, rng=rng)
        assert isinstance(strategy, RepresentationStrategy)
        assert strategy.embedding_provider.name == extractor
        assert isinstance(strategy.selection, HierarchicalKMeansSelection)


def test_build_strategy_hc_preset_raises_config_error_without_hierarchy():
    # CIFAR10 in params_micro.json-equivalent shape has no config_kmh/selection
    # key, so its SelectionConfig defaults to flat_closest/hierarchy=None
    # (dalmax.config.loader's documented default) — requesting a HC preset
    # for it must raise a clear ConfigError, never a bare AttributeError.
    import json
    import tempfile
    from pathlib import Path

    with open(PARAMS_MICRO) as f:
        raw = json.load(f)
    raw["DANINHAS"] = dict(raw["DANINHAS"])
    raw["DANINHAS"].pop("config_kmh", None)

    with tempfile.TemporaryDirectory() as tmp_dir:
        params_path = Path(tmp_dir) / "params_no_hierarchy.json"
        params_path.write_text(json.dumps(raw))

        config = load_experiment_config(
            params_path,
            "DANINHAS",
            strategy_name="SSRAEKmeansHCSampling",
            seed=1,
            n_init_labeled=10,
            n_query=5,
            n_round=1,
            dir_results="results/unused/",
            device="cpu",
        )
        dataset, net = _dataset_and_net()
        rng = np.random.default_rng(0)
        with pytest.raises(ConfigError):
            build_strategy(
                "SSRAEKmeansHCSampling", dataset, net, config, logger=None, rng=rng
            )


def test_build_strategy_generic_name_uses_config_embedding_and_selection_verbatim():
    dataset, net = _dataset_and_net()
    config = _config(GENERIC_REPRESENTATION_NAME)
    # params_micro.json's DANINHAS entry has a config_kmh block -> hierarchical
    # selection by backward-compat default (dalmax.config.loader docstring).
    assert config.dataset.selection.method == "hierarchical"
    rng = np.random.default_rng(0)
    strategy = build_strategy(
        GENERIC_REPRESENTATION_NAME, dataset, net, config, logger=None, rng=rng
    )
    assert isinstance(strategy, RepresentationStrategy)
    assert strategy.embedding_provider.name == config.dataset.embedding.extractor
    assert isinstance(strategy.selection, HierarchicalKMeansSelection)


def test_build_strategy_two_calls_same_seed_derive_same_selection_rng_state():
    dataset1, net1 = _dataset_and_net()
    dataset2, net2 = _dataset_and_net()
    config = _config("SSRAEKmeansSampling", seed=7)

    strategy1 = build_strategy(
        "SSRAEKmeansSampling", dataset1, net1, config, logger=None, rng=np.random.default_rng(0)
    )
    strategy2 = build_strategy(
        "SSRAEKmeansSampling", dataset2, net2, config, logger=None, rng=np.random.default_rng(0)
    )
    # The selection rng is derived from config.seed (dalmax.seeding.derive_seed),
    # not from the `rng` argument passed to build_strategy, so two builds with
    # the same config.seed must draw identical integers from their respective
    # (independently constructed) selection rngs.
    assert strategy1.rng.integers(0, 10_000) == strategy2.rng.integers(0, 10_000)


def test_build_strategy_rejects_unknown_name():
    dataset, net = _dataset_and_net()
    config = _config("RandomSampling")
    with pytest.raises(KeyError):
        build_strategy("__nope__", dataset, net, config, logger=None, rng=np.random.default_rng(0))
