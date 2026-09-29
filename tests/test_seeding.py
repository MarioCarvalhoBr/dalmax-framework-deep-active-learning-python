"""Tests for `dalmax.seeding`.

`seed_everything` must make `random`, `np.random` (global), and `torch` draws
reproducible across two independent calls with the same seed, and must return
a seeded `np.random.Generator`. `derive_seed` must be deterministic for a
given `(seed, tag)` pair and differ across tags/seeds.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from dalmax.seeding import derive_seed, seed_everything


def test_seed_everything_returns_a_numpy_generator():
    rng = seed_everything(123)
    assert isinstance(rng, np.random.Generator)


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(42)
    first = [random.random() for _ in range(5)]

    seed_everything(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_seed_everything_makes_global_numpy_random_reproducible():
    seed_everything(7)
    first = np.random.rand(5)

    seed_everything(7)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_seed_everything_makes_returned_generator_reproducible():
    rng1 = seed_everything(99)
    draw1 = rng1.random(5)

    rng2 = seed_everything(99)
    draw2 = rng2.random(5)

    np.testing.assert_array_equal(draw1, draw2)


def test_seed_everything_makes_torch_reproducible():
    seed_everything(2024)
    first = torch.rand(5)

    seed_everything(2024)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_seed_everything_sets_cudnn_determinism_flags():
    seed_everything(1)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # Never the blanket cudnn.enabled = False shortcut this replaces
    # (`.claude/rules/reproducibility.md`).
    assert torch.backends.cudnn.enabled is True


def test_different_seeds_produce_different_draws():
    rng1 = seed_everything(1)
    rng2 = seed_everything(2)
    assert not np.array_equal(rng1.random(10), rng2.random(10))


# --- derive_seed --------------------------------------------------------------


def test_derive_seed_is_deterministic():
    a = derive_seed(1, "hierarchical_kmeans")
    b = derive_seed(1, "hierarchical_kmeans")
    assert a == b


def test_derive_seed_differs_per_tag():
    a = derive_seed(1, "tag_a")
    b = derive_seed(1, "tag_b")
    assert a != b


def test_derive_seed_differs_per_seed():
    a = derive_seed(1, "same_tag")
    b = derive_seed(2, "same_tag")
    assert a != b


def test_derive_seed_returns_a_valid_uint32_range_int():
    value = derive_seed(1, "some_tag")
    assert isinstance(value, int)
    assert 0 <= value <= 2**32 - 1


# --- GPU determinism settings (2026-09-29) --------------------------------------


def test_seed_everything_enables_deterministic_algorithms_warn_only(monkeypatch) -> None:
    import os

    import torch

    from dalmax.seeding import CUBLAS_WORKSPACE_CONFIG, determinism_state, seed_everything

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(7)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG == ":4096:8"
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert determinism_state(7) == {
            "seed": 7,
            "deterministic_algorithms": "warn_only",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "fill_uninitialized_memory": False,
            "nondeterministic_op_warnings": [],
        }
    finally:
        torch.use_deterministic_algorithms(previous)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_seed_everything_does_not_clobber_an_operator_set_cublas_config(monkeypatch) -> None:
    import os

    import torch

    from dalmax.seeding import seed_everything

    previous = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    try:
        seed_everything(1)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        torch.use_deterministic_algorithms(previous)


def test_record_nondeterministic_ops_collects_unique_op_names_and_forwards_others() -> None:
    import warnings

    from dalmax.seeding import record_nondeterministic_ops

    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        with record_nondeterministic_ops() as ops:
            for _ in range(2):
                warnings.warn("cumsum_cuda_kernel does not have a deterministic implementation, but", UserWarning, stacklevel=1)
            warnings.warn("index_add_cuda_ does not have a deterministic implementation, but", UserWarning, stacklevel=1)
            warnings.warn("something unrelated", UserWarning, stacklevel=1)
    assert ops == ["cumsum_cuda_kernel", "index_add_cuda_"]
    assert [str(w.message) for w in outer] == ["something unrelated"]
