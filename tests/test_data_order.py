"""The data-order contract (KI-47): loaders list files in sorted name order, independent of the
file system, and `dalmax.data_order` hashes the ordered image list machine-independently.

The loader tests build a tiny class-folder dataset in a temporary directory, creating the files in
a shuffled order, and additionally make `os.listdir` return a shuffled listing: the loaded order
must still be the sorted one. `Data.__init__` writes `results/...` relative to the working
directory (KI-21), so every loader test runs inside `tmp_path`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dalmax.data import loaders
from dalmax.data_order import (
    ORDER_CONTRACT,
    data_order_record,
    ordered_paths_hash,
    relative_image_path,
    relative_image_paths,
)

CLASSES = ("DATASET_B", "DATASET_A")
# Names chosen so that the sorted order differs from creation, numeric and case-insensitive orders.
NAMES = ("img10.png", "img2.png", "Img1.png", "a_3.png", "_z.png", "img1.png")


# --- identity helpers -------------------------------------------------------------


def test_relative_path_drops_the_dataset_location() -> None:
    assert relative_image_path("DATA/daninhas_full_fixed/train/DATASET_MAMONA/x.jpg") == "DATASET_MAMONA/x.jpg"
    assert relative_image_path("/content/dalmax/DATA/d/test/C/y.jpg") == "C/y.jpg"
    assert relative_image_paths(["a/b/c.png", "d/e.png"]) == ["b/c.png", "d/e.png"]


def test_order_hash_ignores_the_location_but_not_the_order() -> None:
    here = ["DATA/d/train/c0/a.png", "DATA/d/train/c1/b.png"]
    there = ["/abs/elsewhere/train/c0/a.png", "/abs/elsewhere/train/c1/b.png"]
    assert ordered_paths_hash(here) == ordered_paths_hash(there)
    assert ordered_paths_hash(here) != ordered_paths_hash(list(reversed(here)))
    assert len(ordered_paths_hash(here)) == 12


def test_data_order_record_fields() -> None:
    record = data_order_record(["t/c/a.png", "t/c/b.png"], ["s/c/x.png"])
    assert record == {
        "contract": ORDER_CONTRACT,
        "train_paths_hash": ordered_paths_hash(["t/c/a.png", "t/c/b.png"]),
        "test_paths_hash": ordered_paths_hash(["s/c/x.png"]),
        "n_train": 2,
        "n_test": 1,
    }


# --- loaders ----------------------------------------------------------------------


def _build_dataset(root: Path, creation_seed: int) -> Path:
    """`root/{train,test}/<class>/<name>` with one distinct 4x4 image per file, created in a shuffled order."""
    entries = [(split, cls, name) for split in ("train", "test") for cls in CLASSES for name in NAMES]
    random.Random(creation_seed).shuffle(entries)
    for split, cls, name in entries:
        folder = root / split / cls
        folder.mkdir(parents=True, exist_ok=True)
        value = (CLASSES.index(cls) * len(NAMES) + NAMES.index(name)) * 9 % 256
        Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(folder / name)
    return root


@pytest.fixture()
def shuffled_listdir(monkeypatch: pytest.MonkeyPatch):
    """Make `os.listdir` return a shuffled listing, as an arbitrary file system may."""
    real_listdir = os.listdir
    rng = random.Random(1234)

    def listdir(path):
        names = list(real_listdir(path))
        rng.shuffle(names)
        return names

    monkeypatch.setattr(os, "listdir", listdir)


def _expected_relative(split_names=NAMES) -> list[str]:
    return [f"{cls}/{name}" for cls in sorted(CLASSES) for name in sorted(split_names)]


@pytest.mark.parametrize("loader", [loaders.get_DANINHAS, loaders.get_CIFAR10], ids=["DANINHAS", "CIFAR10"])
def test_loader_order_is_sorted_and_independent_of_creation_and_listing_order(
    loader, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shuffled_listdir
) -> None:
    monkeypatch.chdir(tmp_path)  # Data.__init__ writes results/... relative to the cwd
    first = loader(None, str(_build_dataset(tmp_path / "ds_a", creation_seed=1)) + "/", img_size=4)
    second = loader(None, str(_build_dataset(tmp_path / "ds_b", creation_seed=2)) + "/", img_size=4)

    for data in (first, second):
        assert relative_image_paths(data.Z_train_paths) == _expected_relative()
        assert relative_image_paths(data.Z_test_paths) == _expected_relative()
    # Same images at the same indices, whatever the creation or listing order.
    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.Y_train, second.Y_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    assert ordered_paths_hash(first.Z_train_paths) == ordered_paths_hash(second.Z_train_paths)
    assert ordered_paths_hash(first.Z_test_paths) == ordered_paths_hash(second.Z_test_paths)


def test_list_image_names_is_a_plain_lexicographic_sort(tmp_path: Path, shuffled_listdir) -> None:
    for name in NAMES:
        (tmp_path / name).write_bytes(b"")
    assert loaders.list_image_names(str(tmp_path)) == sorted(NAMES)
    assert loaders.list_image_names(str(tmp_path))[:2] == ["Img1.png", "_z.png"]  # code-point order, not casefold
