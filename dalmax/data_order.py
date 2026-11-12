"""Machine-independent identity of a loaded image list (KI-47).

Pool and test indices are positions in the lists the dataset loaders build
(`dalmax/data/loaders.py`). Until 2026-11-12 the file names inside each class
folder came from an unsorted `os.listdir`, whose order depends on the file
system and differed between Colab sessions, so the same index named different
images in different sessions. The loaders now sort the names (contract below);
this module gives every consumer one definition of "the same images in the same
order":

- `relative_image_path`: `<class folder>/<file name>`, the part of a loaded path
  that does not depend on where the dataset lives (`DATA/...`, an absolute path,
  another machine);
- `ordered_paths_hash`: a short hash of the ordered relative paths of one split.
  It is recorded in `run_metadata.json` (`data_order`, via `data_order_record`),
  it is part of the embedding-cache key (`dalmax/embeddings/cache.py`), and
  `dalmax.campaign.audit_data_order` compares it across the leaves of a campaign.

Ordering contract (`ORDER_CONTRACT`): classes = the sorted folder names of
`train/`; inside each class folder, files in plain lexicographic order of their
names (`sorted()` on `str`, i.e. by Unicode code point), never the file system's
listing order. Importing this module has no side effects.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

ORDER_CONTRACT = "sorted-names-v1"
HASH_LENGTH = 12


def relative_image_path(path: str | PurePath) -> str:
    """`<class folder>/<file name>` of a loaded image path (POSIX separators)."""
    return "/".join(PurePath(path).parts[-2:])


def relative_image_paths(paths: Sequence[str | PurePath]) -> list[str]:
    """`relative_image_path` of every path, in the given order."""
    return [relative_image_path(p) for p in paths]


def ordered_paths_hash(paths: Sequence[str | PurePath]) -> str:
    """Short SHA-256 of the ORDERED relative paths: equal iff the same images sit at the same indices."""
    joined = "\n".join(relative_image_paths(paths))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def data_order_record(
    train_paths: Sequence[str | PurePath], test_paths: Sequence[str | PurePath]
) -> dict[str, Any]:
    """The `data_order` block of `run_metadata.json`: which images the pool and test indices of a run name."""
    return {
        "contract": ORDER_CONTRACT,
        "train_paths_hash": ordered_paths_hash(train_paths),
        "test_paths_hash": ordered_paths_hash(test_paths),
        "n_train": len(train_paths),
        "n_test": len(test_paths),
    }
