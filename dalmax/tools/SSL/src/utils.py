# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from pathlib import Path

import numpy as np
import torch


def create_clusters_from_cluster_assignment(
    cluster_assignment: np.array,
    num_clusters: int,
    return_object_array: bool = True,
):
    """
    Build clusters from cluster assignment.
    """
    ID = np.argsort(cluster_assignment)
    sorted_cluster_assigment = cluster_assignment[ID]
    index_split = np.searchsorted(sorted_cluster_assigment, list(range(num_clusters)))
    clusters = np.split(ID, index_split[1:])
    if return_object_array:
        return to_object_array(clusters)
    else:
        return clusters


def to_object_array(arrays) -> np.ndarray:
    """1-D ``dtype=object`` array whose elements are the given index arrays.

    DalMax fix (KI-30, 2026-11-07): ``np.array(list_of_arrays, dtype=object)``
    silently builds a 2-D object array when every array has the same length,
    which later breaks ``torch`` indexing (``TypeError: can't convert np.ndarray
    of type numpy.object_``). Pre-allocating the outer array keeps one
    ``int64`` array per element whatever the lengths are.
    """
    out = np.empty(len(arrays), dtype=object)
    for i, arr in enumerate(arrays):
        out[i] = np.asarray(arr)
    return out


def find_all_checkpoints(save_dir, pattern):
    """
    Parameters:
        pattern: str
            checkpoint name format <filename>_%d.<file extension>,
            e.g., kmpp_checkpoint_%d.pth
    """
    save_dir = Path(save_dir)
    ckpt_list = [str(el.stem) for el in save_dir.glob(pattern.replace("%d", "*"))]
    ckpt_list = [int(el.split("_")[-1]) for el in ckpt_list]
    ckpt_list = sorted(ckpt_list)
    return [Path(save_dir, pattern % el) for el in ckpt_list]


def get_last_valid_checkpoint(save_dir, pattern):
    """
    Find path to the last checkpoint.
    """
    ckpt_list = find_all_checkpoints(save_dir, pattern)
    for ckpt_path in ckpt_list[::-1]:
        try:
            if ".pth" in pattern:
                _ = torch.load(ckpt_path, map_location="cpu")
            elif ".npy" in pattern:
                _ = np.load(ckpt_path)
            else:
                raise ValueError("Pattern not recognized!")
            return ckpt_path
        except Exception:
            continue
    return None


def _delete_old_checkpoint(
    save_dir, current_iter, checkpoint_period, max_num_checkpoints, pattern
):
    Path(
        save_dir, pattern % (current_iter - checkpoint_period * max_num_checkpoints)
    ).unlink(missing_ok=True)


def setup_logging(
    *,
    name: str = None,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> None:
    """
    Basic setting for logger.
    """
    logging.captureWarnings(capture_warnings)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return

    fmt_prefix = (
        "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    )
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.propagate = False
    logger.addHandler(handler)
    return
