"""Generate a tiny, deterministic subset of DATA/daninhas_full for local smoke tests.

Motivation
----------
`.specs/infrastructure/execution-environments.md` forbids any real training run
on the local (no-GPU) dev notebook. `make smoke` still needs a true end-to-end
`demo.py` run to catch integration breakage that unit tests miss (see
`.specs/architecture/refactor-plan.md` Phase 1). This script builds a
2-class/70-image subset of the real dataset that a CPU can train on in a few
seconds per epoch.

Selection is deterministic: for each of the two source classes, filenames are
sorted (so the selection does not depend on filesystem iteration order) and
then a fixed-seed `random.Random` instance samples the requested count without
replacement. Re-running this script always selects the same files.

Rules respected (see `.claude/rules/data-safety.md`):
- `DATA/daninhas_full/` is **never** written to, only read.
- `DATA/daninhas_micro/` (this script's output) lives under the gitignored
  `DATA/` directory, so it is never committed; it is regenerated on demand.

If `DATA/daninhas_full` is not present (e.g. a fresh clone with no dataset
copied in yet), this script prints a message and exits 0 rather than failing
`make smoke` outright.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "DATA" / "daninhas_full"
DEST_ROOT = REPO_ROOT / "DATA" / "daninhas_micro"

# The two classes used for the micro dataset. Any two classes from
# daninhas_full would do; these were picked because both have comfortably
# more than the required counts in both `train/` and `test/`.
CLASSES = ("DATASET_BRACHIARIA", "DATASET_GRAMINEA")

N_TRAIN_PER_CLASS = 25
N_TEST_PER_CLASS = 10

# Fixed seed for the sampling below. This is a data-generation seed, not the
# experiment `--seed` CLI argument, so it is intentionally a stable literal:
# changing it would change what "the micro dataset" means for every fixture
# that depends on it (see tests/golden/*.json).
SAMPLING_SEED = 20260823


def _select_files(class_dir: Path, count: int, seed: int) -> list[Path]:
    """Deterministically pick `count` files from `class_dir`.

    Sorting first removes any dependency on filesystem iteration order;
    `random.Random(seed)` then makes the sample reproducible across machines
    and runs.
    """
    files = sorted(p for p in class_dir.iterdir() if p.is_file())
    if len(files) < count:
        raise ValueError(
            f"{class_dir} has only {len(files)} files, need at least {count}"
        )
    rng = random.Random(seed)
    return rng.sample(files, count)


def _populate_split(split: str, count_per_class: int) -> None:
    for class_name in CLASSES:
        src_dir = SOURCE_ROOT / split / class_name
        dst_dir = DEST_ROOT / split / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic per-(split, class) seed so train/test selections don't
        # collide with each other.
        seed = SAMPLING_SEED + hash((split, class_name)) % 10_000
        selected = _select_files(src_dir, count_per_class, seed)
        selected_names = {p.name for p in selected}

        # Idempotent: remove any stale file left over from a previous
        # selection (e.g. after this script's logic changes) and copy only
        # what's currently selected, skipping files already in place.
        for existing in dst_dir.iterdir():
            if existing.name not in selected_names:
                existing.unlink()
        for src_file in selected:
            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)

        print(f"{split}/{class_name}: {len(selected)} files -> {dst_dir}")


def generate_micro_dataset() -> Path | None:
    """Build DATA/daninhas_micro from DATA/daninhas_full. Returns the
    destination path, or None if the source dataset is not present."""
    if not SOURCE_ROOT.is_dir():
        print(
            f"NOTE: {SOURCE_ROOT} not found. Skipping micro-dataset generation "
            "(this is expected on a fresh clone with no dataset copied in yet)."
        )
        return None

    _populate_split("train", N_TRAIN_PER_CLASS)
    _populate_split("test", N_TEST_PER_CLASS)
    print(f"Micro dataset ready at {DEST_ROOT}")
    return DEST_ROOT


if __name__ == "__main__":
    generate_micro_dataset()
