"""Generate a small, deterministic 10%-stratified replica of DATA/daninhas_full
for local smoke tests.

Motivation
----------
`.specs/infrastructure/execution-environments.md` forbids any real training run
on the local (no-GPU) dev notebook. `make smoke` still needs a true end-to-end
`demo.py` run to catch integration breakage that unit tests miss (see
`.specs/architecture/refactor-plan.md` Phase 1). This script builds a
10%-stratified subset of the real dataset -- all 5 classes, both splits -- that
a CPU can train on in a few seconds per epoch, giving a realistic pre-lab
end-to-end check instead of the original 2-class/70-image placeholder (see
`.specs/adr/0004-micro-dataset-and-golden-run.md`'s 2026-08-23 "redefinition"
amendment).

Selection is deterministic: for each `(split, class)` pair, filenames are
sorted (so the selection does not depend on filesystem iteration order) and
then a fixed-seed `random.Random` instance samples
`max(1, floor(FRACTION * n_files))` files without replacement. Re-running this
script always selects the same files and rewrites `DATA/daninhas_micro/arquivos.txt`
to match, in the same format as `DATA/daninhas_full/arquivos.txt`.

Rules respected (see `.claude/rules/data-safety.md`):
- `DATA/daninhas_full/` is **never** written to, only read.
- `DATA/daninhas_micro/` (this script's output) lives under the gitignored
  `DATA/` directory, so it is never committed; it is regenerated on demand.

If `DATA/daninhas_full` is not present (e.g. a fresh clone with no dataset
copied in yet), this script prints a message and exits 0 rather than failing
`make smoke` outright.
"""

from __future__ import annotations

import hashlib
import math
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "DATA" / "daninhas_full"
DEST_ROOT = REPO_ROOT / "DATA" / "daninhas_micro"
ARQUIVOS_TXT = DEST_ROOT / "arquivos.txt"

# All 5 daninhas_full classes, in the same order as the "Train" section of
# DATA/daninhas_full/arquivos.txt, so DATA/daninhas_micro/arquivos.txt reads as
# a direct per-class analog of the source file.
CLASSES = (
    "DATASET_GRAMINEA",
    "DATASET_MAMONA",
    "DATASET_BRACHIARIA",
    "DATASET_COLONIAO",
    "DATASET_OUTRAS_FOLHAS_LARGAS",
)

# Stratified sampling fraction: every class, in both splits, keeps
# max(1, floor(FRACTION * n_files_in_class)) files. This makes
# DATA/daninhas_micro/ a genuine 10%-stratified replica of the full dataset --
# all 5 classes, both splits -- rather than a hand-picked 2-class subset. See
# .specs/adr/0004-micro-dataset-and-golden-run.md's 2026-08-23 amendment.
FRACTION = 0.10

# Fixed seed for the sampling below. This is a data-generation seed, not the
# experiment `--seed` CLI argument, so it is intentionally a stable literal:
# changing it would change what "the micro dataset" means for every fixture
# that depends on it (see tests/golden/*.json).
SAMPLING_SEED = 20260823


def _select_files(class_dir: Path, seed: int) -> list[Path]:
    """Deterministically pick `max(1, floor(FRACTION * n))` files from `class_dir`.

    Sorting first removes any dependency on filesystem iteration order;
    `random.Random(seed)` then makes the sample reproducible across machines
    and runs.
    """
    files = sorted(p for p in class_dir.iterdir() if p.is_file())
    count = max(1, math.floor(FRACTION * len(files)))
    if len(files) < count:
        raise ValueError(f"{class_dir} has only {len(files)} files, need at least {count}")
    rng = random.Random(seed)
    return rng.sample(files, count)


def _seed_for(split: str, class_name: str) -> int:
    """Deterministic per-(split, class) seed so train/test selections don't
    collide with each other. Built-in `hash()` on a tuple is salted
    per-process (PYTHONHASHSEED randomization) and must never be used here:
    two processes running this exact script would silently select different
    files. `hashlib.sha256` is stable across processes and machines.
    """
    digest = hashlib.sha256(f"{split}:{class_name}".encode()).hexdigest()
    return SAMPLING_SEED + int(digest, 16) % 10_000


def _populate_split(split: str) -> dict[str, int]:
    """Populate DEST_ROOT/split/<class>/ for every class in CLASSES.

    Returns the per-class selected file count (used to write
    DATA/daninhas_micro/arquivos.txt).
    """
    counts: dict[str, int] = {}
    for class_name in CLASSES:
        src_dir = SOURCE_ROOT / split / class_name
        dst_dir = DEST_ROOT / split / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        seed = _seed_for(split, class_name)
        selected = _select_files(src_dir, seed)
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

        counts[class_name] = len(selected)
        print(f"{split}/{class_name}: {len(selected)} files -> {dst_dir}")
    return counts


def _write_arquivos_txt(train_counts: dict[str, int], test_counts: dict[str, int]) -> None:
    """Write DATA/daninhas_micro/arquivos.txt, in exactly the same format as
    DATA/daninhas_full/arquivos.txt (see that file for the reference format)."""

    def _section(title: str, split: str, counts: dict[str, int]) -> str:
        lines = [title]
        for class_name in CLASSES:
            lines.append(f"DATA/daninhas_micro/{split}//{class_name}: {counts[class_name]}")
        total = sum(counts.values())
        average = total / len(counts)
        lines.append(f"Total: {total}")
        lines.append(f"Average: {average:.1f}")
        return "\n".join(lines)

    content = (
        _section("Train", "train", train_counts) + "\n\n" + _section("Test", "test", test_counts)
    )
    ARQUIVOS_TXT.write_text(content)


def generate_micro_dataset() -> Path | None:
    """Build DATA/daninhas_micro from DATA/daninhas_full. Returns the
    destination path, or None if the source dataset is not present."""
    if not SOURCE_ROOT.is_dir():
        print(
            f"NOTE: {SOURCE_ROOT} not found. Skipping micro-dataset generation "
            "(this is expected on a fresh clone with no dataset copied in yet)."
        )
        return None

    train_counts = _populate_split("train")
    test_counts = _populate_split("test")
    _write_arquivos_txt(train_counts, test_counts)
    print(f"Micro dataset ready at {DEST_ROOT}")
    return DEST_ROOT


if __name__ == "__main__":
    generate_micro_dataset()
