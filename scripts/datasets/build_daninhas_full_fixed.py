#!/usr/bin/env python3
"""Build the canonical, duplicate-free weed dataset ``daninhas_full_fixed`` (KI-46).

The source dataset (``DATA/daninhas_full``) holds byte-identical copies of the same image under
different file names, inside each split and across train and test. This script writes a NEW
directory and never touches the source:

1. Two files are the same image when their MD5 digests are equal (author's definition).
2. Test: one file per distinct MD5 of the source test split.
3. Train: one file per distinct MD5 of the source train split, minus every MD5 that also
   occurs in the test split (no train/test leakage; the test split keeps the image).
4. Train additionally drops the MD5s listed in ``--exclude-train-md5`` (near-duplicates of
   test images found by the author's verifier; not byte-identical, so rule 1 does not catch them).
5. Among identical copies the kept file is the one with the lexicographically smallest name.

It refuses to run when the destination exists, verifies the result (no repeated MD5 anywhere,
one class per MD5) and writes ``MANIFEST.csv`` and ``README.txt`` into the destination.

Usage (from the repository root):
    python scripts/datasets/build_daninhas_full_fixed.py
    python scripts/datasets/build_daninhas_full_fixed.py --check   # verify an existing build
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "test")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "DATA" / "daninhas_full"
DEFAULT_DST = REPO_ROOT / "DATA" / "daninhas_full_fixed"
DEFAULT_EXCLUDE = REPO_ROOT / "files_config" / "datasets" / "daninhas_full_fixed_near_duplicate_leaks.csv"

Row = tuple[str, str, str, str]


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan(root: Path) -> list[Row]:
    """Return ``(split, class, file name, md5)`` for every file below ``root/{train,test}/<class>/``."""
    rows: list[Row] = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            sys.exit(f"missing split directory: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for file in sorted(p for p in class_dir.iterdir() if p.is_file()):
                rows.append((split, class_dir.name, file.name, md5_of(file)))
    return rows


def verify(rows: list[Row], excluded: set[str]) -> list[str]:
    """Return the violations of the dataset contract (empty list = clean)."""
    problems: list[str] = []
    by_md5: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for split, cls, name, digest in rows:
        by_md5[digest].append((split, cls, name))
    for digest, members in by_md5.items():
        if len(members) > 1:
            problems.append(f"md5 {digest} occurs {len(members)} times: {members}")
    for digest in excluded & {d for s, _, _, d in rows if s == "train"}:
        problems.append(f"excluded md5 {digest} is present in train")
    return problems


def summarize(rows: list[Row]) -> str:
    lines: list[str] = []
    for split in SPLITS:
        counts = Counter(cls for s, cls, _, _ in rows if s == split)
        lines.append(f"{split}: {sum(counts.values())} images")
        lines.extend(f"  {cls}: {n}" for cls, n in sorted(counts.items()))
    return "\n".join(lines)


def read_excluded(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["md5"] for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--exclude-train-md5", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--check", action="store_true", help="only verify an existing destination")
    args = parser.parse_args()
    excluded = read_excluded(args.exclude_train_md5)

    if args.check:
        rows = scan(args.dst)
        problems = verify(rows, excluded)
        print(summarize(rows))
        print("\n".join(problems) if problems else "OK: no repeated md5 inside or across splits")
        return 1 if problems else 0

    if args.dst.exists():
        sys.exit(f"destination already exists, refusing to write: {args.dst}")

    source = scan(args.src)
    groups: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for split, cls, name, digest in source:
        groups[digest][split].append((cls, name))
    for digest, per_split in groups.items():
        classes = {cls for members in per_split.values() for cls, _ in members}
        if len(classes) > 1:
            sys.exit(f"md5 {digest} carries more than one class label: {sorted(classes)}")

    kept: list[tuple[str, str, str, str, int, int, str]] = []
    dropped: Counter[str] = Counter()
    for digest, per_split in sorted(groups.items()):
        n_train, n_test = len(per_split.get("train", [])), len(per_split.get("test", []))
        if n_test:
            split = "test"
            dropped["test_copies"] += n_test - 1
            dropped["train_files_also_in_test"] += n_train
        elif digest in excluded:
            dropped["train_near_duplicates_of_test"] += n_train
            continue
        else:
            split = "train"
            dropped["train_copies"] += n_train - 1
        cls, name = min(per_split[split], key=lambda member: member[1])
        aliases = ";".join(
            f"{s}/{c}/{n}" for s in SPLITS for c, n in sorted(per_split.get(s, [])) if (s, n) != (split, name)
        )
        kept.append((split, cls, name, digest, n_train, n_test, aliases))

    for split, cls, name, *_ in kept:
        target_dir = args.dst / split / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.src / split / cls / name, target_dir / name)

    kept.sort()
    with (args.dst / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "class", "file", "md5", "source_train_copies", "source_test_copies", "source_aliases_dropped"]
        )
        writer.writerows(kept)

    built = scan(args.dst)
    problems = verify(built, excluded)
    if {(s, c, n, d) for s, c, n, d, *_ in kept} != set(built):
        problems.append("files on disk differ from the manifest")
    readme = [
        "daninhas_full_fixed: duplicate-free version of daninhas_full (KI-46).",
        "Built by scripts/datasets/build_daninhas_full_fixed.py; the source dataset is unchanged.",
        "Identity rule: two files are the same image when their MD5 digests are equal.",
        "Test = one file per distinct MD5 of the source test split.",
        "Train = one file per distinct MD5 of the source train split, minus every image present in test",
        "and minus the near-duplicates of test images listed in the exclusion file.",
        "",
        f"source files: {len(source)}",
        summarize(built),
        "",
        "files removed from the source:",
        *(f"  {reason}: {count}" for reason, count in sorted(dropped.items())),
    ]
    (args.dst / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print("\n".join(readme))
    if problems:
        print("\n".join(problems))
        return 1
    print("OK: no repeated md5 inside or across splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
