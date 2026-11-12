#!/usr/bin/env python3
"""Build, verify or migrate the canonical, duplicate-free weed dataset ``daninhas_full_fixed`` (KI-46).

The source dataset (``DATA/daninhas_full``) holds byte-identical copies of the same image under
different file names, inside each split and across train and test. A build writes a NEW
directory and never touches the source:

1. Two files are the same image when their MD5 digests are equal (author's definition).
2. Test: one file per distinct MD5 of the source test split.
3. Train: one file per distinct MD5 of the source train split, minus every MD5 that also
   occurs in the test split (no train/test leakage; the test split keeps the image).
4. Train additionally drops the MD5s listed in ``--exclude-train-md5`` (near-duplicates of
   test images found by the author's verifier; not byte-identical, so rule 1 does not catch them).
5. Among identical copies the kept file is the one with the lexicographically smallest name.

Layout of the destination (author instruction 2026-11-12: the top-level files describe the
dataset as it is; the provenance lives under ``logs/``)::

    train/<class>/<file>          the images
    test/<class>/<file>
    README.txt                    what the dataset is: identity rule, counts per split and class, how to verify
    MANIFEST.csv                  split,class,file,md5 -- one row per image on disk
    logs/build_history.txt        provenance: builder, exclusion list, source file count, what was removed and why
    logs/source_manifest.csv      MANIFEST.csv plus the source copies and the aliases dropped per kept image

Every metadata file is a deterministic function of the build's data, so a fresh build and a
folder converted by ``--migrate-metadata`` are byte-identical in all four files.

Modes:

- (default) build: refuses to run when the destination exists, copies the kept images, writes the
  metadata and verifies the result like ``--check``.
- ``--check``: read only. The metadata layout above, the files on disk equal the rows of
  ``MANIFEST.csv`` (split, class, name and MD5: a corrupted or incomplete copy fails), no repeated
  MD5 inside or across the splits, no excluded MD5 in train, ``README.txt`` and
  ``logs/source_manifest.csv`` consistent with ``MANIFEST.csv``.
- ``--migrate-metadata [--dry-run]``: converts an EXISTING destination of the previous layout
  (``MANIFEST.csv`` with the source columns and ``README.txt`` with the source counts, no ``logs/``)
  in place, touching only metadata. Precondition, verified before anything is written (abort with
  exit 1 and nothing written otherwise): the top level holds exactly ``train/``, ``test/``,
  ``MANIFEST.csv`` and ``README.txt``; the image files on disk are exactly the rows of the old
  ``MANIFEST.csv`` (split, class, name, MD5); no MD5 repeats; the old ``MANIFEST.csv`` and
  ``README.txt`` are byte-identical to what the builder writes from that manifest (so the
  history generated from it says exactly what the old README said). It then moves the old
  manifest to ``logs/source_manifest.csv``, writes ``logs/build_history.txt``, the new
  ``README.txt`` and ``MANIFEST.csv``, and re-runs ``--check``. It never creates, deletes, renames
  or rewrites an image. If ``logs/`` already exists it writes nothing: exit 0 with a message when
  the folder passes ``--check`` (already migrated), exit 1 otherwise. ``--dry-run`` stops after
  the precondition and writes nothing. The one sanctioned use is the author's one-off migration
  of ``DATA/daninhas_full_fixed`` of 2026-11-12, before any run (``.claude/rules/data-safety.md``).

``scan()`` and every check read only ``train/`` and ``test/`` for images; ``logs/`` is never scanned.

Usage (from the repository root):
    python scripts/datasets/build_daninhas_full_fixed.py
    python scripts/datasets/build_daninhas_full_fixed.py --check                       # verify an existing build
    python scripts/datasets/build_daninhas_full_fixed.py --migrate-metadata --dry-run  # precondition only
    python scripts/datasets/build_daninhas_full_fixed.py --migrate-metadata            # one-off layout migration
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

SPLITS = ("train", "test")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "DATA" / "daninhas_full"
DEFAULT_DST = REPO_ROOT / "DATA" / "daninhas_full_fixed"
DEFAULT_EXCLUDE = REPO_ROOT / "files_config" / "datasets" / "daninhas_full_fixed_near_duplicate_leaks.csv"
BUILDER = "scripts/datasets/build_daninhas_full_fixed.py"
DATASET_NAME = "daninhas_full_fixed"

MANIFEST_NAME = "MANIFEST.csv"
README_NAME = "README.txt"
LOGS_DIR = "logs"
HISTORY_NAME = "build_history.txt"
SOURCE_MANIFEST_NAME = "source_manifest.csv"
MANIFEST_HEADER: tuple[str, ...] = ("split", "class", "file", "md5")
SOURCE_MANIFEST_HEADER: tuple[str, ...] = (
    *MANIFEST_HEADER, "source_train_copies", "source_test_copies", "source_aliases_dropped",
)
# Top-level entries of the layout, and of the previous layout a migration starts from.
LAYOUT_TOP = frozenset({*SPLITS, README_NAME, MANIFEST_NAME, LOGS_DIR})
LAYOUT_LOGS = frozenset({HISTORY_NAME, SOURCE_MANIFEST_NAME})
OLD_LAYOUT_TOP = frozenset({*SPLITS, README_NAME, MANIFEST_NAME})
NEAR_DUPLICATE_REASON = "train_near_duplicates_of_test"
MAX_LISTED = 10  # differences listed per kind before "... and N more"

Row = tuple[str, str, str, str]  # (split, class, file name, md5)
SourceRow = tuple[str, str, str, str, int, int, str]  # Row + (source train copies, source test copies, aliases)


class MigrationError(RuntimeError):
    """The migration precondition does not hold; nothing was written."""


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan(root: Path) -> list[Row]:
    """Return ``(split, class, file name, md5)`` for every file below ``root/{train,test}/<class>/``.

    Only the two split directories are read: the metadata files and ``logs/`` are never scanned.
    """
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


def summarize(rows: Iterable[Row]) -> str:
    rows = list(rows)
    lines: list[str] = []
    for split in SPLITS:
        counts = Counter(cls for s, cls, _, _ in rows if s == split)
        lines.append(f"{split}: {sum(counts.values())} images")
        lines.extend(f"  {cls}: {n}" for cls, n in sorted(counts.items()))
    return "\n".join(lines)


def read_excluded(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["md5"] for row in csv.DictReader(handle)}


def read_exclusion_rows(path: Path) -> list[tuple[str, str]]:
    """``(md5, reason)`` rows of the exclusion list, in file order."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row["md5"], row.get("reason", "")) for row in csv.DictReader(handle)]


# --- rendering (deterministic: the same data gives the same bytes) ------------------------------------


def render_csv(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """CSV text exactly as ``csv.writer`` writes it to a file opened with ``newline=""``."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def project(kept: Iterable[SourceRow]) -> list[Row]:
    return [(s, c, n, d) for s, c, n, d, *_ in kept]


def removal_counts(kept: Sequence[SourceRow], near_duplicates: int) -> Counter[str]:
    """Files of the source the build did not keep, by reason, derived from the source-manifest rows
    (plus the near-duplicate count, which no kept row records). A reason is listed when the build met
    its case at least once (a zero count included), as the build's own counter does."""
    counts: Counter[str] = Counter()
    for split, _cls, _name, _digest, n_train, n_test, _aliases in kept:
        if split == "test":
            counts["test_copies"] += n_test - 1
            counts["train_files_also_in_test"] += n_train
        else:
            counts["train_copies"] += n_train - 1
    if near_duplicates:
        counts[NEAR_DUPLICATE_REASON] += near_duplicates
    return counts


def build_report(kept: Sequence[SourceRow], source_count: int, near_duplicates: int) -> str:
    """The build report: the README text of the previous layout, kept word for word inside
    ``logs/build_history.txt`` (and what ``--migrate-metadata`` requires the old README to be)."""
    lines = [
        f"{DATASET_NAME}: duplicate-free version of daninhas_full (KI-46).",
        f"Built by {BUILDER}; the source dataset is unchanged.",
        "Identity rule: two files are the same image when their MD5 digests are equal.",
        "Test = one file per distinct MD5 of the source test split.",
        "Train = one file per distinct MD5 of the source train split, minus every image present in test",
        "and minus the near-duplicates of test images listed in the exclusion file.",
        "",
        f"source files: {source_count}",
        summarize(project(kept)),
        "",
        "files removed from the source:",
        *(f"  {reason}: {count}" for reason, count in sorted(removal_counts(kept, near_duplicates).items())),
    ]
    return "\n".join(lines) + "\n"


def render_history(
    kept: Sequence[SourceRow],
    source_count: int,
    near_duplicates: int,
    exclusion_label: str,
    exclusion_rows: Sequence[tuple[str, str]],
) -> str:
    """``logs/build_history.txt``: the provenance of the build."""
    lines = [
        f"{DATASET_NAME} build history (provenance; the dataset itself is described in ../{README_NAME}).",
        "",
        f"builder: {BUILDER}",
        f"source manifest: {LOGS_DIR}/{SOURCE_MANIFEST_NAME} (one row per kept image: its copies in the source"
        " splits and the source files dropped as its aliases)",
        f"exclusion list (--exclude-train-md5, train near-duplicates of test images): {exclusion_label}",
        *(f"  {digest}: {reason}" for digest, reason in exclusion_rows),
        "",
        "build report (written as README.txt by the build; moved here by the metadata migration of 2026-11-12):",
        "",
        build_report(kept, source_count, near_duplicates).rstrip("\n"),
    ]
    return "\n".join(lines) + "\n"


def render_readme(rows: Sequence[Row]) -> str:
    """``README.txt``: what the dataset is now (no reference to how it was obtained)."""
    n_classes = len({cls for _, cls, _, _ in rows})
    lines = [
        f"{DATASET_NAME}: weed images of sugarcane fields captured by UAV, {n_classes} classes (one folder per class).",
        "",
        "Layout:",
        "  train/<class>/<file>   training pool",
        "  test/<class>/<file>    test set",
        f"  {MANIFEST_NAME}           one row per image: {','.join(MANIFEST_HEADER)}",
        f"  {LOGS_DIR}/                  provenance of the build ({HISTORY_NAME}, {SOURCE_MANIFEST_NAME})",
        "",
        "Identity rule: two files are the same image when their MD5 digests are equal.",
        "No MD5 occurs twice in this folder, inside a split or across train and test.",
        "",
        summarize(rows),
        "",
        f"Verify (read only): python {BUILDER} --check --dst <this folder>",
        f"It checks that the files on disk are exactly the rows of {MANIFEST_NAME} (name and MD5), that no MD5",
        "repeats, and this metadata layout.",
    ]
    return "\n".join(lines) + "\n"


def exclusion_label(path: Path) -> str:
    """The exclusion list's path as recorded in the history: repo-relative when inside the repository."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def metadata_files(
    kept: Sequence[SourceRow], source_count: int, near_duplicates: int, exclude_path: Path
) -> dict[str, str]:
    """Relative path -> text of the four metadata files of the layout."""
    kept = sorted(kept)
    rows = project(kept)
    return {
        README_NAME: render_readme(rows),
        MANIFEST_NAME: render_csv(MANIFEST_HEADER, rows),
        f"{LOGS_DIR}/{HISTORY_NAME}": render_history(
            kept, source_count, near_duplicates, exclusion_label(exclude_path), read_exclusion_rows(exclude_path)
        ),
        f"{LOGS_DIR}/{SOURCE_MANIFEST_NAME}": render_csv(SOURCE_MANIFEST_HEADER, kept),
    }


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` through a temporary file of the same directory and ``os.replace``."""
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


# --- reading and checking ----------------------------------------------------------------------------


def read_exact(path: Path) -> str:
    """The file's text without newline translation (``\r\n`` of a CSV stays ``\r\n``)."""
    return path.read_bytes().decode("utf-8")


def read_csv(path: Path) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        return header, [tuple(row) for row in reader]


def _listed(kind: str, items: Sequence[object]) -> list[str]:
    shown = [f"{kind}: {item}" for item in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        shown.append(f"{kind}: ... and {len(items) - MAX_LISTED} more")
    return shown


def compare_disk_to_manifest(disk: Sequence[Row], manifest: Sequence[Row]) -> list[str]:
    """Differences between the image files on disk and the manifest rows (empty list = equal)."""
    problems: list[str] = []
    repeated = sorted(row for row, n in Counter(manifest).items() if n > 1)
    problems += _listed(f"row repeated in {MANIFEST_NAME}", repeated)
    on_disk = {(s, c, n): d for s, c, n, d in disk}
    listed = {(s, c, n): d for s, c, n, d in manifest}
    missing = sorted("/".join(k) for k in listed.keys() - on_disk.keys())
    extra = sorted("/".join(k) for k in on_disk.keys() - listed.keys())
    changed = sorted(f"{'/'.join(k)} (disk {on_disk[k]}, manifest {listed[k]})"
                     for k in listed.keys() & on_disk.keys() if listed[k] != on_disk[k])
    problems += _listed(f"in {MANIFEST_NAME} but missing on disk", missing)
    problems += _listed(f"on disk but not in {MANIFEST_NAME}", extra)
    problems += _listed("md5 on disk differs from the manifest", changed)
    return problems


def top_level_problems(root: Path, expected: frozenset[str]) -> list[str]:
    present = {p.name for p in root.iterdir()}
    problems = [f"missing at the top level: {name}" for name in sorted(expected - present)]
    problems += [f"unexpected at the top level: {name}" for name in sorted(present - expected)]
    return problems


def check_destination(root: Path, excluded: set[str]) -> tuple[list[Row], list[str]]:
    """``--check``: the files on disk and the problems found (empty list = the dataset is canonical)."""
    disk = scan(root)
    problems = verify(disk, excluded)
    layout = top_level_problems(root, LAYOUT_TOP)
    logs = root / LOGS_DIR
    if logs.is_dir():
        layout += [f"{LOGS_DIR}/: {p}" for p in top_level_problems(logs, LAYOUT_LOGS)]
    elif logs.exists():
        layout.append(f"{LOGS_DIR} is not a directory")
    if layout:
        hint = ("previous metadata layout (no logs/): run --migrate-metadata"
                if not logs.exists() and (root / MANIFEST_NAME).is_file() else "metadata layout differs")
        return disk, problems + layout + [hint]
    header, raw = read_csv(root / MANIFEST_NAME)
    if header != MANIFEST_HEADER:
        return disk, problems + [f"{MANIFEST_NAME} header is {list(header)}, expected {list(MANIFEST_HEADER)}"]
    manifest: list[Row] = [tuple(r) for r in raw if len(r) == len(MANIFEST_HEADER)]  # type: ignore[misc]
    problems += _listed(f"malformed row in {MANIFEST_NAME}", [r for r in raw if len(r) != len(MANIFEST_HEADER)])
    problems += compare_disk_to_manifest(disk, manifest)
    if read_exact(root / README_NAME) != render_readme(sorted(manifest)):
        problems.append(f"{README_NAME} does not describe the rows of {MANIFEST_NAME}")
    source_header, source_raw = read_csv(logs / SOURCE_MANIFEST_NAME)
    if source_header != SOURCE_MANIFEST_HEADER or sorted(r[:4] for r in source_raw) != sorted(manifest):
        problems.append(f"{LOGS_DIR}/{SOURCE_MANIFEST_NAME} does not match {MANIFEST_NAME}")
    return disk, problems


def _parse_source_rows(raw: Sequence[tuple[str, ...]]) -> list[SourceRow]:
    kept: list[SourceRow] = []
    for row in raw:
        if len(row) != len(SOURCE_MANIFEST_HEADER):
            raise MigrationError(f"malformed row in {MANIFEST_NAME}: {list(row)}")
        split, cls, name, digest, n_train, n_test, aliases = row
        try:
            kept.append((split, cls, name, digest, int(n_train), int(n_test), aliases))
        except ValueError as exc:
            raise MigrationError(f"non-integer copy count in {MANIFEST_NAME} row {list(row)}") from exc
    return kept


def _readme_number(text: str, pattern: str, default: int | None = None) -> int:
    match = re.search(pattern, text, flags=re.M)
    if match:
        return int(match.group(1))
    if default is None:
        raise MigrationError(f"old {README_NAME} has no line matching {pattern!r}")
    return default


def migration_plan(root: Path, exclude_path: Path) -> tuple[dict[str, str], int]:
    """Check the migration precondition on an old-layout ``root`` (read only); return the four
    metadata files to write and the number of image files. Raises ``MigrationError`` when the
    precondition does not hold."""
    layout = top_level_problems(root, OLD_LAYOUT_TOP)
    if layout:
        raise MigrationError("not the previous metadata layout: " + "; ".join(layout))
    old_manifest_text = read_exact(root / MANIFEST_NAME)
    header, raw = read_csv(root / MANIFEST_NAME)
    if header != SOURCE_MANIFEST_HEADER:
        raise MigrationError(f"old {MANIFEST_NAME} header is {list(header)}, expected {list(SOURCE_MANIFEST_HEADER)}")
    kept = _parse_source_rows(raw)
    disk = scan(root)
    problems = compare_disk_to_manifest(disk, project(kept)) + verify(disk, read_excluded(exclude_path))
    if problems:
        raise MigrationError("the files on disk do not match the old MANIFEST.csv:\n" + "\n".join(problems))
    if render_csv(SOURCE_MANIFEST_HEADER, sorted(kept)) != old_manifest_text:
        raise MigrationError(f"old {MANIFEST_NAME} is not byte-identical to the builder's rendering of its rows")
    old_readme = read_exact(root / README_NAME)
    source_count = _readme_number(old_readme, r"^source files: (\d+)$")
    near_duplicates = _readme_number(old_readme, rf"^  {NEAR_DUPLICATE_REASON}: (\d+)$", default=0)
    if build_report(kept, source_count, near_duplicates) != old_readme:
        raise MigrationError(
            f"old {README_NAME} differs from the build report generated from {MANIFEST_NAME} "
            f"(source files {source_count}, {NEAR_DUPLICATE_REASON} {near_duplicates})"
        )
    kept_source_files = sum(n_train + n_test for *_, n_train, n_test, _ in kept)
    if kept_source_files + near_duplicates != source_count:
        raise MigrationError(
            f"source count {source_count} != {kept_source_files} files behind the kept images + "
            f"{near_duplicates} near-duplicates"
        )
    return metadata_files(kept, source_count, near_duplicates, exclude_path), len(disk)


def migrate(root: Path, exclude_path: Path, *, dry_run: bool) -> int:
    """``--migrate-metadata``: see the module docstring. Never touches an image."""
    excluded = read_excluded(exclude_path)
    if (root / LOGS_DIR).exists():
        _disk, problems = check_destination(root, excluded)
        if problems:
            print("\n".join(problems))
            print(f"{root / LOGS_DIR} exists but the folder fails --check: refusing to migrate, nothing written")
            return 1
        print(f"already migrated ({root / LOGS_DIR} exists and the folder passes --check): nothing written")
        return 0
    try:
        files, n_images = migration_plan(root, exclude_path)
    except MigrationError as exc:
        print(f"migration precondition FAILED, nothing written: {exc}")
        return 1
    print(f"precondition OK: {n_images} image files on disk == the {n_images} rows of the old {MANIFEST_NAME}; "
          "no repeated md5; old MANIFEST.csv and README.txt are the builder's own output")
    if dry_run:
        print("dry run: nothing written. The migration would write:")
        print(f"  {LOGS_DIR}/{SOURCE_MANIFEST_NAME}  (the old {MANIFEST_NAME}, byte for byte)")
        print(f"  {LOGS_DIR}/{HISTORY_NAME}  (the old {README_NAME}, word for word, under a provenance header)")
        print(f"  {MANIFEST_NAME}  ({','.join(MANIFEST_HEADER)})")
        print(f"  {README_NAME}  (the dataset as it is)")
        return 0
    logs_tmp = root / f".{LOGS_DIR}.migrate.tmp"
    if logs_tmp.exists():
        print(f"{logs_tmp} exists (an interrupted migration?): remove it by hand and re-run; nothing written")
        return 1
    logs_tmp.mkdir()
    for rel in (f"{LOGS_DIR}/{HISTORY_NAME}", f"{LOGS_DIR}/{SOURCE_MANIFEST_NAME}"):
        write_atomic(logs_tmp / Path(rel).name, files[rel])
    os.replace(logs_tmp, root / LOGS_DIR)
    write_atomic(root / MANIFEST_NAME, files[MANIFEST_NAME])
    write_atomic(root / README_NAME, files[README_NAME])
    disk, problems = check_destination(root, excluded)
    print(summarize(disk))
    if problems:
        print("\n".join(problems))
        print("migration written, but the re-check FAILED (see above)")
        return 1
    print("OK: migrated; files on disk match MANIFEST.csv, no repeated md5 inside or across splits")
    return 0


# --- build ----------------------------------------------------------------------------------------------


def build(src: Path, dst: Path, exclude_path: Path) -> int:
    if dst.exists():
        sys.exit(f"destination already exists, refusing to write: {dst}")
    excluded = read_excluded(exclude_path)
    source = scan(src)
    groups: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for split, cls, name, digest in source:
        groups[digest][split].append((cls, name))
    for digest, per_split in groups.items():
        classes = {cls for members in per_split.values() for cls, _ in members}
        if len(classes) > 1:
            sys.exit(f"md5 {digest} carries more than one class label: {sorted(classes)}")

    kept: list[SourceRow] = []
    dropped: Counter[str] = Counter()
    for digest, per_split in sorted(groups.items()):
        n_train, n_test = len(per_split.get("train", [])), len(per_split.get("test", []))
        if n_test:
            split = "test"
            dropped["test_copies"] += n_test - 1
            dropped["train_files_also_in_test"] += n_train
        elif digest in excluded:
            dropped[NEAR_DUPLICATE_REASON] += n_train
            continue
        else:
            split = "train"
            dropped["train_copies"] += n_train - 1
        cls, name = min(per_split[split], key=lambda member: member[1])
        aliases = ";".join(
            f"{s}/{c}/{n}" for s in SPLITS for c, n in sorted(per_split.get(s, [])) if (s, n) != (split, name)
        )
        kept.append((split, cls, name, digest, n_train, n_test, aliases))

    near_duplicates = dropped[NEAR_DUPLICATE_REASON]
    if removal_counts(kept, near_duplicates) != dropped or sum(dropped.values()) + len(kept) != len(source):
        sys.exit("internal error: the removal counts derived from the manifest differ from the build's own")

    for split, cls, name, *_ in kept:
        target_dir = dst / split / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / split / cls / name, target_dir / name)
    (dst / LOGS_DIR).mkdir()
    files = metadata_files(kept, len(source), near_duplicates, exclude_path)
    for rel, text in files.items():
        write_atomic(dst / rel, text)

    disk, problems = check_destination(dst, excluded)
    print(files[f"{LOGS_DIR}/{HISTORY_NAME}"], end="")
    if problems:
        print("\n".join(problems))
        return 1
    print("OK: files on disk match MANIFEST.csv, no repeated md5 inside or across splits")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--exclude-train-md5", type=Path, default=DEFAULT_EXCLUDE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="only verify an existing destination (read only)")
    mode.add_argument("--migrate-metadata", action="store_true",
                      help="convert an existing previous-layout destination in place (metadata only)")
    parser.add_argument("--dry-run", action="store_true", help="with --migrate-metadata: check the precondition only")
    args = parser.parse_args(argv)
    if args.dry_run and not args.migrate_metadata:
        parser.error("--dry-run applies to --migrate-metadata only")

    if args.check:
        disk, problems = check_destination(args.dst, read_excluded(args.exclude_train_md5))
        print(summarize(disk))
        print("\n".join(problems) if problems else
              "OK: files on disk match MANIFEST.csv, no repeated md5 inside or across splits")
        return 1 if problems else 0
    if args.migrate_metadata:
        return migrate(args.dst, args.exclude_train_md5, dry_run=args.dry_run)
    return build(args.src, args.dst, args.exclude_train_md5)


if __name__ == "__main__":
    raise SystemExit(main())
