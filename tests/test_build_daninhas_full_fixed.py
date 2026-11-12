"""`scripts/datasets/build_daninhas_full_fixed.py`: the metadata layout (2026-11-12), `--check`
against `MANIFEST.csv`, and the one-off `--migrate-metadata` of a previous-layout folder.

Fast, synthetic: a tiny source dataset is written into `tmp_path` (never `DATA/`). The previous
layout is produced by the builder as committed before the layout change (`git show`), the exact
code that built the real `DATA/daninhas_full_fixed`; those tests skip outside a git checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER = REPO_ROOT / "scripts/datasets/build_daninhas_full_fixed.py"
# Last commit of the builder with the previous layout (MANIFEST.csv with source columns, no logs/).
OLD_BUILDER_REF = "c7d2e9a"
METADATA = ("README.txt", "MANIFEST.csv", "logs/build_history.txt", "logs/source_manifest.csv")


def _builder():
    spec = importlib.util.spec_from_file_location("build_daninhas_full_fixed", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_daninhas_full_fixed"] = module
    spec.loader.exec_module(module)
    return module


B = _builder()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _source(root: Path) -> tuple[Path, Path]:
    """A source with every case the builder handles; returns (source dir, exclusion csv)."""
    src = root / "src"
    # test copies of one image (2 in test, 1 in train) -> one test file, 1 test copy + 1 train file dropped
    _write(src / "test/CLS_A/t2.jpg", b"img-1")
    _write(src / "test/CLS_A/t1.jpg", b"img-1")
    _write(src / "train/CLS_A/x9.jpg", b"img-1")
    _write(src / "test/CLS_B/t3.jpg", b"img-2")
    # train copies (3) -> one train file
    for name in ("b.jpg", "a.jpg", "c.jpg"):
        _write(src / f"train/CLS_A/{name}", b"img-3")
    _write(src / "train/CLS_B/d.jpg", b"img-4")
    _write(src / "train/CLS_B/e.jpg", b"img-5")
    # a near-duplicate of a test image (2 train copies), excluded by md5
    _write(src / "train/CLS_B/n1.jpg", b"img-near")
    _write(src / "train/CLS_B/n2.jpg", b"img-near")
    exclude = root / "exclude.csv"
    digest = hashlib.md5(b"img-near").hexdigest()
    exclude.write_text(f"md5,reason\n{digest},near-duplicate of a test image\n", encoding="utf-8")
    return src, exclude


def _run(args: list[str]) -> int:
    return B.main(args)


def _old_build(src: Path, dst: Path, exclude: Path, tmp_path: Path) -> None:
    proc = subprocess.run(["git", "show", f"{OLD_BUILDER_REF}:scripts/datasets/build_daninhas_full_fixed.py"],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"git history with {OLD_BUILDER_REF} not available: {proc.stderr.strip()}")
    old = tmp_path / "old_builder" / "build_daninhas_full_fixed.py"
    _write(old, proc.stdout.encode())
    subprocess.run([sys.executable, "-I", str(old), "--src", str(src), "--dst", str(dst),
                    "--exclude-train-md5", str(exclude)], check=True, capture_output=True)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file below `root` -> (bytes, inode); plus the directory list."""
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else b"<dir>", p.stat().st_ino)
            for p in sorted(root.rglob("*"))}


def _images(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
            for split in ("train", "test") for p in sorted((root / split).rglob("*")) if p.is_file()}


# --- fresh build ---------------------------------------------------------------------------------------


def test_fresh_build_writes_the_layout_and_passes_check(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src, exclude = _source(tmp_path)
    dst = tmp_path / "fixed"
    assert _run(["--src", str(src), "--dst", str(dst), "--exclude-train-md5", str(exclude)]) == 0
    assert {p.name for p in dst.iterdir()} == {"train", "test", "README.txt", "MANIFEST.csv", "logs"}
    assert {p.name for p in (dst / "logs").iterdir()} == {"build_history.txt", "source_manifest.csv"}
    manifest = (dst / "MANIFEST.csv").read_bytes().decode()
    assert manifest.splitlines()[0] == "split,class,file,md5"
    assert sorted(manifest.splitlines()[1:]) == sorted(
        f"{s},{c},{n},{d}" for s, c, n, d in B.scan(dst))
    readme = (dst / "README.txt").read_text()
    # No reference to the old dataset or to how this one was obtained (that is logs/build_history.txt).
    assert "source files" not in readme and "removed" not in readme and "KI-46" not in readme
    assert re.search(r"daninhas_full(?!_fixed)", readme) is None
    assert "train: 3 images" in readme and "test: 2 images" in readme
    history = (dst / "logs/build_history.txt").read_text()
    assert "source files: 11" in history and "train_near_duplicates_of_test: 2" in history
    assert "test_copies: 1" in history and "train_files_also_in_test: 1" in history and "train_copies: 2" in history
    assert hashlib.md5(b"img-near").hexdigest() in history  # the exclusion list used
    assert "source_aliases_dropped" in (dst / "logs/source_manifest.csv").read_text()
    capsys.readouterr()
    assert _run(["--check", "--dst", str(dst), "--exclude-train-md5", str(exclude)]) == 0
    assert "OK: files on disk match MANIFEST.csv" in capsys.readouterr().out


def test_check_detects_a_corrupted_missing_or_extra_image(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src, exclude = _source(tmp_path)
    dst = tmp_path / "fixed"
    _run(["--src", str(src), "--dst", str(dst), "--exclude-train-md5", str(exclude)])
    check = ["--check", "--dst", str(dst), "--exclude-train-md5", str(exclude)]

    image = dst / "train/CLS_B/d.jpg"
    good = image.read_bytes()
    image.write_bytes(b"truncated")  # a corrupted upload: same name, other bytes
    capsys.readouterr()
    assert _run(check) == 1
    assert "md5 on disk differs from the manifest: train/CLS_B/d.jpg" in capsys.readouterr().out
    image.unlink()  # an incomplete upload
    assert _run(check) == 1
    assert "in MANIFEST.csv but missing on disk: train/CLS_B/d.jpg" in capsys.readouterr().out
    image.write_bytes(good)
    _write(dst / "test/CLS_A/stray.jpg", b"stray")
    assert _run(check) == 1
    assert "on disk but not in MANIFEST.csv: test/CLS_A/stray.jpg" in capsys.readouterr().out
    (dst / "test/CLS_A/stray.jpg").unlink()
    assert _run(check) == 0


def test_scan_ignores_logs_and_metadata(tmp_path: Path) -> None:
    src, exclude = _source(tmp_path)
    dst = tmp_path / "fixed"
    _run(["--src", str(src), "--dst", str(dst), "--exclude-train-md5", str(exclude)])
    rows = B.scan(dst)
    assert {s for s, *_ in rows} == {"train", "test"} and len(rows) == 5
    assert not any(n in {"README.txt", "MANIFEST.csv", "build_history.txt", "source_manifest.csv"}
                   for _, _, n, _ in rows)


# --- migration -----------------------------------------------------------------------------------------


def test_migration_of_an_old_layout_equals_a_fresh_build_and_touches_no_image(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    src, exclude = _source(tmp_path)
    fresh, old = tmp_path / "fresh", tmp_path / "old"
    _run(["--src", str(src), "--dst", str(fresh), "--exclude-train-md5", str(exclude)])
    _old_build(src, old, exclude, tmp_path)
    assert {p.name for p in old.iterdir()} == {"train", "test", "README.txt", "MANIFEST.csv"}
    old_manifest, old_readme = (old / "MANIFEST.csv").read_bytes(), (old / "README.txt").read_bytes()
    images = _images(old)
    migrate = ["--migrate-metadata", "--dst", str(old), "--exclude-train-md5", str(exclude)]

    # The old layout fails --check with a pointer to the migration.
    capsys.readouterr()
    assert _run(["--check", "--dst", str(old), "--exclude-train-md5", str(exclude)]) == 1
    assert "run --migrate-metadata" in capsys.readouterr().out

    before = _snapshot(old)
    assert _run([*migrate, "--dry-run"]) == 0
    assert "precondition OK: 5 image files" in capsys.readouterr().out
    assert _snapshot(old) == before  # the dry run wrote nothing

    assert _run(migrate) == 0
    assert "OK: migrated" in capsys.readouterr().out
    for rel in METADATA:
        assert (old / rel).read_bytes() == (fresh / rel).read_bytes(), rel  # byte-identical to a fresh build
    assert (old / "logs/source_manifest.csv").read_bytes() == old_manifest  # moved byte for byte
    assert old_readme.decode() in (old / "logs/build_history.txt").read_text()  # old README word for word
    assert _images(old) == images  # no image created, deleted, renamed or rewritten (bytes, inode, mtime)
    assert not [p for p in old.rglob("*.tmp")]

    # Idempotent-safe: a second call writes nothing and exits cleanly.
    after = _snapshot(old)
    assert _run(migrate) == 0
    assert "already migrated" in capsys.readouterr().out
    assert _snapshot(old) == after
    assert _run(["--check", "--dst", str(old), "--exclude-train-md5", str(exclude)]) == 0


@pytest.mark.parametrize("damage", ["remove_image", "corrupt_image", "extra_image", "edit_readme", "edit_manifest"])
def test_migration_aborts_without_writing_when_the_precondition_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture, damage: str
) -> None:
    src, exclude = _source(tmp_path)
    old = tmp_path / "old"
    _old_build(src, old, exclude, tmp_path)
    if damage == "remove_image":
        (old / "train/CLS_B/e.jpg").unlink()
    elif damage == "corrupt_image":
        (old / "train/CLS_B/e.jpg").write_bytes(b"corrupt")
    elif damage == "extra_image":
        _write(old / "train/CLS_B/z.jpg", b"img-new")
    elif damage == "edit_readme":
        (old / "README.txt").write_text((old / "README.txt").read_text() + "edited by hand\n")
    else:
        text = (old / "MANIFEST.csv").read_bytes().decode()
        (old / "MANIFEST.csv").write_bytes(text.replace(",1,", ",7,", 1).encode())
    before = _snapshot(old)
    capsys.readouterr()
    for extra in (["--dry-run"], []):
        assert _run(["--migrate-metadata", "--dst", str(old), "--exclude-train-md5", str(exclude), *extra]) == 1
        assert "precondition FAILED, nothing written" in capsys.readouterr().out
        assert _snapshot(old) == before


def test_migration_refuses_a_logs_folder_that_fails_check(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    src, exclude = _source(tmp_path)
    dst = tmp_path / "fixed"
    _run(["--src", str(src), "--dst", str(dst), "--exclude-train-md5", str(exclude)])
    (dst / "train/CLS_B/d.jpg").write_bytes(b"corrupt")
    before = _snapshot(dst)
    capsys.readouterr()
    assert _run(["--migrate-metadata", "--dst", str(dst), "--exclude-train-md5", str(exclude)]) == 1
    assert "refusing to migrate, nothing written" in capsys.readouterr().out
    assert _snapshot(dst) == before


def test_dry_run_requires_the_migrate_mode() -> None:
    with pytest.raises(SystemExit):
        _run(["--check", "--dry-run"])
    with pytest.raises(SystemExit):
        _run(["--check", "--migrate-metadata"])
