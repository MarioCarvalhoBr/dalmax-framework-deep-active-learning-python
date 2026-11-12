#!/bin/bash
# Idempotent Colab-session setup: verifies Drive is mounted, builds/reuses a
# single dataset zip on Drive and unzips it onto the local runtime disk, checks
# the local copy (images per split, files == MANIFEST.csv, metadata layout, no
# repeated MD5), and wires results/ to a Google Drive symlink.
#
# Implements the hybrid layout decision documented in COLAB_RUNBOOK.md's
# introduction: the repo, its .venv, and DATA/daninhas_full_fixed all live on the
# Colab runtime's local disk (/content) for fast reads and a normal `poetry
# install`, while results/ is a symlink to Drive so every artifact
# (results.json, saved_model.pth checkpoints, run_metadata.json,
# log-dalmax.log, results/cache/embeddings/ caches) persists across a
# session disconnect -- Colab Pro gives no guaranteed background execution,
# so a dropped session must not lose finished work. Rejected alternative:
# putting the repo/.venv inside Drive itself (FUSE latency on every import,
# and a ~2.5 GB `poetry install` into Drive is impractically slow).
#
# Dataset (ADR 0016): the canonical, duplicate-free DATA/daninhas_full_fixed
# (KI-46; 7,230 train + 1,008 test images, plus README.txt, MANIFEST.csv and
# logs/{build_history.txt,source_manifest.csv} since the metadata layout of
# 2026-11-12 = 8,242 files). The first session to run this script builds
# ${DRIVE_ROOT}/DATA/daninhas_full_fixed.zip from the Drive folder and stores the
# zip back on Drive; every later session just copies that one zip to /content
# and unzips it locally in seconds, instead of reading thousands of small files
# through Drive's FUSE mount (the well-known slow path -- see COLAB_RUNBOOK.md's
# troubleshooting section). After unzipping, the local copy must have exactly
# the expected number of images per split AND pass
# `scripts/datasets/build_daninhas_full_fixed.py --check` (files on disk ==
# MANIFEST.csv by name and MD5, so a corrupted or incomplete upload fails; the
# metadata layout; no repeated MD5 inside or across the splits; standard library
# only). A Drive zip without logs/ predates the layout and is refused.
#
# Usage: DRIVE_ROOT=/content/drive/MyDrive/<folder holding DATA/> bash scripts/colab/setup_colab.sh
# (run from the repo root, e.g. after `%cd /content/dalmax` in the notebook,
# which exports DRIVE_ROOT from its Parameters cell).
#
# Idempotent: safe to re-run after a disconnect/reconnect --
#   - reuses the Drive-side zip if it already exists (does not rebuild it);
#   - skips the local unzip if DATA/daninhas_full_fixed already has the expected
#     images per split and its logs/ folder;
#   - leaves an already-correct results/ symlink alone;
#   - refuses (rather than silently clobbering) if a real, non-empty
#     results/ directory already exists locally -- see .claude/rules/
#     data-safety.md ("results/ is append-only").
#
# Env:
#   DRIVE_ROOT - REQUIRED: the Google Drive folder that holds DATA/ (and gets
#                results/) on THIS account; no default (it differs per account).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${DRIVE_ROOT:-}" ]; then
    echo "ERROR: DRIVE_ROOT is not set. Set it to the Drive folder that holds DATA/ on this account, e.g."
    echo "  DRIVE_ROOT=/content/drive/MyDrive/<your folder> make colab-setup"
    echo "(the notebook's Parameters cell exports it)."
    exit 1
fi

DATASET="daninhas_full_fixed"
EXPECTED_TRAIN_IMAGES=7230
EXPECTED_TEST_IMAGES=1008
LAYOUT_MARKER="logs/build_history.txt"   # present since the metadata layout of 2026-11-12
DRIVE_DATA_DIR="${DRIVE_ROOT}/DATA"
DRIVE_ZIP="${DRIVE_DATA_DIR}/${DATASET}.zip"
DRIVE_RESULTS="${DRIVE_ROOT}/results"
LOCAL_DATASET="DATA/${DATASET}"

echo "============================================================"
echo "DalMax Colab setup"
echo "============================================================"
echo "Repo root:   $REPO_ROOT"
echo "Drive root:  $DRIVE_ROOT"
echo "Dataset:     $DATASET (expected $EXPECTED_TRAIN_IMAGES train + $EXPECTED_TEST_IMAGES test images)"
echo "============================================================"

# --- 1. Assert Drive is mounted and the dataset folder exists there --------
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: /content/drive/MyDrive not found -- mount Google Drive first:"
    echo "  from google.colab import drive; drive.mount('/content/drive')"
    exit 1
fi

if [ ! -d "$DRIVE_DATA_DIR" ]; then
    echo "ERROR: $DRIVE_DATA_DIR not found on Drive."
    echo "Check DRIVE_ROOT and that ${DATASET} was uploaded to Drive under it (DATA/${DATASET}/)."
    exit 1
fi

# --- 2. Create the Drive results destination (never emptied/recreated if it
#        already exists -- results/ is append-only, .claude/rules/data-safety.md) ---
mkdir -p "$DRIVE_RESULTS"

# --- 3. Build (first session) or reuse (later sessions) the dataset zip on
#        Drive, then copy the single zip to /content and unzip it locally. --
if [ ! -f "$DRIVE_ZIP" ]; then
    if [ ! -d "${DRIVE_DATA_DIR}/${DATASET}" ]; then
        echo "ERROR: neither $DRIVE_ZIP nor ${DRIVE_DATA_DIR}/${DATASET} exist on Drive."
        exit 1
    fi
    echo "First session: building ${DRIVE_ZIP} from ${DRIVE_DATA_DIR}/${DATASET}"
    echo "(one-time, slower over Drive's FUSE mount -- later sessions reuse this zip)..."
    ( cd "$DRIVE_DATA_DIR" && zip -qr "${DATASET}.zip" "${DATASET}" ) || { echo "ERROR: zip failed"; exit 1; }
    echo "Zip built and stored back on Drive for future sessions."
else
    echo "Reusing existing zip: $DRIVE_ZIP"
fi
if ! unzip -l "$DRIVE_ZIP" "${DATASET}/${LAYOUT_MARKER}" > /dev/null 2>&1; then
    echo "ERROR: $DRIVE_ZIP has no ${DATASET}/${LAYOUT_MARKER}: it predates the metadata layout of 2026-11-12."
    echo "Upload the migrated DATA/${DATASET}/ folder (with logs/) to Drive, delete $DRIVE_ZIP, and re-run"
    echo "this script: it rebuilds the zip from the folder."
    exit 1
fi

LOCAL_ZIP="/content/${DATASET}.zip"
echo "Copying zip to local runtime disk ($LOCAL_ZIP)..."
cp "$DRIVE_ZIP" "$LOCAL_ZIP" || { echo "ERROR: copying the zip from Drive failed"; exit 1; }

mkdir -p DATA
count_images() {  # files below one split directory (0 when it does not exist)
    find "$LOCAL_DATASET/$1" -type f 2>/dev/null | wc -l
}

if [ "$(count_images train)" -eq "$EXPECTED_TRAIN_IMAGES" ] && [ "$(count_images test)" -eq "$EXPECTED_TEST_IMAGES" ] \
        && [ -f "$LOCAL_DATASET/$LAYOUT_MARKER" ]; then
    echo "$LOCAL_DATASET already has the expected images and logs/ locally -- skipping unzip."
else
    echo "Unzipping into DATA/ ..."
    unzip -q -o "$LOCAL_ZIP" -d DATA/ || { echo "ERROR: unzip failed"; exit 1; }
fi

TRAIN_COUNT=$(count_images train)
TEST_COUNT=$(count_images test)
echo "$LOCAL_DATASET images: train $TRAIN_COUNT (expected $EXPECTED_TRAIN_IMAGES), test $TEST_COUNT (expected $EXPECTED_TEST_IMAGES)"
if [ "$TRAIN_COUNT" -ne "$EXPECTED_TRAIN_IMAGES" ] || [ "$TEST_COUNT" -ne "$EXPECTED_TEST_IMAGES" ]; then
    echo "ERROR: image count mismatch -- the local copy is incomplete or the zip is stale."
    echo "If the Drive zip is stale, delete $DRIVE_ZIP so the next run rebuilds it from the folder."
    exit 1
fi

# The dataset contract: files == MANIFEST.csv (name and MD5), the metadata layout, no repeated MD5 (KI-46).
echo "Checking $LOCAL_DATASET against its MANIFEST.csv (scripts/datasets/build_daninhas_full_fixed.py --check)..."
if ! python3 scripts/datasets/build_daninhas_full_fixed.py --check --dst "$LOCAL_DATASET"; then
    echo "ERROR: $LOCAL_DATASET fails the dataset check (lines above) -- a corrupted or incomplete upload, a stale"
    echo "zip, or not the canonical dataset. Re-upload DATA/${DATASET}/ to Drive and delete $DRIVE_ZIP to rebuild it."
    exit 1
fi

# --- 4. Wire results/ to Drive via a symlink so every artifact persists
#        across session disconnects. -----------------------------------
if [ -L "results" ]; then
    current_target="$(cd "$(dirname "$(readlink results)")" 2>/dev/null && pwd)/$(basename "$(readlink results)")"
    if [ "$current_target" = "$DRIVE_RESULTS" ]; then
        echo "results/ is already a symlink to $DRIVE_RESULTS -- nothing to do."
    else
        echo "ERROR: results/ is already a symlink, but points at $current_target, not $DRIVE_RESULTS."
        echo "Remove it by hand (rm results) if you intend to repoint it, then re-run this script."
        exit 1
    fi
elif [ -d "results" ]; then
    if [ -n "$(ls -A results 2>/dev/null)" ]; then
        echo "ERROR: a real, non-empty results/ directory already exists locally at"
        echo "  $REPO_ROOT/results"
        echo "Refusing to replace it with a symlink (would shadow its contents)."
        echo "Move it aside first (e.g. mv results results_local_backup) if you really"
        echo "intend to switch to the Drive symlink."
        exit 1
    fi
    echo "Removing empty local results/ directory before symlinking..."
    rmdir results
    ln -s "$DRIVE_RESULTS" results
else
    ln -s "$DRIVE_RESULTS" results
fi

echo "============================================================"
echo "Setup complete."
echo "  $LOCAL_DATASET: $TRAIN_COUNT train + $TEST_COUNT test images, matches MANIFEST.csv, no repeated MD5 (local disk)"
echo "  results/ -> $DRIVE_RESULTS (Drive, persists across disconnects)"
echo "============================================================"
