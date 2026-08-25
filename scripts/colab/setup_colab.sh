#!/bin/bash
# Idempotent Colab-session setup: verifies Drive is mounted, builds/reuses a
# single dataset zip on Drive and unzips it onto the local runtime disk, and
# wires results/ to a Google Drive symlink.
#
# Implements the hybrid layout decision documented in COLAB_RUNBOOK.md's
# introduction: the repo, its .venv, and DATA/daninhas_full all live on the
# Colab runtime's local disk (/content) for fast reads and a normal `poetry
# install`, while results/ is a symlink to Drive so every artifact
# (results.json, saved_model.pth checkpoints, run_metadata.json,
# log-dalmax.log, results/cache/embeddings/ SSRAE cache) persists across a
# session disconnect -- Colab Pro gives no guaranteed background execution,
# so a dropped session must not lose finished work. Rejected alternative:
# putting the repo/.venv inside Drive itself (FUSE latency on every import,
# and a ~2.5 GB `poetry install` into Drive is impractically slow).
#
# Dataset transfer strategy: the first session to run this script builds
# DATA/daninhas_full.zip from the Drive folder and stores the zip back on
# Drive (${DRIVE_ROOT}/DATA/daninhas_full.zip); every later session just
# copies that one ~47 MB zip to /content and unzips it locally in seconds,
# instead of copying 10,193 small files through Drive's FUSE mount one at a
# time (the well-known slow path -- see COLAB_RUNBOOK.md's troubleshooting
# section and .specs/infrastructure/execution-environments.md's Colab
# checklist).
#
# Usage: bash scripts/colab/setup_colab.sh   (no arguments; run from the repo
# root, e.g. after `%cd /content/dalmax` in the notebook).
#
# Idempotent: safe to re-run after a disconnect/reconnect --
#   - reuses the Drive-side zip if it already exists (does not rebuild it);
#   - skips the local unzip if DATA/daninhas_full already has the expected
#     file count;
#   - leaves an already-correct results/ symlink alone;
#   - refuses (rather than silently clobbering) if a real, non-empty
#     results/ directory already exists locally -- see .claude/rules/
#     data-safety.md ("results/ is append-only").
#
# Env overrides:
#   DRIVE_ROOT - root of this project's Google Drive folder. Default matches
#                the path this PhD project actually uses (see
#                COLAB_RUNBOOK.md's "User facts" / cell 1).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/UFMS/Pós-graduação/Doutorado/FINAL/PROJETO/DALMAX}"
DRIVE_DATA_DIR="${DRIVE_ROOT}/DATA"
DRIVE_ZIP="${DRIVE_DATA_DIR}/daninhas_full.zip"
DRIVE_RESULTS="${DRIVE_ROOT}/results"

EXPECTED_FILE_COUNT=10193

echo "============================================================"
echo "DalMax Colab setup"
echo "============================================================"
echo "Repo root:   $REPO_ROOT"
echo "Drive root:  $DRIVE_ROOT"
echo "============================================================"

# --- 1. Assert Drive is mounted and the dataset folder exists there --------
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: /content/drive/MyDrive not found -- mount Google Drive first:"
    echo "  from google.colab import drive; drive.mount('/content/drive')"
    exit 1
fi

if [ ! -d "$DRIVE_DATA_DIR" ]; then
    echo "ERROR: $DRIVE_DATA_DIR not found on Drive."
    echo "Check DRIVE_ROOT and that daninhas_full was uploaded to Drive at that path."
    exit 1
fi

# --- 2. Create the Drive results destination (never emptied/recreated if it
#        already exists -- results/ is append-only, .claude/rules/data-safety.md) ---
mkdir -p "$DRIVE_RESULTS"

# --- 3. Build (first session) or reuse (later sessions) the dataset zip on
#        Drive, then copy the single zip to /content and unzip it locally. --
if [ ! -f "$DRIVE_ZIP" ]; then
    if [ ! -d "${DRIVE_DATA_DIR}/daninhas_full" ]; then
        echo "ERROR: neither $DRIVE_ZIP nor ${DRIVE_DATA_DIR}/daninhas_full exist on Drive."
        exit 1
    fi
    echo "First session: building ${DRIVE_ZIP} from ${DRIVE_DATA_DIR}/daninhas_full"
    echo "(one-time, slower over Drive's FUSE mount -- later sessions reuse this zip)..."
    ( cd "$DRIVE_DATA_DIR" && zip -qr daninhas_full.zip daninhas_full )
    echo "Zip built and stored back on Drive for future sessions."
else
    echo "Reusing existing zip: $DRIVE_ZIP"
fi

LOCAL_ZIP="/content/daninhas_full.zip"
echo "Copying zip to local runtime disk ($LOCAL_ZIP)..."
cp "$DRIVE_ZIP" "$LOCAL_ZIP"

mkdir -p DATA
CURRENT_COUNT=0
if [ -d "DATA/daninhas_full" ]; then
    CURRENT_COUNT=$(find DATA/daninhas_full -type f | wc -l)
fi

if [ "$CURRENT_COUNT" -eq "$EXPECTED_FILE_COUNT" ]; then
    echo "DATA/daninhas_full already has $EXPECTED_FILE_COUNT files locally -- skipping unzip."
else
    echo "Unzipping into DATA/ ..."
    unzip -q -o "$LOCAL_ZIP" -d DATA/
fi

ACTUAL_COUNT=$(find DATA/daninhas_full -type f | wc -l)
echo "DATA/daninhas_full file count: $ACTUAL_COUNT (expected $EXPECTED_FILE_COUNT)"
if [ "$ACTUAL_COUNT" -ne "$EXPECTED_FILE_COUNT" ]; then
    echo "ERROR: file count mismatch -- the local copy is incomplete or the zip is stale."
    echo "If DATA/daninhas_full/arquivos.txt exists, diff it against a known-good copy"
    echo "(e.g. the lab machine's) to find the missing/extra files, per LAB_RUNBOOK.md's"
    echo "dataset-transfer verification step."
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
echo "  DATA/daninhas_full: $ACTUAL_COUNT files (local disk, fast reads)"
echo "  results/ -> $DRIVE_RESULTS (Drive, persists across disconnects)"
echo "============================================================"
