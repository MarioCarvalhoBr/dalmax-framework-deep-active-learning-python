#!/bin/bash
# Run the full ablation sweep (33 runs for METHOD=rnhal, 36 for
# METHOD=texhal -- see scripts/ablations/run_ablation_gpu_{0,1}.sh's headers)
# sequentially on Colab's single GPU, by running both lab scripts (normally
# split across two GPUs) back-to-back on GPU 0.
#
# METHOD (2026-08-30): forwarded through to both underlying scripts --
# "rnhal" (default, paper 3, SSRAE) or "texhal" (paper 2, VCTex). See
# files_config/ablations/README.md and .specs/experiments/papers-roadmap.md.
#
# Safe to re-run after a Colab disconnect: both scripts default to
# SKIP_EXISTING=1 (see scripts/ablations/run_ablation_gpu_{0,1}.sh's headers),
# so any (study, config, seed) triple that already has a results.json under
# results/ablations/${METHOD}/ (which persists on Drive across sessions, per
# scripts/colab/setup_colab.sh) is skipped instead of re-run.
#
# Usage: bash scripts/colab/run_ablations_colab.sh   (no arguments; run from
# the repo root, after scripts/colab/setup_colab.sh has wired up DATA/ and
# results/). METHOD=texhal bash scripts/colab/run_ablations_colab.sh to run
# the TexHAL suite instead of the default RNHAL one.
#
# Env overrides: forwarded to both underlying scripts if set (e.g.
# DRY_RUN=1 METHOD=texhal bash scripts/colab/run_ablations_colab.sh for a dry
# run of the TexHAL suite).

set -uo pipefail

# Headless plotting: notebook front-ends (Colab) export an MPLBACKEND that the
# Poetry .venv cannot import (see dalmax/__init__.py). Belt-and-braces here.
export MPLBACKEND=Agg

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:-rnhal}"
if [ "$METHOD" != "rnhal" ] && [ "$METHOD" != "texhal" ]; then
    echo "ERROR: METHOD must be 'rnhal' or 'texhal', got '$METHOD'" >&2
    exit 2
fi
export METHOD

LOG_DIR="results/ablations/${METHOD}"
LOG_FILE="${LOG_DIR}/colab.log"
mkdir -p "$LOG_DIR"

{
    echo "============================================================"
    echo "DalMax Colab ablation sweep (METHOD=${METHOD}) -- both GPU halves on GPU 0"
    echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================================"

    echo ""
    echo ">>> Running run_ablation_gpu_0.sh's configs (METHOD=${METHOD}) on GPU 0..."
    GPU_NUMBER=0 bash scripts/ablations/run_ablation_gpu_0.sh

    echo ""
    echo ">>> Running run_ablation_gpu_1.sh's configs (METHOD=${METHOD}) on GPU 0..."
    GPU_NUMBER=0 bash scripts/ablations/run_ablation_gpu_1.sh

    echo ""
    echo "============================================================"
    echo "Colab ablation sweep (METHOD=${METHOD}) finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Check results/ablations/${METHOD}/gpu{0,1}_failures.log for any failed runs."
    echo "============================================================"
} 2>&1 | tee -a "$LOG_FILE"
