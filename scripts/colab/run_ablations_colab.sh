#!/bin/bash
# Run the full 11-config / 33-run Phase 3 ablation sweep sequentially on
# Colab's single GPU, by running both lab scripts (normally split across two
# GPUs) back-to-back on GPU 0.
#
# Safe to re-run after a Colab disconnect: both scripts default to
# SKIP_EXISTING=1 (see scripts/ablations/run_ablation_gpu_{0,1}.sh's headers),
# so any (study, config, seed) triple that already has a results.json under
# results/ablations/ (which persists on Drive across sessions, per
# scripts/colab/setup_colab.sh) is skipped instead of re-run.
#
# Usage: bash scripts/colab/run_ablations_colab.sh   (no arguments; run from
# the repo root, after scripts/colab/setup_colab.sh has wired up DATA/ and
# results/).
#
# Env overrides: forwarded to both underlying scripts if set (e.g.
# DRY_RUN=1 bash scripts/colab/run_ablations_colab.sh for a dry run).

set -uo pipefail

# Headless plotting: notebook front-ends (Colab) export an MPLBACKEND that the
# Poetry .venv cannot import (see dalmax/__init__.py). Belt-and-braces here.
export MPLBACKEND=Agg

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="results/ablations"
LOG_FILE="${LOG_DIR}/colab.log"
mkdir -p "$LOG_DIR"

{
    echo "============================================================"
    echo "DalMax Colab ablation sweep -- both GPU halves on GPU 0"
    echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================================"

    echo ""
    echo ">>> Running run_ablation_gpu_0.sh's 5 configs on GPU 0..."
    GPU_NUMBER=0 bash scripts/ablations/run_ablation_gpu_0.sh

    echo ""
    echo ">>> Running run_ablation_gpu_1.sh's 6 configs on GPU 0..."
    GPU_NUMBER=0 bash scripts/ablations/run_ablation_gpu_1.sh

    echo ""
    echo "============================================================"
    echo "Colab ablation sweep finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Check results/ablations/gpu{0,1}_failures.log for any failed runs."
    echo "============================================================"
} 2>&1 | tee -a "$LOG_FILE"
