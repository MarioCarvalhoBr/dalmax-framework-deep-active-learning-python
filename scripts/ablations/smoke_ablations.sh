#!/bin/bash
# CPU-only smoke test for every Phase 3 ablation config, against the tiny
# DATA/daninhas_micro dataset (see scripts/make_micro_dataset.py). Never runs
# against DATA/daninhas_full — see .specs/infrastructure/execution-environments.md
# ("never run a full training/experiment locally").
#
# Runs all 11 files_config/ablations/micro/*.json configs (6.1 representation,
# 6.2 hierarchy, 6.3 stage-contribution) through demo.py with a tiny budget
# (--n_init_labeled 10 --n_query 5 --n_round 1), one round, seed 1, --device
# cpu, RepresentationStrategy. Fails on the first error (set -e) -- this is a
# smoke test, not a batch job that should keep going after a broken config;
# contrast with scripts/ablations/run_ablation_gpu_*.sh, which are meant to
# survive one bad config across a multi-hour lab run and therefore do NOT use
# set -e.
#
# Each config gets its own results subfolder, nested as
# results/smoke_ablations/<study>/<config_name>/ -- the SAME <study>/<config>
# layout scripts/ablations/run_ablation_gpu_*.sh use under results/ablations/,
# so results/smoke_ablations/ is real, walkable output for
# dalmax/reporting/ablation_report.py's `--root` (see
# tests/test_ablation_report.py and the "run it against results/smoke_ablations/"
# check in .specs/experiments/ablation-study.md's "Materialized files"
# section), not just a directory-collision fix (every config would otherwise
# share the same dataset_folder/SEED_1/NQ_5_NIL_10_NR_1_NE_1/
# RepresentationStrategy/ leaf -- see dalmax/experiment/runner.py::results_dir_for).
#
# Usage: bash scripts/ablations/smoke_ablations.sh
# Wired as `make smoke-ablations`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="files_config/ablations/micro"
RESULTS_ROOT="results/smoke_ablations"

# (study, config_name) pairs -- same study grouping as
# scripts/ablations/run_ablation_gpu_*.sh.
CONFIGS=(
    "6_1:rep_full"
    "6_1:rep_spatial"
    "6_1:rep_spectral"
    "6_2:hier_L1"
    "6_2:hier_L2a"
    "6_2:hier_L2b"
    "6_2:hier_L3"
    "6_2:hier_L4"
    "6_3:stage_full"
    "6_3:stage_no_representation"
    "6_3:stage_no_hierarchy"
)

echo "Generating DATA/daninhas_micro/ (no-op if DATA/daninhas_full/ is absent)..."
poetry run python scripts/make_micro_dataset.py

if [ ! -d "DATA/daninhas_micro/train" ]; then
    echo "NOTE: DATA/daninhas_full not present, skipping the ablation smoke runs."
    exit 0
fi

TOTAL=${#CONFIGS[@]}
PASSED=0

for entry in "${CONFIGS[@]}"; do
    study="${entry%%:*}"
    name="${entry#*:}"
    params_json="${CONFIG_DIR}/${name}.json"
    dir_results="${RESULTS_ROOT}/${study}/${name}/"

    echo ""
    echo "------------------------------------------------------------"
    echo "SMOKE: ${study}/${name} (${params_json})"
    echo "------------------------------------------------------------"

    poetry run python demo.py \
        --params_json "$params_json" \
        --dataset_name DANINHAS \
        --strategy_name RepresentationStrategy \
        --n_init_labeled 10 \
        --n_query 5 \
        --n_round 1 \
        --seed 1 \
        --device cpu \
        --dir_results "$dir_results"

    PASSED=$((PASSED + 1))
    echo "PASSED: ${study}/${name} (${PASSED}/${TOTAL})"
done

echo ""
echo "------------------------------------------------------------"
echo "All ${TOTAL} ablation smoke configs passed."
echo "------------------------------------------------------------"
