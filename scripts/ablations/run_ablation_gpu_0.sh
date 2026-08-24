#!/bin/bash
# Lab-machine batch run for Phase 3 ablation configs, GPU 0's half of the
# 11-config split (see the split rationale below and
# .specs/experiments/ablation-study.md's "Materialized files" section).
# Mirrors scripts/benchmark/run_pipe_gpu_0.sh's style (SEEDS sweep, one dir per config,
# ExperimentNotifier at the end) but iterates over a list of
# (study, config_name) pairs instead of a single params JSON, since every
# ablation config is its own file under files_config/ablations/.
#
# GPU split rationale (by expected relative cost -- hierarchical selection at
# a large/multi-level hierarchy is the dominant cost, not training):
#   - §6.1 (3 configs) shares the heaviest hierarchy (n_clusters=[600,200,100],
#     the scripts/benchmark/run_pipe_gpu_0.sh reference) across all three variants; split 2/1
#     across the two GPUs rather than putting all three on one.
#   - §6.3's `stage_no_representation` reuses that same heavy reference
#     hierarchy (plus a ResNet50 forward pass per image); `stage_full` also
#     uses it; `stage_no_hierarchy` (flat k-means, no multi-level clustering)
#     is the cheapest config in the whole sweep.
#   - §6.2's 5 rows range from cheap (`hier_L1`, k=[50], 1 level) to
#     expensive (`hier_L4`, k=[300,100,50,25], 4 levels); spread across GPUs
#     rather than stacking the heavy ones together.
# This GPU (0) gets: rep_full, rep_spatial (2 of 6.1's 3 heavy-hierarchy
# configs), stage_no_representation (heavy hierarchy + ResNet50 forward pass),
# hier_L1 (cheapest 6.2 row), hier_L2b (light-medium 6.2 row) -- 5 configs,
# a comparable total load to GPU 1's 6 configs (run_ablation_gpu_1.sh), which
# gets fewer heavy-hierarchy configs (2 vs this GPU's 3) to compensate for
# having more configs overall.
#
# Unlike scripts/benchmark/run_pipe_gpu_*.sh (which lets any single `python demo.py` failure
# kill the whole batch), this script logs a failing (study, config, seed) to
# a failure file and continues to the next iteration -- an 8-hour, 15-run
# lab batch should not be lost to one bad config. `set -uo pipefail` (not
# `-e`) plus an explicit `if ! ...; then ...` per run gives this "set -e-safe
# but keep going" behavior deliberately, not by omission.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_NUMBER=0
SEEDS=(1 2 3)
DATASET_NAME="DANINHAS"
STRATEGY_NAME="RepresentationStrategy"
N_QUERY=100
N_ROUND=8
# n_init_labeled: not passed explicitly, relying on the CLI default of 100 --
# matches scripts/benchmark/run_pipe_gpu_0.sh/scripts/benchmark/run_pipe_gpu_1.sh and
# .specs/experiments/experimental-protocol.md's "n_init_labeled" section.

CONFIG_DIR="files_config/ablations"
RESULTS_ROOT="results/ablations"
FAILURE_LOG="results/ablations/gpu0_failures.log"

# (study, config_name) pairs -- config_name matches
# files_config/ablations/<config_name>.json and becomes the results
# subfolder: results/ablations/<study>/<config_name>/.
CONFIGS=(
    "6_1:rep_full"
    "6_1:rep_spatial"
    "6_3:stage_no_representation"
    "6_2:hier_L1"
    "6_2:hier_L2b"
)

mkdir -p "$(dirname "$FAILURE_LOG")"
: > "$FAILURE_LOG"

echo "Iniciando bateria de ablações (GPU ${GPU_NUMBER})..."

for entry in "${CONFIGS[@]}"; do
    study="${entry%%:*}"
    config_name="${entry#*:}"
    params_json="${CONFIG_DIR}/${config_name}.json"
    dir_results="${RESULTS_ROOT}/${study}/${config_name}/"

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "EXECUTANDO: study=$study config=$config_name seed=$seed (GPU $GPU_NUMBER)"
        echo "------------------------------------------------------------"

        if ! CUDA_VISIBLE_DEVICES=$GPU_NUMBER poetry run python demo.py \
            --params_json "$params_json" \
            --dataset_name="$DATASET_NAME" \
            --strategy_name "$STRATEGY_NAME" \
            --n_query $N_QUERY \
            --seed "$seed" \
            --n_round $N_ROUND \
            --dir_results="$dir_results" \
            --device cuda
        then
            echo "FAILED: study=$study config=$config_name seed=$seed" | tee -a "$FAILURE_LOG"
        else
            echo "Comando executado: CUDA_VISIBLE_DEVICES=$GPU_NUMBER poetry run python demo.py --params_json $params_json --dataset_name=$DATASET_NAME --strategy_name $STRATEGY_NAME --n_query $N_QUERY --seed $seed --n_round $N_ROUND --dir_results=$dir_results --device cuda"
            echo "(study=$study, config=$config_name, seed=$seed) finalizado."
        fi
    done
done

echo ""
echo "------------------------------------------------------------"
echo "Todas as ablações da GPU ${GPU_NUMBER} foram concluídas."
if [ -s "$FAILURE_LOG" ]; then
    echo "FALHAS registradas em $FAILURE_LOG:"
    cat "$FAILURE_LOG"
else
    echo "Nenhuma falha registrada."
fi
echo "------------------------------------------------------------"

# Run ExperimentNotifier to send email notification (same pattern as
# scripts/benchmark/run_pipe_gpu_0.sh/scripts/benchmark/run_pipe_gpu_1.sh).
poetry run python ExperimentNotifier/main.py --dir_results="${RESULTS_ROOT}/" --args "GPU_NUMBER=$GPU_NUMBER, ABLATION_BATCH=gpu0, FAILURE_LOG=$FAILURE_LOG"
