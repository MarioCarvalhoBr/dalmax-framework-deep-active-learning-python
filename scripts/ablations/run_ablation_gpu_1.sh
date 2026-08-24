#!/bin/bash
# Lab-machine batch run for Phase 3 ablation configs, GPU 1's half of the
# 11-config split. See run_ablation_gpu_0.sh's header for the full split
# rationale (by expected relative cost, hierarchical selection dominates
# runtime, not training) and .specs/experiments/ablation-study.md's
# "Materialized files" section.
#
# This GPU (1) gets: rep_spectral (last of 6.1's 3 heavy-hierarchy configs),
# stage_full (heavy reference hierarchy -- the RNHAL-full row re-run through
# the new RepresentationStrategy pipeline for cross-checking, see
# files_config/ablations/README.md), stage_no_hierarchy (the cheapest config
# in the whole sweep -- flat k-means, no multi-level clustering), and
# hier_L2a/hier_L3/hier_L4 (the three heavier/medium 6.2 rows) -- 6 configs,
# a comparable total load to GPU 0's 5 configs (run_ablation_gpu_0.sh), which
# carries 3 heavy-hierarchy configs vs this GPU's 2 to compensate for having
# fewer configs overall.
#
# Same "keep going on failure" behavior as run_ablation_gpu_0.sh: `set -uo
# pipefail` (not `-e`), explicit `if ! ...; then ...` per run, failures
# logged to a file instead of aborting the batch.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_NUMBER=1
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
FAILURE_LOG="results/ablations/gpu1_failures.log"

# (study, config_name) pairs -- config_name matches
# files_config/ablations/<config_name>.json and becomes the results
# subfolder: results/ablations/<study>/<config_name>/.
CONFIGS=(
    "6_1:rep_spectral"
    "6_3:stage_full"
    "6_3:stage_no_hierarchy"
    "6_2:hier_L2a"
    "6_2:hier_L3"
    "6_2:hier_L4"
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
poetry run python ExperimentNotifier/main.py --dir_results="${RESULTS_ROOT}/" --args "GPU_NUMBER=$GPU_NUMBER, ABLATION_BATCH=gpu1, FAILURE_LOG=$FAILURE_LOG"
