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
#
# Environment overrides (all optional, defaults match the original lab-only
# behavior exactly) -- see run_ablation_gpu_0.sh's header for the full
# rationale of each:
#   GPU_NUMBER    - which CUDA device to target (default 1). Colab has a
#                   single GPU 0, so scripts/colab/run_ablations_colab.sh
#                   sets GPU_NUMBER=0 for both this and run_ablation_gpu_0.sh.
#   SKIP_EXISTING - "1" (default) skips a (study, config, seed) triple whose
#                   results.json already exists under dir_results ("SKIP
#                   (already completed)"), making relaunch after a Colab
#                   disconnect or a lab crash idempotent. Set "0" to force.
#   DRY_RUN       - "1" echoes commands (training + ExperimentNotifier)
#                   instead of executing them; default "0".
#   RESULTS_ROOT  - base results directory (default results/ablations).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_NUMBER="${GPU_NUMBER:-1}"
SEEDS=(1 2 3)
DATASET_NAME="DANINHAS"
STRATEGY_NAME="RepresentationStrategy"
N_QUERY=100
N_ROUND=8
# n_init_labeled: not passed explicitly, relying on the CLI default of 100 --
# matches scripts/benchmark/run_pipe_gpu_0.sh/scripts/benchmark/run_pipe_gpu_1.sh and
# .specs/experiments/experimental-protocol.md's "n_init_labeled" section.

SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG_DIR="files_config/ablations"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablations}"
FAILURE_LOG="${RESULTS_ROOT}/gpu1_failures.log"

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

        # SKIP_EXISTING: glob for this triple's results.json rather than
        # hardcoding NIL/NE, since both come from the params JSON / CLI
        # default (see dalmax/experiment/runner.py::results_dir_for), not
        # this script.
        existing_glob="${dir_results}*/SEED_${seed}/NQ_${N_QUERY}_NIL_*_NR_${N_ROUND}_NE_*/${STRATEGY_NAME}/results.json"
        if [ "$SKIP_EXISTING" = "1" ] && compgen -G "$existing_glob" > /dev/null; then
            echo "SKIP (already completed): study=$study config=$config_name seed=$seed"
            continue
        fi

        cmd=(poetry run python trainer.py
            --params_json "$params_json"
            --dataset_name="$DATASET_NAME"
            --strategy_name "$STRATEGY_NAME"
            --n_query "$N_QUERY"
            --seed "$seed"
            --n_round "$N_ROUND"
            --dir_results="$dir_results"
            --device cuda)

        if [ "$DRY_RUN" = "1" ]; then
            echo "DRY-RUN: CUDA_VISIBLE_DEVICES=$GPU_NUMBER ${cmd[*]}"
            continue
        fi

        if ! CUDA_VISIBLE_DEVICES=$GPU_NUMBER "${cmd[@]}"; then
            echo "FAILED: study=$study config=$config_name seed=$seed" | tee -a "$FAILURE_LOG"
        else
            echo "Comando executado: CUDA_VISIBLE_DEVICES=$GPU_NUMBER ${cmd[*]}"
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
# Guarded: ExperimentNotifier/ is a separate, gitignored sibling repo
# (.claude/rules/data-safety.md) that is absent on a fresh Colab clone -- skip
# the notification there instead of failing the whole batch on a missing
# file. Also skipped under DRY_RUN, so a dry run never sends a real email.
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY-RUN: skipping ExperimentNotifier call."
elif [ -f ExperimentNotifier/main.py ]; then
    poetry run python ExperimentNotifier/main.py --dir_results="${RESULTS_ROOT}/" --args "GPU_NUMBER=$GPU_NUMBER, ABLATION_BATCH=gpu1, FAILURE_LOG=$FAILURE_LOG"
else
    echo "ExperimentNotifier/main.py not found (expected on Colab) -- skipping email notification."
fi
