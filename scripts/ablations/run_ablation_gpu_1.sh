#!/bin/bash
# Lab-machine batch run for the ablation configs, GPU 1's half of the split
# (the suite is 10 configs for METHOD=rnhal, 11 for METHOD=texhal; `stage_no_representation`
# is the campaign's shared run, ADR 0009). See
# run_ablation_gpu_0.sh's header for the full split rationale (by expected
# relative cost, hierarchical selection dominates runtime, not training) and
# .specs/experiments/ablation-study.md / ablation-study-texhal.md's
# "Materialized files" sections.
#
# METHOD (2026-08-30): this script now serves BOTH ablation suites -- `rnhal`
# (paper 3, SSRAE, already executed 2026-08-25/26 on Colab) and `texhal`
# (paper 2, VCTex, not yet run) -- see files_config/ablations/README.md and
# .specs/experiments/papers-roadmap.md.
#
# This GPU (1) gets, per METHOD:
#   rnhal:  rep_spectral (last of 6.1's 3 heavy-hierarchy configs),
#           stage_full (heavy reference hierarchy -- the RNHAL-full row
#           re-run through the new RepresentationStrategy pipeline for
#           cross-checking, see files_config/ablations/README.md),
#           stage_no_hierarchy (the cheapest config in the whole sweep --
#           flat k-means, no multi-level clustering), and hier_L2a/hier_L3/
#           hier_L4 (the three heavier/medium 6.2 rows) -- 6 configs.
#   texhal: rep_q17, rep_q13 (the remaining two of 6.1's 4 heavy-hierarchy
#           configs -- rep_q13 added 2026-08-30 per the VCTex method
#           authors, see files_config/ablations/README.md), stage_full,
#           stage_no_hierarchy, hier_L2a, hier_L3, hier_L4 -- 7 configs.
# a comparable total load to GPU 0's half (run_ablation_gpu_0.sh, 5 configs
# for either method), which carries more heavy-hierarchy configs per config
# to compensate for having fewer configs overall.
#
# Same "keep going on failure" behavior as run_ablation_gpu_0.sh: `set -uo
# pipefail` (not `-e`), explicit `if ! ...; then ...` per run, failures
# logged to a file instead of aborting the batch.
#
# Environment overrides (all optional, defaults match the original lab-only
# behavior exactly) -- see run_ablation_gpu_0.sh's header for the full
# rationale of each:
#   METHOD        - "rnhal" (default) or "texhal" -- selects
#                   files_config/ablations/${METHOD}/ as the config source
#                   and results/ablations/${METHOD}/ as the default results
#                   root. Any other value exits 2.
#   GPU_NUMBER    - which CUDA device to target (default 1). Colab has a
#                   single GPU 0, so scripts/colab/run_ablations_colab.sh
#                   sets GPU_NUMBER=0 for both this and run_ablation_gpu_0.sh.
#   SKIP_EXISTING - "1" (default) skips a (study, config, seed) triple whose
#                   results.json already exists under dir_results ("SKIP
#                   (already completed)"), making relaunch after a Colab
#                   disconnect or a lab crash idempotent. Set "0" to force.
#   DRY_RUN       - "1" echoes commands (training + ExperimentNotifier)
#                   instead of executing them; default "0".
#   RESULTS_ROOT  - base results directory (default results/ablations/${METHOD}).
#
# IMPORTANT back-compat note: the already-executed RNHAL batch (2026-08-25/26)
# lives at the LEGACY root `results/ablations/{6_1,6_2,6_3}/` (no `rnhal/`
# segment) -- append-only, left in place. A new `METHOD=rnhal` run of this
# script writes to `results/ablations/rnhal/{6_1,6_2,6_3}/` by default;
# `SKIP_EXISTING` will NOT see the legacy tree (its glob is scoped to
# `RESULTS_ROOT`). Pass `RESULTS_ROOT=results/ablations` explicitly to
# extend the legacy layout instead of starting a new one.

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

CONFIG_DIR="files_config/ablations/${METHOD}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablations/${METHOD}}"
FAILURE_LOG="${RESULTS_ROOT}/gpu1_failures.log"

# (study, config_name) pairs -- config_name matches
# files_config/ablations/${METHOD}/<config_name>.json and becomes the results
# subfolder: results/ablations/${METHOD}/<study>/<config_name>/. Per-method
# arrays because §6.1's basenames (and row count) differ between rnhal and
# texhal (see files_config/ablations/README.md's naming-convention table);
# §6.2/§6.3 basenames are identical across methods.
CONFIGS_RNHAL=(
    "6_1:rep_spectral"
    "6_3:stage_full"
    "6_3:stage_no_hierarchy"
    "6_2:hier_L2a"
    "6_2:hier_L3"
    "6_2:hier_L4"
)
CONFIGS_TEXHAL=(
    "6_1:rep_q17"
    "6_1:rep_q13"
    "6_3:stage_full"
    "6_3:stage_no_hierarchy"
    "6_2:hier_L2a"
    "6_2:hier_L3"
    "6_2:hier_L4"
)

if [ "$METHOD" = "rnhal" ]; then
    CONFIGS=("${CONFIGS_RNHAL[@]}")
else
    CONFIGS=("${CONFIGS_TEXHAL[@]}")
fi

mkdir -p "$(dirname "$FAILURE_LOG")"
: > "$FAILURE_LOG"

echo "Iniciando bateria de ablações (METHOD=${METHOD}, GPU ${GPU_NUMBER})..."

for entry in "${CONFIGS[@]}"; do
    study="${entry%%:*}"
    config_name="${entry#*:}"
    params_json="${CONFIG_DIR}/${config_name}.json"
    dir_results="${RESULTS_ROOT}/${study}/${config_name}/"

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "EXECUTANDO: method=$METHOD study=$study config=$config_name seed=$seed (GPU $GPU_NUMBER)"
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
echo "Todas as ablações da GPU ${GPU_NUMBER} (METHOD=${METHOD}) foram concluídas."
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
    poetry run python ExperimentNotifier/main.py --dir_results="${RESULTS_ROOT}/" --args "METHOD=$METHOD, GPU_NUMBER=$GPU_NUMBER, ABLATION_BATCH=gpu1, FAILURE_LOG=$FAILURE_LOG"
else
    echo "ExperimentNotifier/main.py not found (expected on Colab) -- skipping email notification."
fi
