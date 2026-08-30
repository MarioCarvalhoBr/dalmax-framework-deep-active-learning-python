#!/bin/bash
# Lab-machine batch run for the ablation configs, GPU 0's half of the split
# (11 configs for METHOD=rnhal, 12 for METHOD=texhal -- see the split
# rationale below and .specs/experiments/ablation-study.md /
# ablation-study-texhal.md's "Materialized files" sections).
# Mirrors scripts/benchmark/run_pipe_gpu_0.sh's style (SEEDS sweep, one dir per config,
# ExperimentNotifier at the end) but iterates over a list of
# (study, config_name) pairs instead of a single params JSON, since every
# ablation config is its own file under files_config/ablations/{rnhal,texhal}/.
#
# METHOD (2026-08-30): this script now serves BOTH ablation suites --
# `rnhal` (paper 3, SSRAE + hierarchical k-means, the batch already executed
# 2026-08-25/26 on Colab) and `texhal` (paper 2, VCTex + hierarchical
# k-means, not yet run) -- see files_config/ablations/README.md and
# .specs/experiments/papers-roadmap.md for the three-paper plan this
# reflects. Every (study, config_name) pair below resolves to
# files_config/ablations/${METHOD}/<config_name>.json.
#
# GPU split rationale (by expected relative cost -- hierarchical selection at
# a large/multi-level hierarchy is the dominant cost, not training). This
# rationale was derived for RNHAL and is kept analogous for TexHAL (the
# config *names* differ for §6.1 -- see files_config/ablations/README.md's
# naming-convention table -- but the GPU split follows the same shape):
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
# This GPU (0) gets, per METHOD:
#   rnhal:  rep_full, rep_spatial (2 of 6.1's 3 heavy-hierarchy configs),
#           stage_no_representation (heavy hierarchy + ResNet50 forward
#           pass), hier_L1 (cheapest 6.2 row), hier_L2b (light-medium 6.2
#           row) -- 5 configs.
#   texhal: rep_full, rep_q5 (2 of 6.1's 4 heavy-hierarchy configs, same
#           shape as rnhal's rep_full/rep_spatial), stage_no_representation,
#           hier_L1, hier_L2b -- 5 configs; the other two 6.1 rows
#           (rep_q13, rep_q17, added 2026-08-30 per the VCTex method
#           authors -- see files_config/ablations/README.md) both go to
#           GPU 1 (run_ablation_gpu_1.sh), which therefore carries 7
#           texhal configs to this GPU's 5.
# a comparable total load to GPU 1's rnhal half (6 configs), which
# gets fewer heavy-hierarchy configs (2 vs this GPU's 3) to compensate for
# having more configs overall.
#
# Unlike scripts/benchmark/run_pipe_gpu_*.sh (which lets any single `python trainer.py` failure
# kill the whole batch), this script logs a failing (study, config, seed) to
# a failure file and continues to the next iteration -- an 8-hour, 15-run
# lab batch should not be lost to one bad config. `set -uo pipefail` (not
# `-e`) plus an explicit `if ! ...; then ...` per run gives this "set -e-safe
# but keep going" behavior deliberately, not by omission.
#
# Environment overrides (all optional, defaults match the original lab-only
# behavior exactly):
#   METHOD        - "rnhal" (default) or "texhal" -- selects
#                   files_config/ablations/${METHOD}/ as the config source
#                   and results/ablations/${METHOD}/ as the default results
#                   root (see RESULTS_ROOT below). Any other value exits 2.
#   GPU_NUMBER    - which CUDA device to target (default 0). Colab has a
#                   single GPU 0, so scripts/colab/run_ablations_colab.sh
#                   sets GPU_NUMBER=0 for both this and run_ablation_gpu_1.sh.
#   SKIP_EXISTING - "1" (default) skips a (study, config, seed) triple whose
#                   results.json already exists under dir_results, logging
#                   "SKIP (already completed)" instead of re-running it. This
#                   is what makes relaunching after a Colab disconnect (no
#                   guaranteed background execution on Colab Pro) or a lab
#                   crash idempotent, and it respects results/'s append-only
#                   policy (.claude/rules/data-safety.md) by never touching an
#                   existing leaf directory. Set to "0" to force-rerun
#                   everything.
#   DRY_RUN       - "1" echoes the CUDA_VISIBLE_DEVICES + poetry run command
#                   (and the ExperimentNotifier call) instead of executing
#                   them; default "0". Used by tests/test_ablation_scripts.py
#                   to exercise the SKIP_EXISTING logic without spending any
#                   GPU time or touching ExperimentNotifier.
#   RESULTS_ROOT  - base results directory (default results/ablations/${METHOD}).
#                   Overridable so tests can point this at a throwaway temp
#                   directory instead of the real results/ tree.
#
# IMPORTANT back-compat note: the already-executed RNHAL batch (2026-08-25/26,
# see .specs/experiments/ablation-study.md's "Execution record") lives at the
# LEGACY root `results/ablations/{6_1,6_2,6_3}/` (no `rnhal/` segment) --
# that tree is append-only (.claude/rules/data-safety.md) and is left exactly
# where it is. A *new* `METHOD=rnhal` run of this script (e.g. re-running a
# failed triple, or extending the batch) writes to the NEW root
# `results/ablations/rnhal/{6_1,6_2,6_3}/` by default -- `SKIP_EXISTING`'s
# glob only looks under `RESULTS_ROOT`, so it will NOT see the legacy tree
# and will happily re-run every triple that already completed there. If you
# want to extend the executed legacy RNHAL batch instead of starting a new
# one, pass `RESULTS_ROOT=results/ablations` explicitly to land in the
# legacy layout (or use `make ablation-report-legacy` to report on it as-is).

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

GPU_NUMBER="${GPU_NUMBER:-0}"
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
FAILURE_LOG="${RESULTS_ROOT}/gpu0_failures.log"

# (study, config_name) pairs -- config_name matches
# files_config/ablations/${METHOD}/<config_name>.json and becomes the results
# subfolder: results/ablations/${METHOD}/<study>/<config_name>/. Per-method
# arrays because §6.1's basenames differ between rnhal and texhal (see
# files_config/ablations/README.md's naming-convention table); §6.2/§6.3
# basenames are identical across methods.
CONFIGS_RNHAL=(
    "6_1:rep_full"
    "6_1:rep_spatial"
    "6_3:stage_no_representation"
    "6_2:hier_L1"
    "6_2:hier_L2b"
)
CONFIGS_TEXHAL=(
    "6_1:rep_full"
    "6_1:rep_q5"
    "6_3:stage_no_representation"
    "6_2:hier_L1"
    "6_2:hier_L2b"
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
    poetry run python ExperimentNotifier/main.py --dir_results="${RESULTS_ROOT}/" --args "METHOD=$METHOD, GPU_NUMBER=$GPU_NUMBER, ABLATION_BATCH=gpu0, FAILURE_LOG=$FAILURE_LOG"
else
    echo "ExperimentNotifier/main.py not found (expected on Colab) -- skipping email notification."
fi
