#!/bin/bash
# CPU-only smoke test for every ablation config, against the tiny
# DATA/daninhas_micro dataset (see scripts/make_micro_dataset.py). Never runs
# against DATA/daninhas_full — see .specs/infrastructure/execution-environments.md
# ("never run a full training/experiment locally").
#
# METHOD (2026-08-30): "rnhal" (paper 3, SSRAE), "texhal" (paper 2, VCTex),
# or "both" (default) -- runs `files_config/ablations/{rnhal,texhal}/micro/*.json`
# through trainer.py (historical `demo.py`) with a tiny budget
# (--n_init_labeled 10 --n_query 5 --n_round 1), one round, seed 1, --device
# cpu, RepresentationStrategy. Fails on the first error (set -e) -- this is a
# smoke test, not a batch job that should keep going after a broken config;
# contrast with scripts/ablations/run_ablation_gpu_*.sh, which are meant to
# survive one bad config across a multi-hour lab run and therefore do NOT use
# set -e.
#
# `METHOD=both` (default) runs rnhal's 11 configs then texhal's 12 configs
# sequentially, 23 total. `METHOD=rnhal` or `METHOD=texhal` restricts to one
# suite -- see files_config/ablations/README.md for what each config's
# embedding/selection block is. This is also the first place VCTex actually
# runs through the generic RepresentationStrategy path (previously only
# through the legacy VCTexKmeansSampling/VCTexKmeansHCSampling presets) --
# see .specs/experiments/ablation-study-texhal.md's "VCTex-through-generic-
# path findings" section for anything this surfaced.
#
# Each config gets its own results subfolder, nested as
# results/smoke_ablations/<method>/<study>/<config_name>/ -- the SAME
# <study>/<config> layout scripts/ablations/run_ablation_gpu_*.sh use under
# results/ablations/<method>/, so results/smoke_ablations/<method>/ is real,
# walkable output for dalmax/reporting/ablation_report.py's `--root` (see
# tests/test_ablation_report.py and the "run it against
# results/smoke_ablations/<method>/" check in
# .specs/experiments/ablation-study.md's "Materialized files" section), not
# just a directory-collision fix (every config would otherwise share the
# same dataset_folder/SEED_1/NQ_5_NIL_10_NR_1_NE_1/RepresentationStrategy/
# leaf -- see dalmax/experiment/runner.py::results_dir_for).
#
# Usage: bash scripts/ablations/smoke_ablations.sh          (both methods)
#        METHOD=rnhal bash scripts/ablations/smoke_ablations.sh
#        METHOD=texhal bash scripts/ablations/smoke_ablations.sh
# Wired as `make smoke-ablations` (passes METHOD through if set).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:-both}"
if [ "$METHOD" != "rnhal" ] && [ "$METHOD" != "texhal" ] && [ "$METHOD" != "both" ]; then
    echo "ERROR: METHOD must be 'rnhal', 'texhal', or 'both', got '$METHOD'" >&2
    exit 2
fi

# (study, config_name) pairs -- same study grouping as
# scripts/ablations/run_ablation_gpu_*.sh. §6.1 basenames differ between
# rnhal and texhal (see files_config/ablations/README.md); texhal has a 4th
# §6.1 row (rep_q13, added 2026-08-30 per the VCTex method authors).
CONFIGS_RNHAL=(
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
CONFIGS_TEXHAL=(
    "6_1:rep_q5"
    "6_1:rep_q13"
    "6_1:rep_q17"
    "6_1:rep_full"
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

run_method_smoke() {
    local method="$1"
    shift
    local configs=("$@")
    local config_dir="files_config/ablations/${method}/micro"
    local results_root="results/smoke_ablations/${method}"
    local total=${#configs[@]}
    local passed=0

    echo ""
    echo "############################################################"
    echo "SMOKE METHOD=${method} (${total} configs)"
    echo "############################################################"

    for entry in "${configs[@]}"; do
        local study="${entry%%:*}"
        local name="${entry#*:}"
        local params_json="${config_dir}/${name}.json"
        local dir_results="${results_root}/${study}/${name}/"

        echo ""
        echo "------------------------------------------------------------"
        echo "SMOKE: ${method}/${study}/${name} (${params_json})"
        echo "------------------------------------------------------------"

        poetry run python trainer.py \
            --params_json "$params_json" \
            --dataset_name DANINHAS \
            --strategy_name RepresentationStrategy \
            --n_init_labeled 10 \
            --n_query 5 \
            --n_round 1 \
            --seed 1 \
            --device cpu \
            --dir_results "$dir_results"

        passed=$((passed + 1))
        echo "PASSED: ${method}/${study}/${name} (${passed}/${total})"
    done

    echo ""
    echo "------------------------------------------------------------"
    echo "All ${total} ${method} ablation smoke configs passed."
    echo "------------------------------------------------------------"
}

if [ "$METHOD" = "rnhal" ] || [ "$METHOD" = "both" ]; then
    run_method_smoke rnhal "${CONFIGS_RNHAL[@]}"
fi

if [ "$METHOD" = "texhal" ] || [ "$METHOD" = "both" ]; then
    run_method_smoke texhal "${CONFIGS_TEXHAL[@]}"
fi

echo ""
echo "------------------------------------------------------------"
echo "All ablation smoke configs passed (METHOD=${METHOD})."
echo "------------------------------------------------------------"
