.PHONY: setup lint format test test-all smoke smoke-ablations clean \
	lab-setup lab-check micro-dataset \
	ablations-gpu0 ablations-gpu1 ablations-all ablation-report \
	benchmark-gpu0 benchmark-gpu1 \
	colab-setup ablations-colab colab-check

# GPU index used by `lab-check` (0 or 1, matching files_config/benchmark/params_df_gpu_{0,1}.json).
# Override on the command line, e.g. `make lab-check GPU=1`.
GPU ?= 0

# Install the Poetry-managed environment (./.venv, see poetry.toml).
setup:
	poetry install

# Report-only lint pass. Existing code (pre-refactor) has known ruff
# violations (see .specs/quality/known-issues.md); this target does not fail
# the build on findings, it just reports them. `make -k` is used so a
# non-zero ruff exit does not stop a `make ci`-style chain; CI itself sets
# `continue-on-error: true` on the equivalent step for the same reason
# (see .github/workflows/ci.yml).
lint:
	poetry run ruff check .

format:
	poetry run ruff format .

# Fast test suite: excludes anything requiring a GPU, the real DATA/ dataset,
# or a slow full-pipeline run. This is what CI runs on every push/PR.
test:
	poetry run pytest -q -m "not gpu and not dataset and not slow"

# Full test suite, including gpu/dataset/slow-marked tests. Meant to be run
# on the lab machine (2x GPU, DATA/ present), not on the local CPU-only
# notebook.
test-all:
	poetry run pytest -q

# `smoke` is a true end-to-end run of trainer.py (historical name `demo.py`,
# renamed 2026-08-23) on a tiny CPU 2-class subset,
# landed in refactor Phase 1 (see .specs/architecture/refactor-plan.md and
# .specs/quality/testing-strategy.md):
#   1. scripts/make_micro_dataset.py deterministically samples 25 train + 10
#      test images per class from DATA/daninhas_full/ (2 classes) into
#      DATA/daninhas_micro/, without ever writing into daninhas_full/ itself.
#      If daninhas_full/ isn't present (e.g. a fresh clone with no dataset
#      copied in), it prints a message and exits 0 instead of failing.
#   2. trainer.py runs against files_config/params_micro.json (n_epoch=1,
#      n_classes=2, batch_size=16) with RandomSampling, into results/smoke/
#      (gitignored, never committed) — a few seconds on CPU, no GPU needed.
#   3. The fast test suite runs on top, including tests/test_cache_paths.py.
# The full golden-run regression (tests/test_golden_run.py, comparing exact
# selected indices + metrics against tests/golden/*.json for both
# RandomSampling and SSRAEKmeansSampling) is `dataset`+`slow`-marked and runs
# via `make test-all`, not here, to keep `make smoke` fast.
smoke:
	poetry run python scripts/make_micro_dataset.py
	@if [ -d DATA/daninhas_micro/train ]; then \
		poetry run python trainer.py \
			--params_json files_config/params_micro.json \
			--dataset_name DANINHAS \
			--strategy_name RandomSampling \
			--n_init_labeled 10 --n_query 5 --n_round 1 --seed 1 \
			--dir_results results/smoke/; \
	else \
		echo "NOTE: DATA/daninhas_full not present, skipping the trainer.py smoke run."; \
	fi
	poetry run pytest -q -m "not gpu and not dataset and not slow"

# `smoke-ablations`: CPU end-to-end smoke test for all 11 Phase 3 ablation
# configs (files_config/ablations/micro/*.json — see
# .specs/experiments/ablation-study.md and that folder's README.md), against
# the same tiny DATA/daninhas_micro dataset as `make smoke`. Each config runs
# with a tiny budget (--n_init_labeled 10 --n_query 5 --n_round 1, seed 1,
# --device cpu) into its own results/smoke_ablations/<config>/ subfolder.
# Fails on the first broken config (see scripts/ablations/smoke_ablations.sh's
# own header for why this is set -e, unlike the lab run scripts).
smoke-ablations:
	bash scripts/ablations/smoke_ablations.sh

clean:
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

# --- Lab machine targets (2x NVIDIA GPU, 10 GB each) -------------------------
# See LAB_RUNBOOK.md for the full step-by-step operator guide these targets
# are called from; do not run any of these on the local no-GPU dev notebook
# (`.claude/rules/data-safety.md` / `.specs/infrastructure/execution-environments.md`).

# One-time (or post-`git pull`) lab environment setup: sync the Poetry env to
# poetry.lock, then print whether CUDA is visible and how many devices torch
# sees. Expected output on the lab machine: `True 2`. Does not run any
# training — see `lab-check` for the first real-data GPU smoke run.
lab-setup:
	poetry install
	poetry run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# The first real-data GPU check after a fresh `lab-setup` (or after any code
# change that touches the training/embedding/selection path): one short
# SSRAEKmeansHCSampling run on the real daninhas_full dataset, n_round=1, into
# a throwaway results/lab_check/ directory (not results/dalmax{1,2}/, so it
# never collides with the real reference runs). `GPU` selects both
# `CUDA_VISIBLE_DEVICES` and which of the two per-GPU params JSONs to use
# (`files_config/benchmark/params_df_gpu_{0,1}.json`) — override with
# `make lab-check GPU=1`. See LAB_RUNBOOK.md step 1 for expected artifacts,
# duration, and how to verify the resulting checkpoint with loader.py/predict.py.
lab-check:
	CUDA_VISIBLE_DEVICES=$(GPU) poetry run python trainer.py \
		--params_json files_config/benchmark/params_df_gpu_$(GPU).json \
		--dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
		--n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
		--device cuda --dir_results results/lab_check/

# Regenerate DATA/daninhas_micro/ (10%-stratified replica of DATA/daninhas_full/,
# see scripts/make_micro_dataset.py) — a no-op that exits 0 if daninhas_full/
# is not present on this machine. Also runs implicitly as part of `smoke`/
# `smoke-ablations`; exposed standalone here for the lab-runbook's data-arrival
# verification step.
micro-dataset:
	poetry run python scripts/make_micro_dataset.py

# Run one GPU's half of the Phase 3 ablation batch (see
# scripts/ablations/run_ablation_gpu_{0,1}.sh's headers for the exact
# (study, config) split and its cost-balancing rationale, and
# .specs/experiments/ablation-study.md for the run tables). Each of these
# survives a single failing (study, config, seed) run and logs it to
# results/ablations/gpu{N}_failures.log instead of aborting the batch, and
# emails a completion notification via ExperimentNotifier/main.py at the end
# (its .env must already be configured on the lab machine — see LAB_RUNBOOK.md
# step 0). Meant to be launched inside its own tmux session/pane, one per GPU.
ablations-gpu0:
	bash scripts/ablations/run_ablation_gpu_0.sh

ablations-gpu1:
	bash scripts/ablations/run_ablation_gpu_1.sh

# Run both GPUs' ablation halves concurrently from a single shell, via `&` +
# `wait` rather than two separate `tmux` panes. Each script's combined
# stdout/stderr is redirected to its own results/ablations/gpu{0,1}.log
# specifically to avoid the two concurrent processes' output interleaving
# unreadably in one terminal (that unredirected interleaving is exactly what
# `&`-backgrounding both would otherwise produce) — tail each log separately
# to monitor progress (`tail -f results/ablations/gpu0.log`), and check
# results/ablations/gpu{0,1}_failures.log once both finish. Prefer two tmux
# panes (`make ablations-gpu0` / `make ablations-gpu1`) when you want to watch
# each GPU's live output directly instead of via a log file; this target is
# for a single unattended `nohup make ablations-all &`-style launch.
ablations-all:
	mkdir -p results/ablations
	( bash scripts/ablations/run_ablation_gpu_0.sh > results/ablations/gpu0.log 2>&1 & \
	  bash scripts/ablations/run_ablation_gpu_1.sh > results/ablations/gpu1.log 2>&1 & \
	  wait )

# Aggregate every discovered results/ablations/<study>/<config>/.../results.json
# into docs/results/ablation_tables/ (ablation_summary.csv + one
# ablation_6_{1,2,3}.md/.tex per sub-study) — see
# dalmax/reporting/ablation_report.py and docs/results/README.md. Run this
# after both ablations-gpu{0,1} batches finish (or against
# results/smoke_ablations/ with --root overridden, for a dry run).
ablation-report:
	poetry run python -m dalmax.reporting.ablation_report --root results/ablations --out docs/results/ablation_tables

# Re-run the reference RNHAL benchmark sweep (SSRAEKmeansHCSampling,
# QUERIES=(10 50 100) x SEEDS=(1 2 3), n_round=8) on one GPU — see
# scripts/benchmark/run_pipe_gpu_{0,1}.sh. Historical saved_model.pth files
# under results/dalmax{1,2}/ predate the checkpoint fix (KI-22 / ADR 0006)
# and contain no weights; re-running these targets regenerates real,
# loadable checkpoints for the same sweep.
benchmark-gpu0:
	bash scripts/benchmark/run_pipe_gpu_0.sh

benchmark-gpu1:
	bash scripts/benchmark/run_pipe_gpu_1.sh

# --- Colab targets (single GPU, session-limited) -----------------------------
# See COLAB_RUNBOOK.md for the full step-by-step notebook-cell guide these
# targets are called from.

# One-time-per-session Colab setup: verify Drive is mounted, build/reuse the
# DATA/daninhas_full.zip on Drive and unzip it onto the local runtime disk,
# and symlink results/ to Drive (see scripts/colab/setup_colab.sh's header
# for the full hybrid-layout rationale). Idempotent -- safe to re-run after a
# disconnect. Override the Drive path with `DRIVE_ROOT=... make colab-setup`.
colab-setup:
	bash scripts/colab/setup_colab.sh

# Run the full 11-config / 33-run Phase 3 ablation sweep sequentially on
# Colab's single GPU (both scripts/ablations/run_ablation_gpu_{0,1}.sh halves,
# both pinned to GPU 0), logging to results/ablations/colab.log. SKIP_EXISTING
# (default on in both underlying scripts) makes relaunching this after a
# disconnect safe -- see scripts/colab/run_ablations_colab.sh.
ablations-colab:
	bash scripts/colab/run_ablations_colab.sh

# The same single real-data GPU sanity check as `lab-check`, but into a
# dedicated results/colab_check/ directory (so it never collides with the
# real ablation/benchmark results trees symlinked to Drive) and pinned to
# Colab's single GPU 0. Run this once per session, right after `colab-setup`
# and before `ablations-colab`, per COLAB_RUNBOOK.md.
colab-check:
	CUDA_VISIBLE_DEVICES=0 poetry run python trainer.py \
		--params_json files_config/benchmark/params_df_gpu_0.json \
		--dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
		--n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
		--device cuda --dir_results results/colab_check/
