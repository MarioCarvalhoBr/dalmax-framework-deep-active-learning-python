.PHONY: setup lint format test test-all smoke clean micro-dataset \
	lab-setup lab-check colab-setup colab-check \
	campaign-list campaign-run campaign-verify campaign-report campaign-smoke campaign-manifest

# GPU index used by `lab-check` (sets CUDA_VISIBLE_DEVICES).
# Override on the command line, e.g. `make lab-check GPU=1`.
GPU ?= 0

# Install the Poetry-managed environment (./.venv, see poetry.toml).
setup:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

# Fast test suite: excludes anything requiring a GPU, the real DATA/ dataset,
# or a slow full-pipeline run. This is what CI runs on every push/PR.
test:
	poetry run pytest -q -m "not gpu and not dataset and not slow"

# Full test suite, including gpu/dataset/slow-marked tests (needs DATA/ present).
test-all:
	poetry run pytest -q

# `smoke`: a true end-to-end run of tools/trainer.py on a tiny CPU 2-class
# subset (see .specs/architecture/refactor-plan.md Phase 1 and
# .specs/quality/testing-strategy.md):
#   1. scripts/make_micro_dataset.py deterministically samples 25 train + 10
#      test images per class from DATA/daninhas_full/ (2 classes) into
#      DATA/daninhas_micro/, without ever writing into daninhas_full/ itself.
#      If daninhas_full/ isn't present it prints a message and exits 0.
#   2. tools/trainer.py runs against files_config/params_micro.json with
#      RandomSampling into results/smoke/ (gitignored) -- a few seconds on CPU.
#   3. The fast test suite runs on top.
# The golden-run regression (tests/test_golden_run.py) is `dataset`+`slow`-marked
# and runs via `make test-all`.
smoke:
	poetry run python scripts/make_micro_dataset.py
	@if [ -d DATA/daninhas_micro/train ]; then \
		poetry run python tools/trainer.py \
			--params_json files_config/params_micro.json \
			--dataset_name DANINHAS \
			--strategy_name RandomSampling \
			--n_init_labeled 10 --n_query 5 --n_round 1 --seed 1 \
			--dir_results results/smoke/; \
	else \
		echo "NOTE: DATA/daninhas_full not present, skipping the tools/trainer.py smoke run."; \
	fi
	poetry run pytest -q -m "not gpu and not dataset and not slow"

clean:
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

# Regenerate DATA/daninhas_micro/ (10%-stratified replica of DATA/daninhas_full/,
# see scripts/make_micro_dataset.py) -- a no-op that exits 0 if daninhas_full/
# is not present on this machine. Also runs implicitly as part of `smoke`.
micro-dataset:
	poetry run python scripts/make_micro_dataset.py

# --- Environment checks (any GPU machine; never run on a no-GPU dev box) ------
# See LAB_RUNBOOK.md / COLAB_RUNBOOK.md. GPU identity is never hardcoded: it is
# captured at runtime into each run's run_metadata.json (`environment.gpus`).

# One-time (or post-`git pull`) setup: sync the Poetry env to poetry.lock, then
# print whether CUDA is visible and how many devices torch sees. No training.
lab-setup:
	poetry install
	poetry run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# One short real-data GPU run (n_round=1) into results/lab_check/ before a full
# batch. `GPU` selects CUDA_VISIBLE_DEVICES. Uses the campaign's own paper-1 params
# (files_config/campaign/params_paper1.json, which carries the config_kmh
# hierarchy SSRAEKmeansHCSampling needs), so the check exercises the same config.
lab-check:
	CUDA_VISIBLE_DEVICES=$(GPU) poetry run python tools/trainer.py \
		--params_json files_config/campaign/params_paper1.json \
		--dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
		--n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
		--device cuda --dir_results results/lab_check/

# One-time-per-session Colab setup: verify Drive is mounted, build/reuse the
# DATA/daninhas_full.zip on Drive, unzip it onto the local runtime disk, and
# symlink results/ to Drive (scripts/colab/setup_colab.sh). Idempotent. Override
# the Drive path with `DRIVE_ROOT=... make colab-setup`.
colab-setup:
	bash scripts/colab/setup_colab.sh

# The same real-data GPU check as `lab-check`, into results/colab_check/ and
# pinned to GPU 0. Run once per session, right after `colab-setup`.
colab-check:
	CUDA_VISIBLE_DEVICES=0 poetry run python tools/trainer.py \
		--params_json files_config/campaign/params_paper1.json \
		--dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
		--n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
		--device cuda --dir_results results/colab_check/

# --- The campaign (ADR 0008/0009) ---------------------------------------------
# One manifest (files_config/campaign/manifest.json) drives the single clean
# execution of everything papers 1-3 need -- no redundant runs. See
# .specs/experiments/campaign.md, COLAB_RUNBOOK.md and LAB_RUNBOOK.md.
# Variables (all optional):
#   PART=all|paper1|upper_bound|rnhal|texhal (comma-separated allowed; default all)
#   MICRO=1     -> files_config/campaign/manifest_micro.json (CPU smoke, results/smoke_campaign/)
#   SEEDS=1     -> restrict to a subset of the seeds (smoke)
#   EXCLUDE=... -> comma-separated strategies to drop (default: none; with MICRO=1 the
#                  two adversarial baselines, far too slow on CPU -- override with
#                  `SMOKE_EXCLUDE=` to include them)
#   DEVICE=cuda|cpu|auto, default cuda (cpu with MICRO=1)
#   RUN_ARGS='--no-skip-existing' / '--dry-run' -> extra flags for `campaign-run`
PART ?= all
SMOKE_EXCLUDE ?= AdversarialBIM,AdversarialDeepFool
EXCLUDE ?= $(if $(MICRO),$(SMOKE_EXCLUDE))
CAMPAIGN_FLAGS = --part $(PART) $(if $(MICRO),--micro) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(EXCLUDE),--exclude-strategy $(EXCLUDE))

# Print the job table and the run counts per part/total.
campaign-list:
	poetry run python -m dalmax.campaign list $(CAMPAIGN_FLAGS)

# Execute the jobs sequentially (skip-existing: re-running resumes after a disconnect).
campaign-run:
	poetry run python -m dalmax.campaign run $(CAMPAIGN_FLAGS) $(if $(DEVICE),--device $(DEVICE)) $(RUN_ARGS)

# Per-job OK/INCOMPLETE/MISSING + the seed-consistency audit.
# Exit 0 all OK, 1 incomplete/missing job or FAIL audit, 2 only audit WARNs.
campaign-verify:
	poetry run python -m dalmax.campaign verify $(CAMPAIGN_FLAGS)

# Tables (md/tex/csv), seed audit and mean confusion matrices from results/campaign/
# into docs/results/campaign/ (MICRO=1: results/smoke_campaign/ -> results/smoke_campaign/report/).
campaign-report:
	poetry run python -m dalmax.reporting.campaign_report $(if $(MICRO),--micro --root results/smoke_campaign --out results/smoke_campaign/report,--root results/campaign --out docs/results/campaign)

# Micro campaign on CPU, seed 1 only, into results/smoke_campaign/ (every run
# group once; the two adversarial baselines are skipped by default: 58 of the 64
# jobs). Verify afterwards with `make campaign-verify MICRO=1 SEEDS=1`.
campaign-smoke:
	poetry run python -m dalmax.campaign run --part all --micro --seeds 1 $(if $(SMOKE_EXCLUDE),--exclude-strategy $(SMOKE_EXCLUDE))

# Regenerate the two committed manifests from scripts/campaign/build_manifest.py.
campaign-manifest:
	poetry run python scripts/campaign/build_manifest.py
