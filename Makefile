.PHONY: setup lint format test test-all smoke clean micro-dataset \
	lab-setup lab-check colab-setup colab-check \
	campaign-list campaign-run campaign-verify campaign-report campaign-smoke campaign-smoke-kmh campaign-manifest kmh-sweep-configs \
	campaign-archive-superseded

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
#   PART=all|paper1|paper1_rerun_a|paper1_rerun_b|upper_bound|rnhal|texhal|texhal_q17|hier_transfer|control|kmh_nq10|kmh_nq50|kmh_nq100
#               (comma-separated allowed; default all). paper1_rerun_a / paper1_rerun_b: the re-run of the seven
#               paper-1 baselines fixed on 2026-11-11 (ADR 0014), one part per Colab account (COLAB_RUNBOOK.md 6d).
#   MICRO=1     -> files_config/campaign/manifest_micro.json (CPU smoke, results/smoke_campaign/)
#   MANIFEST=path -> any other manifest (overrides MICRO), e.g. the KI-43 control (3 RandomSampling
#                  nq100 runs into results/campaign_control/, see .specs/experiments/campaign.md):
#                    make campaign-list MANIFEST=files_config/campaign/manifest_control_ki43.json
#                    make campaign-run  MANIFEST=files_config/campaign/manifest_control_ki43.json PART=control
#                  The KMH parameter sweep (paper 1, prepared, not executed; 135 runs, 45 groups, results/campaign_kmh/):
#                    make campaign-list   MANIFEST=files_config/campaign/manifest_kmh_sweep.json
#                    make campaign-run    MANIFEST=files_config/campaign/manifest_kmh_sweep.json PART=kmh_nq10  # kmh_nq50|kmh_nq100
#                    make campaign-verify MANIFEST=files_config/campaign/manifest_kmh_sweep.json
#                    make campaign-report MANIFEST=files_config/campaign/manifest_kmh_sweep.json REPORT_ROOT=results/campaign_kmh REPORT_OUT=docs/results/campaign_kmh
#                  (its configs: `make kmh-sweep-configs`; CPU smoke: `make campaign-smoke-kmh`)
#   REPORT_ROOT / REPORT_OUT -> `campaign-report` results root / output dir (defaults: results/campaign, docs/results/campaign)
#   SEEDS=1     -> restrict to a subset of the seeds (smoke)
#   EXCLUDE=... -> comma-separated strategies to drop (default: none; with MICRO=1 the
#                  two adversarial baselines, far too slow on CPU -- override with
#                  `SMOKE_EXCLUDE=` to include them)
#   DEVICE=cuda|cpu|auto, default cuda (cpu with MICRO=1)
#   RUN_ARGS='--no-skip-existing' / '--dry-run' -> extra flags for `campaign-run`
PART ?= all
SMOKE_EXCLUDE ?= AdversarialBIM,AdversarialDeepFool
EXCLUDE ?= $(if $(MICRO),$(SMOKE_EXCLUDE))
CAMPAIGN_FLAGS = --part $(PART) $(if $(MANIFEST),--manifest $(MANIFEST),$(if $(MICRO),--micro)) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(EXCLUDE),--exclude-strategy $(EXCLUDE))

# Print the job table and the run counts per part/total.
campaign-list:
	poetry run python -m dalmax.campaign list $(CAMPAIGN_FLAGS)

# Execute the jobs sequentially (skip-existing: re-running resumes after a disconnect).
campaign-run:
	poetry run python -m dalmax.campaign run $(CAMPAIGN_FLAGS) $(if $(DEVICE),--device $(DEVICE)) $(RUN_ARGS)

# Per-job OK/INCOMPLETE/MISSING/STALE (STALE: superseded re-run leaf, ADR 0014) + the seed-consistency audit.
# Exit 0 all OK, 1 incomplete/missing job or FAIL audit, 2 only audit WARNs.
campaign-verify:
	poetry run python -m dalmax.campaign verify $(CAMPAIGN_FLAGS)

# Tables (md/tex/csv), seed audit and mean confusion matrices from results/campaign/
# into docs/results/campaign/ (MICRO=1: results/smoke_campaign/ -> results/smoke_campaign/report/).
REPORT_ROOT ?= results/campaign
REPORT_OUT ?= docs/results/campaign
campaign-report:
	poetry run python -m dalmax.reporting.campaign_report $(if $(MANIFEST),--manifest $(MANIFEST) --root $(REPORT_ROOT) --out $(REPORT_OUT),$(if $(MICRO),--micro --root results/smoke_campaign --out results/smoke_campaign/report,--root $(REPORT_ROOT) --out $(REPORT_OUT)))

# Micro campaign on CPU, seed 1 only, into results/smoke_campaign/ (every run
# group once; the two adversarial baselines are skipped by default: 68 of the 74
# jobs). Verify afterwards with `make campaign-verify MICRO=1 SEEDS=1`.
campaign-smoke:
	poetry run python -m dalmax.campaign run --part all --micro --seeds 1 $(if $(SMOKE_EXCLUDE),--exclude-strategy $(SMOKE_EXCLUDE))

# Micro KMH sweep on CPU, seed 1 only, into results/smoke_campaign_kmh/ (45 jobs).
# Verify afterwards with
# `make campaign-verify MANIFEST=files_config/campaign/manifest_kmh_sweep_micro.json SEEDS=1`.
campaign-smoke-kmh:
	poetry run python -m dalmax.campaign run --part all --manifest files_config/campaign/manifest_kmh_sweep_micro.json --seeds 1 --device cpu

# Re-run of the seven fixed paper-1 baselines (ADR 0014): MOVE the superseded leaves of the parts
# paper1_rerun_a / paper1_rerun_b (run_metadata.json started_at before the fix cutoff) from the live
# tree to results/old_results_broken/<same path below results/>, never deleting or overwriting
# anything (scripts/campaign/archive_superseded_runs.py; log: results/old_results_broken/ARCHIVE_LOG.tsv).
#   DRY=1    -> report what would move, move nothing
#   CHECK=1  -> move nothing; exit 1 if a live leaf of those parts predates the cutoff (run it before
#               campaign-run and after merging downloaded results)
#   BEFORE=<ISO-8601 UTC> -> override the cutoff; MANIFEST=path -> another manifest (e.g. the micro one)
campaign-archive-superseded:
	poetry run python scripts/campaign/archive_superseded_runs.py $(if $(MANIFEST),--manifest $(MANIFEST),$(if $(MICRO),--manifest files_config/campaign/manifest_micro.json)) $(if $(DRY),--dry-run) $(if $(CHECK),--check) $(if $(BEFORE),--before $(BEFORE))

# Regenerate the KMH sweep params files (files_config/ablations/kmh_sweep/, full + micro).
kmh-sweep-configs:
	poetry run python scripts/campaign/build_kmh_sweep_configs.py

# Regenerate the committed manifests (and the sweep configs they reference) from scripts/campaign/.
campaign-manifest: kmh-sweep-configs
	poetry run python scripts/campaign/build_manifest.py
