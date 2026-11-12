.PHONY: setup lint format test test-all smoke clean micro-dataset \
	lab-setup lab-check colab-setup colab-check \
	campaign-list campaign-run campaign-verify campaign-report campaign-report-all campaign-smoke campaign-smoke-kmh \
	campaign-list-gpu-part campaign-run-gpu-part campaign-verify-gpu-part campaign-verify-all campaign-smoke-gpu-part \
	campaign-manifest kmh-sweep-configs gpu-time-estimates dataset-check

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
# (files_config/campaign/params_paper1.json, DATA/daninhas_full_fixed/, which carries
# the config_kmh hierarchy SSRAEKmeansHCSampling needs), so the check exercises the
# same config and dataset as the campaign.
lab-check:
	CUDA_VISIBLE_DEVICES=$(GPU) poetry run python tools/trainer.py \
		--params_json files_config/campaign/params_paper1.json \
		--dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
		--n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
		--device cuda --dir_results results/lab_check/

# One-time-per-session Colab setup: verify Drive is mounted, build/reuse the
# DATA/daninhas_full_fixed.zip on Drive, unzip it onto the local runtime disk,
# check it (file count + scripts/datasets/build_daninhas_full_fixed.py --check: no
# repeated MD5), and symlink results/ to Drive (scripts/colab/setup_colab.sh).
# Idempotent. DRIVE_ROOT (the Drive folder holding DATA/) is REQUIRED:
# `DRIVE_ROOT=/content/drive/MyDrive/<your folder> make colab-setup` (the notebook exports it).
colab-setup:
	bash scripts/colab/setup_colab.sh

# Verify the canonical dataset (KI-46, ADR 0016): no repeated MD5 inside or across
# DATA/daninhas_full_fixed/{train,test}. Read-only.
dataset-check:
	poetry run python scripts/datasets/build_daninhas_full_fixed.py --check

# The same real-data GPU check as `lab-check`, into results/colab_check/ and
# pinned to GPU 0. Run once per session, right after `colab-setup`.
colab-check:
	CUDA_VISIBLE_DEVICES=0 poetry run python tools/trainer.py \
		--params_json files_config/campaign/params_paper1.json \
		--dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
		--n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
		--device cuda --dir_results results/colab_check/

# --- The campaign (ADR 0008/0009/0016) ----------------------------------------
# One manifest (files_config/campaign/manifest.json) drives the clean execution of
# everything papers 1-3 need on DATA/daninhas_full_fixed/ -- no redundant runs --
# into results/campaign_fixed/; the KMH sweep (manifest_kmh_sweep.json) goes to
# results/campaign_kmh_fixed/. See .specs/experiments/campaign.md, COLAB_RUNBOOK.md
# and LAB_RUNBOOK.md.
# Variables (all optional):
#   PART=all|paper1|upper_bound|rnhal|texhal|texhal_q17|hier_transfer|control|kmh_nq10|kmh_nq50|kmh_nq100
#               (comma-separated allowed; default all), or a GPU part: part_1_gpu|part_2_gpu (ADR 0016).
#   MICRO=1     -> files_config/campaign/manifest_micro.json (CPU smoke, results/smoke_campaign_fixed/)
#   MANIFEST=path -> any other manifest (overrides MICRO), e.g. the KMH sweep or the KI-43 control:
#                    make campaign-list   MANIFEST=files_config/campaign/manifest_kmh_sweep.json
#                    make campaign-run    MANIFEST=files_config/campaign/manifest_kmh_sweep.json PART=kmh_nq10
#                    make campaign-list   MANIFEST=files_config/campaign/manifest_control_ki43.json
#                  (its configs: `make kmh-sweep-configs`; CPU smoke: `make campaign-smoke-kmh`)
#   REPORT_ROOT / REPORT_OUT -> `campaign-report` results root / output dir
#                  (defaults: results/campaign_fixed, docs/results/campaign_fixed)
#   SEEDS=1     -> restrict to a subset of the seeds (smoke)
#   EXCLUDE=... -> comma-separated strategies to drop (default: none; with MICRO=1 the
#                  baselines far too slow on CPU, SMOKE_EXCLUDE -- override with
#                  `SMOKE_EXCLUDE=` to include them)
#   DEVICE=cuda|cpu|auto, default cuda (cpu with MICRO=1)
#   RUN_ARGS='--no-skip-existing' / '--dry-run' -> extra flags for `campaign-run`
PART ?= all
SMOKE_EXCLUDE ?= AdversarialBIM,AdversarialDeepFool
EXCLUDE ?= $(if $(MICRO),$(SMOKE_EXCLUDE))
CAMPAIGN_FLAGS = --part $(PART) $(if $(MANIFEST),--manifest $(MANIFEST),$(if $(MICRO),--micro)) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(EXCLUDE),--exclude-strategy $(EXCLUDE))

# Print the job table, the run counts and the estimated GPU hours per part / GPU part / total.
campaign-list:
	poetry run python -m dalmax.campaign list $(CAMPAIGN_FLAGS)

# Execute the jobs sequentially (skip-existing: re-running resumes after a disconnect).
campaign-run:
	poetry run python -m dalmax.campaign run $(CAMPAIGN_FLAGS) $(if $(DEVICE),--device $(DEVICE)) $(RUN_ARGS)

# Per-job OK/INCOMPLETE/MISSING + the seed-consistency audit.
# Exit 0 all OK, 1 incomplete/missing job or FAIL audit, 2 only audit WARNs.
campaign-verify:
	poetry run python -m dalmax.campaign verify $(CAMPAIGN_FLAGS)

# --- One GPU part across BOTH manifests (ADR 0016: one Colab account per GPU part) ---
#   GPU_PART=part_1_gpu|part_2_gpu (required). The main manifest and the KMH sweep are merged into one
#   plan ordered by execution stage: cheap table-critical groups, ablations, the KMH sweep, then the long
#   baselines (AdversarialBIM, VAAL/WAAL, AdversarialDeepFool) last. Resume = re-run the same command.
#   MICRO=1 -> the two micro manifests (CPU smoke). SEEDS / EXCLUDE / DEVICE / RUN_ARGS as above.
GPU_MANIFESTS = $(if $(MICRO),files_config/campaign/manifest_micro.json$(,)files_config/campaign/manifest_kmh_sweep_micro.json,files_config/campaign/manifest.json$(,)files_config/campaign/manifest_kmh_sweep.json)
, := ,
GPU_PART_FLAGS = --part $(GPU_PART) --manifest $(GPU_MANIFESTS) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(EXCLUDE),--exclude-strategy $(EXCLUDE))
define require_gpu_part
	@case "$(GPU_PART)" in part_1_gpu|part_2_gpu) ;; *) echo "ERROR: set GPU_PART=part_1_gpu or GPU_PART=part_2_gpu (got '$(GPU_PART)')"; exit 2;; esac
endef

campaign-list-gpu-part:
	$(require_gpu_part)
	poetry run python -m dalmax.campaign list $(GPU_PART_FLAGS)

campaign-run-gpu-part:
	$(require_gpu_part)
	poetry run python -m dalmax.campaign run $(GPU_PART_FLAGS) $(if $(DEVICE),--device $(DEVICE),$(if $(MICRO),--device cpu)) $(RUN_ARGS)

campaign-verify-gpu-part:
	$(require_gpu_part)
	poetry run python -m dalmax.campaign verify $(GPU_PART_FLAGS)

# Both manifests, every group (after merging the two GPU parts locally): expect OK=459 and the seed audit PASS.
campaign-verify-all:
	poetry run python -m dalmax.campaign verify --part all --manifest $(GPU_MANIFESTS) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(EXCLUDE),--exclude-strategy $(EXCLUDE))

# Tables (md/tex/csv), seed audit and mean confusion matrices from results/campaign_fixed/
# into docs/results/campaign_fixed/ (MICRO=1: results/smoke_campaign_fixed/ -> results/smoke_campaign_fixed/report/).
REPORT_ROOT ?= results/campaign_fixed
REPORT_OUT ?= docs/results/campaign_fixed
campaign-report:
	poetry run python -m dalmax.reporting.campaign_report $(if $(MANIFEST),--manifest $(MANIFEST) --root $(REPORT_ROOT) --out $(REPORT_OUT),$(if $(MICRO),--micro --root results/smoke_campaign_fixed --out results/smoke_campaign_fixed/report,--root $(REPORT_ROOT) --out $(REPORT_OUT)))

# Both reports of the clean campaign: main -> docs/results/campaign_fixed/, KMH sweep -> docs/results/campaign_kmh_fixed/.
campaign-report-all:
	poetry run python -m dalmax.reporting.campaign_report --root results/campaign_fixed --out docs/results/campaign_fixed
	poetry run python -m dalmax.reporting.campaign_report --manifest files_config/campaign/manifest_kmh_sweep.json \
		--root results/campaign_kmh_fixed --out docs/results/campaign_kmh_fixed

# Micro campaign on CPU, seed 1 only, into results/smoke_campaign_fixed/ (every run
# group once; the baselines in SMOKE_EXCLUDE are skipped by default).
# Verify afterwards with `make campaign-verify MICRO=1 SEEDS=1`.
campaign-smoke:
	poetry run python -m dalmax.campaign run --part all --micro --seeds 1 $(if $(SMOKE_EXCLUDE),--exclude-strategy $(SMOKE_EXCLUDE))

# Micro KMH sweep on CPU, seed 1 only, into results/smoke_campaign_kmh_fixed/ (55 jobs).
# Verify afterwards with
# `make campaign-verify MANIFEST=files_config/campaign/manifest_kmh_sweep_micro.json SEEDS=1`.
campaign-smoke-kmh:
	poetry run python -m dalmax.campaign run --part all --manifest files_config/campaign/manifest_kmh_sweep_micro.json --seeds 1 --device cpu

# Micro run of one GPU part across both micro manifests (the same plan as campaign-run-gpu-part), seed 1.
campaign-smoke-gpu-part:
	$(MAKE) campaign-run-gpu-part MICRO=1 SEEDS=1 GPU_PART=$(GPU_PART)

# Regenerate the KMH sweep params files (files_config/ablations/kmh_sweep/, full + micro).
kmh-sweep-configs:
	poetry run python scripts/campaign/build_kmh_sweep_configs.py

# Regenerate the committed manifests (and the sweep configs they reference) from scripts/campaign/.
campaign-manifest: kmh-sweep-configs
	poetry run python scripts/campaign/build_manifest.py

# Rebuild files_config/campaign/gpu_time_estimates.json from the first execution's trees
# (results/campaign/, results/campaign_kmh/; read only). A no-op where those trees are absent.
gpu-time-estimates:
	poetry run python scripts/campaign/build_gpu_time_estimates.py
