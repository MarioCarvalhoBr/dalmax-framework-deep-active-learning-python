.PHONY: setup lint format test test-all smoke smoke-ablations clean

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
