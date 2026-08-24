---
description: Run make smoke (fast CPU pipeline on a tiny subset) and interpret failures.
---

Run the project's smoke test and interpret the results.

Steps:

1. Run `make smoke`. This is documented in the Makefile as a tiny CPU
   end-to-end run of `demo.py` on a small subset of `daninhas_full` (or a
   generated micro-dataset) — see `.specs/quality/testing-strategy.md` and
   `.specs/quality/known-issues.md` for whether it is a real run or a
   documented stub in the current state of the repo.
2. If `make smoke` is not yet a real end-to-end run (check the Makefile
   target body), fall back to:
   - `poetry run pytest -m "not gpu and not dataset and not slow"` (the fast unit tests:
     `tests/test_imports.py`, `tests/test_strategy_registry.py`,
     `tests/test_ssrae_embedding_layout.py`), and
   - `poetry run python -c "import demo"` plus `poetry run python -c "from
     dalmax.query_strategies.registry import build_strategy; from
     dalmax.data.registry import get_dataset; from dalmax.models.registry import
     get_network"` as a minimal import sanity check.
3. On failure, read the traceback and classify it:
   - Import error → likely a missing dependency (check whether `pyproject.toml`
     declares it and re-run `poetry install`; `pandas` was historically missing
     from the project's dependency list — see `.specs/quality/known-issues.md`).
   - Registry error (`KeyError`/`dalmax.config.schema.ConfigError` from
     `dalmax/query_strategies/registry.py` or another `dalmax/*/registry.py`) →
     likely a strategy/dataset name mismatch between `dalmax/cli.py` choices and the
     registry — see `.claude/skills/adding-query-strategy/SKILL.md`.
   - Anything touching `DATA/` or `results/` unexpectedly → stop and flag,
     per `.claude/rules/data-safety.md`; do not let the smoke test write into
     those directories.
4. Report pass/fail per check, and for failures, the root cause classification
   and suggested next action (fix directly if trivial, else delegate to
   `implementer` or `mechanic` per `.claude/rules/model-delegation.md`).

`$ARGUMENTS`, if given, is passed through as extra `pytest` args (e.g. `-k
registry` to narrow the run).
