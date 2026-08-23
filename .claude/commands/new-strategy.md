---
description: Guided flow to add a new query strategy to DalMax, wiring it into every place it needs to be registered.
argument-hint: <ClassName>
---

Add the query strategy `$ARGUMENTS` end-to-end. Follow the checklist in
`.claude/skills/adding-query-strategy/SKILL.md` (load it first) and delegate the
actual coding to the `implementer` agent per `.claude/rules/model-delegation.md`
unless the change is trivial enough to do directly.

Steps:

1. Confirm the class name `$ARGUMENTS` is not already used — check
   `dalmax/query_strategies/__init__.py` and `dalmax/cli.py`'s `--strategy_name`
   `choices=[...]` list.
2. Create the new module `dalmax/query_strategies/<snake_case_name>.py`, following
   the pattern in one of the 12 baseline strategy files (e.g.
   `dalmax/query_strategies/random_sampling.py`; subclass `Strategy` from
   `dalmax/query_strategies/base.py`, implement `query(self, n)`). If the new
   strategy is representation-based (embedding + selection), prefer adding it as
   a preset/config on `dalmax/query_strategies/representation.py::RepresentationStrategy`
   instead of a new fixed subclass — see that module's docstring.
   Do **not** copy any legacy bugs (hardcoded `random_state=3`, hardcoded
   dataset-name keys) — see `.claude/rules/reproducibility.md` and
   `.claude/rules/code-quality.md`.
3. Register it in `dalmax/query_strategies/__init__.py` (add the import/export).
4. Register it in `dalmax/query_strategies/registry.py::STRATEGY_REGISTRY`
   (add the dict entry — this is a plain dict lookup, not an if/elif chain;
   see that module's docstring for the difference between `LEGACY_STRATEGY_REGISTRY`
   entries and `REPRESENTATION_PRESETS`).
5. Add `$ARGUMENTS` to `dalmax/cli.py`'s `--strategy_name` `choices=[...]` list.
6. Add a smoke-level test entry: extend `tests/test_strategy_registry.py` so
   `build_strategy("$ARGUMENTS", ...)` resolves without error.
7. Update `.specs/use-cases/add-new-strategy.md` (checklist should already
   describe this flow — update it if this run revealed the checklist was
   incomplete) and `.specs/architecture/current-state.md` if the module list
   there is out of date.
8. Run `poetry run pytest -k registry` and `poetry run ruff check
   dalmax/query_strategies/<snake_case_name>.py`, report the results.

Report back: files created/modified, test results, and any `.specs/` files
updated.
