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
   `core/query_strategies/__init__.py` and `demo.py`'s `--strategy_name`
   `choices=[...]` list.
2. Create the new module `core/query_strategies/<snake_case_name>.py`, following
   the pattern in `core/query_strategies/ssrae_kmeans_sampling.py` (subclass
   `Strategy` from `core/query_strategies/strategy.py`, implement `query(self, n)`).
   Do **not** copy its known bugs (hardcoded `random_state=3`, hardcoded
   `self.params['DANINHAS']`) — see `.claude/rules/reproducibility.md` and
   `.claude/rules/code-quality.md`.
3. Register it in `core/query_strategies/__init__.py` (add the import/export).
4. Register it in `utils/orchestrator.get_strategy` (add the branch — or, if the
   registry has already been migrated to a dict per the refactor plan, add the
   dict entry instead).
5. Add `$ARGUMENTS` to `demo.py`'s `--strategy_name` `choices=[...]` list.
6. Add a smoke-level test entry: extend `tests/test_registry.py` so
   `get_strategy("$ARGUMENTS")` resolves without error.
7. Update `.specs/use-cases/add-new-strategy.md` (checklist should already
   describe this flow — update it if this run revealed the checklist was
   incomplete) and `.specs/architecture/current-state.md` if the module list
   there is out of date.
8. Run `poetry run pytest -k registry` and `poetry run ruff check
   core/query_strategies/<snake_case_name>.py`, report the results.

Report back: files created/modified, test results, and any `.specs/` files
updated.
