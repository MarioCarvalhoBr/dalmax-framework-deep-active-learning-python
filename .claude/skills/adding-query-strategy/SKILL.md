---
name: adding-query-strategy
description: Step-by-step checklist of every file that must change to add a new active-learning query strategy to DalMax, derived from how SSRAEKmeansSampling is wired today.
---

# Adding a query strategy

DalMax wires a query strategy into the system through a fixed set of files —
missing any one of them leaves the strategy partially registered (importable
but unreachable from the CLI, or reachable but untested). This skill lists
every touch point, using `SSRAEKmeansSampling` as the reference example of how
a strategy is wired today.

## Files to touch, in order

1. **`core/query_strategies/<snake_case_name>.py`** — the new strategy module.
   Pattern (from `core/query_strategies/ssrae_kmeans_sampling.py`):
   ```python
   from .strategy import Strategy

   class YourStrategyName(Strategy):
       def __init__(self, dataset, net, logger):
           super(YourStrategyName, self).__init__(dataset, net, logger)

       def query(self, n):
           ...  # return an array of selected sample ids
   ```
   Do not copy known bugs from the reference implementation:
   - `SSRAEKmeansSampling.query` hardcodes `KMeans(random_state=3, ...)` —
     ignores the experiment seed. Thread the real seed through instead.
   - `SSLStrategy.query` (`core/query_strategies/ssl_ssrae_sampling.py`) reads
     `self.params['DANINHAS']['config_kmh']` — hardcodes the dataset name.
     Read the dataset name from `self.dataset` or an injected config instead.

2. **`core/query_strategies/__init__.py`** — add the import/export line:
   ```python
   from .<snake_case_name> import YourStrategyName
   ```

3. **`utils/orchestrator.py`** — add `YourStrategyName` to the `get_strategy`
   function's branches (currently an if/elif chain ending in
   `raise NotImplementedError`):
   ```python
   elif name == "YourStrategyName":
       return YourStrategyName
   ```
   (If the refactor has migrated this to a registry dict by the time you read
   this, add a dict entry instead — check the current file before assuming
   the if/elif shape.)

4. **`demo.py`** — add `"YourStrategyName"` to the `--strategy_name`
   `argparse` `choices=[...]` list so the CLI accepts it.

5. **`tests/test_registry.py`** — add (or confirm the existing parametrized
   test covers) a case asserting
   `utils.orchestrator.get_strategy("YourStrategyName")` resolves without
   raising. This test iterates `demo.py`'s `choices` list, so step 4 must be
   done first for it to pick up the new name automatically — verify it does.

6. **`.specs/use-cases/add-new-strategy.md`** — update the checklist/example
   list of strategies if it enumerates them, and note anything this specific
   addition revealed that the generic checklist was missing.

## Verification

```bash
poetry run python -c "from utils.orchestrator import get_strategy; get_strategy('YourStrategyName')"
poetry run pytest -k registry
poetry run ruff check core/query_strategies/<snake_case_name>.py
```

## Do not skip spec sync

Per `.claude/rules/spec-sync.md`, steps 5 and 6 are not optional cleanup — they
are part of the same task as steps 1-4. A strategy present in
`core/query_strategies/__init__.py` but absent from `demo.py` choices (or vice
versa) is a bug, and `.claude/agents/code-reviewer.md` checks for exactly this
kind of drift.

See also `.claude/commands/new-strategy.md` for the command form of this
checklist.
