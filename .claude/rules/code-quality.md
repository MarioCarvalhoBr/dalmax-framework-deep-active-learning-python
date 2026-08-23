# Code quality — the soul of the project

The target is a **new, elegant, scalable, fast, and organized codebase**. DalMax's
current code (see `.specs/quality/known-issues.md`) violates most of these
principles today; every change should move the codebase closer to this target,
not further from it.

## Principles

- **Single responsibility.** `demo.py` today mixes CLI parsing, training loop,
  plotting, and persistence in ~290 lines — new code must not add to that pile.
  Split concerns into `runner` / `reporting` / thin CLI as the refactor plan
  (`.specs/architecture/refactor-plan.md`) describes.
- **No duplicated strategy boilerplate.** Every `core/query_strategies/*.py` file
  currently repeats the same `__init__(self, dataset, net, logger)` /
  `super().__init__(...)` pattern. New strategies should factor shared logic into
  `Strategy` (`core/query_strategies/strategy.py`) rather than copy-pasting.
- **Registry pattern over if/elif chains.** `utils/orchestrator.get_strategy` and
  `get_dataset`/`get_network_deep_learning` are currently `if name == "X": return X`
  chains with 16+ branches. Do not add a 17th `elif` — new code in this area should
  move toward a dict-based or decorator-based registry (see refactor Phase 2).
- **Explicit dependency injection — no `setattr` magic.** `demo.py:79` does
  `setattr(strategy, "params", params)` after construction instead of passing
  `params` into `Strategy.__init__`. Do not repeat this pattern in new code; pass
  dependencies through constructors.
- **Type hints on all new/edited code.** None of the current strategy or utils
  code has type hints. Any new function or edited function signature must be typed.
- **No hardcoded dataset names, paths, or seeds inside logic.** Example violation
  to never repeat: `core/query_strategies/ssl_ssrae_sampling.py` does
  `self.params['DANINHAS']['config_kmh']` — this breaks for `CIFAR10` or any future
  dataset. New/edited code must read the active dataset name from `self.dataset`
  or an injected config, never a literal string.
- **Caches must have explicit keys and invalidation.** `utils/data.py` writes
  `results/features_dict_ssrae.pkl` and `results/features_dict_vctex.pkl` with no
  key for `(dataset, Q, embedding_variant)`. A cache file from a `daninhas_full`
  run with `Q=13` will be silently reused for a different `Q` or dataset. New
  caching code must key the cache path/filename on those parameters explicitly.
- **English-only code and comments.** Existing code mixes Portuguese and English
  (e.g. print statements in `utils/data.py`, `core/query_strategies/*`). New or
  edited lines must be English.
- **Fail-fast errors over silent fallbacks.** `utils/orchestrator.get_strategy`
  raises `NotImplementedError` for unknown names — keep that pattern; do not add
  a default/fallback branch that silently picks a strategy.
- **Every module importable without side effects.** No top-level training,
  plotting, or file I/O at import time — see `tests/test_imports.py`, which
  imports every module under `core/` and `utils/` and must stay green.

## Enforcement

- `make lint` (`ruff check`) must pass (or violations must be pre-existing and
  tracked in `.specs/quality/known-issues.md`, not newly introduced).
- The `code-reviewer` agent (`.claude/agents/code-reviewer.md`) checks new diffs
  against this rule set before merge.
