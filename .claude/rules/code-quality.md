# Code quality — the soul of the project

The target is a **new, elegant, scalable, fast, and organized codebase**. DalMax's
code (see `.specs/quality/known-issues.md`) still violates some of these
principles; every change should move the codebase closer to this target,
not further from it.

## Principles

- **Single responsibility.** `demo.py` historically mixed CLI parsing, training loop,
  plotting, and persistence in ~290 lines. That was split into `dalmax/experiment/runner.py`
  (round loop) / `dalmax/experiment/reporter.py` (plotting/JSON/CSV) / a thin `dalmax/cli.py`
  (argparse only) — `demo.py` is now a 12-line shim (`from dalmax.cli import main`). New code
  must not add mixed-concern logic back into `dalmax/cli.py`.
- **No duplicated strategy boilerplate.** Every `dalmax/query_strategies/*.py` file
  repeats the same `__init__(self, dataset, net, logger)` /
  `super().__init__(...)` pattern. New strategies should factor shared logic into
  `Strategy` (`dalmax/query_strategies/base.py`) rather than copy-pasting.
- **Registry pattern over if/elif chains.** The old `utils/orchestrator.get_strategy`/
  `get_dataset`/`get_network_deep_learning` if/elif chains (16+ branches each) were deleted
  in Phase 4 and replaced by plain dict registries: `dalmax/data/registry.py::DATASET_REGISTRY`,
  `dalmax/models/registry.py::MODEL_REGISTRY`, `dalmax/query_strategies/registry.py::STRATEGY_REGISTRY`,
  `dalmax/embeddings/registry.py::EMBEDDING_REGISTRY`, `dalmax/selection/registry.py::SELECTION_REGISTRY`.
  Do not add a new `if/elif` branch to any of them — add a registry entry.
- **Explicit dependency injection — no `setattr` magic.** The old `demo.py` did
  `setattr(strategy, "params", params)` after construction instead of passing
  `params` into `Strategy.__init__` — that line no longer exists (`demo.py` is a shim).
  `dalmax/query_strategies/registry.py::build_strategy` constructs every strategy with
  everything it needs at `__init__` time. Do not repeat the `setattr` pattern in new code;
  pass dependencies through constructors.
- **Type hints on all new/edited code.** Most of the moved legacy strategy/model
  code still has no type hints. Any new function or edited function signature must be typed.
- **No hardcoded dataset names, paths, or seeds inside logic.** Example violation
  to never repeat: the now-deleted `ssl_ssrae_sampling.py` did
  `self.params['DANINHAS']['config_kmh']` — this broke for `CIFAR10` or any future
  dataset. Its replacement, `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection`,
  takes `hierarchy` via constructor injection instead. New/edited code must read the active
  dataset name from `self.dataset` or an injected `ExperimentConfig`, never a literal string.
- **Caches must have explicit keys and invalidation.** The old `utils/data.py` (deleted
  in Phase 4) wrote `results/features_dict_ssrae.pkl` and `results/features_dict_vctex.pkl`
  with no key for `(dataset, Q, embedding_variant)`. `dalmax/embeddings/cache.py::EmbeddingCache`
  replaces it, keyed on `(dataset, extractor, Q, variant, split, pool_hash)`. New caching code
  must key the cache path/filename on its parameters explicitly, the same way.
- **English-only code and comments.** Some moved code still mixes Portuguese and English
  (e.g. `dalmax/data/datasets.py`, was `utils/data.py`, still has Portuguese comments —
  see `.specs/quality/known-issues.md` KI-12). New or edited lines must be English.
- **Fail-fast errors over silent fallbacks.** `dalmax/query_strategies/registry.py`
  (replacing the deleted `utils/orchestrator.get_strategy`) raises a clear `KeyError`/
  `dalmax.config.schema.ConfigError` naming the invalid name — keep that pattern; do not add
  a default/fallback branch that silently picks a strategy.
- **Every module importable without side effects.** No top-level training,
  plotting, or file I/O at import time — see `tests/test_imports.py`, which
  imports every module under `dalmax/` and must stay green.

## Enforcement

- `make lint` (`ruff check`) must pass (or violations must be pre-existing and
  tracked in `.specs/quality/known-issues.md`, not newly introduced).
- The `code-reviewer` agent (`.claude/agents/code-reviewer.md`) checks new diffs
  against this rule set before merge.
