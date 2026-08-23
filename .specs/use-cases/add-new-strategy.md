# Use case: add a new query strategy

**Status: rewritten 2026-08-23 for Phase 2.** The if/elif registry pattern this file originally
described (`utils/orchestrator.py::get_strategy`) is now dead code, unreachable from
`demo.py`/`dalmax.cli` (`.specs/architecture/current-state.md` §0) — `demo.py` routes through
`dalmax/query_strategies/registry.py::build_strategy` instead. There are now **two** distinct ways
to add a strategy, depending on what it actually needs; pick the right one before writing code.

## Which path do you need?

- **Your strategy is "an embedding source + a batch-selection rule"** (i.e. it fits the RNHAL
  pattern: compute/cache embeddings for the unlabeled pool, then cluster/pick a batch from them) —
  use **Path A** below. This covers everything the ablation study needs (`.specs/experiments/ablation-study.md`)
  and almost certainly covers any future representation-based strategy: you add a
  `dalmax.embeddings.EmbeddingProvider` and/or a `dalmax.selection.SelectionStrategy`, not a new
  `Strategy` subclass — `dalmax.query_strategies.representation.RepresentationStrategy` already
  composes the two (ADR 0005).
- **Your strategy is genuinely something else** (a new uncertainty measure, a new adversarial
  method, anything that isn't "embed the pool, then pick a batch") — use **Path B**: a new legacy-style
  `Strategy` subclass, registered in `dalmax/query_strategies/registry.py::LEGACY_STRATEGY_REGISTRY`.

## Path A — new embedding provider or selection strategy (the common case)

### A.1 New embedding provider

1. Implement `dalmax/embeddings/<name>_provider.py`, subclassing
   `dalmax.embeddings.base.EmbeddingProvider`: constructor `__init__(self, q, device: str = "cpu")`,
   and `embed(self, images: np.ndarray) -> np.ndarray` taking `(N, H, W, 3)` `uint8` and returning
   `(N, D)` `float32`. Follow `dalmax/embeddings/resnet_imagenet_provider.py` as the reference
   example (no model weights loaded at import time — only inside `__init__`, per
   `.claude/rules/code-quality.md`).
2. Register it in `dalmax/embeddings/registry.py::EMBEDDING_REGISTRY` (a plain dict entry —
   `"<name>": <ProviderClass>`).
3. If your provider needs a per-extractor default `q` (like SSRAE's `13`/VCTex's `(5, 17)`), add it
   to `dalmax/config/loader.py::_EXTRACTOR_DEFAULT_Q` and to
   `dalmax/query_strategies/registry.py::REPRESENTATION_PRESET_Q` **only if** you are also adding a
   legacy-style preset name for it (most new providers won't need a preset — they are reachable via
   the generic `RepresentationStrategy` CLI name and the params JSON's `"embedding": {"extractor":
   "<name>", ...}` block without any preset).
4. Add `"<name>"` to `dalmax/config/schema.py::VALID_EXTRACTORS`.
5. Add a test mirroring `tests/test_resnet_provider.py`/`tests/test_embeddings_ssrae.py`: construct
   the provider, call `embed()` on a small synthetic image batch, assert the output shape/dtype.

### A.2 New selection strategy

1. Implement `dalmax/selection/<name>.py`, subclassing `dalmax.selection.base.SelectionStrategy`:
   `select(self, embeddings: np.ndarray, ids: np.ndarray, budget: int, rng: np.random.Generator) ->
   np.ndarray`, returning a subset of `ids` of length `min(budget, len(ids))`, no duplicates. **Never**
   hardcode a `random_state`/seed literal — derive every stochastic decision from the `rng` argument
   (`.claude/rules/reproducibility.md`; this is exactly the `KMeans(random_state=3)` bug, KI-5, that
   Phase 2 fixed). See `dalmax/selection/flat_kmeans_proportional.py` for a worked example of a
   from-scratch new strategy (not a variant of an existing one).
2. Register it in `dalmax/selection/registry.py::SELECTION_REGISTRY`.
3. Add `"<name>"` to `dalmax/config/schema.py::VALID_SELECTION_METHODS`.
4. If your method needs no extra config (like `flat_closest`/`flat_proportional`), no schema change
   is needed beyond step 3. If it needs a hierarchy-like config block, follow
   `dalmax/config/schema.py::HierarchyConfig`/`SelectionConfig` as the pattern (duck-typed, validated
   in `__post_init__`, never a hardcoded dataset-name key lookup — the KI-4 bug Phase 2 fixed).
5. Add a test mirroring `tests/test_selection_flat.py`/`tests/test_selection_proportional.py`:
   synthetic embeddings (e.g. `sklearn.datasets.make_blobs`), assert the returned size/subset
   property, and — critically — assert that two `rng`s seeded from **different** seeds produce
   **different** selections (this is the automated version of the "does it actually use the seed"
   check that would have caught KI-5 originally).

### A.3 Wire it into the CLI (only needed if you want a **preset** shortcut name)

If your new `(extractor, selection)` combination should be reachable under its own
`--strategy_name` (like the four legacy `*Kmeans*Sampling` presets), add an entry to
`dalmax/query_strategies/registry.py::REPRESENTATION_PRESETS` and to `dalmax/cli.py`'s
`--strategy_name` `choices=[...]` list. **This is usually unnecessary** — the generic
`RepresentationStrategy` CLI name already reaches any `(extractor, selection)` combination via the
params JSON's `"embedding"`/`"selection"` blocks (see `.specs/experiments/ablation-study.md` for the
exact syntax); only add a preset if the combination needs a fixed-forever, seed/params-JSON-proof
identity (the way the four legacy names must keep reproducing their exact historical behavior).

## Path B — new legacy-style `Strategy` subclass (uncertainty/adversarial/etc.)

Unchanged from before Phase 2, except step 3:

1. Create `core/query_strategies/<new_strategy_file>.py`, subclassing
   `Strategy` (`core/query_strategies/strategy.py`). Constructor signature:
   `def __init__(self, dataset, net, logger): super().__init__(dataset, net, logger)`. Implement
   `query(self, n)` returning an array of `n` unlabeled sample indices.
2. Add `from .<new_strategy_file> import <NewStrategyClass>` to
   `core/query_strategies/__init__.py`.
3. **Register in `dalmax/query_strategies/registry.py::LEGACY_STRATEGY_REGISTRY`** (a plain dict
   entry, `"<NewStrategyClass>": <NewStrategyClass>`) — **not** `utils/orchestrator.py::get_strategy`,
   which is dead code as of Phase 2 (still present, still works if called directly, but unreachable
   from `demo.py`/`dalmax.cli`; do not add new entries to it).
4. Add `"<NewStrategyClass>"` to `dalmax/cli.py`'s `--strategy_name` `choices=[...]` list.
5. If the strategy needs new hyperparameters, add a typed field to
   `dalmax/config/schema.py::DatasetConfig` (or a nested config dataclass) and read it via
   `config.dataset.<field>` — never `self.params[<dataset_name>][<key>]` string-keyed dict access
   (the KI-4 anti-pattern), and never a `setattr(strategy, ...)` post-construction injection (the
   KI-7 anti-pattern this refactor removed).
6. Add a test asserting `dalmax.query_strategies.registry.build_strategy("<NewStrategyClass>", ...)`
   resolves without error — see `tests/test_strategy_registry.py` for the pattern (stub
   dataset/net, no real training). `tests/test_registry.py`/`tests/test_strategy_registry.py` parse
   `dalmax/cli.py`'s `choices=[...]` via `ast` (without importing `dalmax.cli`, to avoid its
   import-time log-file side effect) — adding your name to step 4's list is picked up automatically.
7. Update this file's "Reference: current full strategy list" below, and
   `.specs/experiments/experimental-protocol.md`'s "Query strategies exercised" section if the new
   strategy is added to any run script's sweep.

## Do not use the pre-Phase-2 flow

The previous version of this checklist described editing `utils/orchestrator.py`'s if/elif chain and
`demo.py`'s `choices=[...]` list directly. `utils/orchestrator.py` is now dead code (unreachable) and
`demo.py` is a 12-line shim with no `choices=[...]` of its own — edit `dalmax/cli.py` instead. Do not
"fix" a strategy by making it reachable only through the legacy path; it must be reachable through
`dalmax.cli.build_arg_parser()`'s `choices=[...]`, since that is what `demo.py`/`run_pipe_gpu_*.sh`
actually invoke.

## Reference: current full strategy list (for cross-checking Path B step 4 / Path A step 3)

`RandomSampling`, `LeastConfidence`, `MarginSampling`, `EntropySampling`,
`LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`,
`KMeansSampling`, `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`,
`AdversarialDeepFool`, `SSRAEKmeansSampling`, `VCTexKmeansSampling`,
`SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`, `RepresentationStrategy`.
