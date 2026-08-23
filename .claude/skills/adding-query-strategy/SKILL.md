---
name: adding-query-strategy
description: Step-by-step checklist of every file that must change to add a new active-learning query strategy to DalMax, via the dalmax/ registries (Phase 2) or the legacy path.
---

# Adding a query strategy

**Rewritten 2026-08-23 for Phase 2.** DalMax now routes `demo.py`/`--strategy_name` through
`dalmax/query_strategies/registry.py::build_strategy`, not `utils/orchestrator.py::get_strategy`
(that if/elif chain is dead code as of Phase 2 — unreachable from `demo.py`/`dalmax.cli`, see
`.specs/architecture/current-state.md` §0). Full spec:
`.specs/use-cases/add-new-strategy.md` — read it first if this summary is not enough context.

## Which path?

- **Embedding + selection combination** (fits the RNHAL pattern: embed the unlabeled pool, then
  cluster/pick a batch) → **Path A**, below. This is almost always the right path for a new
  representation-based strategy, including anything the ablation study needs.
- **Something else entirely** (new uncertainty measure, new adversarial method) → **Path B**, below.

## Path A — new `EmbeddingProvider` or `SelectionStrategy`

New embedding provider:
1. `dalmax/embeddings/<name>_provider.py`, subclass `dalmax.embeddings.base.EmbeddingProvider`
   (`embed(self, images: np.ndarray) -> np.ndarray`, `(N,H,W,3)` uint8 → `(N,D)` float32). No model
   weights loaded at import time. Reference: `dalmax/embeddings/resnet_imagenet_provider.py`.
2. Register in `dalmax/embeddings/registry.py::EMBEDDING_REGISTRY`.
3. Add to `dalmax/config/schema.py::VALID_EXTRACTORS`; add a default `q` to
   `dalmax/config/loader.py::_EXTRACTOR_DEFAULT_Q` if needed.
4. Test: construct it, call `embed()` on synthetic images, assert shape/dtype (see
   `tests/test_resnet_provider.py`).

New selection strategy:
1. `dalmax/selection/<name>.py`, subclass `dalmax.selection.base.SelectionStrategy`
   (`select(self, embeddings, ids, budget, rng) -> np.ndarray`). **Never hardcode a random-state
   literal** — derive every stochastic choice from `rng` (the `KMeans(random_state=3)` bug, KI-5,
   Phase 2 fixed this pattern; do not reintroduce it). Reference:
   `dalmax/selection/flat_kmeans_proportional.py`.
2. Register in `dalmax/selection/registry.py::SELECTION_REGISTRY`; add to
   `dalmax/config/schema.py::VALID_SELECTION_METHODS`.
3. Test: synthetic embeddings (`sklearn.datasets.make_blobs`), assert subset/size property, **and**
   assert two `rng`s from different seeds give different selections (the automated KI-5 regression
   check — see `tests/test_selection_flat.py`).

Optional — a fixed `--strategy_name` preset for one specific `(extractor, selection)` combination:
add to `dalmax/query_strategies/registry.py::REPRESENTATION_PRESETS` and to `dalmax/cli.py`'s
`choices=[...]`. Usually unnecessary: the generic `RepresentationStrategy` CLI name plus a params
JSON `"embedding"`/`"selection"` block already reaches any combination (see
`.specs/experiments/ablation-study.md` for exact JSON syntax).

## Path B — new legacy-style `Strategy` subclass

1. `core/query_strategies/<snake_case_name>.py`:
   ```python
   from .strategy import Strategy

   class YourStrategyName(Strategy):
       def __init__(self, dataset, net, logger):
           super().__init__(dataset, net, logger)

       def query(self, n):
           ...  # return an array of selected sample ids
   ```
2. Add the import/export line to `core/query_strategies/__init__.py`.
3. **Register in `dalmax/query_strategies/registry.py::LEGACY_STRATEGY_REGISTRY`** — a plain dict
   entry (`"YourStrategyName": YourStrategyName`). Do **not** add to `utils/orchestrator.py`
   (dead code, unreachable — see `.specs/architecture/current-state.md` §0).
4. Add `"YourStrategyName"` to `dalmax/cli.py`'s `--strategy_name` `choices=[...]` list —
   `demo.py` itself has no `choices=[...]` of its own any more (12-line shim).
5. If the strategy needs new hyperparameters, add a typed field to `dalmax/config/schema.py`,
   read via `config.dataset.<field>` — never a hardcoded dataset-name string key (the
   `self.params['DANINHAS']['config_kmh']` anti-pattern, KI-4, that Phase 2 fixed) and never a
   post-construction `setattr` (KI-7, also fixed — strategies get everything through the
   constructor now, via `build_strategy`).
6. Test: `dalmax/query_strategies/registry.py::build_strategy("YourStrategyName", ...)` resolves
   without raising — see `tests/test_strategy_registry.py` (stub dataset/net, no real training).
7. Update `.specs/use-cases/add-new-strategy.md`'s strategy list and
   `.specs/experiments/experimental-protocol.md`'s "Query strategies exercised" section.

## Verification

```bash
poetry run python -c "from dalmax.query_strategies.registry import STRATEGY_REGISTRY; print(sorted(STRATEGY_REGISTRY))"
poetry run pytest -k "registry or strategy_registry"
poetry run ruff check dalmax/ core/query_strategies/<touched files>
```

Do not verify against `utils.orchestrator.get_strategy` — it no longer reflects what `demo.py`
actually resolves.

## Do not skip spec sync

Per `.claude/rules/spec-sync.md`, the test and spec-update steps above are not optional cleanup —
they are part of the same task as the code changes. A strategy present in a registry but absent
from `dalmax/cli.py`'s `choices=[...]` (or vice versa) is a bug;
`tests/test_strategy_registry.py`/`tests/test_registry.py` (parsing `dalmax/cli.py`'s
`choices=[...]` via `ast`) and `.claude/agents/code-reviewer.md` both check for exactly this kind
of drift.

See `.specs/use-cases/add-new-strategy.md` for the full version of this checklist and
`.claude/commands/new-strategy.md` for the command form.
