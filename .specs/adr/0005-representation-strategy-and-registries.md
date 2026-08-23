# ADR 0005: One generic `RepresentationStrategy` + dict registries replacing the four fixed embedding-based strategy classes and `utils/orchestrator.py`'s if/elif chains

- **Status:** Accepted
- **Date:** 2026-08-23

## Context

Before Phase 2, four concrete `Strategy` subclasses duplicated the same two-step logic — "compute
embeddings for the pool" then "cluster/select a batch from them" — with the embedding source and
the selection method baked into the class itself: `SSRAEKmeansSampling` (SSRAE + flat k-means,
closest-to-centroid), `VCTexKmeansSampling` (VCTex + flat k-means), `SSRAEKmeansHCSampling`/
`VCTexKmeansHCSampling` (SSRAE/VCTex + hierarchical k-means, via the shared `SSLStrategy` base in
`core/query_strategies/ssl_ssrae_sampling.py`). Adding a fifth combination — e.g. the ablation
study's "ResNet-ImageNet embeddings + hierarchical selection" (§6.3) or "SSRAE + proportional-random
flat k-means" (§6.3) — meant writing a fifth near-duplicate class, copying whichever bugs the
closest sibling had (the hardcoded `KMeans(random_state=3)`, KI-5; the hardcoded
`self.params['DANINHAS']`, KI-4).

Separately, `utils/orchestrator.py` mapped `--strategy_name`/dataset/model name strings to classes
via four `if/elif` chains (`get_handler`, `get_dataset`, `get_network_deep_learning`,
`get_strategy`), each requiring hand-editing to add a new name, with no compile-time or import-time
check that `utils/orchestrator.py`, `core/query_strategies/__init__.py`, and `demo.py`'s
`choices=[...]` stayed in sync (KI-9).

ADR 0003 already introduced the `EmbeddingProvider` abstraction (embedding source) and
`.specs/architecture/target-architecture.md` §6 already specified a `SelectionStrategy` abstraction
(clustering/picking logic) as two separate concerns. This ADR covers the third piece: how a
`Strategy` (the interface `demo.py`/`dalmax.cli` actually calls `query()` on) is built by composing
one of each, and how every `--strategy_name` string — old and new — resolves to a constructed
instance.

## Decision

We will:

1. Implement **one** concrete strategy class, `dalmax.query_strategies.representation.
   RepresentationStrategy(dataset, net, config, logger, *, embedding_provider, selection, rng,
   cache=None)`, that subclasses the unmodified, vendored `core.query_strategies.strategy.Strategy`
   and overrides only `query(n)`: compute/load the pool's embeddings once (via the injected
   `EmbeddingProvider` + `EmbeddingCache`), slice by `embedding_variant` if needed, and delegate
   picking to the injected `SelectionStrategy`. No strategy-specific subclass exists for any
   `(extractor, selection)` combination, past or future.
2. Introduce `dalmax.query_strategies.registry.STRATEGY_REGISTRY` / `build_strategy(name, dataset,
   net, config, logger, rng)` as the **single** place that resolves a `--strategy_name` string to a
   constructed `Strategy`, replacing `utils/orchestrator.py::get_strategy`'s if/elif chain for the
   `dalmax`-routed CLI. It recognizes three kinds of names:
   - The 12 legacy strategies (`RandomSampling` … `AdversarialDeepFool`): constructed exactly as
     before, `LegacyClass(dataset, net, logger)`, `dalmax` uninvolved.
   - The 4 legacy `*Kmeans*Sampling`/`*KmeansHCSampling` names: **presets** —
     `REPRESENTATION_PRESETS: dict[str, tuple[extractor, selection_method]]` — that build a
     `RepresentationStrategy` with a fixed `(embedding_provider, selection)` pair and a pinned
     legacy `Q` (`REPRESENTATION_PRESET_Q`), so these CLI names keep their exact historical meaning
     regardless of what a params JSON's generic `"embedding"` block says.
   - The new generic `RepresentationStrategy` CLI name: builds from `config.dataset.embedding`/
     `config.dataset.selection` verbatim — this is what the ablation study's params JSON
     `"embedding"`/`"selection"` blocks drive.
   Every `SelectionStrategy`'s RNG is derived from `config.seed` via
   `dalmax.seeding.derive_seed(config.seed, "selection")` — never a literal — closing KI-5/KI-29.
3. Introduce the same dict-registry pattern for the other three axes `utils/orchestrator.py` used to
   handle: `dalmax.data.registry.DATASET_REGISTRY`/`get_dataset`, `dalmax.models.registry.
   MODEL_REGISTRY`/`get_network`, and (from ADR 0003/selection design) `dalmax.embeddings.registry.
   EMBEDDING_REGISTRY`/`get_embedding_provider` and `dalmax.selection.registry.SELECTION_REGISTRY`/
   `get_selection_strategy`. Each raises a `KeyError` (or, where config validity is at stake,
   `dalmax.config.schema.ConfigError`) naming the valid names — never a bare `NotImplementedError`
   or a `KeyError` several calls removed from the actual bad input.
4. Leave `utils/orchestrator.py` and the four superseded strategy files
   (`ssrae_kmeans_sampling.py`, `vctex_kmeans_sampling.py`, `ssl_ssrae_sampling.py`) in place,
   unmodified, per ADR 0002's Phase-2/Phase-4 staging — they become dead code (unreachable from
   `demo.py`/`dalmax.cli`), not deleted, until Phase 4.

## Consequences

- Positive: adding a new `(embedding, selection)` combination — e.g. every cell of the ablation
  study's §6.3 table — is a params-JSON/CLI change, not a new Python class; this is the direct
  enabler for `.specs/architecture/refactor-plan.md` Phase 3 being config-only.
- Positive: the `KMeans(random_state=3)` (KI-5) and hardcoded-`'DANINHAS'` (KI-4) bugs cannot recur
  in new combinations, since the fix lives once in `FlatKMeansClosest`/`HierarchicalKMeansSelection`
  (ADR 0003's siblings), not once per strategy subclass.
- Positive: `tests/test_strategy_registry.py` and `tests/test_registry.py` can assert every
  `--strategy_name` choice resolves without constructing real datasets/models, closing the
  "no test for registry/CLI drift" gap noted in KI-9.
- Negative: a preset name (`SSRAEKmeansSampling`, etc.) now means two things that must never be
  confused — "reproduce this exact historical `(extractor, Q, selection)` combination" (preset path,
  ignores `config.dataset.embedding.q`) vs. "whatever the params JSON's `embedding`/`selection`
  blocks say" (generic `RepresentationStrategy` path). `REPRESENTATION_PRESET_Q`'s docstring
  documents this explicitly because getting it backwards silently reintroduces a Q-mixup bug
  (VCTex accidentally getting SSRAE's `Q=13`).
- Negative: four legacy files are now dead code sitting alongside their replacement with no compiler
  or import-time signal that they are unreachable — a future contributor could accidentally import
  and use them directly (they still work standalone). Tracked for deletion in
  `.specs/architecture/refactor-plan.md` Phase 4.
