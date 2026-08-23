# ADR 0003: Introduce an `EmbeddingProvider` abstraction with a keyed cache

- **Status:** Accepted
- **Date:** 2026-08-23

## Context

Today, embedding computation and caching live inside `utils/data.py`'s `Data` class:
`create_feature_maps_ssrae` (`utils/data.py:117-184`, `Q=13` hardcoded at line 139) and
`create_feature_maps_vctex` (`utils/data.py:46-115`, `Q=[5,17]` hardcoded at line 70). Both cache to
a **fixed, unkeyed** pickle path (`results/features_dict_ssrae.pkl`, `results/features_dict_vctex.pkl`)
— confirmed by direct inspection. There is no encoding of dataset name, `Q`, or any notion of an
"embedding variant" in the cache path, so switching dataset or `Q` while a stale cache file exists
silently loads the wrong features. `Data.initialize_labels` (`utils/data.py:211-229`) decides which
extractor to run by string-matching the **strategy class name**, not by any embedding configuration.

The advisor-requested ablation study (`.specs/experiments/ablation-study.md` §6.1, §6.3) requires:
running SSRAE at a fixed `Q`, slicing its output into `spatial`/`spectral`/`full` variants without
recomputation, and adding a **new** embedding source (ImageNet-pretrained ResNet penultimate-layer
features) as a drop-in alternative to SSRAE for the "without representation module" condition. None
of this is expressible in the current design without more hardcoded string matches.

## Decision

We will introduce an `EmbeddingProvider` abstract base class (`dalmax/embeddings/base.py`) with
`SSRAEProvider`, `VCTexProvider`, and `ResNetImageNetProvider` implementations, backed by an
`EmbeddingCache` (`dalmax/embeddings/cache.py`) keyed by the tuple
`(dataset, extractor, Q, variant, split)`. The **full** embedding is always computed and cached
once per key; `embedding_variant` (`full | spatial | spectral`) is applied as a pure slicing
function (`dalmax/embeddings/variants.py`) on top of the cached full vector, never as a separate
extraction run. See `.specs/architecture/target-architecture.md` §4-5 for the full design and file
mapping.

## Consequences

- Positive: directly unblocks ablation 6.1 (representation) and 6.3 (representation vs. hierarchy
  contribution) as described in `.specs/architecture/refactor-plan.md`'s "ablation enablers"
  checklist, with no risk of stale-cache cross-contamination between datasets/`Q` values/variants.
- Positive: adding `ResNetImageNetProvider` is additive — no existing provider or strategy code
  needs to change to support it, since strategies only depend on the `EmbeddingProvider` interface.
- Negative: requires migrating existing cached pickle files (`results/features_dict_*.pkl`) to the
  new keyed layout, or accepting a one-time recomputation after Phase 2 lands; this should be called
  out explicitly to the advisor before the lab machine's next batch (`experiment-auditor`'s
  pre-flight check in `refactor-plan.md` Phase 3 covers this).
- Negative: the `Data`/`PoolDataset` class loses direct ownership of feature extraction, which is a
  behavior change to `Data.initialize_labels` — must be covered by the golden-run regression test
  from `.specs/quality/testing-strategy.md` before merging.
