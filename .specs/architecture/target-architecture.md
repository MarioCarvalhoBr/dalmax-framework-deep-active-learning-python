# Architecture — Target State

Status: design target for the refactor described in `refactor-plan.md` (Phases 2-4). This is
the end state after Phase 4 (package consolidation, ADR 0002); intermediate phases may keep
the new modules under `core/`/`utils/` before the physical rename. Every module below is
mapped to the current file(s) it replaces so the move is traceable.

## 1. Goals (traced to `prompt-master.md` §4.3 / `.claude/rules/code-quality.md`)

- Single responsibility per module; no duplicated strategy boilerplate.
- Registry pattern over `if/elif` chains, for datasets, models, and strategies.
- Explicit dependency injection — no `setattr` magic (fixes coupling point #1 in `current-state.md`).
- A typed config layer that validates the params JSON instead of raw dict indexing, with no
  hardcoded dataset keys (fixes coupling point #2).
- An `EmbeddingProvider` abstraction with an explicitly-keyed cache (fixes coupling points #3, #4).
- A `SelectionStrategy` abstraction that separates "how points are clustered" from "which strategy
  class you subclass" (fixes coupling point #5, and is the direct enabler for ablations §6.2/§6.3
  in `ablation-study.md`).
- All randomness derived from the experiment seed; no literal `random_state=3` (fixes #3, #6).
- Run metadata (config snapshot + git commit hash) saved into every results directory.

## 2. Target package layout

```
dalmax/
├── cli.py                        # thin CLI: argparse -> ExperimentConfig -> ExperimentRunner.run()
├── config/
│   ├── schema.py                 # dataclasses/pydantic: ExperimentConfig, DatasetConfig,
│   │                              #   ModelConfig, HierarchyConfig, EmbeddingConfig
│   └── loader.py                 # load_config(params_json, dataset_name, cli_args) -> ExperimentConfig
│                                  #   validates presence of keys instead of raising KeyError deep in query()
├── seeding.py                    # seed_everything(seed): np, torch, python random, deterministic cudnn flags
├── data/
│   ├── datasets.py                # PoolDataset (was utils/data.py: Data), dataset-agnostic
│   ├── handlers.py                # torch Dataset wrappers (was utils/dataset.py)
│   ├── loaders.py                 # load_daninhas(), load_cifar10(), load_cifar10_download()
│   │                              #   (was utils/data.py: get_DANINHAS/get_CIFAR10/get_CIFAR10_Download)
│   └── registry.py                # DATASET_REGISTRY: name -> (loader, handler)
├── embeddings/
│   ├── base.py                    # EmbeddingProvider(ABC): compute(images) -> np.ndarray; cache_key()
│   ├── ssrae_provider.py           # wraps core/tools/SSRAE/{extractor,rnn,splitter}.py, Q from config
│   ├── vctex_provider.py           # wraps core/tools/VCTex/VCTexMethod.py, Q from config
│   ├── resnet_imagenet_provider.py # NEW (ablation 6.3): penultimate-layer features from the
│   │                              #   ImageNet-pretrained ResNet50 already used in daninhas_resnet50.py
│   ├── variants.py                 # slice_embedding(vector, variant, Q): "full"|"spatial"|"spectral"
│   │                              #   full = vector; layout is ROW-INTERLEAVED (verified 2026-08-23,
│   │                              #   see §5) — spatial/spectral are column-group slices of
│   │                              #   vector.reshape(9, 6*(Q+1)), NOT vector[:len//2]/vector[len//2:]
│   └── cache.py                    # EmbeddingCache keyed by (dataset, extractor, Q, variant, split)
│                                  #   path template: results/.cache/embeddings/{dataset}__{extractor}__Q{Q}__{variant}__{split}.pkl
│                                  #   explicit invalidation: cache.invalidate(key) / cache.exists(key)
├── selection/
│   ├── base.py                     # SelectionStrategy(ABC): select(embeddings, ids, budget, rng) -> ids
│   ├── flat_kmeans_closest.py       # today's SSRAEKmeansSampling/VCTexKmeansSampling logic:
│   │                              #   k = budget, pick closest-to-centroid sample per cluster, seeded via rng
│   ├── flat_kmeans_proportional.py  # NEW (ablation 6.3 "without hierarchical module"):
│   │                              #   k = budget, pick RANDOM samples per cluster, proportional to
│   │                              #   cluster size relative to the budget — explicitly distinct from
│   │                              #   flat_kmeans_closest.py (see ablation-study.md §6.3 note)
│   ├── hierarchical_kmeans.py       # wraps core/tools/SSL/src/{hierarchical_kmeans_gpu,hierarchical_sampling,
│   │                              #   clusters}.py; device resolved from ExperimentConfig.device, no hardcoded
│   │                              #   "cuda"; n_clusters/n_levels/sample_sizes entirely from HierarchyConfig
│   └── registry.py                  # SELECTION_REGISTRY: name -> SelectionStrategy class
├── query_strategies/
│   ├── base.py                      # QueryStrategy(ABC), constructor takes (dataset, net, config, logger)
│   │                              #   — config passed explicitly, no post-hoc setattr
│   ├── uncertainty.py                # LeastConfidence, MarginSampling, EntropySampling (+ Dropout variants)
│   ├── diversity.py                   # KMeansSampling, KCenterGreedy (operate on net embeddings, unchanged)
│   ├── bayesian.py                     # BALDDropout
│   ├── adversarial.py                   # AdversarialBIM, AdversarialDeepFool
│   ├── representation.py                 # ONE generic RepresentationStrategy(embedding_provider, selection_strategy)
│   │                              #   replaces SSRAEKmeansSampling, VCTexKmeansSampling, SSRAEKmeansHCSampling,
│   │                              #   VCTexKmeansHCSampling, and SSLStrategy as 4 fixed subclasses; the 4
│   │                              #   existing CLI strategy names become config presets (embedding=ssrae|vctex,
│   │                              #   selection=flat_closest|hierarchical) resolved through the two registries above
│   └── registry.py                      # STRATEGY_REGISTRY: name -> factory(dataset, net, config, logger)
│                                        #   replaces utils/orchestrator.py:get_strategy's if/elif
├── models/
│   ├── base.py                        # DeepLearning (was core/deep_learning.py), takes seeded rng,
│   │                              #   no global cudnn side effect, load_model TODO resolved
│   ├── daninhas_resnet50.py            # was core/daninhas_model.py, duplicate imports removed
│   ├── cifar10_cnn.py                   # was core/cifar10_model.py, duplicate imports removed
│   └── registry.py                       # MODEL_REGISTRY: dataset name -> model class
├── experiment/
│   ├── runner.py                        # ExperimentRunner.run(config) — the round loop extracted from demo.py main()
│   ├── reporter.py                       # per-run reporting: confusion matrix, accuracy/precision/recall/F1 plots,
│   │                              #   results.json, predictions.csv (was demo.py tail half)
│   └── run_metadata.py                    # snapshot(config) -> {config_dict, git_commit_hash, timestamp}
│                                        #   written as run_metadata.json into every results dir
├── reporting/                             # cross-run aggregation (was utils/report/*.py, renumbered to
│   ├── extract_confusion_matrices.py       #   descriptive names; CLI behavior preserved)
│   ├── chunk_results.py
│   ├── average_confusion_matrices.py
│   └── average_results.py
└── tools/                                 # vendored/adapted third-party code, unchanged in Phase 2-3
    ├── ssrae/                              # was core/tools/SSRAE/
    ├── vctex/                               # was core/tools/VCTex/
    └── ssl/                                  # was core/tools/SSL/ (hierarchical_kmeans_gpu, hierarchical_sampling,
                                              #   clusters, kmeans_gpu — Meta-licensed code kept as-is per its LICENSE)
```

`tests/` mirrors this package layout 1:1 (see `.specs/quality/testing-strategy.md`).

## 3. Component / class diagram

```mermaid
classDiagram
    class ExperimentConfig {
      +DatasetConfig dataset
      +ModelConfig model
      +EmbeddingConfig embedding
      +HierarchyConfig hierarchy
      +int seed
      +int n_init_labeled
      +int n_query
      +int n_round
      +str strategy_name
    }
    class ConfigLoader {
      +load(params_json, dataset_name, cli_args) ExperimentConfig
    }
    class DatasetRegistry {
      +get(name) DatasetLoader
    }
    class ModelRegistry {
      +get(dataset_name) ModelClass
    }
    class StrategyRegistry {
      +get(name) StrategyFactory
    }
    class EmbeddingProvider {
      <<abstract>>
      +compute(images) ndarray
      +cache_key(dataset, split) EmbeddingKey
    }
    class SSRAEProvider
    class VCTexProvider
    class ResNetImageNetProvider
    class EmbeddingCache {
      +get(key) ndarray
      +put(key, value)
      +exists(key) bool
      +invalidate(key)
    }
    class EmbeddingKey {
      +dataset str
      +extractor str
      +Q int|list
      +variant str
      +split str
    }
    class SelectionStrategy {
      <<abstract>>
      +select(embeddings, ids, budget, rng) ids
    }
    class FlatKMeansClosest
    class FlatKMeansProportionalRandom
    class HierarchicalKMeansSelection
    class QueryStrategy {
      <<abstract>>
      +query(n) ids
      +update(ids)
      +train()
      +predict(data)
    }
    class RepresentationStrategy {
      -EmbeddingProvider embedding_provider
      -SelectionStrategy selection_strategy
      +query(n) ids
    }
    class DeepLearning {
      +train(data)
      +predict(data)
      +get_embeddings(data)
    }
    class ExperimentRunner {
      +run(config) RunResult
    }
    class Reporter {
      +write_plots(metrics, dir)
      +write_results_json(metrics, dir)
      +write_predictions_csv(preds, dir)
    }
    class RunMetadata {
      +snapshot(config) dict
    }

    EmbeddingProvider <|-- SSRAEProvider
    EmbeddingProvider <|-- VCTexProvider
    EmbeddingProvider <|-- ResNetImageNetProvider
    EmbeddingProvider --> EmbeddingCache
    EmbeddingCache --> EmbeddingKey
    SelectionStrategy <|-- FlatKMeansClosest
    SelectionStrategy <|-- FlatKMeansProportionalRandom
    SelectionStrategy <|-- HierarchicalKMeansSelection
    QueryStrategy <|-- RepresentationStrategy
    RepresentationStrategy --> EmbeddingProvider
    RepresentationStrategy --> SelectionStrategy
    ExperimentRunner --> ConfigLoader
    ExperimentRunner --> DatasetRegistry
    ExperimentRunner --> ModelRegistry
    ExperimentRunner --> StrategyRegistry
    ExperimentRunner --> QueryStrategy
    ExperimentRunner --> DeepLearning
    ExperimentRunner --> Reporter
    ExperimentRunner --> RunMetadata
    ConfigLoader --> ExperimentConfig
```

## 4. Embedding cache contract

Cache key is the tuple `(dataset, extractor, Q, variant, split)`:

- `dataset`: e.g. `"daninhas_full"`, `"cifar10"` — never a strategy name.
- `extractor`: `"ssrae" | "vctex" | "resnet_imagenet"`.
- `Q`: extractor hyperparameter (`13` for SSRAE, `[5,17]` for VCTex, `None`/embedding-dim for
  ResNet-ImageNet) — serialized into the key so changing `Q` cannot silently reuse a stale cache.
- `variant`: `"full" | "spatial" | "spectral"` (§5 below). Only meaningful for SSRAE today; VCTex
  and ResNet-ImageNet providers always use `"full"`.
- `split`: `"train"` (only the pool is embedded today; kept explicit for future test-time use).

The **full** embedding is always computed and cached once; `spatial`/`spectral` variants are
**slices of the cached full vector**, never recomputed — this directly satisfies the "never
recompute SSRAE three times" requirement in `ablation-study.md` §6.1. `EmbeddingCache` stores one
file per key (JSON- or hash-encoded key in the filename, e.g.
`results/.cache/embeddings/daninhas_full__ssrae__Q13__full__train.pkl`), replacing the two fixed
paths `results/features_dict_ssrae.pkl` / `results/features_dict_vctex.pkl`.

## 5. `embedding_variant` slicing (ablation 6.1 enabler)

**Layout caveat (verified 2026-08-23)**: the SSRAE embedding `emb =
hstack[β_R, β_G, β_B, β_S_RG, β_S_GB, β_S_BR]` (`core/tools/SSRAE/extractor.py:99`) is
**row-interleaved, not six contiguous blocks**. Each `beta` block has shape `(9, Q+1)`;
`torch.hstack` concatenates along columns (axis=1) into a `(9, 6*(Q+1))` matrix, and the
subsequent row-major `.reshape((1, -1))` interleaves all six blocks per row. So
`vector[:len(vector)//2]` is **not** spatial-only and `vector[len(vector)//2:]` is
**not** spectral-only — the correct slice operates on column-groups of the reshaped
`(9, 6*(Q+1))` matrix, which requires knowing `Q`:

```python
def slice_embedding(vector: np.ndarray, variant: str, Q: int) -> np.ndarray:
    matrix = vector.reshape(9, 6 * (Q + 1))   # rows = patch dims, column groups = [R,G,B,RG,GB,BR]
    if variant == "full":
        return vector
    if variant == "spatial":     # R, G, B column-groups (columns [0, 3*(Q+1)))
        return matrix[:, : 3 * (Q + 1)].reshape(-1)
    if variant == "spectral":    # S_RG, S_GB, S_BR column-groups (columns [3*(Q+1), 6*(Q+1)))
        return matrix[:, 3 * (Q + 1) :].reshape(-1)
    raise ValueError(f"unknown embedding_variant: {variant}")
```
This function lives in `dalmax/embeddings/variants.py` and is applied by `RepresentationStrategy`
right before handing embeddings to the `SelectionStrategy` — the cache always stores `full`. `Q`
must be threaded through from `EmbeddingConfig`/`EmbeddingKey` since it is required to recover the
column-group boundaries. **Requirement**: prefer having `SSRAEProvider` return an embedding object
that already exposes block-contiguous accessors (or explicit column-group index ranges) rather than
relying on every call site to know `Q` and re-derive the reshape — see
`.specs/quality/known-issues.md` for the tracked issue and `tests/test_ssrae_embedding_layout.py`
for a test that pins down the actual layout.

## 6. Selection module abstraction (ablation 6.2 / 6.3 enabler)

| Selection class | Behavior | Replaces / relates to |
|---|---|---|
| `FlatKMeansClosest` | `k = budget`; pick the sample closest to each centroid | today's `SSRAEKmeansSampling`, `VCTexKmeansSampling` (seeded via `rng`, fixes coupling #3) |
| `FlatKMeansProportionalRandom` | `k = budget`; pick **random** samples per cluster, proportional to relative cluster size | **NEW** — required by ablation 6.3 "without hierarchical module"; explicitly different from `FlatKMeansClosest` |
| `HierarchicalKMeansSelection` | wraps `hierarchical_kmeans_with_resampling` + `hierarchical_sampling`, config-driven `n_clusters`/`n_levels`/`sample_sizes`, device from config | today's `SSLStrategy` (`SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`), fixes coupling #2 and #8 (no hardcoded `'DANINHAS'`, no hardcoded `'cuda'`) |

Hierarchy depth (`L=1..4`) and cluster counts become plain `HierarchyConfig` fields validated by
`config/schema.py` — running ablation 6.2's four configurations is then a matter of four config
files/CLI overrides, not code changes.

## 7. Seed propagation contract

`dalmax/seeding.py:seed_everything(seed)` is the **single** place that touches global RNG state:
`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and
`torch.backends.cudnn.deterministic = True` / `torch.backends.cudnn.benchmark = False` (replacing
the blanket `cudnn.enabled = False`). Every `SelectionStrategy` that uses `KMeans` receives an
`rng`/`random_state` derived from the experiment seed via its constructor — no strategy is allowed
to hardcode a `random_state` literal. `ExperimentRunner.run(config)` calls `seed_everything` exactly
once, before any dataset/model/strategy object is constructed.

## 8. Run metadata (reproducibility)

`dalmax/experiment/run_metadata.py:snapshot(config)` writes `run_metadata.json` into the results
directory alongside `results.json`, containing: the fully-resolved `ExperimentConfig` (as loaded,
not the raw JSON), the current git commit hash (`git rev-parse HEAD`, best-effort if unavailable),
Python/torch/CUDA versions, and the wall-clock start/end time. This is what makes "every experiment
fully determined by (params JSON + CLI args + seed + git commit)" (`.claude/rules/reproducibility.md`)
verifiable after the fact.

## 9. Registries replacing `if/elif`

`DATASET_REGISTRY`, `MODEL_REGISTRY`, `STRATEGY_REGISTRY`, `SELECTION_REGISTRY` are plain
`dict[str, Callable]` populated via a `@register("name")` decorator at import time in each module
(no metaclass magic — keeps `code-quality.md`'s "every module importable without side effects").
`utils/orchestrator.py`'s four functions collapse into thin `registry.get(name)` lookups that raise
a clear `KeyError` listing valid names, instead of a bare `NotImplementedError`.

## 10. `ExperimentRunner` + `Reporter` + thin CLI

`demo.py`'s 289-line `main()` splits three ways:
- `dalmax/cli.py`: argparse only, then `ExperimentRunner(ConfigLoader.load(...)).run()`.
- `dalmax/experiment/runner.py`: the round loop (init labels → train → query → update → train →
  metrics), returns a `RunResult` dataclass (metrics per round, final predictions, trained net).
- `dalmax/experiment/reporter.py`: takes a `RunResult` and a results directory, writes all plots,
  `results.json`, `predictions.csv`, and delegates to `run_metadata.snapshot`.

This mirrors the acceptance criterion in `refactor-plan.md` Phase 2 ("split `demo.py` into
`runner` + `reporting` + thin CLI").
