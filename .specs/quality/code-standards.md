# Code Standards

This mirrors `.claude/rules/code-quality.md` with concrete before/after examples taken from this
repository, so "elegant, scalable, fast, and organized" has a checkable meaning for this codebase
specifically. Applies to all new/edited code from Phase 1 of `.specs/architecture/refactor-plan.md`
onward; existing code is not retroactively rewritten outside that plan.

## 1. Registry pattern over `if/elif` chains

**Before** (`utils/orchestrator.py:45-80`, confirmed as-is at the time; file deleted in Phase 4):
```python
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    # ... 14 more branches ...
    else:
        raise NotImplementedError
```
Every new strategy required editing this function, `core/query_strategies/__init__.py`, and
`demo.py` (historical)'s `choices=[...]` list by hand, with nothing enforcing they stay consistent. (Both
`utils/orchestrator.py` and the old `core/query_strategies/__init__.py` no longer exist — see the
`dalmax/query_strategies/__init__.py`/`registry.py` implementation below, landed for real, not just
sketched.)

**After** (target, `dalmax/query_strategies/registry.py`):
```python
STRATEGY_REGISTRY: dict[str, StrategyFactory] = {}

def register(name: str):
    def _wrap(factory: StrategyFactory) -> StrategyFactory:
        STRATEGY_REGISTRY[name] = factory
        return factory
    return _wrap

def get_strategy(name: str) -> StrategyFactory:
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}") from None
```
A new strategy self-registers with `@register("MyStrategy")` at definition time; `demo.py` (historical)'s CLI
`choices` can be generated from `STRATEGY_REGISTRY.keys()` instead of hand-duplicated, which is what
`test_registry.py` (see `testing-strategy.md`) enforces.

## 2. Explicit dependency injection, not `setattr` magic

**Before** (`demo.py (historical):76-79`, confirmed as-is):
```python
strategy = get_strategy(args.strategy_name)(dataset, net, logger)
setattr(strategy, "params", params)  # strategy.params only exists after this line
```
`SSLStrategy.__init__` (`core/query_strategies/ssl_ssrae_sampling.py:29`, file deleted in Phase 4)
had to defend against this
with `self.params = self.params if hasattr(self, 'params') else None`, which only worked because of
call-order luck, not a guarantee.

**After:**
```python
config = ConfigLoader.load(params_json, dataset_name, cli_args)
strategy = STRATEGY_REGISTRY[args.strategy_name](dataset, net, config, logger)
```
`config` is a required constructor argument on every `QueryStrategy` subclass — there is no state
for which "was `params` already attached?" is even a question.

## 3. No hardcoded dataset names inside logic

**Before** (`core/query_strategies/ssl_ssrae_sampling.py:66`, confirmed as-is at the time; file
deleted in Phase 4):
```python
config_kmh = self.params['DANINHAS']['config_kmh']
```
This raised `KeyError` for any dataset other than DANINHAS, including `CIFAR10`, which is a
documented CLI choice in `demo.py` (historical).

**After:**
```python
config_kmh = self.config.dataset.hierarchy  # HierarchyConfig, resolved for whichever dataset was run
```
The dataset the experiment is actually running is threaded through `ExperimentConfig`, never
re-derived from a literal string inside strategy logic.

## 4. Caches must have explicit keys and invalidation

**Before** (`utils/data.py:117-121`, confirmed as-is at the time; file deleted in Phase 4):
```python
path_pkl = 'results/features_dict_ssrae.pkl'
if os.path.exists(path_pkl):
    with open(path_pkl, 'rb') as f:
        self.features_dict = pickle.load(f)
```
Same fixed path regardless of dataset, `Q`, or embedding variant — see ADR 0003 and
`known-issues.md` item KI-3.

**After** (`dalmax/embeddings/cache.py`, see `target-architecture.md` §4):
```python
key = EmbeddingKey(dataset="daninhas_full", extractor="ssrae", Q=13, variant="full", split="train")
if cache.exists(key):
    features = cache.get(key)
else:
    features = provider.compute(images)
    cache.put(key, features)
```

## 5. All randomness derives from the experiment seed

**Before** (`core/query_strategies/ssrae_kmeans_sampling.py:23`, confirmed as-is at the time; file
deleted in Phase 4):
```python
kmeans = KMeans(n_clusters=n, random_state=3, n_init=10)
```
Literal `3`, independent of `--seed`; every seed in `SEEDS=(1 2 3)` (`scripts/benchmark/run_pipe_gpu_0.sh`) clustered
identically for this strategy.

**After:**
```python
kmeans = KMeans(n_clusters=n, random_state=self.rng_seed, n_init=10)
```
where `self.rng_seed` is derived once from `ExperimentConfig.seed` in `seed_everything()`
(`target-architecture.md` §7) and passed down to every `SelectionStrategy`.

## 6. Fail-fast errors over silent fallbacks

**Before**: a missing `config_kmh` for a dataset raises a bare `KeyError` deep inside `query()`
(round N of an already-running experiment), after potentially hours of training.

**After:** `config/loader.py` validates the full `ExperimentConfig` (including
`HierarchyConfig` presence when `selection_strategy == "hierarchical"`) **before** `ExperimentRunner`
constructs the dataset or trains anything, with a message naming the missing field and the JSON file
it should be added to.

## 7. English-only code and comments; type hints on new/edited code

**Before**: `utils/data.py` mixed Portuguese docstrings/comments (`"""Carrega o dataset..."""`,
`# Salva em um arquivo indices.txt...`) with English ones throughout the same file; no type hints
anywhere in `utils/data.py`, `utils/orchestrator.py`, or `core/query_strategies/*.py`. **Status
(Phase 4)**: `utils/orchestrator.py` and the legacy `core/query_strategies/*.py` files with this
issue were deleted; the Portuguese comment quoted above is still present, unchanged, in its Phase 4
destination, `dalmax/data/datasets.py` (KI-12, still open).

**After:** new/edited functions get English docstrings/comments and full type hints, e.g.:
```python
def slice_embedding(vector: np.ndarray, variant: Literal["full", "spatial", "spectral"]) -> np.ndarray:
    """Slice a cached full SSRAE embedding into the requested variant without recomputation."""
```

## 8. Every module importable without side effects

**Before**: `utils/data.py` module-level code has no import-time side effects itself, but
`Data.__init__` unconditionally writes `results/Y_train.pkl` and `results/original_indices.txt` to
disk as a side effect of *object construction* (`utils/data.py:22-27,44,187-195` at the time). **Status
(Phase 4)**: the file moved (not deleted) to `dalmax/data/datasets.py`, unchanged — the same
side effect is still present there (KI-21, still open), which means even
building a `Data` instance for a unit test touches the filesystem.

**After:** persistence (writing indices/labels to disk) is a method the caller invokes explicitly
(`dataset.save_indices(path)`), never something that happens implicitly inside `__init__`.

## 9. No duplicated strategy boilerplate

**Before** (all three files below deleted in Phase 4): `SSRAEKmeansSampling` (`ssrae_kmeans_sampling.py`) and `VCTexKmeansSampling`
(`vctex_kmeans_sampling.py`) were near-identical copy-pasted `query()` methods differing only in
which `features_dict` populated them; likewise `SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling` shared
100% of `SSLStrategy.query()`.

**After:** one `RepresentationStrategy(embedding_provider, selection_strategy)` class
(`target-architecture.md` §2, §6) parameterizes what today requires four near-duplicate classes.

## 10. Duplicate/unused imports (mechanical, but part of "organized")

Originally confirmed duplicate imports (same symbol imported twice in one file), **all resolved by
Phase 4's move — see `known-issues.md` KI-15 through KI-20**: the destination files
(`dalmax/models/daninhas_resnet50.py`, `dalmax/models/cifar10_cnn.py`) were cleaned during the move
and have no duplicate imports; the three `core/query_strategies/*.py` files below were deleted
outright, taking their duplicate/unused imports with them:
- `core/daninhas_model.py:1-8` — `torch`, `torch.nn as nn`, and `ViT_B_16_Weights` each imported twice.
- `core/cifar10_model.py:1-9` — `torch`, `torch.nn as nn` each imported twice.
- `core/query_strategies/ssl_ssrae_sampling.py:1-14` — `numpy as np` (lines 2, 8) and
  `matplotlib.pyplot as plt` (lines 3, 13) each imported twice.

Originally confirmed unused imports, likewise resolved by deletion (Phase 4):
- `core/query_strategies/ssl_ssrae_sampling.py:1` (`abstractmethod`), `:12` (`TSNE`), `:14`
  (`matplotlib.colors as mcolors`) — none of the three were referenced anywhere in the file.
- `core/query_strategies/ssrae_kmeans_sampling.py:2,5` (`matplotlib.pyplot as plt`, `time`) — neither
  was used in the file's body.
- `core/query_strategies/vctex_kmeans_sampling.py:2,5` (`matplotlib.pyplot as plt`, `time`) — same.

These were tracked as KI items in `known-issues.md` and were exactly the kind of change the
`mechanic` agent would have made (mechanical, low-risk, English-only rule already satisfied) — moot
now that the files are gone.
