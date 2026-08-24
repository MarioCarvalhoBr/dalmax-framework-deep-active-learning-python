# Testing Strategy

Status: Phase 1 is complete as of 2026-08-23 — `test_imports.py`, `test_registry.py`,
`test_ssrae_embedding_layout.py`, `test_cache_paths.py` (85 tests total under
`pytest -m "not gpu and not dataset and not slow"`) plus the golden-run fixtures and
`test_golden_run.py` described in item 4 below. Runs on the no-GPU local dev notebook unless marked
otherwise. **Phase 4 (2026-08-23) moved `core/`/`utils/` into `dalmax/` and deleted the superseded
files** — 277 tests now pass under the same fast-marker selection; `test_imports.py`/`test_registry.py`
were updated in that batch to target `dalmax/` instead of the deleted legacy packages (see items 1-2
below for what changed).

## Markers

Register in `pyproject.toml`'s `[tool.pytest.ini_options]` (or a `pytest.ini`):

| Marker | Meaning | Runs where |
|---|---|---|
| `gpu` | requires CUDA (e.g. hierarchical k-means today hardcodes `device="cuda"`, see `current-state.md` §4) | lab machine only |
| `dataset` | requires the real `DATA/daninhas_full/` or `DATA/DATA_CIFAR10/` on disk | local dev (if dataset present) + lab machine |
| `slow` | anything taking more than a few seconds (e.g. a multi-round tiny smoke run) | local dev optionally, always in CI's extended job if one exists |

CI (`.github/workflows/ci.yml`) runs `pytest -m "not gpu and not dataset"` — fast, no dataset, no
GPU. The lab machine, when convenient, can run the full suite including `gpu`/`dataset`-marked
tests before a batch (this is part of what the `experiment-auditor` agent checks pre-flight).

## Now (Phase 1) — the safety net

1. **`tests/test_imports.py`** — **updated in Phase 4**: dynamically walks every module under
   `dalmax/` (`pkgutil.walk_packages`, not a hardcoded list) and asserts each imports cleanly, with
   a small `SKIP_MODULES` dict for standalone vendored scripts that are only meant to run directly
   (`dalmax.tools.SSRAE.classification`, `dalmax.tools.VCTex.classification` — non-relative imports
   + module-level side effects). Before Phase 4 this test walked the legacy `core`/`utils` packages
   (`import core.query_strategies`, `import core.tools.SSRAE.extractor`, etc.); those packages no
   longer exist.
2. **`tests/test_registry.py`** — **updated in Phase 4**: every `--strategy_name` choice parsed out
   of `dalmax/cli.py`'s source (via `ast`, never importing `dalmax.cli` directly — see that test's
   module docstring) matches `dalmax.query_strategies.registry.STRATEGY_REGISTRY`'s keys exactly.
   This file previously also tested the legacy `utils.orchestrator.get_strategy` `if/elif` registry
   directly against `core.query_strategies.strategy.Strategy` — both `utils/orchestrator.py` and
   `core/query_strategies/` were deleted in Phase 4 once the `dalmax` registries fully replaced them
   (see `current-state.md` §0). `build_strategy`'s own behavior (presets, `ConfigError`s, seed
   derivation, constructing every legacy class) is now covered by `tests/test_strategy_registry.py`.
   This is the regression test for the strategy registry staying in sync with the CLI, called out
   explicitly in `.claude/agents/code-reviewer.md`'s responsibilities.
3. **`tests/test_ssrae_embedding_layout.py`** — on a tiny random RGB array (e.g. `16x16x3`) and a
   small `Q` (e.g. `Q=4`), assert `ColorFeatureExtractor(Q).extract(image)`:
   - has length divisible by 6 (six blocks: `β_R, β_G, β_B, β_S_R, β_S_G, β_S_B`, confirmed at
     `dalmax/tools/SSRAE/extractor.py` (was `core/tools/SSRAE/extractor.py:99`, moved in Phase 4)),
   - **Layout caveat (verified 2026-08-23)**: the layout is **row-interleaved**, not six
     contiguous blocks — `emb.reshape(9, 6*(Q+1))` gives column-groups `[R, G, B, RG, GB, BR]`
     (each width `Q+1`); `emb[:len(emb)//2]` is NOT spatial-only and `emb[len(emb)//2:]` is NOT
     spectral-only. The corrected spatial slice is `emb.reshape(9, 6*(Q+1))[:, :3*(Q+1)].reshape(-1)`
     (columns for R,G,B) and spectral is `emb.reshape(9, 6*(Q+1))[:, 3*(Q+1):].reshape(-1)`
     (columns for RG,GB,BR) — see `.specs/experiments/ablation-study.md` §6.1 and
     `.specs/quality/known-issues.md`,
   - runs in well under a second on CPU (no GPU, no dataset — safe for CI).
4. **Golden-run regression fixture** — **done**. `scripts/make_micro_dataset.py` deterministically
   samples a stratified 10% of every class (all 5: `DATASET_BRACHIARIA`, `DATASET_COLONIAO`,
   `DATASET_GRAMINEA`, `DATASET_MAMONA`, `DATASET_OUTRAS_FOLHAS_LARGAS`) in `DATA/daninhas_full/`
   into `DATA/daninhas_micro/` (~806 train + ~209 test images total; sorted filenames, fixed
   sampling seed, idempotent, never writes into `daninhas_full/`, exits 0 with a message if
   `daninhas_full/` isn't present, also writes `DATA/daninhas_micro/arquivos.txt` in the same format
   as the source file's). **2026-08-23 redefinition** (see
   `.specs/adr/0004-micro-dataset-and-golden-run.md`'s "micro-dataset redefinition" amendment):
   this used to be a hand-picked 2-class, fixed 25-train/10-test-per-class subset; it is now the
   all-5-class 10%-stratified replica described above, for a more realistic pre-lab/pre-Colab
   end-to-end check. `files_config/params_micro.json` points DANINHAS at that micro-dataset with
   `n_epoch=1`, `n_classes=5`, `batch_size=16`.
   `demo.py --strategy_name RandomSampling --n_init_labeled 10 --n_query 5 --n_round 1 --seed 1` and
   the same CLI with `--strategy_name SSRAEKmeansSampling` were each run twice in separate processes
   after the redefinition; the initial labeled indices, per-round query indices (both logged via two
   `logger.warning` lines in `demo.py`, the only way to observe them), and every `results.json`
   metric were **bit-identical across both runs for both strategies** — no CPU-training
   nondeterminism was found in this configuration. Recorded as
   `tests/golden/random_sampling_micro_seed1.json` and `tests/golden/ssrae_kmeans_micro_seed1.json`
   (indices + metrics + exact CLI/params + git commit + notes); the pre-redefinition (2-class) values
   are kept in each fixture's `previous_micro_2class_values` field for the historical record, not
   deleted. The SSRAE golden run originally required a minimal, explicitly-scoped Phase 1 exception:
   `utils/data.py`'s `cache_file_path()` helper (this whole file was deleted in Phase 4, superseded by
   `dalmax/embeddings/cache.py::EmbeddingCache` — see KI-3 in `known-issues.md`) keyed the
   SSRAE/VCTex/`Y_train` pickle caches on dataset folder name (+ `Q`) instead of a fixed path, so the
   micro-dataset run computed/read `results/cache/features_ssrae_daninhas_micro_Q13.pkl` and never
   touched the pre-existing full-dataset `results/features_dict_ssrae.pkl` (left on disk, untouched;
   partial fix of KI-3, see `known-issues.md`). Note the SSRAE query indices are a function of
   `SSRAEKmeansSampling`'s hardcoded `KMeans(random_state=3)` (KI-5), not of `--seed` — expected to
   change (and the fixture to be regenerated) once Phase 2's seed-propagation audit lands.
   `tests/test_golden_run.py` (marked `dataset`+`slow`) regenerates the micro dataset if needed, runs
   both strategies via `subprocess` into a `tmp_path` (never `results/`), and asserts indices match
   exactly and metrics match within `1e-6`. Not run by CI's fast job; run via `make test-all` or
   `pytest -m dataset` on a machine with `DATA/daninhas_full/` present.

## After Phase 2 — abstraction-level unit tests

5. **Embedding slicing**: `slice_embedding(vector, "spatial")` and `slice_embedding(vector, "spectral")`
   partition a full vector with no overlap and no gap (`spatial + spectral == full` as index ranges);
   `slice_embedding(vector, "full")` is the identity. Pure function, CPU, milliseconds — no marker
   needed.
6. **Embedding cache keys**: two calls with different `Q` (or different `dataset`, or different
   `variant`) never collide on the same cache file/key; a call with identical
   `(dataset, extractor, Q, variant, split)` hits the cache and does not recompute (assert the
   provider's `compute()` is called exactly once across two `get_or_compute` calls with the same key,
   using a mock/spy).
7. **Hierarchy config validation**: `config/schema.py`'s `HierarchyConfig` rejects
   `len(n_clusters) != n_levels` or `len(sample_sizes) != n_levels` at load time (mirrors the
   `assert` in `dalmax/tools/SSL/src/hierarchical_kmeans_gpu.py` (was `core/tools/SSL/src/
   hierarchical_kmeans_gpu.py:134-135`, moved in Phase 4), but surfaced as a clear
   validation error instead of a bare `AssertionError` deep in `query()`).
8. **Registry completeness**: `STRATEGY_REGISTRY`, `DATASET_REGISTRY`, `MODEL_REGISTRY`,
   `SELECTION_REGISTRY` each have at least the entries the current CLI/`config_kmh` schema requires;
   a test iterates `demo.py`'s (or its replacement `cli.py`'s) declared choices and asserts each
   resolves.
9. **Seed determinism**: `FlatKMeansClosest` (or its current equivalent) run twice with the same
   seed on the same tiny embedding matrix produces identical selected ids; run with two different
   seeds produces different assignments at least once (guards against a reintroduced
   `random_state=3`-style literal).

## After Phase 3 — ablation-specific checks

10. A `slow`+`dataset`-marked test (lab machine or opt-in locally) that runs each of the three
    ablation sub-studies (`.specs/experiments/ablation-study.md` §6.1-6.3) for exactly one round on
    a small subset and asserts the pipeline completes and macro F1 is computed and finite — not a
    correctness check on the science, just a "the plumbing works end-to-end" check.
11. `run_metadata.json` is written and contains the expected config snapshot + a non-empty git
    commit hash string when run inside a git repo.

## What deliberately stays manual / lab-machine-only

- Full multi-seed, multi-strategy sweeps (`scripts/benchmark/run_pipe_gpu_*.sh`) are experiments, not tests — they are
  covered by `.specs/experiments/experimental-protocol.md` and the `experiment-auditor` agent's
  pre-flight checklist, not by `pytest`.
- `make smoke` (see `.specs/architecture/refactor-plan.md` Phase 1 and the root `Makefile`) is a real
  end-to-end check as of Phase 1's closeout: it generates `DATA/daninhas_micro/` (or skips with a
  message if `DATA/daninhas_full/` isn't present), runs `demo.py --strategy_name RandomSampling` on
  it (~30-35 CPU seconds since the 2026-08-23 micro-dataset redefinition to a 10%-stratified,
  all-5-class replica — see `.specs/adr/0004-micro-dataset-and-golden-run.md`'s amendment), then runs
  the fast test suite. This only covers `RandomSampling`
  end-to-end; the fuller golden-run regression (both `RandomSampling` and `SSRAEKmeansSampling`,
  exact index/metric comparison) is `tests/test_golden_run.py`, `dataset`+`slow`-marked and run via
  `make test-all`, not `make smoke`, to keep the latter fast. The hierarchical strategies
  (`*HCSampling`) remain untestable locally (hardcoded `device="cuda"`, KI-13) until Phase 2.
