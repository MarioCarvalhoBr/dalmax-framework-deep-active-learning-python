# Testing Strategy

Status: Phase 1 tests are TBD-implementation (not yet written as of 2026-08-23); this document
specifies what they must cover. Runs on the no-GPU local dev notebook unless marked otherwise.

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

1. **`tests/test_imports.py`** — every module under `core/` and `utils/` imports cleanly
   (`import core.query_strategies`, `import core.tools.SSRAE.extractor`, etc.), with GPU-only
   modules (anything importing `core.tools.SSL.src.kmeans_gpu` at module scope, if it turns out to
   require CUDA at import time — verify before marking) skipped or marked `gpu` as needed.
2. **`tests/test_registry.py`** — every `strategy_name` in `demo.py`'s `argparse` `choices=[...]`
   list resolves via `utils/orchestrator.get_strategy(name)` without raising, and returns a subclass
   of `core.query_strategies.strategy.Strategy`. This is the regression test for the
   `if/elif` chains staying in sync with the CLI, called out explicitly in
   `.claude/agents/code-reviewer.md`'s responsibilities.
3. **`tests/test_ssrae_embedding_layout.py`** — on a tiny random RGB array (e.g. `16x16x3`) and a
   small `Q` (e.g. `Q=4`), assert `ColorFeatureExtractor(Q).extract(image)`:
   - has length divisible by 6 (six blocks: `β_R, β_G, β_B, β_S_R, β_S_G, β_S_B`, confirmed at
     `core/tools/SSRAE/extractor.py:99`),
   - **Layout caveat (verified 2026-08-23)**: the layout is **row-interleaved**, not six
     contiguous blocks — `emb.reshape(9, 6*(Q+1))` gives column-groups `[R, G, B, RG, GB, BR]`
     (each width `Q+1`); `emb[:len(emb)//2]` is NOT spatial-only and `emb[len(emb)//2:]` is NOT
     spectral-only. The corrected spatial slice is `emb.reshape(9, 6*(Q+1))[:, :3*(Q+1)].reshape(-1)`
     (columns for R,G,B) and spectral is `emb.reshape(9, 6*(Q+1))[:, 3*(Q+1):].reshape(-1)`
     (columns for RG,GB,BR) — see `.specs/experiments/ablation-study.md` §6.1 and
     `.specs/quality/known-issues.md`,
   - runs in well under a second on CPU (no GPU, no dataset — safe for CI).
4. **Golden-run regression fixture**: a tiny subset (2 classes, ~50 images sampled from
   `DATA/daninhas_full/`, or a synthetic tiny image set if avoiding real data in the fixture is
   preferred), fixed seed, `n_init_labeled` small, `n_round=1`, `RandomSampling` (and
   `SSRAEKmeansSampling` once it no longer hardcodes `device="cuda"` anywhere in its path — it
   doesn't today, only the *hierarchical* variant does) — record the exact selected indices and
   final metrics as a committed JSON fixture. Marked `dataset` if it needs real images, otherwise
   runs in CI.

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
   `assert` in `core/tools/SSL/src/hierarchical_kmeans_gpu.py:134-135`, but surfaced as a clear
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

- Full multi-seed, multi-strategy sweeps (`run_pipe_gpu_*.sh`) are experiments, not tests — they are
  covered by `.specs/experiments/experimental-protocol.md` and the `experiment-auditor` agent's
  pre-flight checklist, not by `pytest`.
- `make smoke` (see `.specs/architecture/refactor-plan.md` Phase 1 and the root `Makefile`) is the
  closest thing to an end-to-end check runnable without a real GPU; if a true smoke run of
  `demo.py`/`cli.py` on a 2-class/50-image subset is not feasible without first landing Phase 2's
  device-agnostic hierarchical selection, the `smoke` Makefile target should be a **documented stub**
  (prints what it would do and why it's blocked) rather than silently no-op — tracked as a
  known-issue until Phase 2 unblocks it.
