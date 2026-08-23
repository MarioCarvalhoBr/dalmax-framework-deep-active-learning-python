# ADR 0004: Deterministic micro-dataset + golden-run fixtures for local smoke testing

- **Status:** Accepted
- **Date:** 2026-08-23

## Context

`.specs/architecture/refactor-plan.md` Phase 1 requires a golden-run fixture (exact selected indices
+ final metrics for a tiny, fixed-seed run) and a real `make smoke` target, but two things blocked
both before this change:

1. There was no way to point `demo.py` at a tiny subset of the real dataset. `get_DANINHAS`
   (`utils/data.py`) only reads a full `DATA/<name>/{train,test}/<class>/` tree; there was no
   micro/synthetic dataset under version control (and `DATA/` is gitignored/immutable per
   `.claude/rules/data-safety.md`, so one couldn't be committed even if built).
2. `SSRAEKmeansSampling` — the only CPU-safe non-random strategy (the hierarchical variants hardcode
   `device="cuda"`, KI-13) — routes through `Data.create_feature_maps_ssrae`, which caches to the
   **fixed** path `results/features_dict_ssrae.pkl`. This machine already has a 130 MB cache file at
   that path from a real `daninhas_full` run. Running a golden-run smoke test unmodified would
   silently load stale full-dataset features into a 40-image micro run, or overwrite a cache another
   experiment depends on — both unacceptable (see `.claude/rules/reproducibility.md`'s embedding
   cache discipline section, KI-3).

## Decision

We will:

1. Add `scripts/make_micro_dataset.py`, which deterministically samples 25 train + 10 test images
   from each of 2 real classes (`DATASET_BRACHIARIA`, `DATASET_GRAMINEA`) in
   `DATA/daninhas_full/` into `DATA/daninhas_micro/{train,test}/<class>/` — sorted filenames, a fixed
   `random.Random` sampling seed, idempotent, read-only with respect to `daninhas_full/`, and a clean
   exit-0-with-message if `daninhas_full/` isn't present. `files_config/params_micro.json` gives
   DANINHAS a smoke-only config pointed at that micro dataset.
2. Add a `cache_file_path(name, dataset_folder, q=None)` helper in `utils/data.py` and thread a
   `dataset_folder` parameter through `Data.__init__`/`get_DANINHAS`/`get_CIFAR10`, so the
   `Y_train`/SSRAE/VCTex pickle caches key on `(dataset_folder, Q)` under `results/cache/` instead of
   a fixed path. This is a **deliberately minimal, pre-Phase-2 exception** to "no source changes in
   the golden-run capture task" — scoped to exactly the cache-path collision that blocked the SSRAE
   golden run, not the full `EmbeddingProvider`/`EmbeddingCache` abstraction from ADR 0003 (still
   Phase 2). The pre-existing full-dataset pkl files at the old fixed paths are left on disk,
   untouched and unread by the new code path.
3. Add two `logger.warning` lines in `demo.py` (initial labeled indices after
   `dataset.initialize_labels`, queried indices after each `strategy.query` call) — the only way to
   observe these indices for a golden-run fixture, since nothing else in the codebase exposes them.
4. Capture `tests/golden/random_sampling_micro_seed1.json` and
   `tests/golden/ssrae_kmeans_micro_seed1.json` (indices, metrics, exact CLI/params, git commit,
   notes — including that the SSRAE fixture's indices are pinned to the hardcoded
   `KMeans(random_state=3)`, KI-5/KI-29, not to `--seed`), regression-tested by
   `tests/test_golden_run.py` (`dataset`+`slow`-marked, subprocess-driven, writes to `tmp_path` not
   `results/`).
5. Make `Makefile`'s `smoke` target a real end-to-end check: generate the micro dataset, run
   `demo.py --strategy_name RandomSampling` on it into `results/smoke/`, then run the fast test
   suite — replacing the previous documented-stub behavior.

## Consequences

- Positive: Phase 1's golden-run acceptance criterion is met with real dataset images (not a
  synthetic/random image set), for both `RandomSampling` and a representation-based strategy
  (`SSRAEKmeansSampling`), without touching `DATA/daninhas_full/` or the pre-existing full-dataset
  embedding caches.
- Positive: `make smoke` now catches real integration breakage (dataset loading, model construction,
  training loop, metrics computation, JSON/CSV persistence) in ~10-15 seconds of CPU time, closing
  the gap the previous "proxy smoke check" (fast tests only) could not.
- Positive: the cache-path fix is additive and backward-compatible — it doesn't change behavior for
  any existing run that never touches `Data.dataset_folder`-based paths differently than before,
  since old cache files are neither read nor deleted.
- Negative: two `logger.warning` calls were added to `demo.py` at the same (arguably wrong, per
  KI-11) logging level as the rest of the file, to stay minimal and consistent with existing style
  rather than fixing logging levels as an unrelated drive-by change; KI-11 remains open.
- Negative: the SSRAE golden fixture's queried indices will legitimately change (and must be
  regenerated in the same change) once Phase 2's seed-propagation audit fixes KI-5 — this is
  documented in the fixture itself and in KI-29 so it isn't mistaken for a regression.
- Follow-up: Phase 2's `EmbeddingProvider`/`EmbeddingCache` (ADR 0003) supersedes `cache_file_path`
  with a fuller `(dataset, extractor, Q, variant, split)`-keyed cache; `cache_file_path` and its
  call sites should be removed/migrated at that point, not kept as a parallel mechanism.
