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
3. Add two `logger.warning` lines in `demo.py` (historical) (initial labeled indices after
   `dataset.initialize_labels`, queried indices after each `strategy.query` call) — the only way to
   observe these indices for a golden-run fixture, since nothing else in the codebase exposes them.
4. Capture `tests/golden/random_sampling_micro_seed1.json` and
   `tests/golden/ssrae_kmeans_micro_seed1.json` (indices, metrics, exact CLI/params, git commit,
   notes — including that the SSRAE fixture's indices are pinned to the hardcoded
   `KMeans(random_state=3)`, KI-5/KI-29, not to `--seed`), regression-tested by
   `tests/test_golden_run.py` (`dataset`+`slow`-marked, subprocess-driven, writes to `tmp_path` not
   `results/`).
5. Make `Makefile`'s `smoke` target a real end-to-end check: generate the micro dataset, run
   `demo.py (historical) --strategy_name RandomSampling` on it into `results/smoke/`, then run the fast test
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
- Negative: two `logger.warning` calls were added to `demo.py` (historical) at the same (arguably wrong, per
  KI-11) logging level as the rest of the file, to stay minimal and consistent with existing style
  rather than fixing logging levels as an unrelated drive-by change; KI-11 remains open.
- Negative: the SSRAE golden fixture's queried indices will legitimately change (and must be
  regenerated in the same change) once Phase 2's seed-propagation audit fixes KI-5 — this is
  documented in the fixture itself and in KI-29 so it isn't mistaken for a regression.
- Follow-up: Phase 2's `EmbeddingProvider`/`EmbeddingCache` (ADR 0003) supersedes `cache_file_path`
  with a fuller `(dataset, extractor, Q, variant, split)`-keyed cache; `cache_file_path` and its
  call sites should be removed/migrated at that point, not kept as a parallel mechanism.

## Update (2026-08-23): deterministic sampling seed + pool-identity cache key

Two follow-up fixes were made against this same decision, found in code review before this branch
merged:

1. **`scripts/make_micro_dataset.py`'s per-`(split, class)` sampling seed used Python's built-in
   `hash((split, class_name))`.** `hash()` on a tuple is salted per-process
   (`PYTHONHASHSEED` randomization by default), so two processes running the exact same script could
   silently select a *different* set of files for `DATA/daninhas_micro/` — directly contradicting
   this ADR's "deterministic" and "re-running always selects the same files" claims. Fixed by
   deriving the seed from `hashlib.sha256(f"{split}:{class_name}".encode())` instead, which is stable
   across processes and machines. `DATA/daninhas_micro/` was deleted and regenerated; determinism was
   re-verified by regenerating it in 2 separate processes (different `PYTHONHASHSEED` values) and
   diffing `find DATA/daninhas_micro -type f | sort | sha256sum` (identical digest both times).
2. **`cache_file_path` (decision item 2) keyed caches on `(dataset_folder, Q)` but not on which
   pool of images the cache was computed for.** SSRAE/VCTex feature extraction runs only over the
   *unlabeled pool* at the time `create_feature_maps_ssrae`/`create_feature_maps_vctex` is called,
   and that pool depends on `--seed` and `--n_init_labeled` — so two runs with the same
   `(dataset_folder, Q)` but a different seed/`n_init_labeled` could silently load each other's
   stale features. Fixed by adding an optional `pool_hash: str | None = None` parameter to
   `cache_file_path`; call sites compute
   `pool_hash = hashlib.sha256(np.where(self.labeled_idxs==0)[0].astype(np.int64).tobytes()).hexdigest()[:12]`
   before building the cache path, giving `results/cache/{name}_{dataset_folder}_Q{q}_pool{pool_hash}.pkl`.
   `Y_train` has no pool concept and is unaffected (still `q=None`, no `pool_hash` segment).

Both `tests/golden/random_sampling_micro_seed1.json` and `tests/golden/ssrae_kmeans_micro_seed1.json`
were regenerated against the corrected micro dataset (same recorded CLI, git commit updated). The
`RandomSampling` fixture's indices/metrics happened to come out unchanged (indices are pool
*positions*, not filenames, and per-class counts didn't change); the `SSRAEKmeansSampling` fixture's
`round_1_query_idxs_sorted` legitimately changed (`[11, 12, 19, 25, 26]` → `[5, 21, 37, 45, 47]`)
because the actual files selected into the micro dataset changed — not a regression in SSRAE/KMeans
behavior. See `.specs/quality/known-issues.md` KI-3 and the fixtures' own `notes` for detail.
`tests/test_cache_paths.py` was extended with `pool_hash` collision/idempotency/omission tests.

## Amendment (2026-08-23, Phase 4)

Every `utils/data.py`/`demo.py` (historical) reference above (`get_DANINHAS`, `cache_file_path`,
`create_feature_maps_ssrae`/`_vctex`, the `logger.warning` lines) describes the codebase as it
existed at the time this ADR was written. Phase 4's package consolidation later deleted
`utils/data.py` outright (its live logic had already moved to `dalmax/data/`/`dalmax/embeddings/` in
Phase 2/3, so nothing from the file itself needed to carry forward) and reduced `demo.py` (historical) to a
12-line shim; see ADR 0002's final amendment. This does not change this ADR's Decision or the
golden-run fixtures/tooling it describes (`scripts/make_micro_dataset.py`,
`tests/golden/*.json`, `tests/test_golden_run.py`), which are unaffected by the later move and still
work exactly as decided here.

## Amendment (2026-08-23, micro-dataset redefinition — `chore/micro-10pct`)

**Purpose**: per user request, `DATA/daninhas_micro/` is redefined so that the local, no-GPU
end-to-end check (`make smoke`, `make smoke-ablations`, `tests/test_golden_run.py`) is a *realistic
pre-lab/pre-Colab validation* — exercising all 5 real classes with their real imbalance shape —
rather than a 2-class toy case that could pass while a 5-class-specific bug (e.g. in
`n_classes`-sized model heads, confusion-matrix plotting, or per-class metric aggregation) slipped
through to the lab machine undetected.

**What changed**: `scripts/make_micro_dataset.py` no longer hardcodes 2 classes
(`DATASET_BRACHIARIA`, `DATASET_GRAMINEA`) and fixed per-class counts (25 train / 10 test). It now
iterates **every** class folder in `DATA/daninhas_full/train/` and `.../test/` (all 5:
`GRAMINEA`, `MAMONA`, `BRACHIARIA`, `COLONIAO`, `OUTRAS_FOLHAS_LARGAS`) and selects
`max(1, floor(0.10 * n_files_in_class))` files each — a genuine 10%-stratified replica of
`DATA/daninhas_full/`, computed from the real on-disk file counts (not `arquivos.txt`, which the
script never reads). The same deterministic mechanism from the earlier amendment above (sorted
filenames, `hashlib.sha256(f"{split}:{class_name}")`-derived per-`(split, class)` seed, idempotent,
read-only w.r.t. `daninhas_full/`) is unchanged. Resulting sizes, measured against the real
`DATA/daninhas_full/` folders on this machine (also matching `DATA/daninhas_full/arquivos.txt`'s
counts exactly):

| Class | Train (10%) | Test (10%) |
|---|---|---|
| GRAMINEA | 119 | 97 |
| MAMONA | 337 | 45 |
| BRACHIARIA | 77 | 33 |
| COLONIAO | 103 | 19 |
| OUTRAS_FOLHAS_LARGAS | 170 | 15 |
| **Total** | **806** | **209** |

The script now also writes/refreshes `DATA/daninhas_micro/arquivos.txt` on every run, in exactly the
same format as `DATA/daninhas_full/arquivos.txt` (same class order, `Total`/`Average` lines).

**Config updates**: `files_config/params_micro.json`'s `DANINHAS.n_classes` changed `2` → `5`;
`config_kmh` rescaled to `n_clusters=[40, 10]`, `n_levels=2`, `sample_sizes=[2, 2]` for the new
~796-image unlabeled pool (806 train - 10 initial labeled) — this block is schema-completeness-only,
not exercised by any smoke-tested strategy (see the file's own `_comment`). All 11
`files_config/ablations/micro/*.json` also changed `n_classes` `2` → `5` and had their hierarchies
rescaled by dividing each full-scale `n_clusters` entry by 10 (rounding, floor 2), since the new
micro pool is almost exactly 1/10th of the full-scale ablation pool — see
`files_config/ablations/README.md`'s `micro/` section for the full derivation and the empirical
re-verification (all 11 configs re-run end-to-end against the real regenerated dataset,
`bash scripts/ablations/smoke_ablations.sh`, 11/11 passing, no vendored equal-cluster-size bug hit,
~4m37s total wall time on the local CPU-only dev notebook).

**Determinism re-verified**: `DATA/daninhas_micro/` was deleted and regenerated; determinism was
re-verified by regenerating it in 2 separate processes (different `PYTHONHASHSEED` values, one via
`poetry run python scripts/make_micro_dataset.py`, one via `PYTHONHASHSEED=42 poetry run python
scripts/make_micro_dataset.py`) and diffing `find DATA/daninhas_micro -type f | sort | sha256sum`
(identical digest both times, including the regenerated `arquivos.txt`, which is itself
byte-identical across the two runs). Stale caches for the old dataset definition
(`results/cache/*daninhas_micro*`, `results/cache/embeddings/*daninhas_micro*`) were deleted before
regenerating, since the pool identity (and therefore `pool_hash`) changed.

**Golden fixtures regenerated**: `tests/golden/random_sampling_micro_seed1.json` and
`tests/golden/ssrae_kmeans_micro_seed1.json` were re-captured against the redefined dataset, same
recorded CLIs (`RandomSampling`, `SSRAEKmeansSampling`), verified deterministic across 2 separate
processes each (both at the `demo.py` (historical) subprocess level via `tests/test_golden_run.py`, run twice
back-to-back, and independently via direct `demo.py` (historical) invocations before the fixtures were written).
The prior (2-class) fixture values are kept in each file's `previous_micro_2class_values` field for
the historical record, not deleted — see those fields' own notes for exactly what changed and why
(the `RandomSampling` fixture's indices/metrics changed because the pool grew from 40 to 796
unlabeled images across a different class count; `SSRAEKmeansSampling`'s queried ids changed because
the underlying SSRAE embeddings/cluster assignments changed with the pool, not because of any change
to `FlatKMeansClosest`'s seed derivation). SSRAE feature extraction over the new ~796-image pool
still completes in ~10-15 seconds (~12 ms/image, consistent with the pre-redefinition per-image
rate), so the CPU smoke-test budget is unaffected.

**`make smoke` timing**: full target (`scripts/make_micro_dataset.py` + one `demo.py (historical)
--strategy_name RandomSampling` run + the fast test suite) measured at ~35 seconds wall time on the
local CPU-only dev notebook after this redefinition (up from the ~10-15 seconds documented for the
old 2-class dataset, since ResNet50 forward/backward now runs over 806 training images instead of
50) — still comfortably fast enough for routine local use before a lab/Colab handoff.

**Consequences**:
- Positive: `make smoke` / `make smoke-ablations` / the golden-run regression now catch integration
  bugs specific to the real 5-class shape (imbalanced per-class counts, a 5-way classification head,
  5-class confusion matrices/macro-F1 aggregation) that a 2-class toy case could not — directly
  serving the "realistic pre-lab check" purpose this amendment was made for.
- Neutral: `make smoke`'s wall time roughly doubled (~15s → ~35s) and
  `make smoke-ablations`'s roughly quadrupled relative to the old 2-class hierarchies' runtime, but
  both remain well within their respective "few seconds"/"~5 minutes" budgets.
- Negative (expected, not a regression): every golden-run index/metric value changed, since the pool
  of images and its class composition changed entirely — this is exactly what
  `previous_micro_2class_values` documents to avoid the change being mistaken for a determinism
  regression by a future reader.
