# Reproducibility

Every experiment must be fully determined by: **(params JSON + CLI args + seed +
git commit)**. Given those four things, the same results.json must be reproducible
(modulo hardware nondeterminism already tracked in known-issues).

## Seed policy

**Phase 2 landed (2026-08-23)**: `dalmax/seeding.py::seed_everything(seed)` is now the single place
that seeds `random`, NumPy's legacy global RNG, and PyTorch (CPU + CUDA), called once by
`dalmax.experiment.runner.ExperimentRunner.run()` before any dataset/model/strategy object is
constructed. `dalmax/seeding.py::derive_seed(seed, tag)` derives a deterministic sub-seed for any
component that needs its own (e.g. `dalmax/selection/flat_kmeans_closest.py`'s `KMeans(random_state=...)`).
Any **new** source of randomness (sampling, clustering, augmentation) must derive from `seed_everything`'s
`np.random.Generator` or `derive_seed`, passed through explicitly — never read global state
ambiently and never a literal. `demo.py` is now a 12-line shim; the two bullets below describe the
pre-Phase-2 state for context, but the actual seeding code they refer to no longer exists at those
locations.

- `demo.py` used to seed `np.random.seed(args.seed)` and `torch.manual_seed(args.seed)`
  from the `--seed` CLI argument (see `run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh`,
  which sweep `SEEDS=(1 2 3)`, still valid — the CLI flag and sweep are unchanged, only where the
  seeding happens moved to `dalmax/seeding.py`).
- **Known violation, now fixed**: the now-deleted `core/query_strategies/ssrae_kmeans_sampling.py`
  used to instantiate `KMeans(n_clusters=n, random_state=3, n_init=10)` — the `3` was a
  literal, completely ignoring `--seed`. That file (and its bug) was deleted in Phase 4, having been
  dead code/unreachable from `dalmax.cli` since Phase 2 (`.specs/architecture/current-state.md`).
  Its replacement, `dalmax/selection/flat_kmeans_closest.py::FlatKMeansClosest`, derives
  `KMeans(random_state=...)` from `np.random.default_rng(dalmax.seeding.derive_seed(config.seed,
  "selection"))` — verified by the regenerated `tests/golden/ssrae_kmeans_micro_seed1.json` fixture.
  Any new strategy code must follow this pattern, not the legacy literal.
- `torch.backends.cudnn.enabled = False` was used as a determinism
  shortcut in `demo.py:59` (line no longer exists); it globally disabled cuDNN and cost performance on the
  10 GB lab GPUs. **Now uses** `torch.backends.cudnn.deterministic = True` +
  `torch.backends.cudnn.benchmark = False` instead (`dalmax/seeding.py::seed_everything`).
  Deliberate consequence: post-refactor GPU runs are not bit-identical to historical GPU runs at the
  same seed — cuDNN's deterministic kernels differ from the non-deterministic ones used previously.
  CPU runs (the golden-run fixtures) are unaffected.

## Config snapshot

- **Resolved (2026-08-23, Phase 2).** `dalmax/experiment/run_metadata.py::write_run_metadata`
  writes `run_metadata.json` into every results directory
  (`{dir_results}/{dataset}/SEED_{seed}/NQ_{n_query}_NIL_{n_init}_NR_{n_round}_NE_{n_epoch}/{strategy}/`),
  containing the fully-resolved config (`to_dict(config)`), a best-effort `git_commit`, Python/torch
  versions, CUDA availability, and start timestamp — called from `ExperimentRunner.run()` before the
  round loop starts. `results.json`, `predictions.csv`, plots, and `log-dalmax.log` are unchanged;
  `run_metadata.json` is the new artifact that closes the gap this section originally flagged (see
  `.specs/quality/known-issues.md` KI-1, and `current-state.md` §8 for the exact field list).

## Embedding cache discipline

**Phase 2 landed (2026-08-23)**: `dalmax/embeddings/cache.py::EmbeddingCache` is now the live cache
for any run through `dalmax.cli`/`demo.py`, keyed on `(dataset, extractor, Q, variant, split,
pool_hash)` — filename `results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{pool_hash}.pkl`.
This closes both gaps this section originally flagged: `embedding_variant` is now a key component
(only `"full"` is ever cached; `"spatial"`/`"spectral"` are sliced from it on load via
`dalmax/embeddings/variants.py::slice_embedding`, never separately cached or recomputed), and `Q` is
config-driven (`EmbeddingConfig.q`, defaulted per-extractor by `dalmax/config/loader.py` —
`ssrae`→`13`, `vctex`→`(5,17)`, `resnet_imagenet`→`None`), not a hardcoded literal. **Never reuse an
embedding cache across a different `(dataset, extractor, Q, pool, variant)` combination** — the
keying above makes this impossible by construction as long as the key components are computed
correctly; still sanity-check the actual filename before trusting an ablation's cached embeddings.

The `utils/data.py::cache_file_path(name, dataset_folder, q=None, pool_hash=None)` helper described
below (Phase 1 fix) was **deleted in Phase 4** along with the rest of `utils/data.py` — it had been
dead code since Phase 2, only reached when `Data.initialize_labels(..., compute_legacy_features=True)`,
which `dalmax.experiment.runner.ExperimentRunner` never passed. Kept for the historical record:

- `utils/data.py`'s (deleted) `cache_file_path(name, dataset_folder, q=None, pool_hash=None)`
  keyed SSRAE/VCTex feature caches and `Y_train` under `results/cache/` as
  `{name}_{dataset_folder}[_Q{q}[_pool{pool_hash}]].pkl` (e.g.
  `results/cache/features_ssrae_daninhas_micro_Q13_pool<12-hex>.pkl`). SSRAE/VCTex
  feature extraction runs only over the **unlabeled pool** at the moment
  `create_feature_maps_ssrae`/`create_feature_maps_vctex` is called, and that pool
  depends on `--seed` and `--n_init_labeled` — so `pool_hash` is
  `hashlib.sha256(np.where(self.labeled_idxs==0)[0].astype(np.int64).tobytes()).hexdigest()[:12]`,
  computed before the cache path is built (the `dalmax/embeddings/cache.py::pool_hash` function is
  byte-identical to this computation, verified by `tests/test_embedding_cache.py`, so pool identity
  hashes the same way in both the legacy and live paths). `Y_train` has no pool
  concept (`q=None`) and never carries a `pool_hash` segment, even if one is passed.
  The old fixed-path caches (`results/features_dict_ssrae.pkl`,
  `results/features_dict_vctex.pkl`, `results/Y_train.pkl`) are orphaned — no
  longer read or written by this code path, nor by the new `EmbeddingCache` — and must never be
  resurrected or deleted automatically (`.claude/rules/data-safety.md`).
- Longer term fix, **done**: the fuller `EmbeddingProvider`/`EmbeddingCache`
  abstraction (ADR 0003, ADR 0005) supersedes `cache_file_path` with a
  `(dataset, extractor, Q, variant, split, pool_hash)`-keyed cache. See
  `.specs/quality/known-issues.md` KI-3.

## Enforcement

- The `experiment-auditor` agent (`.claude/agents/experiment-auditor.md`) checks
  seed propagation, params/spec consistency, and cache validity before any
  experiment handoff to the lab machine or Colab.
