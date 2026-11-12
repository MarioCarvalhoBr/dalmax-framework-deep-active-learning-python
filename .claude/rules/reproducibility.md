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
ambiently and never a literal. `trainer.py` (historically `demo.py`) is now a 12-line shim; the two bullets below describe the
pre-Phase-2 state for context, but the actual seeding code they refer to no longer exists at those
locations.

- The historical `demo.py` used to seed `np.random.seed(args.seed)` and `torch.manual_seed(args.seed)`
  from the `--seed` CLI argument (the campaign manifest sweeps seeds `(1, 2, 3)`, as the retired
  `scripts/benchmark/run_pipe_gpu_*.sh` did — the CLI flag is unchanged, only where the seeding
  happens moved to `dalmax/seeding.py`).
- **Known violation, now fixed**: the now-deleted `core/query_strategies/ssrae_kmeans_sampling.py`
  used to instantiate `KMeans(n_clusters=n, random_state=3, n_init=10)` — the `3` was a
  literal, completely ignoring `--seed`. That file (and its bug) was deleted in Phase 4, having been
  dead code/unreachable from `dalmax.cli` since Phase 2 (`.specs/architecture/current-state.md`).
  Its replacement, `dalmax/selection/flat_kmeans_closest.py::FlatKMeansClosest`, derives
  `KMeans(random_state=...)` from `np.random.default_rng(dalmax.seeding.derive_seed(config.seed,
  "selection"))` — verified by the regenerated `tests/golden/ssrae_kmeans_micro_seed1.json` fixture.
  Any new strategy code must follow this pattern, not the legacy literal.
- **Adversarial tie-break (2026-11-11, ADR 0014, KI-45)**: `AdversarialBIM` breaks exact distance ties with
  `np.random.default_rng(derive_seed(config.seed, "adversarial_ties"))`, created once per run by
  `dalmax/query_strategies/registry.py::build_strategy` and injected through the constructor (it advances across
  rounds). Without ties the ranking equals the old stable argsort. `AdversarialDeepFool` is built without it
  (`rng=None`: stable argsort, its executed runs stay reproducible; its only ties are byte-identical duplicate images). `KMeansSampling` keeps
  `KMeans(n_clusters=n)` without `random_state`: it draws from NumPy's legacy global RNG, which `seed_everything`
  seeds, so it is reproducible (it is the one legacy strategy still reading that ambient state; do not copy that
  pattern into new code).
- **Auxiliary networks and loaders of a strategy (2026-11-11, ADR 0015)**: a strategy that builds its own
  network (a loss-prediction module, a VAE, a discriminator, a critic) constructs it inside
  `dalmax.seeding.isolated_torch_seed(derive_seed(config.seed, "<tag>"))`: the CPU torch RNG is forked and seeded
  from the sub-seed, so the auxiliary weights never consume the run's global torch stream and the task model's
  initialisation and data order stay exactly where they would be without the strategy (build on CPU inside the
  block, move to the device afterwards). Every other draw it makes is explicit: `dalmax.seeding.torch_generator(seed)`
  for noise (e.g. the VAE reparameterisation) and an explicit `generator=` on every auxiliary `DataLoader` (never the
  ambient default generator). Tags in use: see `.specs/adr/0015-deepalplus-strategies.md`. New strategies follow the
  same pattern; a strategy that only ranks the task model's outputs needs none of this.
- **Per-round history (2026-11-11, ADR 0016)**: every run also writes `round_predictions.json` (the test predictions of
  every round) and `round_queries.json` (the pool indices labeled in every round, with labels and image paths), so a
  later finding about the test set or the pool can be re-scored from the saved runs instead of re-running them.
- **Data order is part of the run's identity (2026-11-12, KI-47)**: a pool or test index means "the image at that
  position of the loader's list", so the order must never depend on the machine. Contract
  (`dalmax/data_order.py::ORDER_CONTRACT`): classes = sorted folder names, files = plain lexicographic sort of the
  names in each class folder (`dalmax/data/loaders.py::list_image_names`), for train and test. Any new loader that
  lists a directory follows it. `run_metadata.json` records `data_order` (hashes of the ordered relative paths of
  both splits); `campaign verify` FAILs when the leaves of one dataset disagree on them, and the seed audit compares
  the round-0 image PATHS (`round_queries.json`), not only indices. Before KI-47 the files came from an unsorted
  `os.listdir`, whose order differed between Colab sessions: equal indices named other images (first campaign).
  Compare runs by `dalmax.data_order.relative_image_path` (`<class>/<file>`), never by index alone.
- **One code version per campaign (2026-11-12, KI-48)**: every Colab session checks out the campaign's tag (`GIT_REF`,
  default `campaign-fixed-v1`) detached, never pulls a branch; `campaign verify` lists the distinct `run_metadata.json`
  `git_commit` values of the OK leaves and a second value (or a missing one) is a WARN that must be explained before
  the results are used. A leaf counts as complete only when its JSON artifacts parse with the expected round count
  and its checkpoint is non-empty (`dalmax/reporting/leaf_check.py::LeafCheck`), and a job's leaf is looked up only
  under the dataset folder of its own params (`RunGroup.dataset_folder`).
- `torch.backends.cudnn.enabled = False` was used as a determinism
  shortcut in the historical `demo.py:59` (line no longer exists); it globally disabled cuDNN and cost performance on the
  the lab GPUs. **Now uses** `torch.backends.cudnn.deterministic = True` +
  `torch.backends.cudnn.benchmark = False` instead (`dalmax/seeding.py::seed_everything`).
  Deliberate consequence: post-refactor GPU runs are not bit-identical to historical GPU runs at the
  same seed — cuDNN's deterministic kernels differ from the non-deterministic ones used previously.
  CPU runs (the golden-run fixtures) are unaffected.

- **GPU determinism settings (2026-09-29, honest version 2026-09-29b)**: `seed_everything` additionally sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` (`os.environ.setdefault`, an operator-set value is kept; the
  campaign runner also exports it into the `trainer.py` subprocess),
  `torch.use_deterministic_algorithms(True, warn_only=True)` (ops without a deterministic kernel warn
  instead of crashing a long run) and `torch.utils.deterministic.fill_uninitialized_memory = False`.
  `run_metadata.json` records a `determinism` block (`dalmax/seeding.py::determinism_state`): `seed`,
  `deterministic_algorithms` (the string `"warn_only"`, not a bool), `cudnn_deterministic`,
  `cudnn_benchmark`, `cublas_workspace_config`, `fill_uninitialized_memory`, and
  `nondeterministic_op_warnings` -- the unique ops torch warned about during the run
  (`dalmax/seeding.py::record_nondeterministic_ops`, filled by `ExperimentRunner.run()` when the run
  ends). **Honest caveat**: `warn_only` means determinism on GPU is **best-effort**, not guaranteed:
  identical seeds give identical initial labeled sets, selection randomness and data order everywhere
  (checked by the campaign's seed audit), and bit-identical metrics are guaranteed only on CPU (golden
  tests); on GPU a non-empty `nondeterministic_op_warnings` list means some op had no deterministic
  kernel, and results may differ run to run and across GPU models/drivers. Papers must report the
  seed-averaged numbers, not claim bit-reproducibility on GPU. The campaign's `verify` runs the
  seed-consistency audit (every run of a seed starts from the identical initial labeled IMAGES, compared by the
  round-0 paths of `round_queries.json`, by the logged indices for older leaves; nothing readable = WARN,
  mismatch = FAIL) and the data-order audit (KI-47) -- see `.specs/experiments/campaign.md`.

- **Hardware-agnostic (2026-09-29, ADR 0010)**: no GPU model name (nor a GPU-specific results directory
  or config name) may appear in code, configs, scripts, Makefile, tests, notebooks, runbooks,
  README/CLAUDE/AGENTS, `.claude/` or `.specs/` (except clearly past-tense execution-history records).
  The GPU actually used is recorded only at runtime, in each run's `run_metadata.json`
  `environment.gpus[*].name` (`dalmax/experiment/environment.py`, no hardcoded defaults); docs use the
  placeholder `<gpu-name>` and tell the operator to check `environment.gpus[0].name` in the first
  run. Tests that need a GPU name use a fake one (`"NVIDIA Fake GPU 40GB"`). Gate: `tests/test_hardware_agnostic.py` greps every tracked file (excluding `poetry.lock`,
  `phd_files/`, `paper_drafts/`, `docs/results/`, `results/`, and past-tense records under `.specs/`) for
  the common GPU model names and fails on any hit.

## Config snapshot

- **Resolved (2026-08-23, Phase 2).** `dalmax/experiment/run_metadata.py::write_run_metadata`
  writes `run_metadata.json` into every results directory
  (`{dir_results}/{dataset}/SEED_{seed}/NQ_{n_query}_NIL_{n_init}_NR_{n_round}_NE_{n_epoch}/{strategy}/`),
  containing the fully-resolved config (`to_dict(config)`), a best-effort `git_commit`, Python/torch
  versions, CUDA availability, start timestamp, and an `environment` block (OS, CPU/RAM, GPU model/memory/driver, CUDA/cuDNN, Colab flag; `dalmax/experiment/environment.py`, KI-34; never the hostname) — called from `ExperimentRunner.run()` before the
  round loop starts. `results.json`, `predictions.csv`, plots, and `log-dalmax.log` are unchanged;
  `run_metadata.json` is the new artifact that closes the gap this section originally flagged (see
  `.specs/quality/known-issues.md` KI-1, and `current-state.md` §8 for the exact field list).

## Embedding cache discipline

**Phase 2 landed (2026-08-23)**: `dalmax/embeddings/cache.py::EmbeddingCache` is now the live cache
for any run through `dalmax.cli`/`trainer.py`, keyed on `(dataset, extractor, Q, variant, split,
pool_hash)` — filename `results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{pool_hash}.pkl`.
This closes both gaps this section originally flagged: `embedding_variant` is now a key component
(only `"full"` is ever cached; `"spatial"`/`"spectral"` are sliced from it on load via
`dalmax/embeddings/variants.py::slice_embedding`, never separately cached or recomputed), and `Q` is
config-driven (`EmbeddingConfig.q`, defaulted per-extractor by `dalmax/config/loader.py` —
`ssrae`→`13`, `vctex`→`(5,17)`, `resnet_imagenet`→`None`), not a hardcoded literal. **Never reuse an
embedding cache across a different `(dataset, extractor, Q, pool, variant)` combination** — the
keying above makes this impossible by construction as long as the key components are computed
correctly; still sanity-check the actual filename before trusting an ablation's cached embeddings.

**Image identity (2026-11-12, KI-47).** `pool_hash` hashes pool INDICES, which name images only under one data
order; the pre-fix key let later Colab sessions load features computed under another order (the first campaign's
KMH sweep, `texhal_q17`, `hier_transfer`, paper-1 KMH nq50/nq10). The key now also carries `order_hash`
(`dalmax.data_order.ordered_paths_hash` of the train split): filename
`{dataset}__{extractor}__Q{q}__{variant}__{split}__order{order_hash}__pool{pool_hash}.pkl`, so no file written
before the fix is ever found. The payload (format `dalmax-embedding-cache` version 2) stores the key, the ids and the
`<class>/<file>` path of every cached image; `EmbeddingCache.load` refuses any mismatch (`EmbeddingCacheError`) and
`compute_pool_embeddings` then logs a warning and recomputes, as it does for an unreadable/truncated file. Writes
are atomic (temporary file in the same directory + `os.replace`). New caching code must store and check the identity
of what it caches the same way, never rely on positional ids alone.

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
