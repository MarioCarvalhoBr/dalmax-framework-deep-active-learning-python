# Reproducibility rules

Expanded version of the standing rule (mirrored in `.claude/rules/reproducibility.md`,
owned by a different batch): every experiment must be fully determined by
**(params JSON + CLI args + seed + git commit)**.

**Phase 2 status (2026-08-23)**: every gap this file originally documented is now resolved for the
live `dalmax.cli`/`demo.py` (historical) path — see the "Resolved" notes inline below and
`.specs/architecture/current-state.md` §5 for the full coupling-point-by-coupling-point account.
The original violation text is kept for the historical record in each subsection. **Phase 4
(2026-08-23)** then physically deleted the legacy files these subsections cite as "unchanged, now
dead code" — `core/` and `utils/` no longer exist; see the inline notes below and ADR 0002's final
amendment.

## Seed policy

- `--seed` (CLI, default 1) is the single source of randomness for a run.
  **Phase 2**: `np.random.seed`/`torch.manual_seed`/etc. are no longer scattered across `demo.py` (historical) —
  `dalmax/seeding.py::seed_everything(seed)` is the single place that seeds `random`, NumPy's legacy
  global RNG, and PyTorch (CPU + all CUDA devices), called exactly once by
  `dalmax.experiment.runner.ExperimentRunner.run()` before any dataset/model/strategy object is
  constructed — same ordering guarantee the original text described for `demo.py` (historical), now centralized.
  `Data.initialize_labels`'s `np.random.shuffle` for the initial labeled pool is unchanged and still
  correctly seed-derived.
- **Violation found, now resolved**: `core/query_strategies/ssrae_kmeans_sampling.py:23`
  and `core/query_strategies/vctex_kmeans_sampling.py:26` both called
  `KMeans(n_clusters=n, random_state=3, n_init=10)` — a **literal hardcoded
  seed**, independent of `--seed`. These two files exhibited this bug until they were unreachable
  (Phase 2) and then **deleted outright (Phase 4)** — they no longer exist on disk. Their replacement,
  `dalmax/selection/flat_kmeans_closest.py::FlatKMeansClosest`, derives `KMeans(random_state=...)`
  from `np.random.default_rng(dalmax.seeding.derive_seed(config.seed, "selection"))` — verified by
  the regenerated `tests/golden/ssrae_kmeans_micro_seed1.json` fixture, whose `round_1_query_idxs_sorted`
  changed once seed-derivation replaced the literal `3` (old value kept in that fixture's
  `legacy_phase1_values` block).
- `SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling`
  (was `core/query_strategies/ssl_ssrae_sampling.py`, deleted in Phase 4) delegated clustering to
  what is now `dalmax/tools/SSL/src/hierarchical_kmeans_gpu.py` (moved, not deleted, in Phase 4) —
  **resolved (Phase 2)**: its replacement, `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection`,
  documents exactly how the vendored pipeline's randomness works (module docstring, "RNG isolation"
  section): `hierarchical_sampling`'s core randomness and `kmeans_gpu.kmeans`'s `random_state=None`
  resolution both consult **global** NumPy legacy random state (not an injectable generator), so
  `HierarchicalKMeansSelection.select` reseeds `random.seed`/`np.random.seed` from a value derived
  from the caller's `rng` immediately before calling into the vendored code — deterministic given the
  same experiment seed, documented as a deliberate, bounded global-state coupling rather than hidden.
- `torch.backends.cudnn.enabled = False` (`demo.py` (historical)) trades GPU performance
  for determinism — **resolved (Phase 2)**: `demo.py` (historical)'s line no longer exists (12-line shim);
  `dalmax/seeding.py::seed_everything` sets `torch.backends.cudnn.deterministic = True` +
  `torch.backends.cudnn.benchmark = False` instead, keeping cuDNN's faster kernels while staying
  deterministic. Deliberate consequence: post-refactor GPU runs are **not** bit-identical to
  historical GPU runs at the same seed (cuDNN's deterministic kernels differ from the
  non-deterministic ones used previously) — the CPU golden-run fixture is unaffected, since cuDNN
  never applies on CPU. See `.specs/architecture/refactor-plan.md` Phase 2 risks.

## Cache policy

**Phase 2 resolved this for the live path**: `dalmax/embeddings/cache.py::EmbeddingCache` keys every
embedding cache file on `(dataset, extractor, Q, variant, split, pool_hash)` —
`results/cache/embeddings/{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{pool_hash}.pkl`.
Only the `"full"` variant is ever written; `"spatial"`/`"spectral"` are always derived from a loaded
`"full"` entry via `dalmax/embeddings/variants.py::slice_embedding`, never recomputed or separately
cached — so running all three §6.1 ablation variants back-to-back extracts SSRAE features exactly
once and cannot cross-contaminate. `Q` is config-driven (`EmbeddingConfig.q`), not a hardcoded
literal, for any run through `dalmax.cli`. **Never reuse an embedding cache across a different
`(dataset, extractor, Q, pool, variant)` combination** — the keying above makes this impossible by
construction (a different combination is a different filename), but always sanity-check the
filename actually used before trusting an ablation's cached embeddings.

The original (pre-Phase-2) description below is kept for historical context — it describes
`utils/data.py`'s caches (that file was deleted in Phase 4, along with the
`compute_legacy_features=True` code path that could still reach them; see
`.specs/architecture/current-state.md` §5 and ADR 0002's final amendment). The original
fixed-path files (`results/features_dict_ssrae.pkl`, `results/features_dict_vctex.pkl`,
`results/Y_train.pkl`) remain on disk, orphaned, not deleted.

`utils/data.py` (deleted in Phase 4) maintained three pickle caches, all under `results/`, with
**no cache key** encoding what was used to produce them:

| Cache file | Written by | Keys it should have but doesn't |
|---|---|---|
| `results/features_dict_ssrae.pkl` | `Data.create_feature_maps_ssrae` | dataset, `Q` (currently 13), embedding_variant |
| `results/features_dict_vctex.pkl` | `Data.create_feature_maps_vctex` | dataset, `Q` (currently `[5,17]`) |
| `results/Y_train.pkl` | `Data.__init__` | dataset |

Each is written **only if the file does not already exist**
(`if not os.path.exists(path_pkl): ...`) — meaning:

- Changing `Q` (e.g. per the advisor's "fix Q per the SSRAE article" request
  — see `experiments/ablation-study.md` §6.1) **silently reuses the stale
  cache** computed at the old `Q` unless the `.pkl` file is deleted by hand
  first. There is no code-level guard against this.
- Switching dataset (`DANINHAS` ↔ `CIFAR10`) with the SSRAE/VCTex strategies
  selected would similarly reuse whatever `features_dict_*.pkl` happens to
  exist, if it does — a stale-cache hazard confirmed by reading the code
  (not merely inferred).
- **Rule going forward**: any change to `Q`, the SSRAE/VCTex extractor
  implementation, the dataset, or (once implemented) the
  `embedding_variant` must either (a) delete the corresponding `.pkl`
  before running, or (b) — the correct fix, required before the ablation
  study can run its three representation variants without manual
  intervention between them — key the cache filename/dict by
  `(dataset, extractor, Q, embedding_variant)` and never silently reuse a
  cache computed under a different key.
- `results/Y_train.pkl` has no dataset key either; same rule applies when
  switching datasets.

## Config snapshot per run

**Resolved (2026-08-23, Phase 2).** `dalmax/experiment/run_metadata.py::write_run_metadata` now
writes `run_metadata.json` into every results directory (called from `ExperimentRunner.run()` before
the round loop starts), containing the fully-resolved config (`to_dict(config)` — every field, not
just the subset `results.json` records), a best-effort `git_commit` (`git rev-parse HEAD`, `None` if
unavailable), Python/torch versions, CUDA availability, the UTC start timestamp, and (2026-09-29, KI-34) a top-level `environment` block (`dalmax/experiment/environment.py`): `os`, `machine` (CPU model/count, RAM GB), `python`, `torch` (version/CUDA/cuDNN), `gpus[]` (name, memory GB, compute capability, SM count, driver, nvidia-smi name/MiB), `cuda_visible_devices`, `current_device`, `runtime.is_colab`; hostname is never recorded. This closes
every gap the original text below (kept for historical context) described. `results.json` itself is
also more complete now: it gained `all_precision_macro`/`all_recall_macro`/`all_f1_macro` (additive,
see `research-rules/metrics.md`), though it still does not duplicate the full params JSON —
`run_metadata.json` is the file to consult for that, not `results.json`.

Original text (pre-Phase-2): **Not currently implemented.** `demo.py` (historical) logs the full `args` and the
selected dataset's `params` block to `log-dalmax.log` via
`logger.warning(json.dumps(...))` (visible in `dir_results/log-dalmax.log`
after the run), and `results.json` records `dataset_name, strategy_name,
n_init_labeled, n_query, n_round, seed`, but:

- `n_epoch` and the rest of the params JSON (optimizer args, `config_kmh`,
  etc.) are **not** duplicated into `results.json` itself — they are only
  recoverable from the run's log file or by knowing which params JSON was
  used for that batch.
- **No git commit hash is recorded anywhere in the results.** This is
  required by the standing rule and is currently a gap: add
  `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` (or equivalent) to
  `dados_config_results` in `demo.py` (historical) before any ablation runs are executed,
  so every `results.json` is traceable to the exact code state that produced
  it. Track this as a refactor-plan Phase 1 item (owned by another batch).

## What "fully determined" means in practice today

**For any run through `dalmax.cli`/`demo.py` (historical) from Phase 2 onward**: open that run's
`run_metadata.json` — it has the fully-resolved config (dataset/embedding/selection/device/seed/
etc.) and the git commit hash directly, no inference needed.

**For pre-Phase-2 runs** (everything under `results/dalmax{1,2}/` and earlier), the original
limitation still applies: to reproduce a given `results.json` leaf directory exactly, you need the
params JSON file used (identified only by filename convention, e.g. `files_config/benchmark/params_df_gpu_0.json` vs
`_1.json` — not embedded in the results), the exact CLI invocation (seed, n_query, n_round,
dataset_name, strategy_name, dir_results — n_init_labeled and n_epoch are NOT in `results.json`,
must be inferred from the directory name / params file), and a best-effort guess at which git commit
produced it (check file timestamps against `git log` as a fallback, since these runs predate
`run_metadata.json`).
