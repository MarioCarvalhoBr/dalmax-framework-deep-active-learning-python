# Known Issues

All items below were originally verified by direct code inspection on 2026-08-23 (paths/lines
referred to the state at commit `554a205` and earlier, when the code still split across `core/` and
`utils/`). **Phase 4 (2026-08-23, branch `refactor/phase-4-package`) physically moved everything
under `dalmax/` and deleted the files that had already been superseded** — every row below has been
re-verified against that new reality: rows whose only blocker was "the legacy file is unreachable
but still on disk" are now marked **Resolved by deletion**, and rows describing files that were
moved (not deleted) have had their paths updated to the `dalmax/` location, with the underlying
issue's open/resolved status unchanged. "Phase" refers to `.specs/architecture/refactor-plan.md`.
Severity: **High** (correctness/reproducibility risk or blocks an ablation), **Medium** (design
debt, works today but limits scalability), **Low** (cosmetic/mechanical).

> **Standing note (2026-08-23): historical `saved_model.pth` files contain no weights.**
> KI-22 below documents a confirmed bug: `DeepLearning.save_model` saved the trained model's
> *class*, not its trained weights, until fixed in this session (ADR 0006,
> `dalmax/models/checkpoint.py`). Every `saved_model.pth` written before 2026-08-23 —
> **including every run under `results/dalmax1/` and `results/dalmax2/`** — is a ~900-byte
> pickled class reference with zero weights and **cannot be recovered**. If a specific historical
> model's weights are needed, it must be **re-trained** (its exact params JSON + CLI args + seed
> are recorded in that run's `run_metadata.json`, so it is fully reproducible from scratch).
> `results.json`, `predictions.csv`, `run_metadata.json`, and the plots from those runs are
> **entirely unaffected** — they were never on the buggy code path — only the weight checkpoint
> itself is lost.

## Seed list from `.specs/history/prompt-master.md` §8.5 — verification status

| # | Issue | Verified? | Evidence | Severity | Fixed in |
|---|---|---|---|---|---|
| KI-1 | `demo.py` (historical) is a monolith mixing CLI, training loop, plotting, and persistence | **Resolved in `dalmax` (2026-08-23, Phase 2)** | `demo.py` is now a 12-line shim (`from dalmax.cli import main`). `dalmax/cli.py` does argparse only; `dalmax/experiment/runner.py::ExperimentRunner` is the round loop; `dalmax/experiment/reporter.py::write_report` is plotting/JSON/CSV persistence. Each is independently unit-testable (`tests/test_runner_n_round_zero.py`, `tests/test_registry.py`) | Medium → **Resolved** | Phase 2 (`experiment/runner.py` + `experiment/reporter.py` + thin `cli.py`) |
| KI-2 | `pandas` imported in `demo.py`/`utils/report/` but was absent from the (since-retired) `requirements.txt` export | **Fully resolved** | `pyproject.toml` declares `pandas==2.2.3`; `dalmax/experiment/reporter.py` (the Phase 2 successor to `demo.py`'s persistence code) also imports `pandas` and installs correctly via `poetry install`. The `requirements.txt` export this row originally referred to was retired entirely (2026-08-23, ADR 0001 amendment) — Poetry is now the only install path, so there is nothing left to keep in sync | High → **Resolved** | Phase 1 (historical fix); requirements.txt export retired (2026-08-23) |
| KI-3 | Pickle caches hardcoded in `utils/data.py` with no cache key for dataset/Q/pool/variant | **Fully resolved (2026-08-23, Phase 2 fix; Phase 4 deleted the superseded legacy path entirely)** | `dalmax/embeddings/cache.py::EmbeddingCache` keys every cache file on `(dataset, extractor, Q, variant, split, pool_hash)` — filename `{dataset}__{extractor}__Q{q}__{variant}__{split}__pool{pool_hash}.pkl` under `results/cache/embeddings/` (since 2026-11-12, KI-47, also on the split's `order_hash`, filename `...__{split}__order{order_hash}__pool{pool_hash}.pkl`, with the image paths stored and validated on load: ids alone did not identify images). Only the `"full"` variant is ever cached; `spatial`/`spectral` are sliced from it on load (`dalmax/embeddings/variants.py`), never separately cached, so they cannot collide. `Q` is config-driven (`EmbeddingConfig.q`, defaulted per-extractor by `dalmax/config/loader.py`), never a hardcoded literal. The Phase 1 `utils/data.py::cache_file_path` path (keyed on dataset/Q/pool, no `embedding_variant`) was **deleted in Phase 4** along with the rest of `utils/data.py` (683 lines) — it had been dead code since Phase 2 (`current-state.md` §0 at the time), so nothing was lost by deleting it. The original unkeyed `results/features_dict_ssrae.pkl`/`results/features_dict_vctex.pkl` remain on disk as **lab-machine housekeeping, not code, debt** — orphaned, not read/written by anything in `dalmax/`, not deleted per `.claude/rules/data-safety.md` (never delete `results/` content automatically). Unit-tested in `tests/test_embedding_cache.py`; `tests/test_cache_paths.py` (which tested the now-deleted legacy path) was updated accordingly. | High → **Resolved** | Phase 1 (cache-path keying: dataset/Q/pool); Phase 2 (`EmbeddingProvider` + `EmbeddingCache`, ADR 0003, `embedding_variant` + config-driven `Q`); Phase 4 (legacy `utils/data.py` path deleted) |
| KI-4 | `ssl_ssrae_sampling.py` hardcodes `self.params['DANINHAS']` | **Resolved (2026-08-23, Phase 2 fix; Phase 4 deleted the file that had the literal)** | The file that had the literal, `core/query_strategies/ssl_ssrae_sampling.py`, no longer exists — deleted in Phase 4 (it had been dead code/unreachable since Phase 2). Its replacement, `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection`, takes `hierarchy` via constructor injection, resolved by `dalmax/config/loader.py` per the actual `dataset_name` argument — no hardcoded key anywhere. Running `SSRAEKmeansHCSampling` against `CIFAR10` raises a clear `dalmax.config.schema.ConfigError` (naming the missing `hierarchy`/`config_kmh` key) instead of `KeyError: 'DANINHAS'`, if that dataset's params JSON entry has no hierarchy block — see KI-23 (still open: the params files themselves need a `CIFAR10.config_kmh`/`selection` block added before a real CIFAR10 hierarchical run, a content gap, not a code gap) | High → **Resolved** (code); KI-23 (params content) remains open | Phase 2 (config layer); Phase 4 (legacy file deleted) |
| KI-5 | `SSRAEKmeansSampling` hardcodes `KMeans(random_state=3)` | **Resolved (2026-08-23, Phase 2 fix; Phase 4 deleted the files that had the literal)** | `core/query_strategies/ssrae_kmeans_sampling.py`/`vctex_kmeans_sampling.py` (the files with the literal) were deleted in Phase 4, having been dead code since Phase 2. Their replacement, `dalmax/selection/flat_kmeans_closest.py::FlatKMeansClosest`, derives `KMeans(random_state=...)` from `rng.integers(...)`, where `rng` is seeded via `dalmax.seeding.derive_seed(config.seed, "selection")` — verified empirically by regenerating `tests/golden/ssrae_kmeans_micro_seed1.json` (`round_1_query_idxs_sorted` changed from the old `[5,21,37,45,47]`, tied to the literal `3`, to `[14,24,37,45,47]` at `--seed 1`; the old value is preserved in the fixture's `legacy_phase1_values` block, not lost) and by `tests/test_selection_flat.py` | High → **Resolved** | Phase 2 (seed propagation audit); Phase 4 (legacy files deleted) |
| KI-6 | Strategies mutate `dataset.features_dict` (deleting selected ids) as a side effect | **Resolved (2026-08-23, Phase 2 fix; Phase 4 deleted the files that had the mutation)** | The three `del features_dict[img_id]` call sites (in `ssrae_kmeans_sampling.py`, `vctex_kmeans_sampling.py`, `ssl_ssrae_sampling.py`) were deleted in Phase 4 along with those files, having been dead code since Phase 2. `dalmax/query_strategies/representation.py::RepresentationStrategy` never touches `dataset.features_dict` at all — it owns its own `{id: embedding}` map populated once from `EmbeddingCache` and re-derives the current unlabeled subset from `dataset.labeled_idxs` on every `query()` call, non-destructively (idempotent given the same labeled/unlabeled state) | Medium → **Resolved** | Phase 2 (`SelectionStrategy` returns ids; caller owns pool bookkeeping, not the strategy); Phase 4 (legacy files deleted) |
| KI-7 | `demo.py` (historical) injects params via `setattr(strategy, "params", params)` instead of the constructor | **Resolved (2026-08-23, Phase 2)** | The `setattr` line no longer exists — `demo.py` is now a 12-line shim with no strategy construction of its own. `dalmax/query_strategies/registry.py::build_strategy` constructs every strategy with everything it needs at `__init__` time (`RepresentationStrategy(dataset, net, config, logger, embedding_provider=..., selection=..., rng=...)` for representation strategies; `LegacyClass(dataset, net, logger)` unchanged for the 12 baseline strategies, which never needed `params` anyway). `SSLStrategy.__init__`'s defensive `hasattr(self, 'params')` check no longer exists on disk — that class and its file were deleted in Phase 4 | High → **Resolved** | Phase 2 (constructor injection); Phase 4 (legacy file deleted) |
| KI-8 | `torch.backends.cudnn.enabled = False` globally disables cuDNN | **Resolved (2026-08-23, Phase 2)** | `demo.py:59`'s line no longer exists (shim). `dalmax/seeding.py::seed_everything` sets `torch.backends.cudnn.deterministic = True` / `benchmark = False` instead, called once per run from `ExperimentRunner.run()` before any dataset/model/strategy object is constructed. Deliberate consequence: post-refactor GPU runs are not bit-identical to historical GPU runs at the same seed (cuDNN's deterministic kernels differ from the previously-used non-deterministic ones) — the CPU golden-run fixture is unaffected, cuDNN never applies on CPU | Medium → **Resolved** (perf gain on 10 GB lab GPUs; GPU bit-identity to pre-refactor runs is knowingly given up, see `refactor-plan.md` Phase 2 risks) | Phase 2 (`seeding.py`: `cudnn.deterministic=True`/`benchmark=False` instead) |
| KI-9 | `utils/orchestrator.py` uses if/elif chains → replace with registries | **Fully resolved (2026-08-23, Phase 2 fix; Phase 4 deleted the file)** | `utils/orchestrator.py` no longer exists — deleted in Phase 4, having been dead code/unreachable since Phase 2 (nothing in `dalmax/` ever imported `utils.orchestrator`). Replaced by `dalmax/data/registry.py::DATASET_REGISTRY`, `dalmax/models/registry.py::MODEL_REGISTRY`, `dalmax/query_strategies/registry.py::STRATEGY_REGISTRY`, `dalmax/embeddings/registry.py::EMBEDDING_REGISTRY`, and `dalmax/selection/registry.py::SELECTION_REGISTRY` — all plain dicts with a `get_*` lookup raising a clear `KeyError`/`ConfigError` listing valid names. `tests/test_strategy_registry.py`/`tests/test_registry.py` (parsing `dalmax/cli.py`'s `choices=[...]` via `ast`, without importing it) catch registry/CLI drift automatically; `tests/test_registry.py` no longer covers the deleted legacy module | Medium → **Resolved** | Phase 2 (registries); Phase 4 (`utils/orchestrator.py` deleted) |
| KI-10 | Dead/experimental files at root | **Resolved by deletion (2026-08-23, Phase 4)** | All of the following were confirmed dead in earlier phases and are now **deleted from the repository**: `temp_teste.py`, `test.py` (root, not a pytest file), `TRASH_TEXT.md`, `sampled_data.pdf`, `TODO.md`, `core/query_strategies/old_functions.py`, `demo_ssl.py`, and `code_kmh.py` (the vendored SSL exploratory script `demo_ssl.py` imported — see KI-27). The four Phase-2-identified dead files/branches (`utils/orchestrator.py`, `core/query_strategies/ssrae_kmeans_sampling.py`, `vctex_kmeans_sampling.py`, `ssl_ssrae_sampling.py`) are likewise deleted, along with `utils/data.py`'s `create_feature_maps_ssrae`/`create_feature_maps_vctex`/`plot_features_tsne_*` methods (deleted with the whole file). Verified: none of these paths exist on disk any more (`ls core utils` → "No such file or directory"). Local orphaned `results/*.pkl` scratch files (not real experiment output) were also removed as part of this cleanup | Low → **Resolved** | Phase 4 (dead-code removal, `.specs/architecture/refactor-plan.md` Phase 4) |
| KI-11 | Logging misuse: informational messages emitted at `warning` level | **Still open (2026-08-23, moved not fixed by Phase 4)** | The original `demo.py` (historical)/`core/query_strategies/strategy.py`/`ssl_ssrae_sampling.py` call sites are gone along with those files, but the pattern itself moved with the logic, not fixed: `logger.warning(...)` for routine informational output is now found in `dalmax/cli.py`, `dalmax/query_strategies/base.py` (the moved `Strategy` base class), `dalmax/query_strategies/representation.py`, and `dalmax/experiment/runner.py` (confirmed by direct grep, 2026-08-23) | Low | Phase 4/future (introduce `logger.info` and reserve `.warning` for actual anomalies) |
| KI-12 | Mixed Portuguese/English comments; no type hints; no tests; heavy result artifacts inside the git repo | **Partially confirmed, path updated (2026-08-23)** | Portuguese comments confirmed still present in `dalmax/data/datasets.py` (was `utils/data.py`), e.g. `"# Salva em um arquivo indices.txt o indice (id) da imagem, classe_nome, e seu caminho"`; `demo.py` itself is now a 12-line shim so its own Portuguese comments are gone with it, but any that moved into `dalmax/experiment/reporter.py`/`runner.py` were not specifically re-audited in this pass. No type hints confirmed in `dalmax/data/datasets.py`, `dalmax/models/base.py`, `dalmax/query_strategies/*.py` (moved unchanged, so pre-existing lack of type hints moved with them). `tests/` now exists (277 tests, `-m "not gpu and not dataset and not slow"`), closing that part of the original claim. **`results/` artifacts inside git: still not tracked** — `results/`, `DATA/`, `local_assets/`, `logs/` remain gitignored and 0 files under `results/` are tracked | Low (comments/type hints) | Phase 2/4 (tests landed); comments/type hints not yet scheduled |

## Additional issues found while reading the code (not in the original seed list)

| # | Issue | Evidence | Severity | Fixed in |
|---|---|---|---|---|
| KI-13 | Hierarchical selection hardcodes `device="cuda"`, no CPU fallback | `core/query_strategies/ssl_ssrae_sampling.py:85` originally: `data=torch.tensor(data, device="cuda", dtype=torch.float32)` — that file was deleted in Phase 4, having been dead code since Phase 2 | High → **Resolved (2026-08-23, Phase 2 fix; Phase 4 deleted the legacy file)** — `dalmax/selection/hierarchical_kmeans.py::HierarchicalKMeansSelection(hierarchy, device="cpu", ...)` takes `device` from `ExperimentConfig.device` (`--device {auto,cuda,cpu}`). CPU support was verified empirically: the vendored `hierarchical_kmeans_with_resampling`/`hierarchical_sampling` pipeline has no hardcoded `.cuda()` call anywhere in its call graph; the sole hardcoded-`cuda` default left in that source tree, `kmeans_gpu.sort_cluster_by_distance` (now `dalmax/tools/SSL/src/kmeans_gpu.py`, was `core/tools/SSL/src/kmeans_gpu.py:407`), is a function this pipeline never calls. `tests/test_selection_hierarchical.py` runs on CPU unconditionally as a result | Phase 2 (`HierarchicalKMeansSelection` takes device from `ExperimentConfig`); Phase 4 (legacy file deleted) |
| KI-14 | `calc_metrics_sklearn` uses `average='weighted'`, but the ablation spec requires macro F1 | Was `utils/data.py:281-295`; `utils/data.py` was deleted in Phase 4, but `calc_metrics_sklearn` itself moved (not deleted) — it now lives at `dalmax/data/datasets.py::Data.calc_metrics_sklearn`, still `average='weighted'`, still unused by any live code path | High → **Resolved additively (2026-08-23, Phase 2 fix, unaffected by Phase 4's move)** — `dalmax/data/datasets.py::Data.calc_metrics` (added alongside the untouched `calc_metrics_sklearn`, both now in the same moved file) returns both weighted and macro precision/recall/F1 in one dict. `dalmax/experiment/runner.py::ExperimentRunner._record_round` persists both; `results.json` has `all_precision_macro`/`all_recall_macro`/`all_f1_macro` in addition to the legacy weighted keys — see `research-rules/metrics.md` for which to report in the ablation tables | Phase 2 (add `average='macro'`, additive, done); Phase 4 (file moved, both methods intact) |
| KI-15 | Duplicate imports in `core/daninhas_model.py` | Lines 1-8: `import torch` (lines 1 and 6), `import torch.nn as nn` (lines 2 and 7), `from torchvision.models import vit_b_16, ViT_B_16_Weights` (lines 4 and 8) | Low → **Resolved by deletion (Phase 4)** — the file was moved+cleaned into `dalmax/models/daninhas_resnet50.py`, which has no duplicate imports (confirmed by direct grep) | Phase 4 (`mechanic`, done as part of the move) |
| KI-16 | Duplicate imports in `core/cifar10_model.py` | Lines 1-9: `import torch` (lines 1 and 8), `import torch.nn as nn` (lines 2 and 9) | Low → **Resolved by deletion (Phase 4)** — moved+cleaned into `dalmax/models/cifar10_cnn.py`, no duplicate imports (confirmed) | Phase 4 (`mechanic`, done as part of the move) |
| KI-17 | Duplicate imports in `core/query_strategies/ssl_ssrae_sampling.py` | `import numpy as np` (lines 2 and 8), `import matplotlib.pyplot as plt` (lines 3 and 13) | Low → **Resolved by deletion (Phase 4)** — the file itself was deleted; its replacement, `dalmax/query_strategies/representation.py`, has no duplicate imports (confirmed) | Phase 4 (legacy file deleted) |
| KI-18 | Unused imports in `core/query_strategies/ssl_ssrae_sampling.py` | `from abc import abstractmethod` (line 1, never used — no `@abstractmethod` decorator anywhere in the file), `from sklearn.manifold import TSNE` (line 12, never called), `import matplotlib.colors as mcolors` (line 14, never referenced) | Low → **Resolved by deletion (Phase 4)** — file deleted; replacement has no such unused imports | Phase 4 (legacy file deleted) |
| KI-19 | Unused imports in `core/query_strategies/ssrae_kmeans_sampling.py` | `import matplotlib.pyplot as plt` (line 2, never called), `import time` (line 5, never used) | Low → **Resolved by deletion (Phase 4)** — file deleted | Phase 4 (legacy file deleted) |
| KI-20 | Unused imports in `core/query_strategies/vctex_kmeans_sampling.py` | `import matplotlib.pyplot as plt` (line 2, never called), `import time` (line 5, never used) | Low → **Resolved by deletion (Phase 4)** — file deleted | Phase 4 (legacy file deleted) |
| KI-21 | `Data.__init__` has filesystem side effects at object-construction time | Was `utils/data.py`; the same logic (writes `results/cache/Y_train_{dataset_folder}.pkl` unconditionally on first construction via `cache_file_path("Y_train", ...)`, and `create_indexes_path` writing `results/original_indices.txt` from `__init__`) now lives at **`dalmax/data/datasets.py`** (moved unchanged in Phase 4, confirmed present) | Medium — still open; makes `Data` impossible to unit-test without touching disk, and doubles as another cache write at construction time (see KI-3) | Phase 2/4 not scoped to fix this; still future work (move persistence out of `__init__` into an explicit method). Observed consequence 2026-11-08: `make campaign-smoke-kmh` (micro dataset) overwrote `results/original_indices.txt` with the micro listing, so the file no longer describes `daninhas_full`; the thesis script `run_log_stats.py` detects this and falls back to size checks. Planned fix: key the file by dataset folder (`results/original_indices_{dataset_folder}.txt`) and update that reader |
| KI-22 | `DeepLearning.load_model` is marked `# TODO: FIX THIS` by the original author — and the deeper bug it was hiding: `DeepLearning.save_model` did `torch.save(self.net, path)`, saving the model *class* (`self.net`), never the trained instance (`self.clf`) | **Confirmed and resolved (2026-08-23)**: re-running `trainer.py` (historical `demo.py`) on the micro dataset and inspecting the resulting `saved_model.pth` showed a ~900-byte file that unpickles to `<class 'dalmax.models.daninhas_resnet50.DaninhasModelResNet50'>` — the class object, zero weights. Every historical `saved_model.pth` this bug ever produced (**including every run under `results/dalmax1/` and `results/dalmax2/`**) is the same 900-byte class-pickle and is **unrecoverable** — there is no weight data hiding in those files to salvage; `results.json`/`predictions.csv`/`run_metadata.json`/plots from those runs are unaffected (never on this code path), only the weight checkpoint is lost. A model must be re-trained (its exact params JSON + CLI args + seed are recorded in that run's `run_metadata.json`) if its weights are needed. Fixed by `dalmax/models/checkpoint.py` (`save_checkpoint`/`load_checkpoint`/`describe_checkpoint`, a self-describing `dalmax-checkpoint` dict format) — `DeepLearning.save_model`/`load_model` (`dalmax/models/base.py`) and `Strategy.save_model` (`dalmax/query_strategies/base.py`) now go through it, with `model_name`/`class_names`/`img_size`/`extra` passed explicitly by `dalmax.experiment.reporter.write_report` (constructor/call-time injection, no `setattr`). New tools consuming valid checkpoints: `predict.py`, `loader.py`, `gui.py` (`dalmax/inference/`). See ADR 0006 and `tests/test_checkpoint.py` / `tests/test_golden_run.py`'s new integration assertion (checkpoint >10 MB + re-prediction matching `predictions.csv`) | High (silent total data loss, not just a TODO) → **Resolved** | Phase 2/4 left open; **fixed 2026-08-23** (ADR 0006, `dalmax/models/checkpoint.py`) |
| KI-23 | `config_kmh` is present only for `DANINHAS` in both params files | `files_config/benchmark/params_df_gpu_0.json` (`DANINHAS` block has `config_kmh`; `CIFAR10` block does not); `files_config/benchmark/params_df_gpu_1.json` has the same shape. Params files themselves were not touched by Phase 4 | High (compounds KI-4 — even after removing the hardcoded `'DANINHAS'` key, CIFAR10 runs of the hierarchical strategies need a `config_kmh` block added) — still open | Phase 2/3 delivered the code-side fix (KI-4); the params-file content gap remains unscheduled |
| KI-24 | `.github/` workflow directory exists but is empty | **Resolved (Phase 1)** — `.github/workflows/ci.yml` now exists (66 lines, runs `ruff check` + `pytest -m "not gpu and not dataset"` on push/PR); re-confirmed present and unaffected by Phase 4 | Medium → **Resolved** | Phase 1 (`.github/workflows/ci.yml`) |
| KI-25 | `DaninhasModelVitB16` is defined but never registered/used | Was `core/daninhas_model.py:47-86` — a second model class in the same file as `DaninhasModelResNet50` | Low → **Resolved by deletion (Phase 4)** — `dalmax/models/daninhas_resnet50.py` (the file's Phase 4 destination) does **not** carry `DaninhasModelVitB16` forward; it was dropped during the move, not wired up (confirmed: only `DaninhasModelResNet50` is defined there) | Phase 4 (dropped during the move) |
| KI-26 | SSRAE embedding vector is row-interleaved; naive half-slicing is wrong | **Resolved (2026-08-23, Phase 2)**: `dalmax/embeddings/variants.py::slice_embedding(vectors, variant, q)` implements the verified column-group reshape (`matrix = vectors.reshape(n, 9, 6*(q+1))`; `spatial = matrix[:, :, :3*(q+1)]`; `spectral = matrix[:, :, 3*(q+1):]`), applied by `RepresentationStrategy.query` on the cached `"full"` embedding — no call site slices a raw embedding vector by hand any more. Original finding, unchanged as history: each `rnn.beta` has shape `(9, Q+1)`; `torch.hstack([beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]).reshape((1, -1))` concatenates along columns (axis=1) into a `(9, 6*(Q+1))` matrix, then flattens row-major — interleaving all six blocks per row, so naive `emb[:len(emb)//2]` was never spatial-only. See `tests/test_ssrae_embedding_layout.py` (layout verification) and `tests/test_embedding_variants.py` (the new slicing function) | High → **Resolved** | Phase 2 (`EmbeddingProvider`/`variants.py` — done: `dalmax/embeddings/variants.py::slice_embedding` is the single, tested slicing implementation; see `.specs/architecture/target-architecture.md` §5) |
| KI-27 | `core/tools/SSL/code_kmh.py` is an exploratory script, not a safe-to-import module | Was confirmed: no `if __name__ == "__main__":` guard anywhere in the file; at module scope it unconditionally loads `results/features_dict_ssrae.pkl` (calling `exit()` if absent), runs a full `sklearn.manifold.TSNE.fit_transform` over the entire cached feature set, builds a `torch.tensor(..., device="cuda", ...)` unconditionally, and ends with a blocking `plt.show()` | Medium → **Resolved by deletion (2026-08-23, Phase 4)** — `code_kmh.py` and its only caller, `demo_ssl.py`, were both deleted rather than guarded/converted; confirmed absent anywhere in the tree (`find . -iname code_kmh.py` → no results) | Phase 4 (deleted, not guarded — see `refactor-plan.md` Phase 4) |
| KI-28 | `utils/report/1_cm_extract_from_pdf.py` imports `PyPDF2`, an undeclared dependency | Was `utils/report/1_cm_extract_from_pdf.py:5` (`from PyPDF2 import PdfReader`); that module is now `dalmax/reporting/extract_confusion_matrices.py` (moved in Phase 4) | Medium → **Resolved (Phase 1 fix; file since moved, unaffected)** — `PyPDF2==3.0.1` is declared in `pyproject.toml` (confirmed, line 25); `dalmax/reporting/extract_confusion_matrices.py` installs correctly from a fresh `poetry install` | Phase 1 (declare `PyPDF2` in `pyproject.toml`); Phase 4 (file moved/renamed, dependency declaration unaffected) |

## Explicitly corrected claims (do not repeat)

- **`results/` is not tracked in git.** `results/`, `DATA/`, `local_assets/`,
  `logs/` are already listed in `.gitignore` (confirmed lines 2-8) and 0 files under `results/` are
  tracked. Any future known-issue about "heavy result artifacts inside the git repo" should refer
  only to the risk of someone force-adding such files, not to a current state of affairs.
- **`requirements.txt` (historical) no longer exists — retired.** `pyproject.toml` already lists
  `pandas==2.2.3` (line 22, retired-`requirements.txt`-era Poetry deliverable). The generated
  `requirements.txt` export this bullet originally tracked as "still stale" was retired entirely on
  2026-08-23 (see KI-2 and ADR 0001's amendment) — Poetry (`poetry install`) is now the only
  supported install path, so there is no export left to go stale.

| KI-29 | `SSRAEKmeansSampling`'s selected indices are a function of the hardcoded `KMeans(random_state=3)`, not `--seed` — golden-run fixture is pinned to that literal | `tests/golden/ssrae_kmeans_micro_seed1.json`: confirmed KI-5 empirically before the fix. **Closed 2026-08-23 (Phase 2)**: the fixture was regenerated (`round_1_query_idxs_sorted` now `[14, 24, 37, 45, 47]`, seed-derived) exactly as this row anticipated — the old value is kept in the fixture's `legacy_phase1_values` block for the historical record, not deleted | High (duplicate severity of KI-5) → **Resolved** | Phase 2 (seed propagation audit — done); `tests/golden/ssrae_kmeans_micro_seed1.json` regenerated in the same change, per this row's own instruction |
| KI-30 | Vendored `SSL.src.utils.create_clusters_from_cluster_assignment` builds a `dtype=object` array when all clusters happen to have equal size, breaking `HierarchicalKMeansSelection` for that specific input | Was `core/tools/SSL/src/utils.py:28`; now **`dalmax/tools/SSL/src/utils.py`** (moved unchanged in Phase 4 — still vendored, Meta-licensed, wrap-don't-edit policy still applies, see ADR 0002/`target-architecture.md`'s `tools/` note). The function builds `np.array(clusters, dtype=object)` from a Python list of per-cluster index arrays; when every cluster has **exactly** the same size, NumPy casts the list to `dtype=object` element-wise instead of keeping each row as an `int64` array, and a later `torch.cdist`/tensor-indexing call on it raises `TypeError: can't convert np.ndarray of type numpy.object_ ...`. Triggered only by perfectly balanced k-means cluster sizes — data-dependent; real embeddings essentially never produce perfectly balanced clusters, so this has not been observed on real data | **Resolved 2026-11-07** (was Medium) | Until 2026-11-07 the plan was no fix (vendored code, wrap-don't-edit policy) and the test fixtures avoided equal-size clusters. **Fixed 2026-11-07 (ADR 0012 preparation)**: the KMH sweep's micro smoke hit it on the budget-matched hierarchy `[6,2,2]` (5 of 45 jobs), and the full-scale `[60,20,10]`/`[300,100,50]` shapes could hit it too (20 centroids into 10 clusters of exactly 2 is plausible), so the wrap-don't-edit policy was set aside for a minimal fix: `dalmax/tools/SSL/src/utils.py::to_object_array` pre-allocates the 1-D object array (one `int64` array per element whatever the lengths) and replaces the four `np.array(..., dtype=object)` sites (`utils.py::create_clusters_from_cluster_assignment`, `kmeans_gpu.py::sort_cluster_by_distance`, `clusters.py::process_clusters`, `distributed_kmeans_gpu.py`), each marked `DalMax fix (KI-30)`. Regression test `tests/test_selection_hierarchical.py::test_perfectly_balanced_clusters_do_not_crash`; golden CPU fixtures unchanged (the unbalanced case yields an identical array) |
| KI-31 | `num_workers > 0` DataLoaders have no `worker_init_fn` — a landmine if image augmentation is ever re-enabled | Was `core/deep_learning.py`/`utils/dataset.py`; now **`dalmax/models/base.py`** (every `DataLoader(...)` call spreads `**self.params['train_args']`/`test_args`, setting `num_workers=4` for DANINHAS/`num_workers=1` for CIFAR10, none pass `worker_init_fn`) and **`dalmax/data/handlers.py`** (every stochastic `transforms.Random*` augmentation still commented out — confirmed, so no augmentation randomness is actually exercised today) | Medium — still open, currently inert since augmentation is disabled; the moment anyone uncomments an augmentation transform in `dalmax/data/handlers.py` with `num_workers>0` still configured, worker processes are seeded independently by PyTorch's default per-worker seeding, which is fine for `torch`-based ops but **not** for any `random`/`numpy`-global-state-based augmentation without an explicit `worker_init_fn` reseeding each library per worker — a reproducibility gap `dalmax/seeding.py::seed_everything` does not currently close (it seeds the main process only) | Not yet planned; flag for whoever re-enables augmentation to add a `worker_init_fn` to `dalmax/models/base.py`'s `DataLoader` calls at that time, and extend `dalmax/seeding.py`'s contract to cover it explicitly |
| KI-32 | `ExperimentNotifier/main.py` invoked a stale `utils/report/build_method_metrics.py` path | **Obsolete 2026-09-29: ExperimentNotifier removed from the project** | The entire `ExperimentNotifier/` folder has been deleted per user decision (2026-09-29). All references to it in operational docs, `.specs/`, `.claude/rules/`, and `.gitignore` have been removed. This issue is now fully closed | High → **Resolved by removal** | 2026-09-29 (user deleted `ExperimentNotifier/` and requested removal of all references) |

## Determinism verification (2026-08-23, Phase 1 golden-run batch)

No CPU-training nondeterminism was found for the golden-run configuration (1 epoch, CPU-only
ResNet50, `batch_size=16`, `num_workers=0`, micro 2-class/70-image DANINHAS subset). Both
`RandomSampling` and `SSRAEKmeansSampling` were run 3 times each with identical CLI args
(`--seed 1`); every run produced bit-identical initial labeled indices, queried indices, and
`results.json` metrics (including exact float repr, e.g. `0.3333333333333333`). The
`SSRAEKmeansSampling` check additionally covered both a cold (recompute) and warm (cached) SSRAE
feature cache path, with identical results either way. `tests/test_golden_run.py` therefore compares
metrics with an `abs=1e-6` tolerance (not because nondeterminism was observed, but as a documented
safety margin — see that test's module docstring) rather than falling back to the
indices-only comparison `.specs/quality/testing-strategy.md` allows for. If a future environment
(different CPU architecture, BLAS backend, or thread count) does observe metric drift here, add a
new known-issue row and loosen `test_golden_run.py` to compare indices only, per the original test
plan.

## Open item count

(Update 2026-11-02: KI-35 to KI-40 are resolved as stated in their own sections; **KI-41 and KI-42 are
accepted as-is (author decision 2026-11-02)**, so they leave the open list (documented, not open): 8 open in total, 43 tracked items.)
(Update 2026-11-02b: **KI-43** (round-0 mismatch, disclosure) added: 10 open, 43 tracked items.)
(Update 2026-11-08: **KI-44** (empty clusters under random k-means init crashed the hierarchical sampling) added and resolved the same day: 10 open, 44 tracked items.)
(Update 2026-11-11: **KI-45** (AdversarialBIM ties broken in pool order) added and accepted as-is, disclosed in the thesis like KI-41/KI-42: 10 open, 45 tracked items. KI-43 gained a dated note on the two executions of the reference KMH configuration.)
(Update 2026-11-11c: **KI-46** (byte-identical duplicate images in DANINHAS, including across train and test) added, OPEN, pending an author decision: 11 open, 46 tracked items.)
(Update 2026-11-11b: author decision of the same day reverses the "accepted as-is" status of **KI-41, KI-42 and KI-45**: all three are **fixed in code (2026-11-11, ADR 0014)** and their 63 runs are prepared for re-execution as campaign parts `paper1_rerun_a` / `paper1_rerun_b` (not executed yet). They stay out of the open count: 10 open, 45 tracked items.)
(Update 2026-11-11d: author decision on **KI-46**: everything is re-run on the new canonical dataset `DATA/daninhas_full_fixed` (ADR 0016), so KI-46 is **resolved for future runs**; the first campaign's results stay as history. The KI-41/KI-42/KI-45 re-run now happens inside that campaign (the parts `paper1_rerun_a/b` and the archive step of ADR 0014 were retired before execution). 10 open, 46 tracked items.)
(Update 2026-11-12: **KI-47** (pool/test order followed the file system; the embedding cache served features of other images across sessions; the seed audit compared indices only) added and **fixed in code** before the clean campaign (ADR 0016 amendment); it affects the first campaign, which stays as history. KI-43 stays open until the clean campaign confirms that its cause 2 is KI-47. 10 open, 47 tracked items.)
(Update 2026-11-12b: **KI-48** (corrupt leaves counted as complete, leaf lookup across dataset folders, mixed code versions invisible, notebook pulled a branch) added and **fixed** before the clean campaign: 10 open, 48 tracked items.)

34 tracked items total (KI-34 added 2026-08-26, while reconciling the Phase 3 ablation execution
batch; KI-33 added/resolved 2026-08-25). Of these, **27 are now fully resolved**: KI-1 through
KI-10, KI-13 through KI-20, KI-22, KI-24 through KI-29, KI-32, and KI-33 (KI-22 fixed 2026-08-23,
`dalmax/models/checkpoint.py`, ADR 0006 — see that row for why every checkpoint saved before this
fix is unrecoverable, not just the ones affected by a bug that is still open; KI-33 fixed
2026-08-25, `dalmax/__init__.py::_ensure_matplotlib_backend()`; KI-32 obsoleted 2026-09-29 by
removal of ExperimentNotifier). The remaining **7 are open**:
KI-11, KI-12 (partially — comments/type hints), KI-21, KI-23, and KI-31 (KI-30 was fixed 2026-11-07; KI-34, the missing GPU model, was resolved 2026-09-29). Treat the per-row tables above as the source of
truth over this summary if they ever disagree after future edits.

**Phase 4 resolution summary (2026-08-23, this docs-sync batch)**: Phase 4's physical move-and-delete
work closed a large batch of known issues simply by deleting the file that had them, since their
Phase 2/3 fixes had already made the legacy file unreachable:
- **Resolved by deletion** (the file with the defect no longer exists on disk): KI-4, KI-5, KI-6,
  KI-7 (partially — the deleted file's part), KI-9, KI-10, KI-13, KI-15, KI-16, KI-17, KI-18, KI-19,
  KI-20, KI-25, KI-27.
- **Resolved, file moved but issue unaffected** (already fixed in Phase 2/1, the file just changed
  location): KI-3, KI-8, KI-14, KI-24, KI-26, KI-28, KI-29.
- **Still open, file moved, issue unchanged** (path updated in this batch, underlying defect not
  touched by Phase 4): KI-11 (now in `dalmax/cli.py`/`dalmax/query_strategies/base.py`/
  `representation.py`/`dalmax/experiment/runner.py`), KI-12 (Portuguese comments now in
  `dalmax/data/datasets.py`), KI-21 (`dalmax/data/datasets.py`),
  KI-30 (`dalmax/tools/SSL/src/utils.py`), KI-31 (`dalmax/models/base.py`/`dalmax/data/handlers.py`).
- **Resolved 2026-08-23, outside the Phase 4 batch** (found and fixed in this session, not a
  move-only closure): KI-22 — see that row and ADR 0006 for the checkpoint-format fix and the
  standing note that every pre-2026-08-23 `saved_model.pth` (including `results/dalmax1/`,
  `results/dalmax2/`) is unrecoverable.
- **Still open, params-file content gap, unaffected by any code move**: KI-23.
- **Obsolete 2026-09-29**: KI-32 (ExperimentNotifier removed from the project).

See `.specs/architecture/current-state.md` §5 for the historical coupling-point-numbered version of
this same resolution story, and ADR 0002's final amendment for the full move/delete inventory.

### KI-33 — Colab-exported `MPLBACKEND` broke every CLI launch inside the `.venv` (FIXED 2026-08-25)

**Symptom**: on Colab, every `poetry run python tools/trainer.py ...` died at `import matplotlib` with
`ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`.
**Cause**: Colab sets `MPLBACKEND=module://matplotlib_inline.backend_inline` process-wide;
`matplotlib_inline` is not a dependency of this project, so the `.venv` matplotlib cannot import it.
**Fix**: `dalmax/__init__.py::_ensure_matplotlib_backend()` (falls back to `Agg` only when the
configured `module://` backend is unimportable; valid/absent values untouched) + `export MPLBACKEND=Agg`
in the ablation/Colab batch scripts. Regression test: `tests/test_matplotlib_backend.py` (subprocess
reproduction of the Colab environment). Documented in `COLAB_RUNBOOK.md` §8.

### KI-34 — `run_metadata.json` does not record the GPU model (RESOLVED 2026-09-29)

**Resolved 2026-09-29**: `snapshot()` now adds a top-level `environment` block from `dalmax/experiment/environment.py::collect_environment()` (best-effort, runtime-probed, failures -> null): `os` (system/release/version/platform/distro), `machine` (arch/cpu_model/cpu_count/ram_total_gb), `python`, `torch` (version/cuda/cudnn/cuda_available), `gpus[]` (index/name/total_memory_gb/compute_capability/multi_processor_count/driver_version/nvidia_smi_name/memory_total_mib), `cuda_visible_devices`, `current_device`, `runtime` (is_colab/colab_release_tag). Hostname is deliberately not recorded. A one-line summary is also logged at run start. Original report follows.

**Found**: 2026-08-26, while reconciling the Phase 3 ablation batch (executed on a Google Colab Pro
GPU session) against the pre-Phase-2 lab-machine reference runs and the same-config re-run across two
ablation sub-studies (`6_1/rep_full` vs. `6_3/stage_full`, both the identical `RepresentationStrategy`
config at `n_query=100`, differing by 0.0032 weighted F1 — see
`.specs/experiments/ablation-study.md`'s "Execution record" section). `run_metadata.json`
(`dalmax/experiment/run_metadata.py::write_run_metadata`) records Python/torch versions, CUDA
availability, `git_commit`, and the resolved config, but **not the GPU model/name**
(`torch.cuda.get_device_name(0)`) actually used for the run — so a `results/` tree mixing lab-machine
and Colab runs (or different Colab session GPU classes, which Colab does not let the
user pin) cannot be told apart by inspecting `run_metadata.json` alone; only the run's containing
directory path or a memory of which environment executed it distinguishes them.

**Consequence**: cross-GPU runs of the same config are **not bit-identical**
(`.claude/rules/reproducibility.md`'s determinism policy already documents that
`cudnn.deterministic=True` does not guarantee cross-run bit-identity even on the *same* GPU model;
different GPU models compound this further) — this is expected and not itself a bug, but the
inability to attribute an observed F1 difference to "different hardware" vs. "ordinary same-hardware
run-to-run variance" from `run_metadata.json` alone is a genuine reproducibility-auditing gap.

**Planned fix**: a one-line addition to `write_run_metadata` —
`torch.cuda.get_device_name(0) if torch.cuda.is_available() else None` — stored as a new
`gpu_model` field, alongside the existing `cuda_available` boolean. Not yet implemented; flag for
`implementer` the next time `dalmax/experiment/run_metadata.py` is touched.

**Severity**: Low-Medium — does not affect correctness of any result, only the ability to audit
*why* two nominally-identical runs differ after the fact. | Resolved 2026-09-29 | Fixed (`environment` block, see above).

### KI-35 — two runs started in the same second shared one `results/logs/<timestamp>-log-dalmax.log` (FIXED 2026-09-29)

**Symptom**: found while testing the campaign: two `trainer.py` processes started within the same
wall-clock second (e.g. two GPU panes launched together, or the preset-vs-generic equivalence check run
concurrently) got the identical log path `results/logs/YYYY-mm-dd-HH-MM-SS-log-dalmax.log`
(`dalmax/logging_utils.py`, second granularity); the first to finish moved it into its results dir and
the other died in `write_report` with `FileNotFoundError` on the `os.rename` — and the surviving
`log-dalmax.log` interleaved both runs' lines (which would also corrupt the campaign's seed-consistency
audit, which parses `Initial labeled idxs` from that file). **Fix**: the PID is now part of the file name.
Severity: Low (needs a same-second start), Resolved 2026-09-29.

### KI-37 — Colab system Python is 3.13; `poetry install` failed on it (FIXED 2026-09-29)

Verified on a real Colab runtime: the system Python moved to 3.13.x, but the project requires
`>=3.10,<3.13` (`torch==2.5.0` has no 3.13 wheels), so the runbook's `poetry install` could not work and
the old notebook check stopped the run. **Fix**: keep the pinned stack and install a separate
uv-managed Python 3.12 (`uv python install 3.12`, `poetry env use "$(uv python find 3.12)"`) in
`notebooks/colab_runbook.ipynb`, `notebooks/colab_local_test.ipynb`, `COLAB_RUNBOOK.md`; the notebook
checks now validate the Poetry venv interpreter, not the kernel's. `LAB_RUNBOOK.md` notes the same fix.
Severity: Medium (blocked Colab runs), Resolved 2026-09-29.

### KI-36 — campaign resume trusted `results.json` alone; seed audit was silent on a missing log; GPU determinism over-claimed (FIXED 2026-09-29, ADR 0010)

Found in the review of the campaign (verdict APPROVE, follow-ups). (1) `campaign.job_done` treated a leaf
as finished as soon as `results.json` existed, but the reporter wrote `results.json` before the
predictions file, so a run killed mid-report was skipped on resume. **Fix**: `job_done` requires the full
artifact set (`dalmax/reporting/leaf_check.py::check_leaf`) and the reporter writes `results.json` last
(`tests/test_campaign.py`). (2) A run with a missing/unreadable `log-dalmax.log` simply dropped out of the
seed audit. **Fix**: the audit reports WARN, and `campaign verify` exits 0/1/2 (ok / FAIL or incomplete /
WARN only). (3) `run_metadata.json` recorded `deterministic_algorithms: true` although
`warn_only=True` allows non-deterministic kernels. **Fix**: it records `"warn_only"`, the ops that warned
(`determinism.nondeterministic_op_warnings`) and `fill_uninitialized_memory=False`; the reproducibility
docs now call GPU determinism best-effort. Severity: Low-Medium, Resolved 2026-09-29.

### KI-39 — AdversarialBIM infinite loop on saturated predictions (FIXED 2026-09-30)

Found on the real Colab campaign (2026-09-30): `AdversarialBIM.cal_dis` hung forever on the daninhas pool
(progress stuck at 3553/7786 after the ">1000 iterations" warning). Cause: for a sample predicted with
probability ~1.0 in float32 the cross-entropy gradient w.r.t. the input is exactly zero, so `sign(0) = 0`,
`eta` never changes, the prediction never flips and the uncapped `while` never ends. The original DeepAL
code has the same defect. **Fix** (`dalmax/query_strategies/adversarial_bim.py`, `adversarial_deepfool.py`,
`adversarial_base.py`): BIM `max_iter` defaults to 50 and the loop also stops on an all-zero gradient; a
sample that does not flip returns `float("inf")` (maximally far, ranked last by `argsort`; if fewer than
`n` finite distances exist the rest of the picks come from the `inf` samples). DeepFool keeps
`max_iter=50` and the original return value, but skips classes with a zero/NaN `||w_k||` and returns `inf`
if no usable class exists in an iteration. One warning per query reports how many samples did not converge.
Covered by `tests/test_adversarial_strategies.py`. Any BIM/DeepFool result produced before this fix is
not comparable for non-converged samples. Severity: High (blocked the campaign), Resolved 2026-09-30.

### KI-38 — batched AdversarialBIM/AdversarialDeepFool deviated from the original DeepAL algorithm (FIXED 2026-09-29)

The earlier implementations batched samples, capped BIM at 10 iterations and stopped only when ALL samples
of a batch had flipped, so per-sample distances no longer matched the original algorithm (DeepAL, Huang
2021). **Fix**: faithful per-sample ports (`dalmax/query_strategies/adversarial_bim.py`,
`adversarial_deepfool.py`, shared loop in `adversarial_base.py`): the loop runs on the model's own device
(the original's hardcoded CPU/`.cuda()` round trip is gone), the model's train/eval mode is restored, BIM
was uncapped by default (superseded by KI-39: it hung on the real campaign; now capped at 50 iterations, non-converged samples get distance `inf`). Covered by
`tests/test_adversarial_strategies.py`. Any result produced with the earlier batched version is invalid.
These two baselines are the slowest campaign groups. Severity: Medium, Resolved 2026-09-29.

### KI-40 — TexHAL canonical `Q=[5,17]` underperforms single-Q VCTex; hierarchy/stage ablations ran with it (RESOLVED 2026-11-02, pending advisor review; found 2026-11-01)

The campaign (`experiments/campaign.md`, executed 2026-09-30 to 2026-10-05) showed that the TexHAL
"full" configuration `Q=[5,17]` (two RAEs, 648-d) scores 0.8259 weighted F1 / 0.7687 macro F1 at nq100,
below `Q=5` (0.8382), `Q=13` (0.8557) and `Q=17` (0.8745 / 0.8262, std 0.0079), and below the best
paper-1 method (LeastConfidence, 0.8716). All TexHAL 6.2 (hierarchy) and 6.3 (stage) rows, and the
shared `shared/texhal_full` run, used `Q=[5,17]`. **Decision (author, 2026-11-01; advisor review
pending)**: TexHAL's canonical configuration becomes `Q=17` (ADR 0011, accepted).

**Resolution**: campaign part `texhal_q17` (18 runs = 6 groups x 3 seeds; the reference/full rows are
aliases of the already-executed `texhal/6_1/rep_q17`; 210 runs / 70 groups total) was executed on
2026-11-01 (about 70 min on Google Colab Pro, commit `442a8cd`) and verified on 2026-11-02
(`make campaign-verify`: 210/210 OK, seed audit PASS for seeds 1-3). Outcome (nq100, weighted F1 / macro F1):
6.2 best row is now `L=3 [300,100,50]` 0.8842 / 0.8398 (reference `[600,200,100]` 0.8745 / 0.8262);
6.3 full 0.8745, w/o hierarchical 0.8454, w/o representation 0.8224 (both ablations now hurt, the
representation module most). TexHAL (Q=17) edges the best paper-1 method (+0.3 points, within std) and is
0.6 points below RNHAL. The `Q=[5,17]` 6.2/6.3 results are kept as history (report tables
`6_2_q5_17`/`6_3_q5_17`); paper 2 uses the Q=17 tables. Severity: Medium (scientific consistency of paper 2,
not a code bug). Status: Resolved (executed); advisor confirmation of the Q=17 decision is still pending.
Related note (no action): the RNHAL pending decision on whether the canonical hierarchy
becomes `L=4 [800,600,200,100]` is tracked in `papers-roadmap.md`.

### KI-41 — no dropout layer in the ResNet-50 head: MC-dropout strategies and BALD are degenerate (FIXED IN CODE 2026-11-11, re-run prepared, not executed; ADR 0014; was accepted as-is 2026-11-02; found 2026-11-02)

**Fix (2026-11-11, ADR 0014; author decision of 2026-11-11 to fix and re-run).** `DaninhasModelResNet50(n_classes,
dropout_p=0.0)` applies `nn.Dropout(dropout_p)` to the 2048-d pooled features right before `classifier` when
`dropout_p > 0`, as in DeepAL; with the default the parameters, `state_dict` keys, construction RNG draws and logits are
the pre-fix ones (tested against a frozen copy of the old class). The value comes from the optional params-JSON block
`"model_args": {"dropout_p": ...}` (`dalmax/config/schema.py::ModelArgs`, validated by the loader, threaded by
`get_network` into `DeepLearning(model_kwargs=...)`, recorded in `run_metadata.json` and in the checkpoint);
`files_config/campaign/params_paper1_mc_dropout.json` sets 0.5 for the four MC-dropout strategies only.
`predict_prob_dropout*` now run under `dalmax/models/mc_dropout.py::mc_dropout_mode` (eval mode, only the dropout
modules in train mode, modes restored, BatchNorm statistics untouched) and raise `MCDropoutError` when the model has no
dropout with p > 0; they use their `n_drop` argument (equal to `params['n_drop']` in every caller). BALD and
`EntropySamplingDropout` use `torch.xlogy(p, p)` (no NaN at p = 0). Tests: `tests/test_mc_dropout_and_embeddings.py`.
Re-run: since ADR 0016 (2026-11-11) inside the clean campaign on `DATA/daninhas_full_fixed` (part `paper1`, new
results root `results/campaign_fixed/`), together with everything else; the ADR 0014 parts `paper1_rerun_a/b` and the
archive step were retired before execution, and the first-execution runs stay in `results/campaign/` as history. The
record below describes the first execution.

**Location**: `dalmax/models/daninhas_resnet50.py::DaninhasModelResNet50` (no `nn.Dropout` anywhere: a
pretrained backbone, `embedding_layer = nn.Linear(2048, 50)` and `classifier = nn.Linear(2048, n_classes)`);
`dalmax/models/base.py::DeepLearning.predict_prob_dropout` (l. 120-134) and `predict_prob_dropout_split`
(l. 136-148); consumers `dalmax/query_strategies/{least_confidence,margin_sampling,entropy_sampling}_dropout.py`
and `bayesian_active_learning_disagreement_dropout.py` (`BALDDropout`).

**Evidence (verified against the code)**: both predict functions only call `self.clf.train()` and then run
`n_drop` (10 in `files_config/campaign/params_paper1.json`) forward passes under `torch.no_grad()` over a
`DataLoader(..., shuffle=False)`; the dataset transform for DANINHAS is deterministic (`ToTensor` +
`Normalize`, the random augmentations in `dalmax/data/handlers.py` are commented out). With no dropout, the
only train-mode effect is BatchNorm using batch statistics (and updating its running statistics, a side
effect of "prediction"). Because the loader is unshuffled, every pass sees the same batches, hence the same
batch statistics: the `n_drop` passes are identical up to GPU floating-point nondeterminism, not merely
similar. Consequences: (i) `LeastConfidenceDropout`, `MarginSamplingDropout` and `EntropySamplingDropout`
equal their non-dropout counterparts computed with BatchNorm in train mode (batch-dependent scores, not
an MC-dropout posterior); (ii) `BALDDropout`'s mutual information `entropy(mean p) - mean(entropy p)`
(code: `entropy2 - entropy1`, ascending sort) is zero for every sample up to float noise (and `0*log(0)`
can give NaN for saturated probabilities), so its ranking is determined by rounding noise, effectively
arbitrary. The original DeepAL nets (Huang 2021, https://github.com/ej0cl6/deep-active-learning) contain
dropout layers, so the strategies were not ported faithfully.

**Impact on the campaign**: `paper1` strategies `LeastConfidenceDropout`, `MarginSamplingDropout`,
`EntropySamplingDropout`, `BALDDropout`, all budgets (nq 10/50/100) x seeds 1-3 = 36 executed runs. Other
strategies are unaffected (the non-dropout ones call `eval()`). Severity: High for paper 1 (4 of the 12
classical baselines do not measure what their names say); no effect on papers 2/3.

**Status**: Fixed in code 2026-11-11 (ADR 0014), re-run prepared, not executed (see the fix paragraph at the top).
History: accepted as-is by the author on 2026-11-02 (runs kept, behaviour disclosed in paper 1, fix deferred to a future
campaign); option (b) below was chosen on 2026-11-11. Options considered: (a) keep the executed runs and disclose the behaviour
in paper 1 (the thesis chapter already does); (b) fix: add `nn.Dropout(p)` before the classifier (and make
`predict_prob_dropout*` use it) and re-run the affected strategies as a new campaign part, keeping the old
runs as history (append-only). Combined with KI-42 option (b): 6 strategies x 3 budgets x 3 seeds = 54 runs
(63 with KI-45's BIM).

### KI-42 — `KMeansSampling` and `KCenterGreedy` cluster an untrained random ReLU projection (FIXED IN CODE 2026-11-11, re-run prepared, not executed; ADR 0014; was accepted as-is 2026-11-02; found 2026-11-02)

**Fix (2026-11-11, ADR 0014).** `DaninhasModelResNet50.forward` returns as `e1` the 2048-d pooled features that feed the
classifier (the trained penultimate representation, as in DeepAL) and `get_embedding_dim()` returns 2048.
`embedding_layer = nn.Linear(2048, 50)` is kept, constructed in the same position and unused: its construction draws
from the torch RNG before the classifier's initialisation and its keys are in every checkpoint, so removing it would
change every strategy's training. Only `get_embeddings` consumes `e1`. `KCenterGreedy` also clamps the squared pairwise
distances at 0 before `np.sqrt` (float cancellation made near-duplicate rows NaN, and a NaN row wins `argmax`);
`KMeansSampling` is unchanged (it draws from NumPy's global RNG, seeded by `seed_everything`). Tests:
`tests/test_mc_dropout_and_embeddings.py`, `tests/test_kcenter_greedy.py`. Re-run: inside the clean campaign of ADR
0016 (part `paper1`, `results/campaign_fixed/`; the ADR 0014 part `paper1_rerun_a` was retired before execution). The
record below describes the first execution.

**Location**: `dalmax/models/daninhas_resnet50.py::DaninhasModelResNet50.forward` (`e1 =
relu(embedding_layer(features))`, `embedding_layer = nn.Linear(2048, 50)`, while `classifier` consumes the
2048-d `features` directly); `dalmax/models/base.py::DeepLearning.train` (l. 62-86: `self.clf =
self.net(n_classes)` re-created at every call, i.e. every round) and `DeepLearning.get_embeddings`;
`dalmax/query_strategies/base.py::Strategy.get_embeddings`; consumers `kmeans_sampling.py`,
`kcenter_greedy.py`.

**Evidence (verified against the code)**: `e1` is returned by `forward` but the training loss is
`F.cross_entropy(out, y)` on the classifier output only, so `embedding_layer` receives no gradient (it is in
the optimizer's parameter list but `.grad` stays `None`, SGD skips it) and keeps its random initialisation.
The model is rebuilt each round (`self.net(n_classes)`), so the projection is re-drawn every round (from
the seeded torch RNG). `KMeansSampling` and `KCenterGreedy` therefore operate on a random 50-d ReLU
projection of the ImageNet-pretrained (then fine-tuned) 2048-d features, not on a learned embedding as in
DeepAL, where `e1` is the trained penultimate layer.

**Impact on the campaign**: `paper1` strategies `KMeansSampling` and `KCenterGreedy`, all budgets x seeds
1-3 = 18 executed runs. Random projections roughly preserve distances (Johnson-Lindenstrauss), so the
strategies still behave as diversity samplers, but not as specified. Severity: Medium for paper 1; no
effect on papers 2/3 (KMH uses ResNet-50 ImageNet features through the embedding provider, not `e1`).

**Status**: Fixed in code 2026-11-11 (ADR 0014), re-run prepared, not executed (see the fix paragraph at the top).
History: accepted as-is by the author on 2026-11-02; option (b) below was chosen on 2026-11-11. Options considered: (a) keep the executed runs and disclose it in paper 1
(the thesis chapter already does); (b) fix: make `e1` the trained penultimate representation, or use the
2048-d features for the embedding strategies, and re-run `KMeansSampling`/`KCenterGreedy` (with KI-41 (b):
6 strategies x 3 budgets x 3 seeds = 54 runs as a new campaign part, old runs kept as history).

### KI-43 — round-0 metrics differ across runs of the same seed and initial labeled set (OPEN, disclosure; found 2026-11-02; cause 2 identified 2026-11-12 as KI-47, pending confirmation by the clean campaign)

Update 2026-11-12 (KI-47): the "same initial labeled set" of this entry was the same set of INDICES only. The loaders
listed each class folder in file-system order, which differs between Colab sessions, so the same indices named other
images in every session after the first; and the representation strategies of later sessions selected on cached
features of other images. That is very likely the true cause 2 below ("run environment drift between Colab
sessions"): the round-0 value is a function of the session because the round-0 training images are, and the
"perfect function of the Colab runtime image" is a function of the session's unzip order (three sessions on one image
tag gave three test orders in the KMH sweep). It also explains the 2026-11-11 note at the end: the reference KMH
configuration reproduced the campaign's query indices in 24/24 rounds because both executions read the same cached
features with the same seed, and those indices named other images in the sweep's session, hence the 1.7-point gap.
Cause 1 (RNG consumption of the ResNet-ImageNet provider) is unaffected. The loaders now sort the file names and the
cache, the seed audit and a new data-order audit check image identity (KI-47); the clean campaign (ADR 0016) should
show round-0 metrics that depend on the seed only (up to the documented best-effort GPU determinism), which would
confirm the diagnosis. The control run below stays prepared, not executed.

Note 2026-11-10: the `hier_transfer` part (ADR 0013) ran on a third runtime image (a different GPU model, see `run_metadata.json` `environment.gpus[0].name`); its 12 runs are compared only within the campaign's nq100 tables, as before.

Control run prepared 2026-11-04 (not executed, pending the author's decision P07): see `.specs/experiments/campaign.md`, section "Control run for KI-43".

Author decision 2026-11-08 (thesis): the KMH sweep runs (one later runtime image) are reported in paper 1 as KMH's results next to the campaign rows, without a runtime-image caveat in the thesis text and without a re-baseline of the other strategies; the facts stay recorded here and in `campaign.md` (reference re-run +0.7 to +1.7 points of weighted F1 vs the campaign rows).

**What was observed** (audit of the 132 `NQ_*` leaves under `results/campaign/` with `run_metadata.json`; the
`FullSupervised` upper bound excluded). In every group the initial labeled set is identical for a seed (md5 of
the logged `Initial labeled idxs` equal for seeds 1, 2, 3 across Random, KMH, TexHAL and `texhal_q17`), yet the
round-0 weighted F1 (model trained only on the 100 initial labels) is not a function of the seed alone. nq100,
all on one 80 GB GPU model (see `environment.gpus[0].name` in each `run_metadata.json`), torch 2.5.0+cu124, `determinism.deterministic_algorithms = warn_only`:

| seed | value | runs | Colab image tag | git commit |
|---|---|---|---|---|
| 1 | 0.308398 (acc 0.34378) | 36 nq100 runs (`shared/random_nq100`, 11 `paper1/nq100`, all `rnhal/*`, `texhal/*`, `shared/texhal_full`) | 20260921 | 9128826 or 705b8e3 |
| 1 | 0.300169 (acc 0.31909) | `shared/kmh_nq100` only | 20260921 | 705b8e3 |
| 1 | 0.297946 (acc 0.32194) | the 6 `texhal_q17/*` runs | 20261029 | 442a8cd |
| 2 | 0.430349 / 0.420118 (kmh) / 0.467697 (`texhal_q17`) | same groups | same | same |
| 3 | 0.616711 / 0.394459 (kmh) / 0.366958 (`texhal_q17`) | same groups | same | same |

(user-reported values confirmed: seed 1 KMH 0.300 vs 0.308, seed 3 KMH 0.394 vs Random = RNHAL = 0.617.) Macro F1
and accuracy deviate in lockstep (seed 3: macro 0.5092 vs 0.3292 vs 0.3400; accuracy 0.5959 vs 0.3751 vs 0.3951).
Within a group all runs are bit-identical to each other, including different strategies and different `Q`/
hierarchy, which confirms round 0 does not depend on the selection configuration. Across budgets the same
holds only when the image is the same: `paper1/nq50` non-ResNet runs (image 20260921) equal nq100 (0.308398 /
0.430349 / 0.616711), while all 11 non-ResNet `paper1/nq10` runs (image 20260930, `NIL=100` as well) give a
different, constant value (0.437548 / 0.093498 / 0.467548), and so does the one nq50 outlier
`paper1/nq50/AdversarialDeepFool` (image 20260930, 0.093498 for seed 2 = the nq10 value, not the nq50 one) and
`paper1/nq10/AdversarialDeepFool` (image 20261001, 0.416078 / 0.303098 / 0.474383). So `p1/RandomSampling` at
seed 2 has round-0 F1 0.0935 (nq10) vs 0.4303 (nq50, nq100) although round 0 cannot depend on `n_query`.

**Two independent causes** (the deviation does not correlate with git commit, GPU model, nondeterministic-op
warning list or strategy type among the "other" runs: 9128826 and 705b8e3 runs of the same image are equal; all
GPUs are the same 80 GB model; `cumsum_cuda_kernel` appears or not without changing the value):

1. **RNG consumption before round-0 training (KMH, `resnet_imagenet` provider)**. `build_strategy`
   (`dalmax/query_strategies/registry.py`, called at `dalmax/experiment/runner.py` after `get_network`, before
   `initialize_labels` and `strategy.train()`) constructs the embedding provider;
   `dalmax/embeddings/resnet_imagenet_provider.py` line 41 `model = resnet50(weights=weights)` first
   random-initialises the whole network with torch's global CPU RNG and only then loads the pretrained weights.
   Verified: after `torch.manual_seed(1)`, `torch.rand(1)` is 0.7576 but 0.8269 after `resnet50()`. The extra
   draws shift the generator state seen by the `DataLoader(shuffle=True)` permutation of round-0 training
   (`dalmax/models/base.py` l. 69), hence a different batch order and different weights. SSRAE (`ssrae_provider`,
   LCG weights) and VCTex (`dalmax/tools/VCTex/rnn.py`, LCG weights) draw nothing from the torch RNG, which is
   why RNHAL, TexHAL and Random coincide. This is a real, deterministic RNG-order difference, not noise.
2. **Run environment drift between Colab sessions (the other groups)**. The round-0 value is a perfect function
   of the Colab runtime image recorded in `environment.runtime.colab_release_tag` (20260921, 20260930,
   20261001, 20261029; OS minor/Python patch also change), not of the code (the `texhal_q17` and `texhal`
   `stage_no_hierarchy` logs are line-for-line identical up to the paths and `Q`, same torch 2.5.0+cu124, same
   driver 580.82.07, and VCTex does not touch the torch RNG, yet 0.2979 vs 0.3084). The mechanism is best-effort
   GPU determinism (`warn_only`, `.claude/rules/reproducibility.md`) plus unrecorded library differences in the
   image (cuDNN/cuBLAS builds, numpy, torchvision, Pillow; only the torch version is stored in
   `run_metadata.json`). Not proven which library; the correlation is.

**Impact on the thesis**: not only the round-0 point. Every round trains a freshly built model, so each round's
metric carries the same kind of run-to-run noise, and the final-round numbers of runs from different images are
not bit-comparable; (a) the round-0 point of the learning curves differs between groups (KMH vs the others at
every budget; `texhal_q17` vs `texhal`/RNHAL; nq10 vs nq50/nq100) and must not be read as a method effect;
(b) comparisons across budgets of the same strategy mix images (nq10: 20260930; nq50, nq100: 20260921; KMH
nq50: 20260930, nq10: 20261001; part `texhal_q17`: 20261029), so image noise is confounded with budget and,
for TexHAL Q=17 vs Q=[5,17] or vs RNHAL, with the method; (c) the seed audit only proves the initial labeled
sets are equal (it does), not that round-0 metrics are. The size of the effect is visible directly: round-0
weighted F1 spans 0.367 to 0.617 for seed 3 from environment drift alone (100 labels, 5 classes, 10 epochs),
which bounds how small a final-round difference the tables can resolve. Seed-averaged final metrics (3 seeds)
are what the papers report; they remain valid but the single-seed round-0 numbers are noisy.

**Recommendation (no code changed here)**: disclose in the thesis that GPU determinism is best-effort and that
the campaign spans four Colab images; report seed means with std, and never attribute a round-0 gap (or a gap
smaller than the seed std) to a method. For KMH, the clean fix is to make provider construction RNG-neutral
(`torch.random.fork_rng()` around `resnet50(...)` in `ResNetImageNetProvider.__init__`, or build the provider
lazily on first `query()`), then re-run `shared/kmh_nq100` and `paper1` `RepresentationStrategy` (12 runs) if
the author wants KMH round 0 aligned; route to `implementer`, with a `run_metadata.json` extension recording
`numpy`/`torchvision`/`scikit-learn`/cuDNN versions so image drift becomes attributable. A clean control for
cause 2 is to re-run `RandomSampling` nq100 seeds 1-3 in the current image and compare with 0.3084 / 0.4303 /
0.6167 (about 3 runs). Executed runs stay as history (append-only).

Note 2026-11-11 (thesis integrity audit, `thesis_drafts/final_thesis/files/audits/2026-11-11-integrity.md`,
F-04/F-06; recomputed from the logs): the reference KMH configuration (ImageNet features, `[600,200,100]`,
`n_resamples=10`) was executed twice, in the campaign (`paper1/nq10` and `paper1/nq50`
`RepresentationStrategy`, `shared/kmh_nq100`; commit 705b8e3) and in the KMH sweep (`campaign_kmh/nq{10,50,100}/hier_L3_ref`;
commit b26d8ca, later runtime image). The logged `Round k query_idxs` are identical in all 24 query rounds of
every budget (8 rounds x 3 seeds; 24/24 at nq10, nq50 and nq100), so the two executions differ only in training.
Their final weighted F1 means still differ: nq10 0.6466 (campaign) vs 0.6539 (sweep), nq50 0.7833 vs 0.7951,
nq100 0.8224 vs 0.8389 (per seed at nq100: 0.7938/0.8685/0.8049 vs 0.8587/0.8635/0.7945, i.e. up to 6.5 points
for one seed). This is the direct measure of the run-to-run training variation of the protocol: about 1.7
points in the seed mean at nq100. The thesis now states it once in paper 1, in the chapter 9 limitations and
in the chapter 10 conclusion, treats differences of that size as ties, and no longer ranks the untuned
reference with one of the two executions (first at nq10, level with Adversarial DeepFool at nq50, below Least
Confidence at nq100). It also confirms that the code-version changes between 705b8e3 and b26d8ca (new
`selection.hierarchy` options with neutral defaults, KI-44 fix) preserved the selection behaviour.

### KI-44 — random k-means init leaves empty clusters that crashed the hierarchical sampling (RESOLVED 2026-11-08; found 2026-11-08)

**Symptom.** The KMH parameter sweep (ADR 0012, executed 2026-11-07 to 2026-11-08) finished 129/135 runs; six runs
failed with exit 1, all of them `init_random` configurations (`init_method: "random"`), with
`ValueError: zero-size array to reduction operation minimum which has no identity`. The same seed failed at the
same round on two attempts (deterministic): `kmh/nq100/hier_L3_ref__init_random` seed 3 (round 1),
`kmh/nq50/hier_L3_ref__init_random` seed 3 (round 1), `kmh/nq10/hier_L3_ref__init_random` seeds 1 (round 5) and
3 (round 1), `kmh/nq10/hier_L3_matched__init_random` seeds 1 (round 7) and 2 (round 3). The other `init_random`
seeds (for example nq100 seeds 1 and 2) completed.

**Root cause.** The vendored k-means (`dalmax/tools/SSL/src/kmeans_gpu.py::_kmeans`) with `init_method="random"`
draws the initial centroids as random data points and never re-seeds a centroid that ends up with no assigned
point, so empty clusters can appear at any level (k-means++ spreads the centroids and practically never produces
them). The clustering tolerates empty clusters (`create_clusters_from_cluster_assignment` and
`HierarchicalCluster.process_clusters` give them size 0, and the water-filling allocation gives them a zero
target), but `dalmax/tools/SSL/src/hierarchical_sampling.py::recursive_hierarchical_sampling` recursed into an
empty cluster above the leaf level and `find_subcluster_target_size` called `np.min` on an empty array.
Reproduced locally on CPU: the unpatched module raised the identical error on a hand-built hierarchy with an
empty coarse cluster.

**Fix** (same file, both edits marked `DalMax fix (KI-44)`): `find_subcluster_target_size` returns an empty int64
allocation when there are zero subclusters and the target is 0 (and raises a clear `ValueError` if the target is
positive); `recursive_hierarchical_sampling` returns an empty int64 array for a cluster with no subclusters.

**Behaviour preservation.** Both branches are reached only where the previous code raised, so the 129 completed
sweep runs and every executed campaign run are unaffected (same RNG consumption, same selections). No run has
to be redone except the six that failed.

**Regression tests** (`tests/test_selection_hierarchical.py`):
`test_hierarchical_sampling_tolerates_empty_clusters`,
`test_find_subcluster_target_size_rejects_positive_target_on_empty_cluster`,
`test_random_init_with_empty_clusters_keeps_the_contract` (three groups of duplicated points clustered into 6
clusters with random init give level-1 sizes such as `[10, 0, 2, 8, 6, 4]`).

**Semantic note for the papers.** With `init_method="random"` an empty cluster means the hierarchy effectively has
fewer clusters at that level in that round. This is inherent to the vendored algorithm (no empty-cluster
re-seeding) and must be stated wherever the `init_random` rows are reported.

**Next step.** The author re-runs the same command; the campaign engine resumes (skip-existing) and executes only
the six incomplete leaves. Then `make campaign-verify MANIFEST=files_config/campaign/manifest_kmh_sweep.json`
expects `OK=135`.

### KI-45 — AdversarialBIM selection is degenerate: ties in the BIM distance are broken in pool order (FIXED IN CODE 2026-11-11, re-run prepared, not executed; ADR 0014; was accepted as-is earlier the same day; found 2026-11-11)

**Fix (2026-11-11, ADR 0014).** `dalmax/query_strategies/adversarial_base.py::rank_by_distance` ranks by distance with
a random permutation as secondary key (`np.lexsort((rng.permutation(len(dis)), dis))`); `rng` is created once per run by
`build_strategy` from `derive_seed(config.seed, "adversarial_ties")` and injected through the constructor of
`AdversarialBIM`, so it advances across rounds. `inf` still ranks last; with no ties the order equals the old
stable argsort. `AdversarialDeepFool` (not re-run) gets no generator and keeps the stable argsort, i.e. its pre-fix code
path: its distances are continuous, but 1,269 of the 8,086 DANINHAS training files are byte-identical copies of another file (588 groups, never across classes) and duplicates tie exactly, so a random tie-break could change
which copy it selects; the pool order among identical copies carries no class bias. Tests:
`tests/test_adversarial_strategies.py`. Re-run: inside the clean campaign of ADR 0016 (part `paper1`,
`results/campaign_fixed/`; the ADR 0014 parts were retired before execution). On `daninhas_full_fixed` no two pool
images are byte-identical, so DeepFool's pool-order tie-break no longer has exact ties to break. The record below
describes the first execution.

**Location**: `dalmax/query_strategies/adversarial_base.py::PerSampleAdversarialStrategy.query`, l. 72
(`return unlabeled_idxs[dis.argsort(kind="stable")[:n]]`); distance computed in
`dalmax/query_strategies/adversarial_bim.py::AdversarialBIM.cal_dis` (l. 59; `eta += self.eps * torch.sign(grad)`,
`eps = 0.05`, returns `(eta * eta).sum()`, l. 80-87); pool order from `dalmax/data/datasets.py::Data.get_unlabeled_data`
(`np.arange(self.n_pool)[~self.labeled_idxs]`, sorted indices).

**Evidence** (thesis integrity audit `thesis_drafts/final_thesis/files/audits/2026-11-11-integrity.md`, F-01;
re-computed from the logged `Round k query_idxs` of `results/campaign/paper1/nq{10,50,100}/**/AdversarialBIM/`).
BIM moves every input element by exactly `eps` per step, so the squared norm of `eta` takes few discrete values
(essentially one per step count) and many unlabeled images tie. The stable `argsort` then keeps the pool order,
and the DANINHAS pool is stored class by class (`Brachiaria` first, indices 0 to 773; blocks of
774/1036/1198/3370/1708 images). Images that hit the 50-iteration cap get `inf` and are never selected
(165 to 368 per round in the logs), so the cap cannot explain the batch. Over the 24 query rounds of each budget
(8 rounds x 3 seeds), 91.7% (nq10), 69.8% (nq50) and 81.2% (nq100) of the BIM queries lie among the `2b` lowest
unlabeled indices, and in 5 rounds the batch is exactly the `b` lowest; the same statistic is 0.0 to 4.3% for
`AdversarialDeepFool` and `RandomSampling` (0.0/1.7/4.3% and 0.4/0.8/2.7%; Least Confidence is in the same range
per the audit). At nq100 86.5% of the BIM queries are `Brachiaria`, and the final weighted F1 falls below its
round-0 value at nq50 and nq100 (0.3006 and 0.2844). The BIM rows of paper 1 therefore measure an index-ordered
selection, an implementation artefact, not the BIM criterion.

**Impact on the campaign**: `paper1` `AdversarialBIM`, nq 10/50/100 x seeds 1-3 = 9 executed runs. No other
strategy is affected in practice (DeepFool distances are continuous and show no index bias); papers 2/3 do not
use BIM.

**Status**: Fixed in code 2026-11-11 (ADR 0014), re-run prepared, not executed (see the fix paragraph at the top).
History: first accepted as-is (orchestrator/author decision 2026-11-11, same policy as KI-41/KI-42): runs kept as
executed (append-only), behaviour disclosed in the thesis (paper 1 results footnote, method and limitations,
appendix C, chapters 1, 9 and 10, where BIM is listed with the other strategies evaluated in a degenerate form);
fix deferred to a future campaign. Fix option: break ties at random (e.g. add a tiny seeded jitter from
`derive_seed(seed, "adversarial_ties")`, or `np.lexsort((rng.permutation(len(dis)), dis))`) and re-run the 9 BIM
runs as a new campaign part, old runs kept as history.

### KI-46 — DANINHAS contains byte-identical duplicate images, also across train and test (RESOLVED for future runs 2026-11-11 by the new dataset `daninhas_full_fixed`, ADR 0016; the first campaign's results are affected; found 2026-11-11)

**Found** while implementing ADR 0014 (checking whether DeepFool distances can tie exactly). Evidence, read-only
`md5sum` over `DATA/daninhas_full/` (2026-11-11): **train** 8,086 files, 7,405 distinct contents; 1,269 files belong to
588 groups of identical copies, never across classes. **Test** 2,106 files, 1,008 distinct contents (per class:
Brachiaria 335/172, Coloniao 190/89, Graminea 970/502, Mamona 456/167, Outras folhas largas 155/78). **Across splits**
171 distinct images occur in both train and test: 394 test files and 198 train files are copies of an image of the
other split, never across classes. The copies share the tile coordinates and differ in the orthophoto number of the
file name (e.g. `test/DATASET_MAMONA/ORTOFOTO_234_9001_43609_1.jpg` and `ORTOFOTO_278_9001_43609_1.jpg`), which points
to overlapping orthophoto exports.

**Consequences for the first campaign** (`results/campaign/`, `results/campaign_kmh/`, executed 2026-09-30 to
2026-11-11): (1) train/test leakage: part of the test set is in the pool, so every method that labels those pool images
is evaluated partly on images it was trained on; (2) duplicated test files weight those images two or more times in
every metric; (3) duplicates tie exactly in any per-sample score, which is why ADR 0014 keeps `AdversarialDeepFool` on
its pool-order tie-break. Severity: High for the absolute numbers of all three papers, unknown for the comparisons
between methods (every method saw the same test set). Only the last round's `predictions.csv` was saved per run, so
the per-round metrics could not be recomputed on the distinct test images.

**Resolution (author decision 2026-11-11, ADR 0016).** A new canonical dataset `DATA/daninhas_full_fixed` was built
by `scripts/datasets/build_daninhas_full_fixed.py` (commit c7d2e9a; the source folder is untouched): test = one file
per distinct MD5 of the source test split (1,008); train = one file per distinct MD5 of the source train split minus
every image also in test and minus four near-duplicates of test images listed in
`files_config/datasets/daninhas_full_fixed_near_duplicate_leaks.csv` (7,230); plus `MANIFEST.csv` and `README.txt`
(8,240 files). `--check` verifies that no MD5 repeats inside or across the splits (the Colab setup runs it after
unzipping). Everything is re-run on it into new roots (`results/campaign_fixed/`, `results/campaign_kmh_fixed/`), no
old result is reused, and every run now also saves the per-round test predictions and queries
(`round_predictions.json`, `round_queries.json`) so a later finding of this kind can be re-scored without re-running.
Remaining: the thesis must replace every number of the first campaign once the new one is executed.

### KI-47 — pool and test order depended on the file system; the embedding cache served features of other images across sessions (FIXED IN CODE 2026-11-12, before the clean campaign; the first campaign is affected; found 2026-11-12)

**Defect.** `dalmax/data/loaders.py` (`get_DANINHAS`, `get_CIFAR10`) sorted the class folder names but listed the files
of each class folder with an unsorted `os.listdir`. Every pool index and test index is a position in the lists built
from that listing, so what an index names followed the file system's listing order, which differs between machines
and between Colab sessions (every session unzips the dataset again onto a fresh local disk). Three consequences:

1. **Same seed, same indices, other images.** `Data.initialize_labels` shuffles indices with the seeded global RNG, so
   a seed gives the same initial INDICES everywhere, but under another order they are other IMAGES: rows of one table
   executed in different sessions started from different labelled sets.
2. **Stale embedding cache.** `dalmax/embeddings/cache.py::EmbeddingCache` (on Drive, under the `results/` symlink)
   was keyed on `(dataset, extractor, Q, variant, split, pool_hash)` with `pool_hash` = a hash of the unlabeled
   INDICES, identical for a seed in every session. A later session therefore loaded the features computed by the
   session that wrote the file: the feature at index `i` belonged to the image that had index `i` in THAT session's
   order, silently.
3. **Blind seed audit.** `dalmax.campaign.audit_seed_consistency` compared the logged initial indices only, so it
   passed (it did, for every verify of the first campaign).

**Evidence (read only, 2026-11-12).** The 222 leaves of `results/campaign/` hold the same set of 2,106 test paths in
`predictions.csv` in 5 distinct orders, which coincide with the 5 Colab runtime image tags recorded in
`run_metadata.json` (145 leaves on 20260921, the first session; 38 on 20260930; 9 on 20261001; the 18 `texhal_q17`
leaves on 20261029; the 12 `hier_transfer` leaves on 20261109). The 165 leaves of `results/campaign_kmh/` hold 4
distinct orders, three of them on the same image tag 20261106 (129 + 26 + 6 leaves) and one on 20261109 (4 leaves), so
the order changes with the session (each session unzips the dataset again), not with the image. Cache hits in
later sessions, from the run logs (time between the `Round 1` line and the first `Embedding matrix shape` line):
`shared/kmh_nq100` computed the ResNet-ImageNet features in 8.4-8.6 s (session 1, 2026-09-30); paper-1 KMH loaded them
in 3.9-5.7 s at nq50 (2026-10-02) and 3.1-3.8 s at nq10 (2026-10-05) and the KMH sweep in 0.25 s (2026-11-07);
`texhal/6_1/rep_q17` computed VCTex Q=17 in 56-57 s (session 1), `texhal_q17` loaded it in 0.9-2.4 s (2026-11-01);
`hier_transfer` loaded SSRAE/VCTex in 1.6-2.0 s (2026-11-11) against about 96 s to compute SSRAE. Decisive: the KI-43
note of 2026-11-11 found the sweep's `hier_L3_ref` with exactly the campaign's query indices in 24/24 rounds of every
budget although its session indexed the pool in another order: the same cached features and seed give the same
indices, which named other images there. The brief that reported the defect also measured cache hits of 0.2 s against
8.6 s of computation in leaves of later sessions.

**Effect on the first campaign** (`results/campaign/`, `results/campaign_kmh/`, history, append-only):

- **Misaligned features** (selection on the features of other images): every leaf of a representation-based strategy
  executed in a session other than the one that wrote its embedding cache: the whole KMH sweep (165 leaves, ADR 0012),
  `texhal_q17` (18, ADR 0011's 6.2/6.3 re-run), `hier_transfer` (12, ADR 0013) and paper-1 KMH at nq50 and nq10 (6):
  201 leaves. The class folders were always sorted and every session holds the same files, so each class keeps its
  block of indices and the misalignment permutes images only WITHIN a class: a selected batch kept the class
  composition the cached features implied, but which images of a class were chosen was unrelated to their own
  features (the same holds for the initial set of every later-session leaf). Open uncertainty: in
  the first session the SSRAE seed-1 features were already cached before the first RNHAL run (0.19 s); most likely the
  same session's `make colab-check` (SSRAE, seed 1) wrote them, but that check leaf was overwritten later, so it cannot
  be confirmed; if they came from an earlier session, RNHAL's seed-1 rows are misaligned too.
- **Different initial images**: every comparison between leaves of different sessions (the per-budget readings of
  paper 1 across nq10/nq50/nq100, the KMH sweep against the campaign, `texhal_q17` and `hier_transfer` against the
  rows of the first session) compares runs that started from different labelled images at the same seed.
- **Decisions to re-read on the clean campaign**: ADR 0012 (the KMH sweep, its winners and paper 1's per-budget KMH
  reference, pendencia P17) and ADR 0013 (hierarchy transfer, "the shape does not transfer") rest entirely on
  misaligned leaves. ADR 0011's deciding comparison (the 6.1 rows Q=5/13/17/[5,17] and the paper-1 nq100 rows) was
  measured inside the first session, on features computed there, but the `texhal_q17` 6.2/6.3 results that followed
  from it are misaligned. None of the first campaign's numbers is used for the papers any more (ADR 0016), and these
  choices must be re-read on the clean campaign.

**Fix (2026-11-12, ADR 0016 amendment; before any run of the clean campaign).**

- Loaders: `list_image_names` = plain lexicographic sort of the file names (`sorted()` on `str`, code-point order) in
  every class folder of both splits, for both loaders (the order contract, `dalmax/data_order.py::ORDER_CONTRACT`).
- Identity: `dalmax/data_order.py` (`relative_image_path` = `<class>/<file>`, `ordered_paths_hash`); every run records
  `run_metadata.json` `data_order` = `{contract, train_paths_hash, test_paths_hash, n_train, n_test}`.
- Cache: the key carries `order_hash` (the train split's ordered-paths hash), so the file name gains an
  `__order<hash>` segment and no file written before the fix (including any `daninhas_full_fixed__*` cache) can be
  hit; the payload (format version 2) stores the key, the ids and the relative paths of the cached images and is
  validated on load (any mismatch or unreadable/truncated file: logged warning, recompute); writes are atomic
  (temporary file + `os.replace`).
- Audits: the seed audit compares the round-0 image paths of `round_queries.json` (indices from the log as fallback
  for older leaves), so equal indices naming other images FAIL; the new data-order audit of `campaign verify` FAILs
  when the leaves of one dataset disagree on the `data_order` hashes (WARN for leaves without the record).
- Tests: `tests/test_data_order.py` (sorted, creation- and listing-order independent loaders), `tests/test_embedding_cache.py`
  (foreign order refused, pre-fix name and payload ignored, truncated file recomputed, atomic write),
  `tests/test_representation_strategy.py`, `tests/test_campaign.py` (audits). The two golden fixtures were regenerated
  deliberately (`pre_ki47_unsorted_listing_values` keeps the old values; the micro dataset was not regenerated).

**Status**: fixed in code; to be confirmed by the clean campaign (expect one `data_order` per dataset over all 459
leaves and, per seed, identical round-0 images across both GPU parts).

### KI-48 — campaign completeness and code-version gaps: corrupt leaves counted as complete, foreign dataset folder picked, mixed commits invisible, notebook pulled a branch (FIXED 2026-11-12, before the clean campaign; found 2026-11-12)

Found by the second pre-flight audit of the clean campaign, each reproduced by execution. (1) `check_leaf` only
warned on an unparsable or empty `results.json` / `run_metadata.json` / round file, a round-count mismatch or a 0-byte
`saved_model.pth`, so resume skipped such a leaf forever and `verify` reported it OK. (2) The leaf glob matched any
dataset folder (`*/SEED_...`), so a leaf under `daninhas_full` (sorted before `daninhas_full_fixed`) was picked first
and verified. (3) Nothing compared the `git_commit` of the leaves. (4) The notebook ran `git pull --ff-only origin
main` on every session, so a push during the campaign would mix code versions, and the run cell did not surface a
non-zero exit of `make campaign-run-gpu-part`.

**Fix** (ADR 0016 amendment, second pre-flight pass): `check_leaf` returns a `LeafCheck` whose integrity `problems` make
the leaf incomplete (re-run into the same leaf; `INCOMPLETE ... (integrity: ...)`; the KI-22 below-10 MB hint stays a
soft warning; round files checked only when required); leaves are looked up under `basename(data_dir)` of the group's
params only (`RunGroup.dataset_folder`) and a plan leaf under another dataset folder FAILs the data-order audit;
`verify` and `pilot-report` print the distinct `git_commit` values of the OK leaves (WARN, exit code unchanged, on more
than one or a missing value); the notebook checks out `GIT_REF` (tag `campaign-fixed-v1`) detached on every session,
prints the commit, and its run cell prints a non-zero exit, re-runs the space check and points to both logs. The
first campaign's 222 + 165 leaves were re-checked read-only with the new rule: all still complete. Tests:
`tests/test_campaign.py` (corruptions parametrized, commit audit, foreign folder), `tests/test_colab_notebook.py`.

**Status**: fixed.
