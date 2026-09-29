# Experimental protocol

> **Update 2026-09-29 (ADR 0008/0009/0010).** The batch scripts this document describes
> (`scripts/benchmark/run_pipe_gpu_{0,1}.sh`, `run_pipline.sh`, `params_dnf.json`-based sweeps) were
> **removed**; everything now runs through the campaign manifest (`experiments/campaign.md`):
> paper-1 = 12 classical strategies + KMH x `n_query` {10,50,100} x seeds {1,2,3}, `n_round 8`,
> `n_init_labeled 100`, `n_epoch 10`, batch 256, plus the `FullSupervised` upper bound; papers 2/3 =
> the ablation configs at `n_query 100`. The CLI entry point is `tools/trainer.py`. Script names below
> are kept as the historical record of what produced `results/dalmax{1,2}/`; the flags, defaults,
> seeds, results layout and `results.json` schema they describe are unchanged.

This describes the active-learning experiment protocol as it exists in code
today (`demo.py` (historical) → `dalmax/cli.py`, `dalmax/data/datasets.py`, `dalmax/data/handlers.py`
— Phase 4 moved these from `utils/data.py`/`utils/dataset.py`, both now deleted) and
as actually invoked by the run scripts. Where the scripts diverge from
`demo.py` (historical)/`dalmax.cli` defaults, both are recorded explicitly — do not
assume the CLI default is what was actually run.

**Phase 2 update (2026-08-23)**: `demo.py` (historical) is now a thin shim routing to
`dalmax.cli.main()`; every CLI flag, results-directory path, and
`results.json` field described below is **unchanged** except two additive
CLI flags and two additive `results.json` keys, called out explicitly in
their own sections below (`--device`/`--embedding_variant`,
`all_precision_macro`/`all_recall_macro`/`all_f1_macro`) and one new
artifact (`run_metadata.json`). `scripts/benchmark/run_pipe_gpu_0.sh`/`scripts/benchmark/run_pipe_gpu_1.sh` and
every existing params JSON keep working unchanged. See
`.specs/architecture/current-state.md` §0 for what changed under the hood.

## Datasets

- **`daninhas_full`** (primary): `DATA/daninhas_full/{train,test}/DATASET_<CLASS>/`,
  5 classes (`BRACHIARIA`, `COLONIAO`, `GRAMINEA`, `MAMONA`,
  `OUTRAS_FOLHAS_LARGAS`), loaded by `dalmax.data.loaders.get_DANINHAS`
  (was `utils.data.get_DANINHAS`, moved in Phase 4), images
  resized to 128×128 RGB. See `research-rules/dataset-protocol.md` for
  per-class counts and imbalance notes.
- **`CIFAR10`**: `DATA/DATA_CIFAR10/{train,test}/`, secondary benchmark, 10
  classes, images resized to 32×32.
- Dataset selection is `--dataset_name {CIFAR10,DANINHAS}` in `demo.py` (historical);
  params are read per-dataset from the same params JSON (`params[dataset_name]`).

## Split

Train/test split is **fixed by directory layout** (`train/` vs `test/` under
each dataset root), not by a random split at run time — `X_train`/`Y_train`
and `X_test`/`Y_test` are loaded independently in `dalmax/data/loaders.py`
(was `utils/data.py`, moved in Phase 4) (`get_DANINHAS`, `get_CIFAR10`). The active-learning pool is the entire
`train/` set; `initialize_labels` randomly selects `n_init_labeled` of it as
the seed labeled set (seeded by `np.random.seed(args.seed)` in `demo.py` (historical)).
The held-out `test/` set is used for every round's evaluation
(`dataset.get_test_data()` → `strategy.predict()` → metrics).

## Seeds

`--seed {1,2,3}` — used both for `np.random.seed`/`torch.manual_seed` in
`demo.py` (historical) and as the AL initial-pool shuffle seed in
`Data.initialize_labels`. All batch runs (`scripts/benchmark/run_pipe_gpu_0.sh`,
`scripts/benchmark/run_pipe_gpu_1.sh`) sweep `SEEDS=(1 2 3)`. `scripts/benchmark/run_pipline.sh` takes
seed as a positional CLI arg (`$2`) and is invoked per seed externally (its
inner loop does not sweep seeds).

## n_init_labeled

`demo.py (historical) --n_init_labeled` default is **100**. **Neither
`scripts/benchmark/run_pipe_gpu_0.sh`/`scripts/benchmark/run_pipe_gpu_1.sh` nor `scripts/benchmark/run_pipline.sh` pass
`--n_init_labeled` explicitly** — both rely on the CLI default of 100. This
is confirmed by the results directory names actually on disk
(`NQ_*_NIL_100_NR_*_NE_10`, see `baseline-results.md`). TBD: if a future run
script changes this, update here.

## n_query (budget per round)

`--n_query {10, 50, 100}` in `scripts/benchmark/run_pipe_gpu_0.sh` / `scripts/benchmark/run_pipe_gpu_1.sh`
(`QUERIES=(10 50 100)`) and in `scripts/benchmark/run_pipline.sh` (looped separately
as 10, then 50, then 100). `demo.py` (historical) CLI default is 10 (used only if not
overridden).

## n_round

- `scripts/benchmark/run_pipe_gpu_0.sh` / `scripts/benchmark/run_pipe_gpu_1.sh`: **`--n_round 8`** (hardcoded in
  the script, not swept).
- `scripts/benchmark/run_pipline.sh`: **`N_ROUND=10`** (hardcoded).
- `demo.py` (historical) CLI default: 10.

These two run scripts are NOT protocol-equivalent — `scripts/benchmark/run_pipe_gpu_*.sh` (RNHAL /
`SSRAEKmeansHCSampling`, 8 rounds, `results/dalmax{1,2}/`) and
`scripts/benchmark/run_pipline.sh` (baseline strategies, 10 rounds, `params_dnf.json`, **file
not present in the repo — TBD** whether it existed only on the lab machine)
were evidently authored at different times. Any cross-strategy comparison
must first confirm `n_round` matches, or normalize on a common round index.

## n_epoch

`n_epoch` comes from the params JSON, **not** a `demo.py` (historical) CLI flag.
`files_config/benchmark/params_df_gpu_0.json` / `files_config/benchmark/params_df_gpu_1.json`: `DANINHAS.n_epoch = 10`,
`CIFAR10.n_epoch = 20`. `params_dnf.json` (used by `scripts/benchmark/run_pipline.sh`):
TBD, file absent from repo.

## Model / classifier

ResNet50 (`dalmax/models/daninhas_resnet50.py`, was `core/daninhas_model.py` before
Phase 4's move, TBD verify exact torchvision variant and
pretrained-weights flag — not read in this batch), trained per round via
`Strategy.train()` → `net.train(labeled_data)`. `optimizer_args`:
`lr=0.05`, `momentum=0.3` (both `files_config/benchmark/params_df_gpu_0.json` and
`files_config/benchmark/params_df_gpu_1.json`, `DANINHAS` section). `train_args`/`test_args` batch
size 256, `num_workers=4` for DANINHAS (64 / 1000, `num_workers=1` for
CIFAR10). `n_classes=5` for DANINHAS, `10` for CIFAR10.

**Phase 2 update**: the old `torch.backends.cudnn.enabled = False` global
disable (previously set in `demo.py` (historical)) is gone — `dalmax/seeding.py::seed_everything`
sets `torch.backends.cudnn.deterministic = True` / `benchmark = False`
instead, preserving cuDNN's faster kernels while staying deterministic. See
`research-rules/reproducibility.md` and `known-issues.md` KI-8 for the
performance implication on the lab GPUs, and `.specs/architecture/refactor-plan.md`
Phase 2 risks for why this means post-refactor GPU runs are not bit-identical
to historical GPU runs at the same seed (CPU runs are unaffected).

## Query strategies exercised

Full `--strategy_name` choice list (from `demo.py` (historical) → `dalmax/cli.py`, unchanged plus one addition):
`RandomSampling`, `LeastConfidence`, `MarginSampling`, `EntropySampling`,
`LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`,
`KMeansSampling`, `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`,
`AdversarialDeepFool`, `SSRAEKmeansSampling`, `VCTexKmeansSampling`,
`SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`, **`RepresentationStrategy`
(NEW, Phase 2)**, **`FullSupervised`** (2026-09-29, the paper-1 upper bound — *not* active
learning, see "Upper bound" below). The last four legacy names plus the new
`RepresentationStrategy` are all served by
`dalmax.query_strategies.representation.RepresentationStrategy` under the
hood (`.specs/architecture/target-architecture.md` §6,
`.specs/adr/0005-representation-strategy-and-registries.md`); the four
legacy names are fixed presets (pinned extractor/`Q`/selection method,
ignoring the params JSON's `"embedding"` block), while `RepresentationStrategy`
reads `"embedding"`/`"selection"` from the params JSON verbatim — see
`.specs/experiments/ablation-study.md` for the exact syntax used by the
ablations.

## New CLI flags (Phase 2, additive — every other flag/default is unchanged)

- `--device {auto,cuda,cpu}` (default `auto`): compute device for the
  network and any embedding provider/selection strategy that runs torch
  ops. `auto` resolves to `cuda` iff `torch.cuda.is_available()`, else
  `cpu` (`dalmax/config/loader.py::_resolve_device`). Existing lab-machine
  scripts should pass `--device cuda` explicitly rather than relying on
  `auto`, per `.specs/infrastructure/execution-environments.md`.
- `--embedding_variant {full,spatial,spectral}` (default `None` = no
  override): overrides `dataset.embedding.variant` from the params JSON's
  `"embedding"` block at the CLI level, for the SSRAE extractor only
  (`dalmax/cli.py::_apply_embedding_variant_override`). Lets a single params
  JSON be reused across the three §6.1 ablation runs via a CLI flag instead
  of three separate JSON files, if preferred over the JSON-per-variant
  approach `ablation-study.md`'s exact configs use.

- `scripts/benchmark/run_pipe_gpu_0.sh` / `scripts/benchmark/run_pipe_gpu_1.sh` run **only**
  `SSRAEKmeansHCSampling` (the RNHAL strategy) across the `n_query × seed`
  grid, each on its own GPU/params file, into `results/dalmax1/` and
  `results/dalmax2/` respectively.
- `scripts/benchmark/run_pipline.sh` runs the ten non-SSRAE/VCTex baseline strategies
  (all except `AdversarialBIM`/`AdversarialDeepFool`, which are commented
  out, and except `SSRAEKmeansSampling`/`VCTexKmeansSampling`/
  `SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling`) — but note this script
  only **echoes** the `poetry run python demo.py (historical) ...` command lines rather than
  executing them directly (no `eval`/backticks); it is a command generator
  meant to be piped to a shell or copy-pasted. TBD confirm this is
  intentional and how it is actually invoked on the lab machine.
- `results/dalmax1/` on disk (see `baseline-results.md`) additionally
  contains results for the full baseline set + `VCTexKmeansHCSampling`, so a
  more complete run than `scripts/benchmark/run_pipe_gpu_0.sh` alone was executed at some
  point — TBD reconcile with the exact script/commit used.

## Metrics

See `research-rules/metrics.md` for exact definitions. Summary: per round,
`demo.py` (historical)/`dalmax.experiment.runner.ExperimentRunner` records `all_acc`
(custom tensor-equality accuracy, `Data.cal_test_acc`), plus
`accuracy, precision, recall, f1_score` — **`average='weighted'`** for
precision/recall/F1 (sklearn), not macro, exactly as before. `acc_skl`
(sklearn accuracy_score) is computed but only the manual `cal_test_acc`
value is stored in `results.json`'s `all_acc` list.

**Phase 2 addition**: `Data.calc_metrics` (new method, `calc_metrics_sklearn`
unchanged and still present) also computes macro-averaged
precision/recall/F1 in the same call; `results.json` now additionally
contains `all_precision_macro`, `all_recall_macro`, `all_f1_macro` (one
value per round, same indexing as the legacy `all_*` lists) for **every**
run through `dalmax.cli`/`demo.py` (historical) from this commit onward. See
`research-rules/metrics.md` for which averaging to report where, and the
"Metrics discrepancy" note in `ablation-study.md` for pre-Phase-2
`results.json` files (which do not have these keys).

## Results directory naming convention (from `demo.py` (historical), unchanged by Phase 2)

```
{dir_results}/{dataset_folder}/SEED_{seed}/NQ_{n_query}_NIL_{n_init_labeled}_NR_{n_round}_NE_{n_epoch}/{strategy_name}/
```

where `dataset_folder = os.path.basename(params[dataset_name]['data_dir'].rstrip('/'))`
— i.e. `daninhas_full` or `DATA_CIFAR10`, taken from the params JSON's
`data_dir`, not from `--dataset_name` directly
(`dalmax/experiment/runner.py::results_dir_for`, byte-identical logic to the
pre-Phase-2 `demo.py` (historical)). Each leaf directory contains:
`results.json` (config + per-round metric lists, now including the macro
keys above), `predictions.csv` (per-test-image prediction record),
`confusion_matrix.pdf`, `accuracy.pdf`/`precision.pdf`/`recall.pdf`/
`f1_score.pdf`, `log-dalmax.log`, the saved model (`saved_model.pth`), and
**`run_metadata.json` (NEW, Phase 2)** — see below.

## Upper bound: `FullSupervised` (2026-09-29)

Paper 1's upper bound is "a full training session using the entire pool of images" (thesis
chapter 2). It is the registered strategy `FullSupervised`: every training image is labeled at
initialization (`n_init_labeled` is forced to the pool size — 8,086 for `daninhas_full`; a different
CLI value is overridden with a logged warning, and `run_metadata.json`/the results dir carry the real
size, `NIL_8086`), only round 0 runs (`--n_round` must be 0, else `ConfigError`), `query()` raises,
`n_epoch=10` as every other run, evaluation on the fixed test set, seeds {1,2,3}; the normal
artifacts are written (`results.json` with one round). **There is no validation split** in this
project's protocol (pool/test only, as in the thesis) and none is invented for the upper bound.
Results leaf: `.../NQ_100_NIL_8086_NR_0_NE_10/FullSupervised/`. Run by the campaign
(`.specs/experiments/campaign.md`); CLI form:
`--strategy_name FullSupervised --n_query 100 --n_init_labeled 8086 --n_round 0`.

## Run metadata (NEW, Phase 2)

Every leaf results directory now also contains `run_metadata.json`
(`dalmax/experiment/run_metadata.py`, written before the round loop starts),
with: the fully-resolved config as loaded (including the new `embedding`/
`selection`/`device` fields, and `params_json_path`), `git_commit`
(best-effort `git rev-parse HEAD`, `None` if unavailable), `python_version`,
`torch_version`, `cuda_available`, `started_at` (UTC ISO-8601), and (2026-09-29) `determinism` (`seed`,
`deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark`, `cublas_workspace_config`). This is
what makes `.claude/rules/reproducibility.md`'s "(params JSON + CLI args +
seed + git commit) fully determine a run" verifiable after the fact — see
`research-rules/reproducibility.md`.

## Params file per GPU / environment

- `files_config/benchmark/params_df_gpu_0.json` → `scripts/benchmark/run_pipe_gpu_0.sh` (`GPU_NUMBER=0`) →
  `results/dalmax1/`. `config_kmh`: `n_clusters=[600,200,100]`,
  `n_levels=3`, `sample_sizes=[30,15,2]`.
- `files_config/benchmark/params_df_gpu_1.json` → `scripts/benchmark/run_pipe_gpu_1.sh` (`GPU_NUMBER=1`) →
  `results/dalmax2/`. `config_kmh`: `n_clusters=[500,200,150]`,
  `n_levels=3`, `sample_sizes=[60,30,2]`.
- `params_dnf.json` → `scripts/benchmark/run_pipline.sh` — **file not in the repo**
  (TBD: lab-machine-only or lost).
- Both `files_config/benchmark/params_df_gpu_*.json` are otherwise identical for DANINHAS except
  `config_kmh`, and identical for CIFAR10.
