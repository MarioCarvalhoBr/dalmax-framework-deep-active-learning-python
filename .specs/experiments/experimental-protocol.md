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

- **`daninhas_full_fixed`** (primary since 2026-11-11, ADR 0016): `DATA/daninhas_full_fixed/{train,test}/DATASET_<CLASS>/`,
  the duplicate-free version of `daninhas_full` (KI-46): 7,230 train / 1,008 test images, no MD5 repeated inside or
  across the splits, plus the metadata of the layout of 2026-11-12: `README.txt` (the dataset as it is: identity
  rule, counts per split and class, how to verify), `MANIFEST.csv` (`split,class,file,md5`, one row per image) and
  `logs/` (`build_history.txt`: builder, exclusion list, source file count and what was removed and why;
  `source_manifest.csv`: the manifest with the dropped source copies and aliases) = 8,242 files (8,240 before the
  layout). Built once by `scripts/datasets/build_daninhas_full_fixed.py` (refuses an existing destination;
  `--check` / `make dataset-check` verifies the layout, the files on disk against `MANIFEST.csv` by name and MD5, and
  no repeated MD5); the existing folder is converted once by `--migrate-metadata` (metadata only, images untouched,
  `.claude/rules/data-safety.md`); immutable afterwards. No loader, order hash or cache key reads the metadata files:
  the data order depends only on the paths under `train/` and `test/`. Every full-scale params JSON of the campaign,
  the KMH sweep, the ablations and the lab/Colab checks points to it (`"data_dir": "DATA/daninhas_full_fixed/"`), so
  the results leaf folder and the embedding cache key read `daninhas_full_fixed`. Same 5 classes and loader as below.
- **`daninhas_full`** (primary until 2026-11-11; the dataset of the first campaign, kept immutable as history):
  `DATA/daninhas_full/{train,test}/DATASET_<CLASS>/` (8,086 train / 2,106 test files, with byte-identical duplicates
  inside and across the splits, KI-46); the source of `daninhas_full_fixed` and of the micro dataset
  `DATA/daninhas_micro/` (golden fixtures, unchanged),
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

**Model head and `model_args` (2026-11-11, ADR 0014).** `DaninhasModelResNet50(n_classes, dropout_p=0.0)`: the
ImageNet-pretrained ResNet-50 up to the global average pool (2048-d features), an optional `nn.Dropout(dropout_p)`
on those features, then `classifier = nn.Linear(2048, n_classes)`. `forward` returns `(logits, e1)` with `e1` = the
2048-d pooled features (what `get_embeddings` returns; KI-42). A legacy `embedding_layer = nn.Linear(2048, 50)` is
still constructed (RNG stream and checkpoint keys) but unused. The optional params-JSON block of a dataset entry
`"model_args": {"dropout_p": <float in [0, 1)>}` sets the model's constructor arguments (unknown keys and invalid
values raise `ConfigError`; `"_comment"` keys allowed); absent = nothing passed = the model's own default (DANINHAS:
no dropout layer, exactly the pre-2026-11-11 model; CIFAR10 CNN: its built-in DeepAL dropout 0.5). The resolved value
appears in `run_metadata.json` under `config.dataset.model_args` (`{"dropout_p": null}` when absent) and in the
checkpoint (`model_kwargs`). Only the four MC-dropout strategies of paper 1 train with `dropout_p = 0.5`
(`files_config/campaign/params_paper1_mc_dropout.json`). MC prediction (`predict_prob_dropout*`) runs with the model
in eval mode and only the dropout modules in train mode, `n_drop` passes (10 in the campaign).

**Phase 2 update**: the old `torch.backends.cudnn.enabled = False` global
disable (previously set in `demo.py` (historical)) is gone — `dalmax/seeding.py::seed_everything`
sets `torch.backends.cudnn.deterministic = True` / `benchmark = False`
instead, preserving cuDNN's faster kernels while staying deterministic. See
`research-rules/reproducibility.md` and `known-issues.md` KI-8 for the
performance implication on the lab GPUs, and `.specs/architecture/refactor-plan.md`
Phase 2 risks for why this means post-refactor GPU runs are not bit-identical
to historical GPU runs at the same seed (CPU runs are unaffected).

## Query strategies exercised

Full `--strategy_name` choice list (from `demo.py` (historical) → `dalmax/cli.py`, unchanged plus one addition;
plus the eight DeepAL+ ports of 2026-11-11 listed after the caveats below):
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

Caveats on three families (found 2026-11-02 and 2026-11-11; **fixed in code 2026-11-11, ADR 0014**; the first
campaign execution ran with the defects; they are re-run inside the clean campaign on `daninhas_full_fixed`, ADR 0016): the
MC-dropout strategies (`LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`, `BALDDropout`)
ran on a model with no dropout layer, so they were not MC-dropout and BALD's mutual information was zero up to
noise (`quality/known-issues.md` KI-41); they now need `"model_args": {"dropout_p": ...}` (campaign:
`files_config/campaign/params_paper1_mc_dropout.json`, p = 0.5) and fail fast without it. `KMeansSampling` and
`KCenterGreedy` used an untrained random projection as embedding (KI-42); they now use the trained 2048-d pooled
features. `AdversarialBIM` broke distance ties in pool order (KI-45); ties are now broken at random with a
generator derived from the seed (`derive_seed(seed, "adversarial_ties")`); `AdversarialDeepFool` keeps the pool-order
tie-break of its executed runs (its only exact ties are byte-identical duplicate images).

**DeepAL+ ports (2026-11-11, ADR 0015; in paper 1 of the campaign since ADR 0016).** Eight further
`--strategy_name` choices port every method of DeepAL+ (Zhan et al., 2022, arXiv 2203.13450) that DalMax lacked.
A ninth new choice, `WAALRandomQuery` (2026-11-12, ADR 0015 amendment), is a CONTROL, not a benchmark strategy:
WAAL's training procedure unchanged (it reads the `"WAAL"` block below; there is no `"WAALRandomQuery"` block, a
params JSON that sets one is rejected) with a uniform random query from the unlabeled pool
(`derive_seed(seed, "waal_random_query")`); for a seed it trains WAAL's round-0 model exactly. It measures how much of
a WAAL gain over `RandomSampling` comes from WAAL's training (about 280 SGD steps per round against 10 to 40) rather
than from its selection. Hyperparameter blocks:
`VarRatio`, `MeanSTD` (both MC dropout: run with `files_config/campaign/params_paper1_mc_dropout.json`,
`dropout_p` 0.5, fail fast without it), `KCenterGreedyPCA`, `BadgeSampling`, `CEALSampling`, `LossPredictionLoss`,
`VAAL` (`files_config/campaign/params_paper1_vaal.json`: its defaults except `image_size: 32`, the author's cost
decision for the campaign on `daninhas_full_fixed`, ADR 0016), `WAAL` (all others: `params_paper1.json`). Since
ADR 0016 all eight are in paper 1 of the campaign at `n_query` 10/50/100. Every protocol knob (seeds, `n_init_labeled`, `n_query`,
`n_round`, `n_epoch`, optimizer) is shared with the other strategies. Three of them change the training of the task
model: `LossPredictionLoss` trains a loss-prediction module jointly (same `n_epoch`, the module's gradient reaches the
backbone during the first `floor(0.6 * n_epoch)` epochs), `WAAL` replaces the cross-entropy training by its min-max
training on labeled + unlabeled data (every round, round 0 included, so its round-0 model differs from the others'),
and `CEALSampling` adds pseudo-labelled samples to the next training only (round 0 is plain training). `VAAL` trains
the task model normally and a VAE + discriminator at every query (images area-resized from 128 to `image_size` for
the VAE: default 64, 32 in the campaign). Deviations
from DeepAL+ (including two upstream defects it fixes: VAAL's discriminator targets and VarRatio collapsing to
LeastConfidence) are listed in ADR 0015. `KMeansSamplingGPU` is not ported (same method as `KMeansSampling`).

**`strategy_args` params block (2026-11-11, ADR 0015).** Optional per-dataset block of per-strategy hyperparameters,
keyed by strategy name; every key optional (defaults below = `dalmax/config/schema.py::StrategyArgs`), unknown
strategy names or keys and wrongly typed values raise `ConfigError` (`"_comment"` keys allowed); the fully
resolved block (all five sub-blocks, defaults included) appears in `run_metadata.json` under
`config.dataset.strategy_args`:

```json
"strategy_args": {
  "CEALSampling": {"base_strategy": "EntropySampling", "delta": 0.05, "delta_decay": 0.0033, "delta_schedule": "linear"},
  "KCenterGreedyPCA": {"n_components": 50},
  "LossPredictionLoss": {"margin": 1.0, "weight": 1.0, "detach_fraction": 0.6, "interm_dim": 128, "module_lr": 0.01},
  "VAAL": {"epochs": 30, "latent_dim": 32, "beta": 1.0, "adversary_param": 1.0,
           "num_vae_steps": 2, "num_adv_steps": 1, "lr": 0.0005, "image_size": 64},
  "WAAL": {"alpha": 0.001, "gp_weight_task": 5.0, "gp_weight_critic": 2.0, "gamma_ratio": 1.0, "selection": 10.0}
}
```

`CEALSampling.base_strategy` is one of `LeastConfidence`, `MarginSampling`, `EntropySampling`; its entropy threshold at
query round `t` (1-based, clamped at 0) is `delta - delta_decay * t` with `delta_schedule: "linear"` (the default) or
`delta - delta_decay * t (t + 1) / 2` with `"cumulative"` (DeepAL+'s literal reading). The defaults `delta` 0.05 and
`delta_decay` 0.0033 are the CEAL paper's CACD setting (DeepAL+'s 5e-5 / 3.3e-7 pseudo-labels nothing on DANINHAS,
ADR 0015).

**`strategy_diagnostics`** (reporting only, never fed back into selection or training): all eight DeepAL+ ports write
per-round series under this optional `results.json` key (every other strategy writes no such key, so their schema is
unchanged). Every port records `pool_order_fraction` per query (the KI-45 signature: fraction of the batch among the
first `2n` unlabeled pool positions; about `2n/|U|` for an informative strategy, 1.0 for a collapse to pool order),
and six of them add their own series:
`VarRatio` `var_ratio_n_nonzero`; `MeanSTD` `mean_std_score_mean`, `mean_std_score_std`; `CEALSampling` `ceal_round`,
`ceal_delta`, `ceal_n_pseudo_labels`, `ceal_pseudo_label_accuracy` (null when nothing was pseudo-labelled) and, per
training, `ceal_train_size`; `LossPredictionLoss` `lpl_labeled_spearman`, `lpl_n_labeled`, `lpl_pool_score_std`,
`lpl_pool_score_n_unique`; `VAAL` `vaal_discriminator_auc`, `vaal_score_std`, `vaal_score_n_unique`, `vaal_mean_kl`,
`vaal_mu_dim_std`; `WAAL` (per training) `waal_n_steps`, `waal_wasserstein_last_epoch` and (per query)
`waal_critic_labeled_mean/std`, `waal_critic_unlabeled_mean/std`, `waal_critic_init_corr`, `waal_uncertainty_std`,
`waal_critic_term_std`; the WAAL control `WAALRandomQuery` the same training series and critic statistics
(`waal_critic_*`), no `waal_uncertainty_std`/`waal_critic_term_std` (its query does not use those terms), and
`query_rule` = `"uniform_random"` per query. `KCenterGreedyPCA` and `BadgeSampling` record `pool_order_fraction` only. The campaign's
pilot report prints them (`.specs/experiments/campaign.md`, "Pilot").

For deduplication (`dalmax.campaign.effective_signature`, ADR 0016) only the block of the strategy being run counts:
two params files that differ only in another strategy's `strategy_args` block are the same computation for every
strategy that does not read it.

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
— i.e. `daninhas_full_fixed` (or the historical `daninhas_full`) or `DATA_CIFAR10`, taken from the params JSON's
`data_dir`, not from `--dataset_name` directly
(`dalmax/experiment/runner.py::results_dir_for`, byte-identical logic to the
pre-Phase-2 `demo.py` (historical)). Each leaf directory contains:
`results.json` (config + per-round metric lists, now including the macro
keys above), `predictions.csv` (per-test-image prediction record),
`confusion_matrix.pdf`, `accuracy.pdf`/`precision.pdf`/`recall.pdf`/
`f1_score.pdf`, `log-dalmax.log`, the saved model (`saved_model.pth`),
**`run_metadata.json` (NEW, Phase 2)** — see below — and, since 2026-11-11 (ADR 0016), the per-round history:

- `round_predictions.json` (compact JSON): `{"format": "dalmax-round-predictions", "version": 1, "class_names":
  [...], "rounds": [0, 1, ...], "test_paths": [...], "test_labels": [...], "predictions": [[...], ...]}`, where
  `predictions[r][i]` is the predicted class index of test image `i` (path `test_paths[i]`, true class
  `test_labels[i]`) after round `r`; the last row equals `predictions.csv`.
- `round_queries.json` (compact JSON): `{"format": "dalmax-round-queries", "version": 1, "class_names": [...],
  "n_pool": N, "rounds": [{"round": 0, "kind": "initial", "pool_idxs": [...], "labels": [...], "paths": [...]},
  {"round": 1, "kind": "query", ...}, ...]}`: the sorted pool indices labeled in each round (round 0 = the initial
  labeled set), their true classes and image paths. Pool indices follow the loaders' sorted order (classes = sorted
  folder names, files = sorted names in each class folder; KI-47, `run_metadata.json` `data_order`); compare runs by
  the paths (`<class>/<file>`), as the campaign's seed audit does.

Both are additive (`results.json`, `predictions.csv` and the log lines are unchanged) and written before
`results.json`. `dalmax/reporting/leaf_check.py` requires them only for manifests that list them in
`required_leaf_files` (every manifest of the clean campaign), so the leaves of the earlier trees still verify.

## Upper bound: `FullSupervised` (2026-09-29)

Paper 1's upper bound is "a full training session using the entire pool of images" (thesis
chapter 2). It is the registered strategy `FullSupervised`: every training image is labeled at
initialization (`n_init_labeled` is forced to the pool size — 7,230 for `daninhas_full_fixed`, 8,086 for the
historical `daninhas_full`; a different CLI value is overridden with a logged warning, and `run_metadata.json`/the
results dir carry the real size, `NIL_7230`), only round 0 runs (`--n_round` must be 0, else `ConfigError`), `query()` raises,
`n_epoch=10` as every other run, evaluation on the fixed test set, seeds {1,2,3}; the normal
artifacts are written (`results.json` with one round). **There is no validation split** in this
project's protocol (pool/test only, as in the thesis) and none is invented for the upper bound.
Results leaf: `.../NQ_100_NIL_7230_NR_0_NE_10/FullSupervised/`. Run by the campaign
(`.specs/experiments/campaign.md`); CLI form:
`--strategy_name FullSupervised --n_query 100 --n_init_labeled 7230 --n_round 0`.

## Run metadata (NEW, Phase 2)

Every leaf results directory now also contains `run_metadata.json`
(`dalmax/experiment/run_metadata.py`, written before the round loop starts),
with: the fully-resolved config as loaded (including the new `embedding`/
`selection`/`device` fields, and `params_json_path`), `git_commit`
(best-effort `git rev-parse HEAD`, `None` if unavailable), `python_version`,
`torch_version`, `cuda_available`, `started_at` (UTC ISO-8601), (2026-09-29) `determinism` (`seed`,
`deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark`, `cublas_workspace_config`), and (2026-11-12,
KI-47) `data_order` (`contract` = `"sorted-names-v1"`, `train_paths_hash` and `test_paths_hash` = the first 12 hex
digits of the SHA-256 of the ordered `<class>/<file>` paths of each split, `n_train`, `n_test`; what the run's pool
and test indices name, compared across leaves by `campaign verify`). This is
what makes `.claude/rules/reproducibility.md`'s "(params JSON + CLI args +
seed + git commit) fully determine a run" verifiable after the fact — see
`research-rules/reproducibility.md`.

## Hierarchy knobs in `selection.hierarchy` / `config_kmh` (2026-11-07)

Besides `n_clusters`, `n_levels`, `sample_sizes`, the hierarchy block accepts four optional keys
(defaults = the previous hard-wired behaviour, so every existing params file is unchanged):
`n_resamples` (int >= 0, default `10`; `0` = no resampling), `init_method` (`"kmeans++"` |
`"random"`, default `"kmeans++"`), `resample_strategy` (`"closest"` | `"random"`, default
`"closest"`), `leaf_sampling` (`"random"` | `"closest"`, default `"random"`). Invalid values raise
`ConfigError` naming the key and the valid values; the resolved values appear in
`run_metadata.json` under `config.dataset.selection.hierarchy`.

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
