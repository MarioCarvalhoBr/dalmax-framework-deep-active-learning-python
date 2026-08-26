# Execution environments

Three environments, with distinct roles. All tooling generated for this
project (Makefile targets, CI, agent/skill definitions owned by other
batches) must respect this split — never assume GPU or a full dataset copy
is available locally.

For the lab machine, this file is the architecture/decision-matrix source of
truth; **[`LAB_RUNBOOK.md`](../../LAB_RUNBOOK.md)** (repo root) is the
operator-facing step-by-step guide built on top of it — one-time setup,
dataset transfer, sanity checks, the Phase 3 ablation batch launch/monitoring,
and results collection, all via the `make lab-setup`/`lab-check`/
`ablations-gpu{0,1}`/`ablations-all`/`ablation-report`/`benchmark-gpu{0,1}`/
`micro-dataset` targets.

For Google Colab Pro, **[`COLAB_RUNBOOK.md`](../../COLAB_RUNBOOK.md)** (repo
root) is the equivalent operator-facing guide (numbered notebook cells), built
on top of the "Colab: hybrid local-disk + Drive-symlink layout" section below,
via the `make colab-setup`/`colab-check`/`ablations-colab` targets.
[`notebooks/colab_runbook.ipynb`](../../notebooks/colab_runbook.ipynb) is the
runnable notebook generated to mirror `COLAB_RUNBOOK.md` cell-for-cell (that
file stays the source of truth).

## Phase 2 lab-handoff hazards (2026-08-23)

Read this before the first lab-machine run after pulling Phase 2 (`refactor/phase-2-core` or later):

- **Pass `--device cuda` explicitly.** `dalmax/cli.py`'s new `--device` flag defaults to `"auto"`,
  which resolves to `"cuda"` iff `torch.cuda.is_available()` — this should always be true on the lab
  machine, but do not rely on `auto` silently doing the right thing during a batch you cannot watch
  live (e.g. if CUDA drivers are misconfigured, `auto` would silently fall back to `cpu` and the run
  would be extremely slow rather than failing loudly). Add `--device cuda` to
  `scripts/benchmark/run_pipe_gpu_0.sh`/`scripts/benchmark/run_pipe_gpu_1.sh` (and any new ablation run scripts) explicitly.
- **`num_workers`/`worker_init_fn` landmine if augmentation is re-enabled.** `dalmax/data/handlers.py`
  (was `utils/dataset.py`) currently has every stochastic `transforms.Random*` augmentation commented
  out, so `train_args`/`test_args`' `num_workers=4` (DANINHAS) / `num_workers=1` (CIFAR10) is safe
  today. If anyone uncomments an augmentation transform, `dalmax/models/base.py`'s (was
  `core/deep_learning.py`) `DataLoader(...)` calls have no `worker_init_fn` — worker processes are
  not independently reseeded for any
  `random`/`numpy`-global-state-based augmentation, which `dalmax/seeding.py::seed_everything` does
  not cover (it seeds the main process only). See `.specs/quality/known-issues.md` KI-31. Do not
  re-enable augmentation on the lab machine without adding a `worker_init_fn` first, or determinism
  claims for that run are unverified.
- **`--n_round 0` is allowed again** (train/evaluate once, no active-learning query rounds at all)
  — `dalmax.config.schema.ExperimentConfig` briefly rejected it during Phase 2 development before
  being corrected to match `demo.py` (historical)'s always-permissive behavior (`n_round >= 0`, `n_query > 0`
  still required). Safe to use for a single-shot baseline run; `ExperimentRunner`/`write_report`
  handle it correctly (one recorded/plotted point, confusion matrix from round 0's predictions —
  see `tests/test_runner_n_round_zero.py`).
- **New embedding cache location**: `results/cache/embeddings/` (not `results/cache/` — that path
  still exists too, from the Phase 1 fix, but is dead code as of Phase 2, see
  `.specs/architecture/current-state.md` §0). If copying `results/` back from the lab machine for
  cache reuse across sessions, include `results/cache/embeddings/`, not just `results/cache/`.
- **`run_metadata.json` is now the fastest way to confirm what actually ran** — before trusting a
  batch of results, spot-check a few leaf directories' `run_metadata.json` (`config.dataset.embedding`,
  `config.dataset.selection`, `config.device`, `git_commit`) against the intended params JSON/CLI
  args, rather than reverse-engineering it from the directory name alone.

## Decision matrix

| Environment | Hardware | Role | Allowed operations |
|---|---|---|---|
| **Local dev notebook** (this machine) | 16 GB RAM, **no GPU**, Python 3.12, Poetry 2.2.1 | Coding, code review, spec/doc authoring, smoke tests on tiny subsets, report generation from results already copied back | `poetry` commands, `ruff`, `pytest` (fast tests only), `python -c` import checks, `dalmax/reporting/*.py` against already-present `results/` data, `make smoke` (see below). **Never** a real training run — no GPU, and a full `daninhas_full` epoch would be impractically slow on CPU. |
| **Lab machine** (primary training) | 2× NVIDIA GPUs, **10 GB VRAM each** | All real experiment execution: baseline sweeps, RNHAL reference runs, and (once implemented) the ablation study runs | `scripts/benchmark/run_pipe_gpu_0.sh` (GPU 0, `files_config/benchmark/params_df_gpu_0.json` → `results/dalmax1/`), `scripts/benchmark/run_pipe_gpu_1.sh` (GPU 1, `files_config/benchmark/params_df_gpu_1.json` → `results/dalmax2/`), long-lived via `tmux`/`nohup`; `ExperimentNotifier/main.py` sends an email after each script's full battery completes. |
| **Google Colab Pro** | Single GPU (T4/L4/A100, varies; **measured on a T4, 2026-08-25/26**), session-limited, no guaranteed background execution, burst/secondary | One-off runs, retries of a specific config, or running the full Phase 3 ablation sweep sequentially on one GPU when the lab machine is busy | `make colab-setup` (repo/`.venv`/`DATA/daninhas_full` on the runtime's local disk for fast reads and a normal `poetry install`; `results/` replaced by a symlink to Drive so artifacts survive a disconnect — see "Colab: hybrid local-disk + Drive-symlink layout" below), then `make colab-check` / `make ablations-colab` (both scripts/ablations/run_ablation_gpu_{0,1}.sh halves, `SKIP_EXISTING=1` by default so a relaunch after a disconnect only re-runs incomplete `(study, config, seed)` triples). See [`COLAB_RUNBOOK.md`](../../COLAB_RUNBOOK.md). **Measured timing (Phase 3 ablation batch, one T4, 33 runs)**: ~5 h 15 min total wall-clock including one disconnect/relaunch, ~9-10 min/run, ~10 GB VRAM at `batch_size=256` — this is close to the 10 GB lab GPUs' full capacity, so a Colab-sized batch run on the lab machine carries real **OOM risk** and should not be assumed to fit headroom-free alongside another job on the same GPU. |

## Local smoke testing: the micro dataset

Since the local dev notebook has no GPU and must never run a real training pass on
`daninhas_full`, `make smoke` (see `.specs/architecture/refactor-plan.md` Phase 1,
`.specs/quality/testing-strategy.md`) instead runs a true end-to-end `demo.py` (historical) pass on a small,
deterministic **10%-stratified replica of `daninhas_full`**, generated by
`scripts/make_micro_dataset.py`:

- Reads only from `DATA/daninhas_full/{train,test}/<class>/` for **all 5** classes
  (`DATASET_BRACHIARIA`, `DATASET_COLONIAO`, `DATASET_GRAMINEA`, `DATASET_MAMONA`,
  `DATASET_OUTRAS_FOLHAS_LARGAS`) — never writes there (`DATA/daninhas_full/` stays immutable per
  `.claude/rules/data-safety.md`).
- Writes a fixed-seed, sorted-filename, per-class-stratified sample
  (`max(1, floor(0.10 * n_files_in_class))` files per `(split, class)`, ~806 train + ~209 test
  images total) into `DATA/daninhas_micro/{train,test}/<class>/` — also under the gitignored
  `DATA/`, so it is regenerated on demand and never committed. `DATA/daninhas_micro/arquivos.txt`
  is written/refreshed on every run, in the same format as `DATA/daninhas_full/arquivos.txt`.
- If `DATA/daninhas_full/` isn't present on a given machine (e.g. a fresh clone with no dataset
  copied in), the script prints a message and exits 0 rather than failing `make smoke`.
- `files_config/params_micro.json` points DANINHAS at `DATA/daninhas_micro/` with `n_epoch=1`,
  `n_classes=5`, `batch_size=16`, `num_workers=0` — a full `demo.py (historical) --strategy_name RandomSampling`
  run on this config takes on the order of 30-35 seconds on CPU (plus a one-time ~98 MB ResNet50
  ImageNet-weights download on the very first run, cached under
  `~/.cache/torch/hub/checkpoints/` afterward).
- `tests/golden/random_sampling_micro_seed1.json` and `tests/golden/ssrae_kmeans_micro_seed1.json`
  pin the exact selected indices and metrics this configuration must reproduce;
  `tests/test_golden_run.py` (marked `dataset`+`slow`, not part of `make smoke`/CI's fast job) is the
  regression test for it. Both strategies were verified deterministic across repeated runs on this
  machine (see `.specs/quality/known-issues.md`'s "Determinism verification" note).
- **2026-08-23 redefinition** (`.specs/adr/0004-micro-dataset-and-golden-run.md`'s "micro-dataset
  redefinition" amendment): this dataset used to be a hand-picked 2-class
  (`DATASET_BRACHIARIA`/`DATASET_GRAMINEA`), fixed 25-train/10-test-per-class subset. It is now the
  10%-stratified, all-5-class replica described above, so local pre-lab/pre-Colab smoke testing
  exercises the same class count and imbalance shape as a real experiment instead of a 2-class toy
  case — a more realistic pre-flight check before handing a change off to the lab machine or Colab.

## Handoff workflow (local ↔ lab)

1. Code changes are made and reviewed on the local notebook.
2. `git push` from local; `git pull` on the lab machine.
3. Run the appropriate `scripts/benchmark/run_pipe_gpu_{0,1}.sh` on the lab machine (one
   script per GPU — each has its own params JSON, so the two GPUs can run
   different hierarchy/model configs concurrently without file contention).
4. Results (`results/dalmax{1,2}/...`) come back to the local machine via
   `git` (if ever tracked — currently `results/` is gitignored, see
   `experiments/baseline-results.md`) or a manual copy
   (`scp`/`rsync`/shared drive) — **TBD**: no committed convention for this
   copy-back step was found in the repo; document it once established
   (likely `rsync` given `results/` is gitignored by design, to avoid
   polluting the git history with large binary artifacts).
5. `ExperimentNotifier/` (a separate, gitignored repo:
   `ExperimentNotifier/main.py --dir_results=... --args=...`) emails a
   notification once a `scripts/benchmark/run_pipe_gpu_*.sh` battery finishes — used as the
   lab-machine-to-human handoff signal; not inspected beyond its CLI
   invocation in the run scripts in this batch (its own README, if any, is
   the source of truth — TBD read it in a future pass).
6. **KNOWN EXTERNAL REFERENCE (Phase 4 follow-up, non-blocking)**:
   `ExperimentNotifier/main.py` calls `utils/report/build_method_metrics.py`
   in a legacy CIFAR10-only email branch. Refactor Phase 4
   (`.specs/architecture/refactor-plan.md`) moved that script to
   `dalmax/reporting/build_method_metrics.py` in **this** repo — but
   `ExperimentNotifier/` is a separate, gitignored repo not touched by this
   refactor, so its own copy of that path must be updated by hand on the lab
   machine the next time `ExperimentNotifier/` is edited there. This is
   non-blocking for DANINHAS runs (that email branch is already inert for
   them, CIFAR10-only), but will break if that branch is ever exercised
   against a `results/` tree produced after this move.

## Colab: hybrid local-disk + Drive-symlink layout (decided 2026-08-25)

**Decision**: on Colab, clone the repo and create `.venv` on the runtime's
**local disk** (`/content/dalmax`), copy `DATA/daninhas_full` onto local disk
too (fast reads during training), and replace `results/` with a **symlink**
to a Google Drive folder — so every artifact a run produces (`results.json`,
`saved_model.pth` checkpoints, `run_metadata.json`, `log-dalmax.log`, plots,
and the `results/cache/embeddings/` SSRAE cache) persists on Drive across a
session disconnect. Implemented by `scripts/colab/setup_colab.sh` (`make
colab-setup`); full operator guide: **[`COLAB_RUNBOOK.md`](../../COLAB_RUNBOOK.md)**.

**Rejected alternative**: cloning the repo and running `poetry install`
directly inside a Drive-mounted folder. Drive is exposed to the Colab runtime
as a FUSE mount — writing `poetry install`'s ~2.5 GB of CUDA wheels through
it is slow, and every subsequent Python `import` during training would pay
FUSE latency on file opens. Local disk avoids both costs; the tradeoff is
that the repo/`.venv` do **not** survive a session disconnect (a fresh
`git clone` + `poetry install` is needed every session) — acceptable, since
that's a ~5 minute cost (network + wheel install), versus training progress
being unrecoverable if it lived only on `/content` without the `results/`
symlink.

**Dataset transfer**: never read `daninhas_full`'s ~10,193 individual files
directly from Drive (one-file-at-a-time I/O over Drive's FUSE mount is a
well-known slow path). Instead: the first Colab session zips the Drive
dataset folder into `$DRIVE_ROOT/DATA/daninhas_full.zip` (~47 MB) and stores
it back on Drive; every later session copies that single zip to `/content`
and unzips it locally in seconds. File count is verified against the known
`10193` after unzip; a mismatch fails the setup script loudly rather than
proceeding with a partial dataset.

**Checkpointing / disconnect recovery**: since `results/` is a Drive symlink
and `trainer.py` already writes `results.json` + `saved_model.pth` per
strategy leaf directory (not just at the end of a whole sweep), a dropped
Colab session loses at most the one `(study, config, seed)` triple that was
mid-run — nothing already written is lost, because it was never only on the
ephemeral `/content` disk in the first place. `scripts/ablations/
run_ablation_gpu_{0,1}.sh`'s `SKIP_EXISTING=1` default (see this file's
"Phase 2 lab-handoff hazards" note above, which the same scripts also serve
on the lab machine) makes relaunching a batch after a disconnect skip every
already-completed triple automatically — see `COLAB_RUNBOOK.md` §6.

**Poetry/torch pin**: `pipx install poetry`/`pip install poetry` then
`poetry install`, same as every other environment — DalMax is Poetry-only,
and `pyproject.toml`/`poetry.lock` exact-pin `torch`/`torchvision`, so this
does not depend on whatever torch build a given Colab image ships
preinstalled. Requires Python 3.10-3.12 (the `torch==2.5.0` pin has no 3.13
wheels) — `COLAB_RUNBOOK.md` §0 checks this before proceeding.

## GPU memory guidance

`files_config/benchmark/params_df_gpu_{0,1}.json` both configure DANINHAS training with
`batch_size=256` (train and test) on ResNet50, `n_epoch=10`, on a 10 GB GPU
— this is the configuration actually used for the lab machine's reference
RNHAL runs (`results/dalmax1/`, `results/dalmax2/`), so **256 is a value
known to fit in 10 GB VRAM for this model/image size (128×128 RGB)**, though
the exact peak memory usage was **not measured in this batch** (would
require running `nvidia-smi` during an actual training step, which needs
GPU access this environment does not have). TBD: measure and record actual
peak VRAM for `batch_size=256` at 128×128 the next time a lab-machine
session is available; also record CIFAR10's smaller `batch_size=64`
(32×32 images) memory footprint for comparison, and whether the two GPUs
running concurrently (one job per GPU, per `scripts/benchmark/run_pipe_gpu_0.sh` /
`scripts/benchmark/run_pipe_gpu_1.sh`) share any resource (e.g. shared dataset loading, disk
I/O) that could bottleneck parallel throughput even with `CUDA_VISIBLE_DEVICES`
correctly isolating compute.

**Proxy measurement available (2026-08-25/26, Colab, not the lab machine)**: the Phase 3 ablation
batch's `RepresentationStrategy` runs used the same `batch_size=256`/128×128-image configuration on
a single NVIDIA T4 and measured ~10 GB VRAM — i.e. essentially the full capacity of the lab
machine's 10 GB GPUs, with little to no headroom. This is a same-model/same-batch-size proxy, not a
lab-machine measurement (still TBD as stated above), but it upgrades the "not measured" status to
"measured on comparable hardware, consistent with the 10 GB budget being tight rather than
comfortable" — see the environment comparison table above and
`.specs/quality/known-issues.md` for any issue this implies for concurrent multi-job scheduling on
the lab machine.

## Lab handoff — Phase 3 ablation batch (pre-flighted 2026-08-23, verdict GO)

```bash
git fetch origin && git checkout refactor/phase-3-ablations && git pull
poetry install                            # lab env, same as local dev — Poetry only
# ExperimentNotifier/.env must exist on the lab machine (gitignored)
tmux new -s gpu0   # poetry run bash scripts/ablations/run_ablation_gpu_0.sh   (5 configs)
tmux new -s gpu1   # poetry run bash scripts/ablations/run_ablation_gpu_1.sh   (6 configs)
# check results/ablations/gpu{0,1}_failures.log afterwards (should be empty)
poetry run python -m dalmax.reporting.ablation_report --root results/ablations --out docs/results/ablation_tables
git add docs/results/ablation_tables/ && git commit -m "Add Phase 3 ablation study results" && git push
```

Audit facts: expected embedding recomputes = 3 SSRAE + 3 ResNet (one per seed; all SSRAE configs
share one cache per pool); SSRAE extraction ~2 min CPU per pool (measured 12.13 ms/image);
GPU load split ratio ~1.1x; no VRAM/MEMORY_LIMIT concern at ~10k x 756-d/2048-d scale.
Non-blocking: ExperimentNotifier email shows a generic message for these runs (no STRATEGY_1 token).

The block above is equivalent to `make lab-setup`, `tmux new -s gpu0` / `make
ablations-gpu0`, `tmux new -s gpu1` / `make ablations-gpu1`, and `make
ablation-report` — see `LAB_RUNBOOK.md` §0/§3/§4 for the fuller step-by-step
version (dataset-arrival verification, a `make lab-check` real-data sanity
run before committing to the full batch, monitoring, and the single-run
re-run command pattern for a failed `(study, config, seed)` triple).
