---
name: running-experiments
description: How to run DalMax experiments in each of the 3 execution environments - local CPU smoke test, lab machine with one or more GPUs, and Google Colab Pro.
---

# Running experiments

DalMax experiments run in three distinct environments. Pick the right one for
the task — never attempt a full training sweep on the local notebook.

For the lab machine, **[`LAB_RUNBOOK.md`](../../../LAB_RUNBOOK.md)** (repo
root) is the step-by-step operator guide (one-time setup, dataset transfer,
sanity checks, the campaign, results collection) built on top
of the summary in §2 below — follow it directly for an actual lab session
rather than reconstructing the steps from this skill's prose.

## 1. Local dev notebook (this machine) — CPU only, smoke tests

- 16 GB RAM, **no GPU**, Python 3.12, Poetry 2.2.1.
- Use for: coding, smoke tests on tiny subsets, running reports, running
  `ruff`/`pytest`.
- Never run a full `n_round`/multi-seed sweep here — there is no GPU, and
  ResNet50 training over `daninhas_full` (~10,193 images) is not CPU-feasible
  for a real experiment.
- Fast checks:
  ```bash
  poetry install
  poetry run pytest -m "not gpu and not dataset"
  poetry run ruff check .
  make smoke   # tiny CPU end-to-end run on a small subset — see Makefile
  ```
- See `.claude/commands/smoke-test.md` for the command form.

## 2. Lab machine -- primary training (one or more GPUs)

- Workflow: commit and push from the notebook, then `git pull` + run on the lab machine. Results come
  back via git or copy (`results/` is gitignored, so "via git" means the code/configs, not the
  artifacts -- copy those back separately, e.g. `scp` or a shared drive).
- Everything runs through the campaign (ADR 0008/0009/0010): `make campaign-list`,
  `make campaign-run PART=all` (skip-existing; resume = re-run), `make campaign-verify`,
  `make campaign-report`. With two GPUs, one process per GPU:
  `CUDA_VISIBLE_DEVICES=0 make campaign-run PART=paper1,upper_bound` and
  `CUDA_VISIBLE_DEVICES=1 make campaign-run PART=rnhal,texhal`. Run under `tmux`/`nohup` for a long
  unattended batch. The hardware-agnostic rule applies: never name a GPU model; it is captured at
  runtime in `run_metadata.json` (`environment.gpus`).
- `files_config/benchmark/params_df_gpu_{0,1}.json` (`DANINHAS`: `n_epoch 10`, `batch_size 256`,
  `lr 0.05`, `momentum 0.3`, `n_classes 5`, `config_kmh`) remain the reference params used by
  `make lab-check GPU=<n>` / `make colab-check`.
- The legacy sweep scripts (`scripts/benchmark/*`, `scripts/ablations/*`) and `ExperimentNotifier`
  hooks in them were retired on 2026-09-29 (ADR 0010).

## 3. Google Colab Pro — secondary/burst, one-off runs

**[`COLAB_RUNBOOK.md`](../../../COLAB_RUNBOOK.md)** (repo root) is the
step-by-step, numbered-notebook-cell operator guide — follow it directly for
an actual Colab session rather than reconstructing the steps from this
skill's summary below.

- **Hybrid layout** (decided 2026-08-25, see
  `.specs/infrastructure/execution-environments.md`'s "Colab: hybrid
  local-disk + Drive-symlink layout" section): the repo, its `.venv`, and
  `DATA/daninhas_full` all live on the Colab runtime's **local disk**
  (`/content/dalmax`) for fast reads and a normal `poetry install`; `results/`
  is replaced by a **symlink to Google Drive** so `results.json`,
  checkpoints, logs, and the embedding cache all persist across a session
  disconnect. `make colab-setup` (`scripts/colab/setup_colab.sh`) wires this
  up idempotently every session.
- Dataset transfer never reads `daninhas_full`'s ~10,193 individual files
  directly from Drive (a well-known slow FUSE path). Instead, the first
  session zips the Drive dataset folder into
  `$DRIVE_ROOT/DATA/daninhas_full.zip` (~47 MB) and stores it back on Drive;
  every later session copies that single zip to `/content` and unzips it
  locally in seconds.
- `pipx install poetry`/`pip install poetry` then `poetry install` — same
  Poetry-only step as every other environment (the `requirements.txt`/pip
  fallback was retired, see `.specs/adr/0001-adopt-poetry.md`'s amendment),
  keeping Colab on the exact `pyproject.toml`/`poetry.lock` pins (requires
  Python 3.10-3.12; `torch==2.5.0` has no 3.13 wheels).
- `make campaign-run PART=all` runs the campaign on Colab's single GPU; skip-existing makes relaunching after a
  disconnect safe -- only incomplete jobs re-run (see `COLAB_RUNBOOK.md`).

## Which environment for which task

| Task | Environment |
|---|---|
| Writing/editing code, unit tests, ruff | Local notebook |
| Smoke-testing a new strategy on a tiny subset | Local notebook |
| Full multi-seed, multi-query sweep for the paper | Lab machine |
| Campaign runs (`.specs/experiments/campaign.md`) | Lab machine or Colab Pro |
| One-off run when the lab machine is busy | Colab Pro |

See `.specs/infrastructure/execution-environments.md` for the full decision
matrix and GPU memory guidance.
