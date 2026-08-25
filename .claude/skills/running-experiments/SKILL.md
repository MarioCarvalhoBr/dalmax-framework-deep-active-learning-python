---
name: running-experiments
description: How to run DalMax experiments in each of the 3 execution environments - local CPU smoke test, lab machine with 2x10GB GPUs, and Google Colab Pro.
---

# Running experiments

DalMax experiments run in three distinct environments. Pick the right one for
the task — never attempt a full training sweep on the local notebook.

For the lab machine, **[`LAB_RUNBOOK.md`](../../../LAB_RUNBOOK.md)** (repo
root) is the step-by-step operator guide (one-time setup, dataset transfer,
sanity checks, the Phase 3 ablation batch, results collection) built on top
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

## 2. Lab machine — primary training, 2x NVIDIA GPUs, 10 GB each

- Workflow: commit and push from the notebook, then `git pull` + run on the lab
  machine. Results come back via git or copy (results are gitignored per
  `SHARED_CONTEXT.md`, so "via git" means the run scripts/configs, not the
  `results/` artifacts themselves — copy those back separately, e.g. `scp` or
  a shared drive).
- One params JSON per GPU: `files_config/benchmark/params_df_gpu_0.json` (used by `scripts/benchmark/run_pipe_gpu_0.sh`,
  `CUDA_VISIBLE_DEVICES=0`, writes to `results/dalmax1/`) and
  `files_config/benchmark/params_df_gpu_1.json` (used by `scripts/benchmark/run_pipe_gpu_1.sh`,
  `CUDA_VISIBLE_DEVICES=1`, writes to `results/dalmax2/`). Both currently
  configure `DANINHAS`: `n_epoch 10`, `batch_size 256`, `lr 0.05`,
  `momentum 0.3`, `n_classes 5`, `config_kmh` (`n_clusters [600,200,100]`,
  `n_levels 3`, `sample_sizes [30,15,2]`).
- Each script sweeps `QUERIES=(10 50 100)` x `SEEDS=(1 2 3)` for one strategy
  (`SSRAEKmeansHCSampling` in both current scripts) with `--n_round 8`:
  ```bash
  poetry run bash scripts/benchmark/run_pipe_gpu_0.sh   # GPU 0, results/dalmax1/
  poetry run bash scripts/benchmark/run_pipe_gpu_1.sh   # GPU 1, results/dalmax2/
  ```
  Run each under `nohup`/`tmux`/`screen` for a long unattended sweep, e.g.
  `tmux new -s gpu0 'poetry run bash scripts/benchmark/run_pipe_gpu_0.sh'`.
- For the older baseline-strategy sweep, `scripts/benchmark/run_pipline.sh` iterates all
  non-adversarial strategies (`RandomSampling`, `LeastConfidence`,
  `MarginSampling`, `EntropySampling`, the dropout variants, `KMeansSampling`,
  `KCenterGreedy`, `BALDDropout`) over `n_query` in `{10, 50, 100}` with
  `N_ROUND=10`, using `params_dnf.json` (usage:
  `bash scripts/benchmark/run_pipline.sh <gpu_id> <seed>`) — note `params_dnf.json` is
  **not committed to this repo**; it must exist locally on the lab machine
  before running this script.
- **ExperimentNotifier**: both `scripts/benchmark/run_pipe_gpu_0.sh` and `scripts/benchmark/run_pipe_gpu_1.sh` call
  `poetry run python ExperimentNotifier/main.py --dir_results=<results_dir> --args "..."`
  after the sweep finishes, which sends an HTML email via SMTP (STARTTLS,
  `smtp.gmail.com:587` by default) summarizing the run. It requires
  `ExperimentNotifier/.env` (gitignored — a separate git repo) with
  `EMAIL_FROM`, `EMAIL_TO`, `EMAIL_PASSWORD` (Gmail App Password, not the main
  account password) configured **on the lab machine independently** — it will
  not come from a `git pull` of this repo. Logs land in
  `ExperimentNotifier/logs/experiment_notifier_<timestamp>_<pid>.log`.
- Before handing off, run `.claude/commands/handoff-lab.md` (which runs the
  `experiment-auditor` agent) to catch seed/cache/params drift before spending
  GPU hours on a misconfigured run.

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
- `make ablations-colab` runs the full 11-config Phase 3 ablation sweep
  sequentially on Colab's single GPU (both
  `scripts/ablations/run_ablation_gpu_{0,1}.sh` halves, pinned to GPU 0).
  `SKIP_EXISTING=1` (default in both scripts) makes relaunching after a
  disconnect safe — only incomplete `(study, config, seed)` triples re-run.

## Which environment for which task

| Task | Environment |
|---|---|
| Writing/editing code, unit tests, ruff | Local notebook |
| Smoke-testing a new strategy on a tiny subset | Local notebook |
| Full multi-seed, multi-query sweep for the paper | Lab machine |
| Ablation study runs (`.claude/skills/ablation-study/SKILL.md`) | Lab machine |
| One-off run when the lab machine is busy | Colab Pro |

See `.specs/infrastructure/execution-environments.md` for the full decision
matrix and GPU memory guidance.
