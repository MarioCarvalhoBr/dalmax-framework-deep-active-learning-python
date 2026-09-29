# Google Colab Pro runbook

Step-by-step notebook guide for running the DalMax **campaign** (the single
manifest that executes everything papers 1-3 need) on Google Colab Pro (single
GPU, session-limited). This is the Colab companion to
[`LAB_RUNBOOK.md`](LAB_RUNBOOK.md) (the lab-machine equivalent) and
[`.specs/infrastructure/execution-environments.md`](.specs/infrastructure/execution-environments.md).
Every command below is copy-pasteable into a notebook cell in order and either
exists verbatim in this repo or is a plain `pip`/`git`/`poetry`/shell
invocation.

**Never run any of these on the local no-GPU dev notebook**
(`.claude/rules/data-safety.md`).

The runnable notebook [`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb)
mirrors every cell below -- open it in Colab and run it top to bottom.
**This file stays the source of truth**; if they disagree, trust this file and
re-sync the notebook.

**Hardware-agnostic.** Select a GPU runtime (any). Nothing in the repo assumes a
GPU model: the GPU actually used is captured at runtime into every run's
`run_metadata.json` (`environment.gpus`), and that record is the only place a GPU
name lives. Placeholder used below: `<gpu-name>`.

## Architecture decision: hybrid local-disk + Drive-symlink layout

Colab Pro gives one GPU per session (the model varies) and **no guaranteed
background execution** — a session can disconnect mid-batch. This runbook
uses a **hybrid layout**, decided as follows:

- **Repo, `.venv`, and `DATA/daninhas_full` live on the runtime's local disk**
  (`/content/dalmax`), not on Drive. Rejected alternative: cloning the repo
  and running `poetry install` directly inside a Drive-mounted folder — Drive
  is a FUSE mount, so `poetry install`'s ~2.5 GB of CUDA wheels would be
  slow to write, and every `import` during training would pay FUSE latency
  on every file open. Local disk avoids both.
- **`results/` is replaced by a symlink to Drive**
  (`$DRIVE_ROOT/results`), so every artifact a run produces —
  `results.json`, `saved_model.pth` checkpoints, `run_metadata.json`,
  `log-dalmax.log`, plots, and the `results/cache/embeddings/` SSRAE cache —
  persists on Drive across a disconnect instead of vanishing with the
  ephemeral `/content` disk. This is what makes relaunching a batch after a
  disconnect cheap (see §6's skip-existing behavior) rather than starting
  over.
- **Dataset transfer avoids reading `daninhas_full`'s ~10,193 individual
  files through Drive's FUSE mount** (a well-known slow path — one-file-at-
  a-time I/O over a network filesystem). Instead: the first Colab session
  builds `daninhas_full.zip` from the Drive folder and stores the zip back
  on Drive (`$DRIVE_ROOT/DATA/daninhas_full.zip`, ~47 MB); every later
  session copies that single zip to `/content` and unzips it locally in
  seconds. `scripts/colab/setup_colab.sh` (§4 below) implements this and is
  idempotent — safe to re-run every session.

This is all wired up by `make colab-setup` (§4) — you don't need to build
any of it by hand.

---

## 0. Runtime check

Cell:
```python
!nvidia-smi
```
Expect one GPU listed (whichever model Colab Pro assigns this session; it is
recorded automatically in each run's `run_metadata.json`).

Cell:
```python
!python3 --version
```
**Must print 3.10, 3.11, or 3.12** — `torch==2.5.0` (pinned in
`poetry.lock`) has no wheels for 3.13+. If this prints 3.13 or higher, **stop**
here; see §8's Python-version troubleshooting entry before proceeding.

---

## 1. Mount Google Drive

Cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Verify the dataset is where this runbook expects it on Drive (quote the path
— it contains spaces and an accented character):
```python
!ls "/content/drive/MyDrive/UFMS/Pós-graduação/Doutorado/FINAL/PROJETO/DALMAX/DATA/daninhas_full" | head
```
Expect to see `train`, `test` (and possibly `arquivos.txt`) listed. If this
path doesn't exist, stop and fix the Drive layout before continuing — every
later step assumes it.

Set `DRIVE_ROOT` once for the rest of this notebook session (`%env` persists
it as a shell environment variable for every later `!`-cell, including
`scripts/colab/setup_colab.sh`'s own `DRIVE_ROOT` override and §6's `cp`):
```python
%env DRIVE_ROOT=/content/drive/MyDrive/UFMS/Pós-graduação/Doutorado/FINAL/PROJETO/DALMAX
```

---

## 2. Clone or pull the repo

Cell (first time in this Drive/account — clones into the *local* runtime
disk, not Drive, per the architecture decision above):
```python
%cd /content
!git clone https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python.git dalmax
%cd /content/dalmax
```

Cell (a later session, if `/content/dalmax` already exists from an earlier
run in the *same* still-alive runtime — rare, since Colab runtimes are
usually fresh each session, but harmless to check):
```python
%cd /content/dalmax
!git pull origin main
```

---

## 3. Install dependencies with Poetry

Cell:
```python
!pip install -q poetry
!poetry install
```
This is a **fresh runtime each session**, so `poetry install` re-downloads
the pinned `torch==2.5.0`/`torchvision==0.20.0` CUDA wheels every time —
expect roughly **5 minutes** (not previously measured on Colab specifically;
treat as an estimate, refine after your first session). It creates an
in-project `.venv/` on the local runtime disk (per `poetry.toml`'s
`in-project = true`), same as the lab machine and local notebook.

**Alternative (not recommended by default)**: `poetry config
virtualenvs.create false --local` installs dependencies into the runtime's
system Python instead of a project `.venv`. Trade-off: slightly faster (skips
creating a venv), but it **downgrades Colab's preinstalled torch build to the
pinned 2.5.0** and loses environment isolation from anything else the
notebook might `pip install`. The default in-project `.venv` (just run
`poetry install` with no config change) is recommended — it's the same
environment shape as the lab machine and local notebook, so behavior stays
consistent across all three environments.

Cell (confirm the GPU is visible to torch):
```python
!poetry run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```
Expect `True 1 <gpu-name>`. If this prints
`False ...`, stop — check the Colab runtime type is set to a GPU runtime
(Runtime → Change runtime type → GPU) and re-run from §0.

---

## 4. Wire up the dataset and results/ (Drive)

Cell:
```python
!make colab-setup
```
This runs `scripts/colab/setup_colab.sh`, which:
1. Asserts Drive is mounted and the dataset path from §1 exists.
2. Creates `$DRIVE_ROOT/results/` on Drive if it doesn't exist yet
   (append-only — never emptied by this script, `.claude/rules/data-safety.md`).
3. **First session only**: builds `daninhas_full.zip` from the Drive
   dataset folder and stores it back on Drive
   (`.../DALMAX/DATA/daninhas_full.zip`) — slower this one time, since it
   reads the ~10,193 files through Drive's FUSE mount to build the zip.
   **Later sessions reuse this zip** and skip straight to the fast path.
4. Copies the ~47 MB zip to `/content` and unzips it into this repo's
   `DATA/daninhas_full/` (local disk, fast reads for training).
5. Verifies the local file count is exactly `10193`; fails loudly if not
   (compare `DATA/daninhas_full/arquivos.txt` against a known-good copy,
   e.g. the lab machine's, if this happens).
6. Symlinks `results/` in the repo to `$DRIVE_ROOT/results` — refuses
   (rather than clobbering) if a real, non-empty `results/` directory
   already exists locally.

Expected output ends with something like:
```
============================================================
Setup complete.
  DATA/daninhas_full: 10193 files (local disk, fast reads)
  results/ -> /content/drive/MyDrive/.../DALMAX/results (Drive, persists across disconnects)
============================================================
```
Override the Drive path if your project folder differs from the default:
`DRIVE_ROOT="/content/drive/MyDrive/other/path" bash scripts/colab/setup_colab.sh`.

---

## 5. Sanity checks

Cell (fast unit tests, no GPU/dataset needed):
```python
!make test
```

Cell (CPU smoke test — generates the micro dataset from the local
`DATA/daninhas_full` copy, then a true end-to-end `trainer.py` run on it):
```python
!make smoke
```

Cell (first real-data GPU check on Colab this session, into
`results/colab_check/` so it never collides with the real campaign
tree living on Drive):
```python
!make colab-check
```
This is the same check as `LAB_RUNBOOK.md` §1's `make lab-check`, pinned to
Colab's single GPU 0. Expect artifacts under
`results/colab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/`
(`results.json`, `predictions.csv`, `run_metadata.json`, plots,
`saved_model.pth`, `log-dalmax.log`) — see `LAB_RUNBOOK.md` §1 for the full
expected-artifact list and durations (same caveats: no exact measured
Colab GPU epoch time is recorded anywhere in this repo yet — **TBD**, record
one the first time this actually runs on Colab).

Cell (verify the checkpoint is real and loadable, same as `LAB_RUNBOOK.md` §1):
```python
!poetry run python tools/loader.py --model results/colab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth
```
Expect `format: dalmax-checkpoint`, a nonzero param count, and
`forward pass OK`.

Cell (single-image prediction, path adjusted to the local dataset copy):
```python
!poetry run python tools/predict.py \
    --model results/colab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth \
    --image DATA/daninhas_full/test/DATASET_GRAMINEA/<some_image>.jpg
```
(substitute any real file under that test folder). Expect a predicted class
+ confidence line and a `predictions.csv` written alongside the image.

Only proceed to §6 (the campaign) once all of the above pass.

---

## 6. The campaign

One manifest (`files_config/campaign/manifest.json`, see
[`.specs/experiments/campaign.md`](.specs/experiments/campaign.md) and ADR 0008/0009)
drives the single, deduplicated execution of **everything the three papers need** -- paper 1
(12 classical strategies + KMH x n_query {10,50,100} + the `FullSupervised` upper bound), paper 2
(TexHAL ablations, nq=100), paper 3 (RNHAL ablations incl. 5 new hierarchy rows, nq=100) -- with
seeds (1,2,3) and one protocol in one environment. Sections 0-5 (runtime, Drive, clone, Poetry,
`make colab-setup`, sanity checks) come first; then:

1. `!make campaign-list` -- prints the job table and the counts. Real numbers (full scale):

   | Part | Runs | Run groups |
   |---|---|---|
   | `paper1` | 117 | 39 |
   | `upper_bound` | 3 | 1 |
   | `rnhal` | 42 | 14 |
   | `texhal` | 30 | 10 |
   | **Total** | **192** | **64** |

2. **Run everything**: `!make campaign-run PART=all`. Jobs run sequentially, skip-existing (a job is
   done only when its leaf holds the **full** artifact set; `results.json` is written last),
   logging to `results/campaign/campaign.log` and failures to `results/campaign/failures.log`
   (the batch continues after a failure). **Resume after a disconnect = redo sections 0-4 and
   re-run the same command** (`results/` is the Drive symlink of `make colab-setup`). Order: paper 1
   at nq100 first (shared runs land early), then rnhal, texhal, paper 1 at nq50 and nq10, then the
   upper bound.
3. **Check the environment before leaving it unattended**: the very first job of `PART=all` is the
   cheapest one (RandomSampling, nq=100, seed 1, group `shared/random_nq100`). Once it has finished,
   verify that its `run_metadata.json` reports the GPU you expect:
   ```
   !python -c "import json,glob; m=json.load(open(glob.glob('results/campaign/shared/random_nq100/*/SEED_1/*/RandomSampling/run_metadata.json')[0])); print(m['environment']['gpus'][0]['name'], m['determinism'])"
   ```
   It prints `<gpu-name>` (whatever Colab assigned; `environment.gpus[0].name`) and the `determinism`
   block (`deterministic_algorithms: "warn_only"`, cuDNN flags, cuBLAS workspace,
   `nondeterministic_op_warnings`). If `gpus` is empty, the runtime has no GPU -- stop and fix it.
   Determinism is best-effort: an op without a deterministic kernel only warns and is listed in
   `nondeterministic_op_warnings`; re-runs on a different GPU model are not bit-identical.
4. **Per-part launches** to split across sessions: `!make campaign-run PART=paper1`,
   `PART=rnhal`, `PART=texhal`, `PART=upper_bound` (comma-separated works: `PART=rnhal,texhal`).
   Papers 2/3 comparison tables need paper 1's `shared/*` runs and `shared/texhal_full`, so run
   `paper1` and `texhal` before reporting.
5. **Monitor**: `!tail -n 30 results/campaign/campaign.log` and `!cat results/campaign/failures.log`.
6. **Verify**: `!make campaign-verify` -- OK/INCOMPLETE/MISSING per job with the artifact checks of
   `dalmax/reporting/leaf_check.py`, plus the **seed-consistency audit** (for every seed, all runs
   must have started from the identical initial labeled set; PASS/WARN/FAIL per seed, offenders
   listed). Exit code 0 = all OK, 1 = an incomplete/missing job or a FAIL audit, 2 = only audit
   WARNs (a run whose log lacks the initial-labeled-set line). `!make campaign-verify PART=rnhal`
   restricts to one part. Re-run failed jobs by re-running `campaign-run`.
7. **Report**: `!make campaign-report` writes `docs/results/campaign/` (per-paper tables in md/tex,
   `summary.csv`, `seed_audit.md`, mean confusion matrices under `confusion_matrices/`), then copy to
   Drive: `!cp -r docs/results/campaign "$DRIVE_ROOT/results/campaign_tables"`.

**CPU smoke (not on Colab).** The micro campaign (`make campaign-smoke`, local CPU, seed 1) skips
the two adversarial baselines (`AdversarialBIM`, `AdversarialDeepFool`) because they are far too slow on
CPU; it runs 58 of the 64 micro jobs. The real GPU campaign above runs them (`--exclude-strategy`
is only passed by the smoke targets).

**Wall-clock: TBD.** Record the measured hours (and the `<gpu-name>` from step 3) here after the
first full run; the adversarial/dropout baselines are likely the slowest jobs (unmeasured).

**Drive quota.** Every run writes a `saved_model.pth` checkpoint (ResNet-50, ~95 MB): 192 runs is
about **18 GB** on Drive, plus the embedding caches and logs/plots. Check your Drive quota before
starting `PART=all` (or split by part across sessions).

The upper bound (`FullSupervised`) trains once on all 8,086 pool images (ResNet-50, 10 epochs, batch
256) -- if the GPU runs out of memory, lower nothing silently: report it, it is the same batch size as
every other run.

---

## 7. Troubleshooting

- **`ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`**
  (seen on Colab, 2026-08-25, on every `trainer.py` launch). The notebook
  front-end exports `MPLBACKEND=module://matplotlib_inline.backend_inline` to
  every subprocess, but that module only exists in Colab's system Python, not
  in the Poetry `.venv`, so `import matplotlib` failed at import time. Fixed
  in `dalmax/__init__.py` (`_ensure_matplotlib_backend` falls back to the
  headless `Agg` backend when the configured module is unimportable) and
  `MPLBACKEND=Agg` is exported into every `dalmax.campaign` subprocess (`dalmax/campaign.py::build_env`). If you see it again, you are
  running a checkout older than that fix: `git pull origin main`.
- **Drive FUSE slowness.** Never read `daninhas_full`'s individual files
  directly from `/content/drive/...` during training — that's exactly why
  §4 copies a single zip to local disk instead. If you ever see very slow
  epoch times, check `--data_dir`/the params JSON actually point at the
  local `DATA/daninhas_full/` copy, not a Drive path.
- **`Transport endpoint is not connected`** (Drive FUSE mount dropped mid-
  session). Remount: re-run §1's `drive.mount('/content/drive')` cell (it's
  safe to call again), then re-run `!make colab-setup` before continuing.
- **`results/` symlink already exists / local `results/` directory
  non-empty.** `scripts/colab/setup_colab.sh` refuses to overwrite an
  existing symlink that points somewhere unexpected, or a real non-empty
  local `results/` directory, rather than silently clobbering either. Read
  the script's error message — it tells you the exact conflicting path;
  resolve manually (e.g. `mv results results_local_backup` if you really
  want to switch to the Drive symlink) and re-run.
- **Out of GPU memory** (VRAM depends on which GPU Colab assigns).
  Copy the relevant params JSON into the gitignored `files_config/local/`
  directory and lower `batch_size` there — same procedure as
  `LAB_RUNBOOK.md`'s OOM entry; never edit a committed file under
  `files_config/` in place.
- **Session limit reached** (Colab Pro sessions have a maximum runtime and
  can be reclaimed). This is exactly what the campaign's skip-existing resume
  flow is for — re-run §0–§4 in a fresh runtime, then re-run
  `!make campaign-run PART=all`.
- **Python 3.13+ runtime.** Do not attempt to force-install an older Python
  inside the Colab image (not a supported/tested path for this repo — do
  not invent a workaround). If Colab's default runtime has moved to 3.13,
  the `torch==2.5.0` pin in `pyproject.toml`/`poetry.lock` has no 3.13
  wheels and must be revisited (a `poetry.lock` upgrade to a torch version
  with 3.13 support) before this runbook works again — that is a real code
  change, not something to patch around in a notebook cell.

---
