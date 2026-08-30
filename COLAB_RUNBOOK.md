# Google Colab Pro runbook

Step-by-step notebook guide for running DalMax on Google Colab Pro (single
GPU, session-limited). This is the Colab companion to
[`LAB_RUNBOOK.md`](LAB_RUNBOOK.md) (the lab-machine equivalent) and
[`.specs/infrastructure/execution-environments.md`](.specs/infrastructure/execution-environments.md)
(architecture/decision-matrix source of truth, Colab section). Every command
below is copy-pasteable into a notebook cell in order and either exists
verbatim in this repo or is a plain `pip`/`git`/`poetry`/shell invocation —
nothing here is invented.

**Never run any of these on the local no-GPU dev notebook**
(`.claude/rules/data-safety.md`).

A runnable notebook that mirrors every cell below already exists —
[`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb) — so you
don't need to copy-paste each command by hand; open it directly in Colab and
run it top to bottom. **This file stays the source of truth**: the notebook
is generated to match it, not the other way around, so if they ever
disagree, trust this file and regenerate the notebook.

**This runbook covers both ablation suites** (§6/§7): `METHOD=rnhal`
(paper 3, SSRAE — already executed, see `.specs/experiments/ablation-study.md`)
and `METHOD=texhal` (paper 2, VCTex — not yet run, see
`.specs/experiments/ablation-study-texhal.md`). Every ablation command below
accepts `METHOD` as an environment override; see §6's "Choosing the method"
note.

## Architecture decision: hybrid local-disk + Drive-symlink layout

Colab Pro gives one GPU per session (T4/L4/A100, varies) and **no guaranteed
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
  disconnect cheap (see §6's `SKIP_EXISTING` behavior) rather than starting
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
Expect one GPU listed (T4, L4, or A100 depending on what Colab Pro assigns
this session).

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
`scripts/colab/setup_colab.sh`'s own `DRIVE_ROOT` override and §7's `cp`):
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
Expect `True 1 <gpu-name>` (e.g. `True 1 Tesla T4`). If this prints
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
`results/colab_check/` so it never collides with the real ablation/benchmark
trees living on Drive):
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
!poetry run python loader.py --model results/colab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth
```
Expect `format: dalmax-checkpoint`, a nonzero param count, and
`forward pass OK`.

Cell (single-image prediction, path adjusted to the local dataset copy):
```python
!poetry run python predict.py \
    --model results/colab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth \
    --image DATA/daninhas_full/test/DATASET_GRAMINEA/<some_image>.jpg
```
(substitute any real file under that test folder). Expect a predicted class
+ confidence line and a `predictions.csv` written alongside the image.

Only proceed to §6 once all of the above pass.

---

## 6. Run the ablation study

### Choosing the method (METHOD=rnhal | texhal)

Two suites, same as `LAB_RUNBOOK.md` §2: `METHOD=rnhal` (default, paper 3,
SSRAE — **already executed**, 11 configs / 33 runs) and `METHOD=texhal`
(paper 2, VCTex — not yet run, 12 configs / 36 runs, §6.1 has a 4th row
`rep_q13` per the VCTex method authors). Pass `METHOD` as an environment
variable to every cell below, e.g. `!METHOD=texhal make ablations-colab`.
Results land under `results/ablations/${METHOD}/`, logs under
`results/ablations/${METHOD}/colab.log`. See `LAB_RUNBOOK.md` §2 for the
full per-config GPU-split tables (identical split logic applies here, just
both halves run on Colab's one GPU 0 sequentially).

**What will run**: the same ablation batch as `LAB_RUNBOOK.md` §2 (11 or 12
configs depending on `METHOD`), but sequentially on Colab's **one** GPU
instead of split across two lab GPUs — expect roughly **2x** the per-GPU lab
wall-clock time as a rough estimate for a suite with no measured Colab
number yet.

**Measured RNHAL (2026-08-26, Colab Pro, NVIDIA T4, batch_size 256, ~10 GB VRAM in use):** the full 33-run sweep took **~5 h 15 min wall-clock** including one disconnect/relaunch (the `SKIP_EXISTING` resume worked as designed: 35 completed triples were skipped on relaunch), i.e. **~9–10 min per run**. A100 would cut the GPU part but not the CPU-bound SSRAE extraction; T4 is the cost-effective choice.

**TexHAL wall-clock: TBD (unmeasured).** VCTex's RAE extraction operates on a
27-dim per-patch input (vs. SSRAE's 9-dim), so its per-image extraction cost
is expected to differ from SSRAE's — not assumed to be faster or slower
without a measurement. Record it after the first `METHOD=texhal` run here
and feed it back into this section and `LAB_RUNBOOK.md`.

Cell (launch):
```python
!make ablations-colab                    # rnhal (default)
!METHOD=texhal make ablations-colab      # texhal
```
This runs `scripts/colab/run_ablations_colab.sh`, which is just
`METHOD=${METHOD} GPU_NUMBER=0 bash scripts/ablations/run_ablation_gpu_0.sh`
followed by `METHOD=${METHOD} GPU_NUMBER=0 bash scripts/ablations/run_ablation_gpu_1.sh`
— i.e. both lab scripts' config lists for the chosen method, both pinned to
GPU 0 — logging combined output to `results/ablations/${METHOD}/colab.log`.

**Disconnect policy**: since `results/` is symlinked to Drive and both
underlying scripts default to `SKIP_EXISTING=1`, a disconnect loses at most
the one `(study, config, seed)` triple that was mid-run when it happened.
To resume:
1. Re-run cells in §0–§4 (all idempotent — `colab-setup` detects the dataset
   and symlink are already in place and does the minimal work).
2. Re-run `!make ablations-colab` (with the same `METHOD` as before). Every
   triple with an existing `results.json` under
   `results/ablations/${METHOD}/<study>/<config>/.../results.json` is logged
   as `SKIP (already completed)` and skipped; only the incomplete or
   not-yet-run triples actually execute.

**Running a single triple manually** (e.g. to retry one that failed — see
monitoring below), same CLI pattern as `LAB_RUNBOOK.md` §3, pinned to GPU 0:
```python
!CUDA_VISIBLE_DEVICES=0 poetry run python trainer.py \
    --params_json files_config/ablations/{METHOD}/<config>.json \
    --dataset_name=DANINHAS \
    --strategy_name RepresentationStrategy \
    --n_query 100 \
    --seed <seed> \
    --n_round 8 \
    --dir_results=results/ablations/{METHOD}/<study>/<config>/ \
    --device cuda
```

**Monitoring** (from a separate cell while the sweep cell above is still
running — or after a disconnect, to see how far it got):
```python
!tail -n 30 results/ablations/{METHOD}/colab.log
!cat results/ablations/{METHOD}/gpu0_failures.log results/ablations/{METHOD}/gpu1_failures.log
```
An empty (or missing, before any run finishes) failures file means no
failures so far.

**Keeping the tab alive**: Colab Pro can still disconnect an idle browser
tab even mid-run; keep the tab focused/active, or use a keep-alive browser
extension, to reduce (not eliminate) the chance of a disconnect — either
way, the `SKIP_EXISTING` resume flow above is what actually makes a
disconnect non-fatal, not tab-keep-alive tricks.

**Drive quota note**: 33 runs x one `saved_model.pth` checkpoint each,
plus the SSRAE/ResNet embedding caches (shared across configs at the same
seed, per `LAB_RUNBOOK.md` §2's "expected embedding recomputes" note) and
logs/plots — budget roughly a few GB of Drive space for this batch (**TBD**:
no measured total exists yet; `LAB_RUNBOOK.md`'s own checkpoint-size
estimate is itself unmeasured, so treat any specific number here as a rough
planning figure, not a verified one).

---

## 7. Collect results

Cell:
```python
!make ablation-report                    # rnhal (default) -> docs/results/ablation_tables/rnhal/
!make ablation-report METHOD=texhal      # texhal -> docs/results/ablation_tables/texhal/
```
Same as `LAB_RUNBOOK.md` §4 — aggregates every discovered
`results/ablations/${METHOD}/<study>/<config>/.../results.json` into
`docs/results/ablation_tables/${METHOD}/` (`ablation_summary.csv` plus
`ablation_6_{1,2,3}.md`/`.tex`). For the already-executed legacy RNHAL batch
(no `rnhal/` segment in its results tree), use `!make ablation-report-legacy`
instead — see `LAB_RUNBOOK.md` §4.

**Committing from Colab requires git credentials** (a GitHub token or SSH
key configured in this ephemeral runtime), which this runbook does not set
up by default. Two options:

- **Recommended**: copy the tables to Drive, then commit from your local
  notebook after downloading/syncing them there:
  ```python
  !cp -r docs/results/ablation_tables/{METHOD} "$DRIVE_ROOT/results/ablation_tables/{METHOD}"
  ```
  Then, on the local notebook: pull the tables down from Drive (or however
  you sync), `git add docs/results/ablation_tables/`, and commit/push per
  `.claude/rules/git-workflow.md` (push requires your explicit confirmation).
- **Alternative**: configure a GitHub personal access token in this Colab
  session (`!git config user.email ...`, `!git config user.name ...`, and
  either `gh auth login` or a token-embedded remote URL) and commit/push
  directly from the notebook. Only do this if you're comfortable putting a
  token into a Colab runtime for the session's lifetime; the Drive-copy
  path above avoids that entirely, which is why it's the default
  recommendation.

---

## 8. Troubleshooting

- **`ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value`**
  (seen on Colab, 2026-08-25, on every `trainer.py` launch). The notebook
  front-end exports `MPLBACKEND=module://matplotlib_inline.backend_inline` to
  every subprocess, but that module only exists in Colab's system Python, not
  in the Poetry `.venv`, so `import matplotlib` failed at import time. Fixed
  in `dalmax/__init__.py` (`_ensure_matplotlib_backend` falls back to the
  headless `Agg` backend when the configured module is unimportable) and
  belt-and-braces `export MPLBACKEND=Agg` in `scripts/colab/run_ablations_colab.sh`
  / `scripts/ablations/run_ablation_gpu_{0,1}.sh`. If you see it again, you are
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
- **OOM on a T4** (10-16 GB VRAM depending on which GPU Colab assigns).
  Copy the relevant params JSON into the gitignored `files_config/local/`
  directory and lower `batch_size` there — same procedure as
  `LAB_RUNBOOK.md` §6's OOM entry; never edit a file under
  `files_config/ablations/`/`files_config/benchmark/` in place.
- **Session limit reached** (Colab Pro sessions have a maximum runtime and
  can be reclaimed). This is exactly what §6's `SKIP_EXISTING`-based resume
  flow is for — re-run §0–§4 in a fresh runtime, then re-run
  `!make ablations-colab`.
- **Python 3.13+ runtime.** Do not attempt to force-install an older Python
  inside the Colab image (not a supported/tested path for this repo — do
  not invent a workaround). If Colab's default runtime has moved to 3.13,
  the `torch==2.5.0` pin in `pyproject.toml`/`poetry.lock` has no 3.13
  wheels and must be revisited (a `poetry.lock` upgrade to a torch version
  with 3.13 support) before this runbook works again — that is a real code
  change, not something to patch around in a notebook cell.
