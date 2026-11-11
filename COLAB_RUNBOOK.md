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
This is the **system** Python and may be anything: since 2026-09-29 Colab ships
3.13.x, which is expected. `torch==2.5.0` (pinned in `poetry.lock`) has no
3.13 wheels, so §3 installs a separate uv-managed Python 3.12 for the Poetry
venv; that venv interpreter (3.10-3.12) is what actually gets validated.

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
!git clone https://github.com/MarioCarvalhoBr/dalmax-framework-deep-active-learning-python.git dalmax
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

Colab's system Python is 3.13, so `poetry install` cannot use it (the project
requires `>=3.10,<3.13`). Install Python 3.12 with `uv` and point Poetry at it.
Cells (run from the repo root; each step is idempotent on re-run):
```python
!pip install -q uv poetry
!uv python install 3.12
%cd /content/dalmax
!poetry env use "$(uv python find 3.12)"
!poetry install
```
Then verify the venv interpreter (must start with `3.12`):
```python
!poetry run python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
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
This is the same check as `LAB_RUNBOOK.md` §1's `make lab-check` (both use
`files_config/campaign/params_paper1.json`), pinned to Colab's single GPU 0. The notebook's
next cell prints the GPU name/memory/driver/CUDA from that run's `run_metadata.json`:
**confirm it is the GPU you intended before launching the campaign.** Expect artifacts under
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
   | `paper1` | 54 | 18 |
   | `paper1_rerun_a` | 24 | 8 |
   | `paper1_rerun_b` | 39 | 13 |
   | `upper_bound` | 3 | 1 |
   | `rnhal` | 42 | 14 |
   | `texhal` | 30 | 10 |
   | `texhal_q17` | 18 | 6 |
   | `hier_transfer` | 12 | 4 |
   | **Total** | **222** | **74** |

2. **Run a part** (or everything with `PART=all`): `!make campaign-run PART=paper1`. Jobs run sequentially, skip-existing (a job is
   done only when its leaf holds the **full** artifact set; `results.json` is written last),
   logging to `results/campaign/campaign.log` and failures to `results/campaign/failures.log`
   (the batch continues after a failure). **Resume after a disconnect = redo sections 0-4 and
   re-run the same command** (`results/` is the Drive symlink of `make colab-setup`). Order: paper 1
   at nq100 first (shared runs land early), then rnhal, texhal, texhal_q17, hier_transfer, paper 1 at nq50 and nq10, then
   `paper1_rerun_a` and `paper1_rerun_b`, then the upper bound. Since 2026-11-11 (ADR 0014) the 21 groups of the seven
   paper-1 baselines fixed that day (63 of the first execution's 117 `paper1` runs) are the parts `paper1_rerun_a` /
   `paper1_rerun_b`; they keep their ids and results paths, see section 6d.
3. **Check the environment before leaving it unattended**: the notebook prints the GPU identity
   right after `!make colab-check` (section 5), from that run's `run_metadata.json`
   (`environment.gpus[0]`: name, `total_memory_gb`, `driver_version`; `environment.torch.cuda`; the
   `determinism` block). Confirm it is the GPU you intended **before launching the campaign**. If
   `gpus` is empty, the runtime has no GPU -- stop and fix it. Every campaign run records the same
   block, so the GPU used is always recoverable per run. Determinism is best-effort: an op without a
   deterministic kernel only warns and is listed in `nondeterministic_op_warnings`; re-runs on a
   different GPU model are not bit-identical.
4. **Per-part launches** to split across sessions (the notebook defaults to `PART = "paper1_rerun_a"` since
   2026-11-11, section 6d). Order used for the first execution: `PART=paper1` (then 117 runs; since ADR 0014
   `paper1` is 54 runs plus `paper1_rerun_a` 24 and `paper1_rerun_b` 39), `PART=rnhal` (42), `PART=texhal` (30),
   `PART=texhal_q17` (18; TexHAL Q=17 re-run of 6.2/6.3, ADR 0011), `PART=upper_bound` (3); comma-separated works (`PART=rnhal,texhal`). Re-running the same `PART`
   resumes (completed runs are skipped), so a disconnect loses at most the run in progress.
   Papers 2/3 comparison tables need paper 1's `shared/*` runs and `shared/texhal_full`, so run
   `paper1` and `texhal` (and `texhal_q17` for the Q=17 6.2/6.3 rows) before reporting.
5. **Progress / monitor** (the notebook's Progress cell): `!make campaign-verify PART=...` plus
   `!tail -n 20 results/campaign/campaign.log`; failures go to `results/campaign/failures.log`.
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
CPU; it runs 68 of the 74 micro jobs. The real GPU campaign above runs them (`--exclude-strategy`
is only passed by the smoke targets).

**Wall-clock: TBD.** Record the measured hours (and the `<gpu-name>` from step 3) here after the
first full run; the adversarial/dropout baselines are likely the slowest jobs (unmeasured).

**Drive quota.** Every run writes a `saved_model.pth` checkpoint (ResNet-50, ~95 MB): 222 runs is
about **20 GB** on Drive, plus the embedding caches and logs/plots. Check your Drive quota before
starting `PART=all` (or split by part across sessions).

The upper bound (`FullSupervised`) trains once on all 8,086 pool images (ResNet-50, 10 epochs, batch
256) -- if the GPU runs out of memory, lower nothing silently: report it, it is the same batch size as
every other run.

---

## 6c. Hierarchy transfer runs (main campaign, `hier_transfer`; ADR 0013)

Twelve runs (4 groups x seeds 1-3, `n_query=100`): the KMH-sweep winner shape (`k=[600,100]`, `sample_sizes=[30,5]`,
no resampling) and the reference shape without resampling, with the SSRAE (RNHAL, `results/campaign/rnhal/6_2/`) and VCTex
Q=17 (TexHAL, `results/campaign/texhal_q17/6_2/`) descriptors. They are a part of the MAIN manifest, not of the sweep.
Prepared 2026-11-10; estimated 2 to 4 GPU hours, about 1.1 GB of checkpoints.

**Executed** 2026-11-11 (UTC) in one Colab session (code commit `eee5d4e`), on a third runtime image with a different GPU model than the earlier campaign runs (see `environment.gpus[0].name` in `run_metadata.json`). `make campaign-verify PART=hier_transfer` gave `OK=12`, and after merging into the main `results/` tree `make campaign-verify` gave `OK=222 INCOMPLETE=0 MISSING=0`, seed audit PASS for seeds 1, 2, 3. Outcome in `.specs/experiments/campaign.md` (section "Hierarchy transfer part").

**Notebook setup.** In the Parameters cell of `notebooks/colab_runbook.ipynb` keep `MANIFEST = ""` and set
`PART = "hier_transfer"` (it was the default from 2026-11-10 until 2026-11-11; the default is now `paper1_rerun_a`, section 6d). Do sections 0-5 first (runtime, Drive, clone,
Poetry, `make colab-setup`, `make test`, `make colab-check`) and confirm the GPU (`<gpu-name>`) as in section 6.

1. Pull the latest commit: `!cd /content/dalmax && git pull` (or redo sections 0-4, which clone `main` fresh).
2. List: `!make campaign-list` (expect `TOTAL 222 runs (74 run groups)`, `hier_transfer 12 runs (4 run groups)`).
3. Run: `!make campaign-run PART=hier_transfer`. Skip-existing: re-running after a disconnect resumes.
4. Verify: `!make campaign-verify PART=hier_transfer` (expect `OK=12 INCOMPLETE=0 MISSING=0`; without `PART`, `OK=222`
   once the 210 runs executed earlier are present on the same `results/` tree) and the seed audit PASS.
5. Report: `!make campaign-report`, then copy to Drive: `!cp -r docs/results/campaign "$DRIVE_ROOT/results/campaign_tables"`.

---

## 6d. Re-run of the seven fixed paper-1 baselines (`paper1_rerun_a` + `paper1_rerun_b`; ADR 0014)

**Prepared 2026-11-11, not executed.** Seven paper-1 baselines ran in a degenerate form in the first execution and were
fixed on 2026-11-11 (`.specs/quality/known-issues.md`): the four MC-dropout strategies had no dropout layer (KI-41),
`KMeansSampling`/`KCenterGreedy` clustered an untrained random projection (KI-42), and `AdversarialBIM` broke its many
distance ties in pool order, i.e. by class (KI-45). Their 21 groups (`p1/<Strategy>/nq{10,50,100}`, 63 runs) are re-run;
everything else of the campaign is not. The new runs land in the **same leaf paths** as the first execution
(`results/campaign/paper1/nq{10,50,100}/daninhas_full/SEED_<s>/NQ_<nq>_NIL_100_NR_8_NE_10/<Strategy>/`), so the tables and
the thesis keep working, and the superseded leaves are first **moved** (never deleted) to `results/old_results_broken/`
with the same path below `results/`.

The work is split over **two Colab accounts running at the same time**, balanced by the GPU time measured in the first
execution (per run: BIM about 2,300 s, MC-dropout strategies about 230 s, KMeans/KCenter about 90 s):

| Account | `PART` | Runs | Groups | Content | GPU time (estimate) | Checkpoints |
|---|---|---|---|---|---|---|
| A | `paper1_rerun_a` | 24 | 8 | `AdversarialBIM` nq100 and nq50; `KMeansSampling`, `KCenterGreedy` at nq 10/50/100 | about 4.3 h (6 x 2,300 s + 18 x 90 s) | about 2.3 GB (24 x ~95 MB) |
| B | `paper1_rerun_b` | 39 | 13 | `AdversarialBIM` nq10; `LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`, `BALDDropout` at nq 10/50/100 | about 4.2 h (3 x 2,300 s + 36 x 230 s) | about 3.7 GB (39 x ~95 MB) |

The times are for the GPU class of the first execution; on another `<gpu-name>` they scale accordingly (check
`environment.gpus[0].name` in the first run's `run_metadata.json`). KMeans/KCenter will be somewhat slower than measured
because they now work on the 2048-d trained features instead of a 50-d projection. Add about 15 minutes per session for
sections 0-5. The MC-dropout groups use `files_config/campaign/params_paper1_mc_dropout.json` (= `params_paper1.json` plus
`"model_args": {"dropout_p": 0.5}`): only these four strategies train with a dropout layer, so their round-0 model differs
from the other strategies' (inherent to MC dropout). The new runs execute on a later runtime image than the five unchanged
paper-1 strategies (KI-43: compare seed means, not single runs).

**Per account (A and B do the same steps with their own `PART`):**

1. **Drive prerequisites.** Each account has its own Google Drive, so the dataset must be on THAT account's Drive:
   the folder `$DRIVE_ROOT/DATA/daninhas_full/` (the section 1 check cell asserts it). `make colab-setup` (section 4)
   builds `$DRIVE_ROOT/DATA/daninhas_full.zip` from that folder on the account's first session (slow once, through Drive's
   FUSE mount) and reuses the zip in every later session; to skip the slow build, also copy `daninhas_full.zip` (~47 MB,
   e.g. from the other account's Drive or the local `DATA/daninhas_full.zip`) next to the folder. If the second account's
   Drive does not use the same folder path, set `DRIVE_ROOT` in the Parameters cell to the folder that holds `DATA/` on
   that Drive. `results/` becomes a symlink to `$DRIVE_ROOT/results` of that account. Free Drive space: 2.3 GB (A) or
   3.7 GB (B) plus the zip of the same size (step 10); archiving the old runs is a rename, it takes no extra space.
2. **Parameters cell** of `notebooks/colab_runbook.ipynb`: `MANIFEST = ""` (default) and `PART = "paper1_rerun_a"` (default,
   account A) or `PART = "paper1_rerun_b"` (account B). The fix must be on the branch the notebook clones (`BRANCH`):
   `main` once `fix/degenerate-baselines` is merged, or set `BRANCH = "fix/degenerate-baselines"` until then.
3. **Sections 0-5** (runtime, Drive, clone, Poetry, `make colab-setup`, `make test`, `make smoke`, `make colab-check`, GPU
   identity check). In a still-alive runtime, update the code instead: `!cd /content/dalmax && git pull`, then
   `!git log --oneline -1`.
4. **Archive, dry run**: `!make campaign-archive-superseded DRY=1`. On the Drive that holds the first-execution results
   (`results/campaign/paper1/` complete) it lists `WOULD MOVE` for the 63 superseded leaves (`would move=63`); on a Drive
   without old campaign results it reports `missing=63` and nothing to move, which is fine.
5. **Archive**: `!make campaign-archive-superseded`. Expect `moved=63` (or `moved=0 ... missing=63` on a Drive without old
   results). Every move is logged in `results/old_results_broken/ARCHIVE_LOG.tsv`. Re-running is safe
   (`already_archived=63`), and leaves started after the cutoff (the new runs) are never moved. Exit status 1 means a
   `CONFLICT` (the destination already exists; nothing is merged or overwritten) or an `UNREADABLE` leaf (no readable
   `run_metadata.json`): read the listed paths and resolve them by hand before going on.
6. **Check**: `!make campaign-archive-superseded CHECK=1` must print `CHECK OK`. A `STALE` leaf would be skipped as
   complete by `campaign-run`, so never launch while the check fails.
7. **List**: `!make campaign-list PART=paper1_rerun_a` (B: `PART=paper1_rerun_b`). Expect `paper1_rerun_a 24 runs (8 run
   groups)` and `TOTAL 24 runs (8 run groups)` (B: 39 runs, 13 groups). Without `PART`: `paper1 54 runs (18 run groups)`,
   `TOTAL 222 runs (74 run groups)`.
8. **Run**: `!make campaign-run PART=paper1_rerun_a` (the notebook's run cell). After a disconnect or a session limit:
   redo sections 0-4, run steps 4-6 again (they find nothing more to move) and the same `campaign-run` command:
   skip-existing resumes, at most the run in progress is lost.
9. **Verify**: `!make campaign-verify PART=paper1_rerun_a`: expect `OK=24 INCOMPLETE=0 MISSING=0` (B: `OK=39`) and the
   seed audit PASS for seeds 1-3. Without `PART` the other parts show up as MISSING on a Drive that does not hold them;
   that is expected.
10. **Zip only the new leaves of the part** (the notebook's 6d zip cell does exactly this; run step 6 first so that the
    globs can only match new runs):

    Account A:
    ```python
    !make campaign-archive-superseded CHECK=1
    !cd /content/dalmax && zip -qr /content/paper1_rerun_a.zip \
        results/campaign/paper1/nq100/*/SEED_*/NQ_100_*/AdversarialBIM \
        results/campaign/paper1/nq50/*/SEED_*/NQ_50_*/AdversarialBIM \
        results/campaign/paper1/nq*/*/SEED_*/NQ_*/KMeansSampling \
        results/campaign/paper1/nq*/*/SEED_*/NQ_*/KCenterGreedy
    !unzip -l /content/paper1_rerun_a.zip | grep -c saved_model.pth   # expect 24
    !cp /content/paper1_rerun_a.zip "$DRIVE_ROOT/"                    # about 2.3 GB
    ```

    Account B:
    ```python
    !make campaign-archive-superseded CHECK=1
    !cd /content/dalmax && zip -qr /content/paper1_rerun_b.zip \
        results/campaign/paper1/nq10/*/SEED_*/NQ_10_*/AdversarialBIM \
        results/campaign/paper1/nq*/*/SEED_*/NQ_*/LeastConfidenceDropout \
        results/campaign/paper1/nq*/*/SEED_*/NQ_*/MarginSamplingDropout \
        results/campaign/paper1/nq*/*/SEED_*/NQ_*/EntropySamplingDropout \
        results/campaign/paper1/nq*/*/SEED_*/NQ_*/BALDDropout
    !unzip -l /content/paper1_rerun_b.zip | grep -c saved_model.pth   # expect 39
    !cp /content/paper1_rerun_b.zip "$DRIVE_ROOT/"                    # about 3.7 GB
    ```

    The zip stores the paths as `results/campaign/paper1/...` (it follows the `results/` symlink); download it from Drive.

**Merging the downloads into the authoritative `results/` tree** (the machine that holds all 222 runs, e.g. the local one),
in this order:

1. `make campaign-archive-superseded DRY=1`, then `make campaign-archive-superseded`: moves the 63 old leaves of that tree
   to `results/old_results_broken/`. **This must happen before unzipping**: unzipping first would overwrite the old files
   in place and the first execution would be lost.
2. At the repo root: `unzip -n paper1_rerun_a.zip` and `unzip -n paper1_rerun_b.zip` (`-n`: never overwrite).
3. `make campaign-archive-superseded CHECK=1`: `CHECK OK`.
4. `make campaign-verify`: `OK=222 INCOMPLETE=0 MISSING=0`, seed audit PASS for seeds 1-3. Then `make campaign-report`.

---

## 6b. The KMH parameter sweep (paper 1; ADR 0012)

The sweep is a separate manifest (`files_config/campaign/manifest_kmh_sweep.json`, ADR 0012,
[`.specs/experiments/campaign.md`](.specs/experiments/campaign.md) "KMH parameter sweep") that varies the
KMH selection (hierarchy shapes including budget-matched ones, k-means init, leaf sampling, resampling)
to decide whether paper 1's KMH reference changes. It runs 3 seeds at n_query 10/50/100 into a NEW tree,
`results/campaign_kmh/`, so it never touches `results/campaign/`. Executed and verified (stage 1 on 2026-11-07/08, stage 2 on 2026-11-09/10; 165/165 OK).

| Part | Runs | Run groups |
|---|---|---|
| `kmh_nq10` | 51 | 17 |
| `kmh_nq50` | 51 | 17 |
| `kmh_nq100` | 63 | 21 |
| **Total** | **165** | **55** |

Stage 2 (prepared 2026-11-08, nq100 only, 10 groups / 30 runs; ADR 0012) is part of `kmh_nq100`: re-running the
same `PART=all` command resumes and executes only the 30 new runs, estimated 8 to 10 GPU hours (the L4/L5
shapes cost about 2 min of clustering per round). Stage 2 was executed 2026-11-09 to 2026-11-10 and verified locally 165/165 on 2026-11-10.

**Notebook setup (the only edit you make).** In [`notebooks/colab_runbook.ipynb`](notebooks/colab_runbook.ipynb)
the Parameters cell (second cell) is the only one to touch. Keep `DRIVE_ROOT`, `REPO_URL`, `REPO_DIR` and
`BRANCH = "main"` as they are and set:

```python
MANIFEST = "files_config/campaign/manifest_kmh_sweep.json"   # already the default
PART = "all"                                                 # already the default
```

These are the notebook's defaults since 2026-11-07, so nothing needs editing: `PART = "all"` runs the
three parts in the order `kmh_nq100`, `kmh_nq50`, `kmh_nq10` (the `PARTS` order of `dalmax/campaign.py`)
and every session resumes where the previous one stopped. Every campaign cell below (list, run, progress,
verify, report, copy to Drive) reads these two variables: with the sweep manifest set, the cells write to
`results/campaign_kmh/` and `docs/results/campaign_kmh/` and copy the tables to
`$DRIVE_ROOT/results/campaign_kmh_tables`. Set `MANIFEST = ""` to go back to the main campaign of
section 6. The numbered steps below are the same cells, in order; `PART=...` is only needed to restrict.

1. Do sections 0-5 first (runtime, Drive, clone, Poetry, `make colab-setup`, `make test`, `make smoke`,
   `make colab-check`), including the GPU identity check (step 3 of section 6): confirm the GPU
   (`<gpu-name>`) before launching.
2. List: `!make campaign-list MANIFEST=files_config/campaign/manifest_kmh_sweep.json PART=kmh_nq10`
   (use any part name).
3. Run: `!make campaign-run MANIFEST=files_config/campaign/manifest_kmh_sweep.json PART=all` (the notebook's
   run cell). Execution order: `kmh_nq100` first (smallest, and the most informative for papers 2 and 3), then
   `kmh_nq50`, then `kmh_nq10`. One session will not hold all 165 runs; resume is the same as in section 6:
   re-running the same command skips complete leaves. Log: `results/campaign_kmh/campaign.log`, failures: `results/campaign_kmh/failures.log`.
4. Verify: `!make campaign-verify MANIFEST=files_config/campaign/manifest_kmh_sweep.json` (add `PART=...` to restrict).
   Expected at the end: `OK=165 INCOMPLETE=0 MISSING=0`, seed audit PASS (per part: 63, 51, 51); after the KI-44 re-run but before stage 2 it is `OK=135` (33, 51, 51).
   If the session dropped: redo sections 0-4 and re-run the same `campaign-run` cell (complete runs are skipped).
5. Report: `!make campaign-report MANIFEST=files_config/campaign/manifest_kmh_sweep.json REPORT_ROOT=results/campaign_kmh REPORT_OUT=docs/results/campaign_kmh`,
   then copy to Drive: `!cp -r docs/results/campaign_kmh "$DRIVE_ROOT/results/campaign_kmh_tables"`.

**Drive quota.** `results/campaign_kmh/` is a new tree on Drive (via the `results/` symlink). Each run writes
a ~95 MB checkpoint, so 165 runs are about **16 GB** (stage 1: 13 GB); check the quota first.

**Wall-clock.** First pass (2026-11-07 to 2026-11-08): about 10 to 12 GPU hours (estimate from log timestamps,
Colab session restarts included). The reference hierarchy costs about 64 s of clustering per round in the
executed campaign; L=1 shapes cost much less.

**Resuming after the KI-44 fix (2026-11-08).** The first pass finished 129/135 runs; six `init_random` runs
failed because random k-means init can leave empty clusters (KI-44, fixed in the vendored
`hierarchical_sampling.py`). In a live session run `cd /content/dalmax && git pull` (or redo sections 0 to 4 in a
new session, which clone `main` fresh), then re-run the same run cell: the engine skips the 129 complete leaves
and executes only the 6 incomplete ones. Expected verify afterwards: `OK=135 INCOMPLETE=0 MISSING=0`. The
notebook text for section 6b still says wall-clock is unmeasured; it is left untouched on purpose (tests pin its
cells), this file is the record.

**CPU smoke (not on Colab).** `make campaign-smoke-kmh` runs the micro sweep on CPU (seed 1) into
`results/smoke_campaign_kmh/`.

---

## Running from VS Code (Colab extension)

You can also open a notebook locally in VS Code with the official **Google Colab**
extension and pick a Colab GPU server as the kernel (Select Kernel -> Colab). Whether
`drive.mount` works from that setup is **not confirmed yet**; that uncertainty is exactly
what [`notebooks/colab_local_test.ipynb`](notebooks/colab_local_test.ipynb) measures. It is a
short diagnostic (runtime, GPU, Drive mount, repo + Poetry, CUDA, data wiring, one tiny
1-round run) that prints `PASS`/`FAIL`/`SKIP` per check plus a recommendation. If Drive
passes, run `notebooks/colab_runbook.ipynb` with the same kernel; if not, use the Colab web
UI, or the notebook's Fallback A (Drive-free dataset; results then stay on the ephemeral
disk and must be downloaded before the runtime ends).

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
- **Python 3.13 system runtime (Colab default since 2026-09-29).** Expected, not
  an error: `poetry install` fails on the system interpreter because
  `torch==2.5.0` has no 3.13 wheels. Fix (already in §3 and the notebooks):
  `!pip install -q uv poetry`, `!uv python install 3.12`, then
  `!poetry env use "$(uv python find 3.12)"` and `!poetry install`. If the
  venv check still reports a non-3.12 Python, re-run `poetry env use` and
  `poetry install` (or `poetry env remove --all` first). The old guidance below
  is superseded and kept only for a future 3.14+ runtime where uv 3.12 is
  unavailable:
  Do not attempt to force-install an older Python
  inside the Colab image (not a supported/tested path for this repo — do
  not invent a workaround). If Colab's default runtime has moved to 3.13,
  the `torch==2.5.0` pin in `pyproject.toml`/`poetry.lock` has no 3.13
  wheels and must be revisited (a `poetry.lock` upgrade to a torch version
  with 3.13 support) before this runbook works again — that is a real code
  change, not something to patch around in a notebook cell.

---
