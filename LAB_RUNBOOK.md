# Lab machine runbook

Step-by-step operator guide for running DalMax on the lab machine (2x NVIDIA
GPUs, 10 GB VRAM each, Linux, Poetry-only). This is the practical companion to
[`.specs/infrastructure/execution-environments.md`](.specs/infrastructure/execution-environments.md)
(architecture/decision-matrix source of truth) and
[`.claude/commands/handoff-lab.md`](.claude/commands/handoff-lab.md) (pre-flight
audit before you get here). Every command below is copy-pasteable and either
exists verbatim in this repo or is a plain `poetry`/`git`/`tmux` invocation —
nothing here is invented.

**Never run any of these on the local no-GPU dev notebook** except where a step
explicitly says "local notebook" (`.claude/rules/data-safety.md`,
`.specs/infrastructure/execution-environments.md`).

---

## 0. Prerequisites & one-time setup

- [ ] **Check the GPUs are visible to the OS:**
  ```bash
  nvidia-smi
  ```
  Expect two GPU entries, each ~10 GB VRAM. If this fails, stop — this is a
  driver/hardware problem outside DalMax's scope.

- [ ] **Check Python version** (upper-bounded by the `torch==2.5.0` pin, per
  `README.md`'s Installation section):
  ```bash
  python3 --version   # must be 3.10, 3.11, or 3.12
  ```

- [ ] **Install Poetry** (once per machine), via `pipx`:
  ```bash
  sudo apt install pipx
  pipx ensurepath
  # log out and back in (or open a new shell) so the PATH change takes effect
  pipx install poetry
  poetry --version    # developed against Poetry 2.x
  ```

- [ ] **Clone the repo** (first time) or **pull the latest `main`** (every
  subsequent visit):
  ```bash
  # first time:
  git clone https://github.com/MarioCarvalhoBr/dalmax-deep-active-learning-python.git
  cd dalmax-deep-active-learning-python

  # every subsequent visit:
  git pull origin main
  ```
  Confirm `git status` is clean before proceeding — no uncommitted local
  changes should exist on the lab machine. `run_metadata.json` records the git
  commit for every run, so an uncommitted or unpushed change here breaks
  reproducibility for anything you run next
  (`.claude/rules/reproducibility.md`).

- [ ] **Install dependencies:**
  ```bash
  make lab-setup
  ```
  This is `poetry install` (creates the in-project `.venv/`, per
  `poetry.toml`'s `in-project = true`) followed by a CUDA sanity check.
  `poetry install` downloads the CUDA 12.4 `torch==2.5.0`/`torchvision==0.20.0`
  wheels pinned in `poetry.lock` — **~2.5 GB total**, so expect this to take a
  while on a fresh machine. Confirm the printed line reads:
  ```
  True 2
  ```
  (`torch.cuda.is_available()` and `torch.cuda.device_count()`). If it prints
  anything else (`False ...`, or a device count other than `2`), stop and fix
  the CUDA/driver setup before running anything else — do **not** rely on
  `--device auto` to "silently do the right thing" on a batch you won't watch
  live (`.specs/infrastructure/execution-environments.md`, "Phase 2 lab-handoff
  hazards").

- [ ] **Confirm the dataset is present** (`DATA/` is gitignored — it is never
  pulled by `git pull`, and must be copied onto the lab machine independently):
  ```bash
  find DATA/daninhas_full -type f | wc -l   # expect 10193
  ```
  If `DATA/daninhas_full/` is missing or the count doesn't match, copy it from
  the source machine, e.g.:
  ```bash
  # from the source machine:
  zip -r daninhas_full.zip DATA/daninhas_full/
  scp daninhas_full.zip <lab-user>@<lab-host>:~/dalmax-deep-active-learning-python/

  # on the lab machine:
  unzip daninhas_full.zip -d .
  ```
  (or `rsync -avz DATA/daninhas_full/ <lab-user>@<lab-host>:.../DATA/daninhas_full/`
  as an alternative to zip+scp). After copying, cross-check
  `DATA/daninhas_full/arquivos.txt` against the source machine's copy (e.g.
  `diff` the two files, or at least compare line counts) to catch a partial
  transfer. **Never write into `DATA/daninhas_full/`** once it's in place
  (`.claude/rules/data-safety.md`) — it is read-only input for every run below.

- [ ] **Set up `ExperimentNotifier`** (a separate, gitignored sibling repo —
  not part of this `git pull`):
  ```bash
  cd ExperimentNotifier
  cp .env.example .env
  # edit .env: EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD (a Gmail App Password,
  # not your main account password)
  cd ..
  ```
  This `.env` is gitignored and lab-machine-local — it does not come from
  `git pull` and must be set up on **this** machine independently
  (`.claude/skills/running-experiments/SKILL.md`).

  **One-time stale-path fix (KI-32):** `ExperimentNotifier/main.py` (lines 222
  and 224) still calls the pre-Phase-4 path:
  ```python
  ["python3", "utils/report/build_method_metrics.py", "--method", "SSRAEKmeansHCSampling", ...]
  ["python3", "utils/report/build_method_metrics.py", "--method", "VCTexKmeansHCSampling", ...]
  ```
  `utils/report/` was moved to `dalmax/reporting/` in this repo's Phase 4;
  `ExperimentNotifier` is a separate repo and was not touched by that move, so
  its own copy of the path is stale. Edit both occurrences in
  `ExperimentNotifier/main.py` to point at
  `dalmax/reporting/build_method_metrics.py` (or invoke it as
  `python3 -m dalmax.reporting.build_method_metrics`, which avoids relying on
  the caller's working directory). This branch only fires for the legacy
  CIFAR10 email path (`SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling` against a
  `DATA_CIFAR10/` results tree) — it is currently inert for the DANINHAS runs
  in this runbook, but will raise a "file not found" error the first time that
  branch is exercised against a post-Phase-4 tree, so fix it now rather than
  waiting to hit it mid-batch.

- [ ] **Optional cleanup: orphaned pre-refactor cache files.** If any of these
  exist under `results/` on this machine:
  ```
  results/features_dict_ssrae.pkl
  results/features_dict_vctex.pkl
  results/Y_train.pkl
  ```
  they are safe to delete. They were written by the pre-Phase-2 unkeyed
  feature cache (`utils/data.py`, deleted in Phase 4) and are **not read by
  any code path in this repo today** — the live cache is
  `results/cache/embeddings/*.pkl` (`dalmax/embeddings/cache.py`, keyed on
  `(dataset, extractor, Q, variant, split, pool_hash)`). This is optional
  housekeeping, not a correctness requirement — leaving them in place changes
  nothing.

---

## 1. Sanity checks (fast, before spending any real GPU time)

- [ ] **Fast unit tests (CPU-only, no dataset needed):**
  ```bash
  make test
  ```

- [ ] **CPU smoke test** (generates the micro dataset if needed, then a true
  end-to-end `trainer.py` run on it):
  ```bash
  make smoke
  ```
  This calls `poetry run python scripts/make_micro_dataset.py` internally,
  which deterministically samples a 10%-stratified replica of
  `DATA/daninhas_full/` into `DATA/daninhas_micro/`. You can also run that
  generation step standalone:
  ```bash
  make micro-dataset
  ```

- [ ] **Real-data GPU check — one short run on the actual `daninhas_full`
  dataset**, before committing to a multi-hour batch:
  ```bash
  make lab-check GPU=0
  ```
  which runs (parametrize `GPU=1` to check the other card):
  ```bash
  CUDA_VISIBLE_DEVICES=0 poetry run python trainer.py \
      --params_json files_config/benchmark/params_df_gpu_0.json \
      --dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
      --n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
      --device cuda --dir_results results/lab_check/
  ```
  **Expected duration:** roughly ~2 minutes on CPU for the one-time SSRAE
  feature extraction over the ~8k-image unlabeled pool (measured elsewhere at
  ~12.13 ms/image, see `.specs/infrastructure/execution-environments.md`'s
  ablation audit facts), plus two training passes of 10 epochs each on the
  GPU (`n_round=1` means one initial-labeled-set training pass plus one
  post-query round) — expect this check to complete in well under the time a
  full multi-seed sweep would take; there is no exact measured GPU epoch time
  recorded anywhere in this repo, so treat "well under the ablation batch's
  total time" as the useful signal, not a specific minute count (**TBD**:
  record an exact wall-clock the first time this is actually run on the lab
  machine).

  **Expected artifacts** under
  `results/lab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/`:
  - `results.json` (now including `all_precision_macro`/`all_recall_macro`/`all_f1_macro`
    alongside the legacy weighted metrics)
  - `predictions.csv`
  - `run_metadata.json` (config snapshot + git commit — spot-check `git_commit`
    matches `git rev-parse HEAD` and `config.device` reads `"cuda"`)
  - `confusion_matrix.pdf`, `accuracy.pdf`/`precision.pdf`/`recall.pdf`/`f1_score.pdf`
  - `log-dalmax.log`
  - `saved_model.pth` — a real `dalmax-checkpoint` (post-KI-22 fix), **not**
    the ~900-byte class-pickle the pre-fix bug used to produce. A ResNet50
    checkpoint's state_dict is on the order of tens of MB (rough estimate from
    parameter count, ~25M params x 4 bytes; not a measured figure in this
    repo — treat any specific size like "~95 MB" as an estimate, and instead
    verify correctness via `loader.py` below, not file size alone).

  **Verify the checkpoint is real and loadable:**
  ```bash
  poetry run python loader.py --model results/lab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth
  ```
  Expect `format: dalmax-checkpoint`, a nonzero `total params`/`trainable
  params` count, and `forward pass OK`. Then run a single-image prediction to
  confirm inference end-to-end:
  ```bash
  poetry run python predict.py \
      --model results/lab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth \
      --image DATA/daninhas_full/test/DATASET_GRAMINEA/<some_image>.jpg
  ```
  (substitute any real file under that test folder). Expect a predicted class
  + confidence line and a `predictions.csv` written alongside the image.

Only proceed to the full ablation batch (§3) once all three checks above pass.

---

## 2. The advisor's ablation study — what will run

### Choosing the method (METHOD=rnhal | texhal)

Two independent ablation suites live side by side under
`files_config/ablations/{rnhal,texhal}/` (2026-08-30 reorganization — see
that folder's `README.md` and `.specs/experiments/papers-roadmap.md` for the
three-paper plan):

- **`METHOD=rnhal`** (default) — paper 3, SSRAE + hierarchical k-means. This
  is the suite **already executed** 2026-08-25/26 on Colab (see
  `.specs/experiments/ablation-study.md`'s "Execution record"); the table
  below documents that run.
- **`METHOD=texhal`** — paper 2, VCTex + hierarchical k-means. Not yet run;
  12 configs (§6.1 has a 4th row, `rep_q13`, added 2026-08-30 per the VCTex
  method authors — see `files_config/ablations/README.md`). Its own table is
  in the "TexHAL (METHOD=texhal)" subsection below.

Every `make ablations-gpu0`/`ablations-gpu1`/`ablations-all`/`ablations-colab`/
`ablation-report` target in this section accepts `METHOD` as an environment
override, e.g.:
```bash
METHOD=texhal make ablations-gpu0
METHOD=texhal make ablations-gpu1
METHOD=texhal make ablation-report   # -> docs/results/ablation_tables/texhal/
```
Results land under `results/ablations/${METHOD}/<study>/<config>/...`
(`METHOD=rnhal`'s new-run root — **not** the already-executed legacy root,
see the note at the end of this subsection).

#### RNHAL (METHOD=rnhal, default) — executed 2026-08-25/26

Three sub-studies (`.specs/experiments/ablation-study.md` §6.1/§6.2/§6.3), 11
configs total, each swept over `seeds=(1 2 3)`, `n_query=100`, `n_round=8`
(`n_init_labeled` left at the CLI default of 100), `--strategy_name
RepresentationStrategy`, `--device cuda` — **33 individual runs** total,
split across the two GPUs by `scripts/ablations/run_ablation_gpu_{0,1}.sh`:

| § | Config | Params JSON | GPU |
|---|--------|--------------|-----|
| 6.1 | Full | `files_config/ablations/rnhal/rep_full.json` | GPU 0 |
| 6.1 | Spatial-only | `files_config/ablations/rnhal/rep_spatial.json` | GPU 0 |
| 6.1 | Spectral-only | `files_config/ablations/rnhal/rep_spectral.json` | GPU 1 |
| 6.2 | L=1, k=[50] | `files_config/ablations/rnhal/hier_L1.json` | GPU 0 |
| 6.2 | L=2, k=[300,100] | `files_config/ablations/rnhal/hier_L2a.json` | GPU 1 |
| 6.2 | L=2, k=[100,50] | `files_config/ablations/rnhal/hier_L2b.json` | GPU 0 |
| 6.2 | L=3, k=[300,100,50] | `files_config/ablations/rnhal/hier_L3.json` | GPU 1 |
| 6.2 | L=4, k=[300,100,50,25] | `files_config/ablations/rnhal/hier_L4.json` | GPU 1 |
| 6.3 | RNHAL (full) | `files_config/ablations/rnhal/stage_full.json` | GPU 1 |
| 6.3 | w/o representation module | `files_config/ablations/rnhal/stage_no_representation.json` | GPU 0 |
| 6.3 | w/o hierarchical module | `files_config/ablations/rnhal/stage_no_hierarchy.json` | GPU 1 |

**IMPORTANT**: this executed batch's results live at the LEGACY root
`results/ablations/{6_1,6_2,6_3}/` (no `rnhal/` segment) — append-only, left
in place (`.claude/rules/data-safety.md`). A *new* `METHOD=rnhal` invocation
of `run_ablation_gpu_{0,1}.sh` writes to the NEW root
`results/ablations/rnhal/{6_1,6_2,6_3}/` by default, which `SKIP_EXISTING`
cannot see in the legacy tree (its glob is scoped to `RESULTS_ROOT`). To
extend the already-executed legacy batch instead of starting a fresh one,
pass `RESULTS_ROOT=results/ablations` explicitly; to report on the legacy
tree, use `make ablation-report-legacy` (§4 below), not `make
ablation-report`.

#### TexHAL (METHOD=texhal) — not yet run

Same three sub-studies, but §6.1 sweeps VCTex's own multi-scale
hyperparameter `Q` instead of an SSRAE spatial/spectral split (`slice_embedding`
is SSRAE-only) — 4 rows instead of 3, so **12 configs / 36 runs** total. See
`.specs/experiments/ablation-study-texhal.md` and
`files_config/ablations/README.md` for the full rationale.

| § | Config | Params JSON | GPU |
|---|--------|--------------|-----|
| 6.1 | Q=5 | `files_config/ablations/texhal/rep_q5.json` | GPU 0 |
| 6.1 | Q=13 | `files_config/ablations/texhal/rep_q13.json` | GPU 1 |
| 6.1 | Q=17 | `files_config/ablations/texhal/rep_q17.json` | GPU 1 |
| 6.1 | Q=[5,17] (full) | `files_config/ablations/texhal/rep_full.json` | GPU 0 |
| 6.2 | L=1, k=[50] | `files_config/ablations/texhal/hier_L1.json` | GPU 0 |
| 6.2 | L=2, k=[300,100] | `files_config/ablations/texhal/hier_L2a.json` | GPU 1 |
| 6.2 | L=2, k=[100,50] | `files_config/ablations/texhal/hier_L2b.json` | GPU 0 |
| 6.2 | L=3, k=[300,100,50] | `files_config/ablations/texhal/hier_L3.json` | GPU 1 |
| 6.2 | L=4, k=[300,100,50,25] | `files_config/ablations/texhal/hier_L4.json` | GPU 1 |
| 6.3 | TexHAL (full) | `files_config/ablations/texhal/stage_full.json` | GPU 1 |
| 6.3 | w/o representation module | `files_config/ablations/texhal/stage_no_representation.json` | GPU 0 |
| 6.3 | w/o hierarchical module | `files_config/ablations/texhal/stage_no_hierarchy.json` | GPU 1 |

GPU 0 gets 5 configs, GPU 1 gets 7 (the two extra §6.1 rows both land on
GPU 1 — see `scripts/ablations/run_ablation_gpu_1.sh`'s header). VCTex
per-run wall-clock is **unmeasured (TBD)** — VCTex's RAE extraction cost
differs from SSRAE's (different input dimensionality: 27 vs 9 per patch, see
`.specs/experiments/ablation-study-texhal.md`); record it on the first real
run and update this section.

---

The rest of this section documents the executed RNHAL batch's numbers
(measured timing, results layout) as a worked example — the same commands
apply verbatim to `METHOD=texhal`, just with different expected wall-clock
(TBD, see above) and, obviously, different (unexecuted) results.

GPU 0 gets 5 configs, GPU 1 gets 6 — balanced by *expected relative cost*, not
raw count: hierarchical selection over a large/multi-level hierarchy dominates
runtime, not training, so GPU 0's 5 configs include 3 of the heaviest
(large/shared reference hierarchy) configs to offset having fewer configs
overall (see `scripts/ablations/run_ablation_gpu_0.sh`'s header for the full
per-config rationale).

**Results layout** (one leaf per `(study, config, seed)`):
```
results/ablations/<study>/<config>/daninhas_full/SEED_<seed>/NQ_100_NIL_100_NR_8_NE_10/RepresentationStrategy/
```
containing the same artifact set as §1's `lab-check` (`results.json`,
`predictions.csv`, `run_metadata.json`, plots, `saved_model.pth`,
`log-dalmax.log`).

**Expected embedding recomputes:** 3 SSRAE extractions (one per seed — every
SSRAE-based config at a given seed shares one cache, since the pool identity
only depends on `(dataset, seed, n_init_labeled)`, not on the embedding
variant or hierarchy) + 3 ResNet-ImageNet extractions (one per seed, for the
`stage_no_representation` row) — per
`.specs/infrastructure/execution-environments.md`'s "Lab handoff — Phase 3
ablation batch" audit facts. This is why running all three `rep_*` configs
back-to-back at the same seed is cheap: only the first extracts SSRAE
features, the other two reuse the cached `results/cache/embeddings/...pkl`.


**Measured reference (Colab, single NVIDIA T4, 2026-08-26):** the same 33-run sweep took ~5 h 15 min on one T4 (~9–10 min/run, ~10 GB VRAM at batch_size 256). Two lab GPUs of 10 GB each should therefore finish in roughly 2.5–3 h if per-GPU speed is comparable — note the VRAM headroom is tight on 10 GB cards; see §6 (OOM) for the batch_size fallback.

**Rough wall-clock — estimate only, not a measured figure.** Reasoning: 33
runs total, each doing `n_round=8` (9 evaluation points: initial + 8 query
rounds) x `n_epoch=10` ResNet50 training passes on up to ~8k images at
`batch_size=256`, plus (for most configs) hierarchical k-means selection at
each round. `.specs/infrastructure/execution-environments.md` records a
GPU-load split ratio of ~1.1x between the two scripts (i.e. GPU 1's 6 configs
and GPU 0's 5 configs were balanced to within ~10% of each other by design)
and a measured SSRAE extraction rate (~12.13 ms/image), but **no measured
per-epoch or per-run GPU training time exists anywhere in this repo** — do not
treat any specific hour count here as verified. **TBD**: record actual
wall-clock for one full `(study, config)` x 3-seeds slice the first time this
batch runs, and update this section with a real number.

---

## 3. Launch

Every command below runs the RNHAL suite by default (`METHOD=rnhal`); prefix
any of them with `METHOD=texhal` to run the TexHAL suite instead (§2's
"Choosing the method" — e.g. `METHOD=texhal make ablations-gpu0`).

Use two `tmux` sessions (or panes), one per GPU, so each script's live output
stays directly watchable and the batch survives an SSH disconnect:

```bash
tmux new -s gpu0
# inside the gpu0 session:
make ablations-gpu0
# or: METHOD=texhal make ablations-gpu0
# detach: Ctrl-b d

tmux new -s gpu1
# inside the gpu1 session:
make ablations-gpu1
# or: METHOD=texhal make ablations-gpu1
# detach: Ctrl-b d
```

Re-attach with `tmux attach -t gpu0` / `tmux attach -t gpu1` to check
progress. Both `make ablations-gpu0`/`make ablations-gpu1` just run
`bash scripts/ablations/run_ablation_gpu_{0,1}.sh` — see that Makefile target
if you'd rather invoke the script directly.

If you'd rather launch both from a single unattended shell instead of two
`tmux` panes (e.g. via `nohup make ablations-all &`), use:
```bash
make ablations-all
```
This runs both scripts concurrently via `&` + `wait`, each redirected to its
own log (`results/ablations/gpu0.log` / `gpu1.log`) — the redirection exists
specifically so the two concurrent processes' output doesn't interleave
unreadably in one terminal. Prefer the two-`tmux`-pane form above when you
want to watch each GPU's output live; use `ablations-all` for a genuinely
unattended launch.

**Monitor:**
```bash
# tmux form: tail the script's own log if you redirected it, or just watch
# the attached pane directly.

# ablations-all form:
tail -f results/ablations/gpu0.log
tail -f results/ablations/gpu1.log

# either form, from a third shell:
watch -n 5 nvidia-smi
```

**Check for failures** (both scripts keep going past a single bad run — see
their headers for why they deliberately don't use `set -e`):
```bash
cat results/ablations/gpu0_failures.log
cat results/ablations/gpu1_failures.log
```
An empty file means no failures. A non-empty line looks like:
```
FAILED: study=6_2 config=hier_L2a seed=2
```

**Re-run only a failed `(study, config, seed)` triple** manually, using the
same command pattern the scripts use internally (substitute the failed
triple's values):
```bash
CUDA_VISIBLE_DEVICES=<gpu> poetry run python trainer.py \
    --params_json files_config/ablations/<config>.json \
    --dataset_name=DANINHAS \
    --strategy_name RepresentationStrategy \
    --n_query 100 \
    --seed <seed> \
    --n_round 8 \
    --dir_results=results/ablations/<study>/<config>/ \
    --device cuda
```
e.g. for the failure line above:
```bash
CUDA_VISIBLE_DEVICES=1 poetry run python trainer.py \
    --params_json files_config/ablations/hier_L2a.json \
    --dataset_name=DANINHAS \
    --strategy_name RepresentationStrategy \
    --n_query 100 \
    --seed 2 \
    --n_round 8 \
    --dir_results=results/ablations/6_2/hier_L2a/ \
    --device cuda
```

**Completion email:** each script calls
`ExperimentNotifier/main.py --dir_results="results/ablations/" --args
"GPU_NUMBER=<n>, ABLATION_BATCH=gpu<n>, FAILURE_LOG=..."` once its half
finishes, sending an HTML summary email (see §0's `ExperimentNotifier` setup —
if you didn't configure `.env` there, this step silently fails to send but
does not fail the batch itself, per `EmailNotifier`'s own guard for missing
`EMAIL_FROM`/`EMAIL_TO`).

---

## 4. Collect results

Once both GPUs' batches finish (and `gpu{0,1}_failures.log` are empty, or any
failures have been re-run per §3):

```bash
make ablation-report                    # rnhal (default) -> results/ablations/rnhal/
make ablation-report METHOD=texhal      # texhal -> results/ablations/texhal/
```
which runs:
```bash
poetry run python -m dalmax.reporting.ablation_report --root results/ablations/${METHOD} --out docs/results/ablation_tables/${METHOD} --method ${METHOD}
```
This walks every discovered
`results/ablations/${METHOD}/<study>/<config>/.../results.json`, computes
mean +/- std macro-F1 (and weighted F1) across seeds, and writes into
`docs/results/ablation_tables/${METHOD}/`:
- `ablation_summary.csv`
- `ablation_6_1.md` / `.tex`, `ablation_6_2.md` / `.tex`, `ablation_6_3.md` / `.tex`

**Reporting on the already-executed legacy RNHAL batch** (2026-08-26, living
at the legacy root `results/ablations/{6_1,6_2,6_3}/`, whose committed
tables are the top-level `docs/results/ablation_tables/*.{csv,md,tex}` files
— see `docs/results/README.md`): use `make ablation-report-legacy` instead,
which runs `--root results/ablations --out docs/results/ablation_tables
--method rnhal` — kept as its own target so this historical path is
unaffected by the 2026-08-30 per-method reorganization.

A config with zero discovered runs renders as `TBD` in these tables rather
than being silently omitted — if you see `TBD` after a full batch, some
`(study, config)` didn't produce any `results.json` and needs investigating
(check `gpu{0,1}_failures.log` first).

```bash
git add docs/results/ablation_tables/
git commit -m "exp: add Phase 3 ablation study results"
git push
```
(`push` requires your explicit confirmation per `.claude/rules/git-workflow.md`
— don't run it unattended as part of a script.)

**Then, on the local notebook:**
```bash
git pull
```
and fill in `paper_drafts/ablation_section.tex` from the new tables (see that
folder's own README for conventions). Run the `/ablation-status` skill
(`.claude/skills/ablation-status/`) beforehand for a checklist of which rows
are implemented/executed/pending against the freshly-pulled results.

---

## 5. Optional: re-run the reference benchmark

The pre-existing RNHAL reference sweep (`SSRAEKmeansHCSampling`,
`QUERIES=(10 50 100)` x `SEEDS=(1 2 3)`, `n_round=8`) can be re-run at any
time:
```bash
make benchmark-gpu0   # GPU 0, files_config/benchmark/params_df_gpu_0.json -> results/dalmax1/
make benchmark-gpu1   # GPU 1, files_config/benchmark/params_df_gpu_1.json -> results/dalmax2/
```
**Why you might want to:** every `saved_model.pth` under the existing
`results/dalmax1/`/`results/dalmax2/` predates the checkpoint fix (KI-22 /
ADR 0006) — those files are ~900-byte pickled class references with **no
weights** and cannot be recovered. `results.json`/`predictions.csv` from those
runs are unaffected and still valid; only the model weights are gone.
Re-running these targets regenerates real, loadable `dalmax-checkpoint` files
for the same sweep, into the same `results/dalmax{1,2}/` tree (a genuinely new
run, not a repair of the old one — `results/` is append-only, so this adds new
leaf directories rather than overwriting anything, per
`.claude/rules/data-safety.md`).

---

## 6. Troubleshooting

- [ ] **CUDA not visible to torch.** `--device cuda` does not itself validate
  `torch.cuda.is_available()` at config-load time (only `--device auto` does —
  see `dalmax/config/loader.py::_resolve_device`); passing `--device cuda` on
  a machine without a working CUDA driver will fail once the code actually
  tries to move a tensor/model to `"cuda"`, which happens early in the run
  (well before any real training time is spent) — expect a `RuntimeError`
  from PyTorch itself (e.g. "Found no NVIDIA driver..."), not a clean
  `dalmax` config error. If you hit this, re-run §0's
  `poetry run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
  check first — it should already have caught this before you got here.

- [ ] **CUDA out-of-memory (OOM).** Reduce `batch_size` in a **copy** of the
  relevant params JSON — never edit a file under `files_config/ablations/` or
  `files_config/benchmark/` in place mid-batch (that file may be referenced by
  a run still in flight on the other GPU, and it's tracked in git). Copy it
  into the gitignored `files_config/local/` directory instead:
  ```bash
  mkdir -p files_config/local
  cp files_config/ablations/<config>.json files_config/local/<config>_bs128.json
  # edit files_config/local/<config>_bs128.json: lower train_args/test_args batch_size
  ```
  then point `--params_json` at the copy for that one re-run. `files_config/local/`
  is gitignored (added alongside this runbook) specifically for this purpose.

- [ ] **`poetry.lock` mismatch** (e.g. after a `git pull` that touched
  `pyproject.toml`):
  ```bash
  poetry install --sync
  ```

- [ ] **Vendored equal-cluster-size `TypeError` (KI-30).** The vendored
  `dalmax/tools/SSL/src/utils.py::create_clusters_from_cluster_assignment`
  breaks if a k-means step ever produces perfectly equal-sized clusters
  (builds a `dtype=object` array instead of a normal int array). This is
  data-dependent and has never been observed on a real embedding at full
  scale — it was only ever hit on the old 2-class/70-image micro dataset
  during config authoring (see `files_config/ablations/README.md`'s
  historical note). If it happens on a real ablation run, the workaround is
  to perturb that config's `n_clusters` slightly (not to edit the vendored
  file — wrap-don't-edit policy, `.claude/rules/data-safety.md`); do not treat
  a single occurrence as a sign anything else is broken.

- [ ] **`git status` is not clean before starting.** Every run's
  `run_metadata.json` records the current git commit as the source of truth
  for what produced it (`.claude/rules/reproducibility.md`). Commit or stash
  any local changes before launching a batch — an uncommitted change means
  the recorded commit hash doesn't actually describe the code that ran.
