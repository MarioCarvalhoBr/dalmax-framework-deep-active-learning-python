# Lab machine runbook

Step-by-step operator guide for running the DalMax campaign on a lab machine
(one or more NVIDIA GPUs, Linux, Poetry-only). This is the practical companion to
[`.specs/infrastructure/execution-environments.md`](.specs/infrastructure/execution-environments.md)
(architecture/decision-matrix source of truth) and
[`.claude/commands/handoff-lab.md`](.claude/commands/handoff-lab.md) (pre-flight
audit before you get here). Every command below is copy-pasteable and either
exists verbatim in this repo or is a plain `poetry`/`git`/`tmux` invocation.

**Hardware-agnostic.** Nothing in the repo assumes a GPU model or count: the GPU(s)
actually used are captured at runtime into every run's `run_metadata.json`
(`environment.gpus`, placeholder `<gpu-name>` below), and that record is the only
place a GPU name lives.

**Never run any of these on the local no-GPU dev notebook** except where a step
explicitly says "local notebook" (`.claude/rules/data-safety.md`,
`.specs/infrastructure/execution-environments.md`).

---

## The campaign

The campaign (`files_config/campaign/manifest.json`, ADR 0008/0009,
[`.specs/experiments/campaign.md`](.specs/experiments/campaign.md)) is the single way to
execute everything the three papers need (192 runs / 64 groups: `paper1` 117,
`upper_bound` 3, `rnhal` 42, `texhal` 30) on ONE environment; see `COLAB_RUNBOOK.md`'s "The campaign"
section for the step list (`make campaign-list` / `campaign-run` / `campaign-verify` /
`campaign-report`). With two GPUs the parts can run concurrently, one process per
GPU (the runner reads `CUDA_VISIBLE_DEVICES`, default 0); with one GPU run `PART=all`:

```bash
# tmux pane 1 (GPU 0): paper 1 (117 runs + upper bound)
CUDA_VISIBLE_DEVICES=0 make campaign-run PART=paper1,upper_bound
# tmux pane 2 (GPU 1): papers 2 and 3
CUDA_VISIBLE_DEVICES=1 make campaign-run PART=rnhal,texhal
# then, from either shell:
make campaign-verify && make campaign-report
```

Papers 2/3's comparison tables read paper 1's `shared/*` runs (KMH@100, RandomSampling@100), so the
report needs both panes finished. Both processes share `results/campaign/campaign.log` (append) and
the same `results/campaign/` tree; groups never overlap between the two commands.

- **Check the environment in the first run**: after the first job finishes, confirm
  `environment.gpus[0].name` in its `run_metadata.json` (`results/campaign/shared/random_nq100/.../RandomSampling/`)
  is the GPU you expect, and read its `determinism` block (best-effort determinism: `warn_only`,
  ops that had no deterministic kernel are listed in `nondeterministic_op_warnings`).
- **Resume**: `campaign-run` skips a job only when its leaf holds the full artifact set
  (`results.json` is written last), so relaunching after a crash or reboot resumes cleanly.
- **`campaign-verify` exit code**: 0 all OK, 1 incomplete/missing job or FAIL seed audit, 2 only
  audit WARNs (log without the initial-labeled-set line).
- **Out of GPU memory**: the same batch size (256) is used for every run, including the
  `FullSupervised` upper bound (all 8,086 images). A job that runs out of memory is logged to
  `failures.log` and the batch continues -- do not lower the batch size silently (it would change
  the protocol); report it (see Troubleshooting).
- **Disk**: 192 checkpoints (~95 MB each) are about 18 GB under `results/campaign/`.
- **CPU smoke** (any machine, no GPU): `make campaign-smoke` runs the micro campaign, skipping the
  two adversarial baselines (`AdversarialBIM`, `AdversarialDeepFool`; far too slow on CPU) -- 58 of the
  64 micro jobs; the real campaign runs them.
- Wall-clock: TBD (unmeasured). Record it here with the `<gpu-name>` after the first full run.

## 0. Prerequisites & one-time setup

- [ ] **Check the GPUs are visible to the OS:**
  ```bash
  nvidia-smi
  ```
  Expect one entry per GPU you intend to use. If this fails, stop — this is a
  driver/hardware problem outside DalMax's scope.

- [ ] **Check Python version** (upper-bounded by the `torch==2.5.0` pin, per
  `README.md`'s Installation section):
  ```bash
  python3 --version   # must be 3.10, 3.11, or 3.12
  ```
  If the system Python is 3.13+, the same fix as on Colab works:
  `pip install uv && uv python install 3.12 && poetry env use "$(uv python find 3.12)"`, then `poetry install`.

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
  while on a fresh machine. Confirm the printed line reads `True <n>` with `<n>` the number of GPUs you
  expect (`torch.cuda.is_available()` and `torch.cuda.device_count()`). If it prints
  anything else (`False ...`, or an unexpected device count), stop and fix
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
  which runs (parametrize `GPU=1` to check the other card; it uses the campaign's own paper-1
  params, `files_config/campaign/params_paper1.json`, which carries the `config_kmh` hierarchy
  `SSRAEKmeansHCSampling` needs):
  ```bash
  CUDA_VISIBLE_DEVICES=0 poetry run python tools/trainer.py \
      --params_json files_config/campaign/params_paper1.json \
      --dataset_name DANINHAS --strategy_name SSRAEKmeansHCSampling \
      --n_query 100 --n_init_labeled 100 --n_round 1 --seed 1 \
      --device cuda --dir_results results/lab_check/
  ```
  **Expected duration:** roughly ~2 minutes on CPU for the one-time SSRAE
  feature extraction over the ~8k-image unlabeled pool (measured elsewhere at
  ~12.13 ms/image, see `.specs/infrastructure/execution-environments.md`'s
  execution-environment facts), plus two training passes of 10 epochs each on the
  GPU (`n_round=1` means one initial-labeled-set training pass plus one
  post-query round) — expect this check to complete in well under the time a
  full multi-seed sweep would take; there is no exact measured GPU epoch time
  recorded anywhere in this repo, so treat "well under the campaign's
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
  poetry run python tools/loader.py --model results/lab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth
  ```
  Expect `format: dalmax-checkpoint`, a nonzero `total params`/`trainable
  params` count, and `forward pass OK`. Then run a single-image prediction to
  confirm inference end-to-end:
  ```bash
  poetry run python tools/predict.py \
      --model results/lab_check/daninhas_full/SEED_1/NQ_100_NIL_100_NR_1_NE_10/SSRAEKmeansHCSampling/saved_model.pth \
      --image DATA/daninhas_full/test/DATASET_GRAMINEA/<some_image>.jpg
  ```
  (substitute any real file under that test folder). Expect a predicted class
  + confidence line and a `predictions.csv` written alongside the image.

Only proceed to the campaign once all three checks above pass.

---

---

## 2. Troubleshooting

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
  relevant params JSON — never edit a committed file under `files_config/` in place mid-batch (that file may be referenced by
  a run still in flight on the other GPU, and it's tracked in git). Copy it
  into the gitignored `files_config/local/` directory instead:
  ```bash
  mkdir -p files_config/local
  cp files_config/ablations/rnhal/<config>.json files_config/local/<config>_bs128.json
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
  historical note). If it happens on a real run, the workaround is
  to perturb that config's `n_clusters` slightly (not to edit the vendored
  file — wrap-don't-edit policy, `.claude/rules/data-safety.md`); do not treat
  a single occurrence as a sign anything else is broken.

- [ ] **`git status` is not clean before starting.** Every run's
  `run_metadata.json` records the current git commit as the source of truth
  for what produced it (`.claude/rules/reproducibility.md`). Commit or stash
  any local changes before launching a batch — an uncommitted change means
  the recorded commit hash doesn't actually describe the code that ran.
