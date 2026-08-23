# Execution environments

Three environments, with distinct roles. All tooling generated for this
project (Makefile targets, CI, agent/skill definitions owned by other
batches) must respect this split — never assume GPU or a full dataset copy
is available locally.

## Decision matrix

| Environment | Hardware | Role | Allowed operations |
|---|---|---|---|
| **Local dev notebook** (this machine) | 16 GB RAM, **no GPU**, Python 3.12, Poetry 2.2.1 | Coding, code review, spec/doc authoring, smoke tests on tiny subsets, report generation from results already copied back | `poetry` commands, `ruff`, `pytest` (fast tests only), `python -c` import checks, `utils/report/*.py` against already-present `results/` data. **Never** a real training run — no GPU, and a full `daninhas_full` epoch would be impractically slow on CPU. |
| **Lab machine** (primary training) | 2× NVIDIA GPUs, **10 GB VRAM each** | All real experiment execution: baseline sweeps, RNHAL reference runs, and (once implemented) the ablation study runs | `run_pipe_gpu_0.sh` (GPU 0, `params_df_gpu_0.json` → `results/dalmax1/`), `run_pipe_gpu_1.sh` (GPU 1, `params_df_gpu_1.json` → `results/dalmax2/`), long-lived via `tmux`/`nohup`; `ExperimentNotifier/main.py` sends an email after each script's full battery completes. |
| **Google Colab Pro** | Single GPU, session-limited, burst/secondary | One-off runs, retries of a specific config, work overflow when the lab machine is busy | Upload dataset as a zip and extract into the Colab runtime's local disk — **never** read the ~10,000 individual `daninhas_full` files directly from a mounted Google Drive (slow, one-file-at-a-time I/O over Drive's virtual filesystem). `pip install -r requirements.txt` (exported from Poetry, see `environment-setup.md`), pin the torch/torchvision/CUDA versions Colab actually ships, checkpoint intermediate results defensively (session can be reclaimed), copy the finished `results/<run>/` folder back out (e.g. to Drive or via `git`) before the session ends. |

## Handoff workflow (local ↔ lab)

1. Code changes are made and reviewed on the local notebook.
2. `git push` from local; `git pull` on the lab machine.
3. Run the appropriate `run_pipe_gpu_{0,1}.sh` on the lab machine (one
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
   notification once a `run_pipe_gpu_*.sh` battery finishes — used as the
   lab-machine-to-human handoff signal; not inspected beyond its CLI
   invocation in the run scripts in this batch (its own README, if any, is
   the source of truth — TBD read it in a future pass).

## Colab checklist

1. Zip the target dataset locally (`daninhas_full/` is ~47 MB, small enough
   to zip and upload directly).
2. Upload/extract the zip into the Colab runtime's local disk (e.g.
   `/content/DATA/daninhas_full/`), not Drive.
3. `pip install -r requirements.txt` (the Poetry-exported file — see
   `environment-setup.md`); **pin torch/torchvision** explicitly since
   Colab images change their pre-installed torch build periodically and a
   naive `pip install torch` can silently upgrade/downgrade CUDA
   compatibility.
4. Point `--dir_results` and the params JSON's `data_dir` at the Colab-local
   paths.
5. Checkpoint: since `demo.py` already saves `results.json` +
   `saved_model.pth` per strategy leaf directory (not just at the very end
   of a whole sweep), a dropped Colab session loses at most the
   in-progress strategy/n_query/seed combination, not the whole batch —
   still, copy finished leaf directories out incrementally rather than
   waiting for the entire sweep to end.
6. Copy `results/` back (Drive mount for the *outgoing* copy is fine —
   writing a handful of larger files back to Drive is not the same
   bottleneck as reading thousands of small dataset files from it).

## GPU memory guidance

`params_df_gpu_{0,1}.json` both configure DANINHAS training with
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
running concurrently (one job per GPU, per `run_pipe_gpu_0.sh` /
`run_pipe_gpu_1.sh`) share any resource (e.g. shared dataset loading, disk
I/O) that could bottleneck parallel throughput even with `CUDA_VISIBLE_DEVICES`
correctly isolating compute.
