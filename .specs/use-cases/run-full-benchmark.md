# Use case: run the full benchmark (all strategies x seeds x budgets)

Since 2026-09-29 (ADR 0008/0009/0010) the paper-1 benchmark is **part of the campaign**
(`experiments/campaign.md`): 12 classical strategies + KMH x `n_query in {10,50,100}` x seeds `{1,2,3}`
+ the `FullSupervised` upper bound. The old batch scripts (`scripts/benchmark/run_pipe_gpu_*.sh`,
`run_pipline.sh`, ...) were removed; `files_config/benchmark/params_df_gpu_{0,1}.json` remain as the
reference params for `make lab-check`/`colab-check`, and the campaign uses the equivalent
`files_config/campaign/params_paper1.json` (n_epoch 10, batch 256, lr 0.05, momentum 0.3).

It must run on a **GPU machine** (lab or Colab; `infrastructure/execution-environments.md`); never
locally (no GPU).

## Preconditions

1. `git pull` so the code matches what was reviewed; `git status` clean (each run records the commit).
2. `DATA/daninhas_full/` present (10,193 files).
3. `make lab-setup` / `make colab-setup`, then `make lab-check GPU=0` (one short real-data run).

## Run

```
make campaign-list                 # 117 paper-1 runs / 39 groups + 3 upper-bound runs
make campaign-run PART=paper1      # nq100 first (shared runs land early), then nq50, nq10
make campaign-run PART=upper_bound # FullSupervised: train once on the whole pool, n_round 0
make campaign-verify PART=paper1
make campaign-report               # docs/results/campaign/paper1/
```

`AdversarialBIM`/`AdversarialDeepFool` are part of the manifest and run on GPU; only the CPU smoke
(`make campaign-smoke`) skips them (`--exclude-strategy`). With two GPUs run one process per GPU
(`CUDA_VISIBLE_DEVICES=0|1`, disjoint `PART` values). After the first job finishes, check
`environment.gpus[0].name` in its `run_metadata.json`.

## After the sweep

Copy `results/campaign/` back if it was produced elsewhere (Colab: it is on Drive), then follow
`use-cases/generate-report.md`; the campaign report already averages across seeds.
