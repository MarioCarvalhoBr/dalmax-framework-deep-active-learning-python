# Use case: run the full benchmark (all strategies × seeds × budgets)

This is the lab-machine workflow to reproduce or extend the baseline
comparison in `experiments/baseline-results.md`. It must run on the **lab
machine** (2× 10 GB GPUs) — see `infrastructure/execution-environments.md`;
do not attempt full training locally (no GPU).

## Preconditions

1. `git pull` on the lab machine so the code matches what was reviewed
   locally.
2. `DATA/daninhas_full/` present on the lab machine at the path referenced
   by the params JSON's `data_dir` (`DATA/daninhas_full/`).
3. Confirm which params JSON to use per GPU: `files_config/benchmark/params_df_gpu_0.json` for GPU
   0, `files_config/benchmark/params_df_gpu_1.json` for GPU 1 (they differ only in `config_kmh` for
   DANINHAS — see `experiments/experimental-protocol.md`).
4. If any SSRAE/VCTex-based strategy will run, check
   `results/features_dict_ssrae.pkl` / `features_dict_vctex.pkl` for
   staleness per `research-rules/reproducibility.md` — delete and
   regenerate if `Q` or the extractor changed since they were last written.

## RNHAL reference sweep (SSRAEKmeansHCSampling only)

```
bash scripts/benchmark/run_pipe_gpu_0.sh   # GPU 0, files_config/benchmark/params_df_gpu_0.json → results/dalmax1/
bash scripts/benchmark/run_pipe_gpu_1.sh   # GPU 1, files_config/benchmark/params_df_gpu_1.json → results/dalmax2/
```

Each script sweeps `n_query ∈ {10, 50, 100}` × `seed ∈ {1, 2, 3}` = 9 runs,
each `--n_round 8`, `--strategy_name SSRAEKmeansHCSampling`,
`--dataset_name DANINHAS`. Both can run **concurrently** (one per GPU,
`CUDA_VISIBLE_DEVICES` isolates them) since they write to separate
`--dir_results` roots (`results/dalmax1/` vs `results/dalmax2/`). Each
script emails a completion notification via `ExperimentNotifier/main.py`
after its own 9-run battery finishes.

## Baseline strategy sweep

`scripts/benchmark/run_pipline.sh` is a **command generator**, not a direct executor
— it `echo`s the `poetry run python demo.py (historical) ...` invocations for the 10 non-SSRAE/
VCTex strategies (`RandomSampling` through `BALDDropout`, with
`AdversarialBIM`/`AdversarialDeepFool` commented out in the current script)
across `n_query ∈ {10, 50, 100}`, for a given `(gpu, seed)` pair passed as
positional args:
```
bash scripts/benchmark/run_pipline.sh <gpu_number> <seed>
```
This prints, but does not execute, the commands — pipe its output to a
shell to actually run them, e.g.:
```
bash scripts/benchmark/run_pipline.sh 0 1 | bash
```
(**TBD**: confirm this is how it is actually invoked on the lab machine —
not verified in this batch; it may instead be intended purely for
inspecting/copy-pasting individual commands.) It references
`params_dnf.json`, which is **not present in this repo** — obtain or
recreate it on the lab machine before running (TBD source).

Repeat for both GPU numbers and all three seeds to cover the full protocol
(`n_query × seed` = 9 combinations per GPU/seed argument set, `N_ROUND=10`
hardcoded inside the script — note this differs from the RNHAL sweep's
`n_round=8`, see `experimental-protocol.md`'s cross-comparability caveat).

## Adversarial strategies

`AdversarialBIM`/`AdversarialDeepFool` are commented out in
`scripts/benchmark/run_pipline.sh` but are valid `demo.py (historical) --strategy_name` choices and
appear in the existing `results/dalmax1/` output — they must have been run
via a direct `demo.py` (historical) invocation outside these two scripts. Invoke
manually if needed:
```
CUDA_VISIBLE_DEVICES=<gpu> poetry run python demo.py (historical) \
  --params_json files_config/benchmark/params_df_gpu_<gpu>.json --dataset_name=DANINHAS \
  --strategy_name AdversarialBIM --n_query <10|50|100> --seed <1|2|3> \
  --n_round 8 --dir_results=results/<target>/
```

## After the sweep

Copy/sync `results/dalmax{1,2}/` (and any manual-strategy output) back to
the local machine per the (currently undocumented — TBD, see
`infrastructure/execution-environments.md`) handoff convention, then follow
`use-cases/generate-report.md` to produce seed-averaged tables.
