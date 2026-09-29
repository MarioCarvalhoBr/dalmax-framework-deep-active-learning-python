# Campaign manifests

- `manifest.json` -- the single source of truth for the campaign (see
  `.specs/experiments/campaign.md`): run groups (one distinct computation each), aliases
  (same computation, different spelling), and per-paper tables. **Generated**: edit
  `scripts/campaign/build_manifest.py` and run `make campaign-manifest`
  (`tests/test_campaign.py` fails if the committed file drifts).
- `manifest_micro.json` -- the same structure on `DATA/daninhas_micro/` (n_query 3/4/5, n_init 10,
  n_round 1, upper bound n_init 806) for `make campaign-smoke`; results under `results/smoke_campaign/`.
- `params_paper1.json` -- DANINHAS reference params (n_epoch 10, batch 256, `config_kmh` reference
  hierarchy); used by the 12 classical strategies and the upper bound (`config_kmh` is ignored by
  them) and by `make lab-check`/`colab-check` (SSRAEKmeansHCSampling reads `config_kmh`).
- `params_kmh.json` -- ResNet-ImageNet features + hierarchical k-means, reference hierarchy;
  used by the KMH groups (paper 1) which are also papers 2/3's "w/o representation module" row.
- `params_paper1_micro.json`, `params_kmh_micro.json` -- micro mirrors (`n_drop` 2 in the paper-1
  micro file for smoke speed).

Run `python -m dalmax.campaign list` for the job table (add `--exclude-strategy A,B` to drop strategies from
`list`/`run`/`verify`). `make campaign-smoke` passes `AdversarialBIM,AdversarialDeepFool` by default: those
two baselines are far too slow on CPU, so the CPU smoke runs 58 of the 64 micro jobs (seed 1); the real GPU
campaign runs them (`SMOKE_EXCLUDE=` includes them in the smoke too). KMH at n_query 10/50 lives under
`paper1/nq{N}/` like the other paper-1 strategies. Each group: `id`, `part`, `params_json`,
`strategy_name`, `n_query`, `n_init_labeled`, `n_round`, `seeds`, `dir_results`, `used_by`, `note`.
Groups consumed by more than one paper live under `shared/`.
