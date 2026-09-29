# Campaign manifests

- `manifest.json` -- the single source of truth for the A100 campaign (see
  `.specs/experiments/campaign-a100.md`): run groups (one distinct computation each), aliases
  (same computation, different spelling), and per-paper tables. **Generated**: edit
  `scripts/campaign/build_manifest.py` and run `make campaign-manifest`
  (`tests/test_campaign.py` fails if the committed file drifts).
- `manifest_micro.json` -- the same structure on `DATA/daninhas_micro/` (n_query 3/4/5, n_init 10,
  n_round 1, upper bound n_init 806) for `make campaign-smoke`; results under `results/smoke_campaign/`.
- `params_paper1.json` -- DANINHAS block of `files_config/benchmark/params_df_gpu_0.json`; used by
  the 12 classical strategies and the upper bound (`config_kmh` is ignored by them).
- `params_kmh.json` -- ResNet-ImageNet features + hierarchical k-means, reference hierarchy;
  used by the KMH groups (paper 1) which are also papers 2/3's "w/o representation module" row.
- `params_paper1_micro.json`, `params_kmh_micro.json` -- micro mirrors (`n_drop` 2 in the paper-1
  micro file for smoke speed).

Run `python -m dalmax.campaign list` for the job table. Each group: `id`, `part`, `params_json`,
`strategy_name`, `n_query`, `n_init_labeled`, `n_round`, `seeds`, `dir_results`, `used_by`, `note`.
Groups consumed by more than one paper live under `shared/`.
