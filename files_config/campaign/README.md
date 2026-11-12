# Campaign manifests

Since 2026-11-11 (ADR 0016) every manifest here describes the **clean re-run on `DATA/daninhas_full_fixed`**
(the duplicate-free dataset, KI-46) into NEW results roots; the manifests of the first campaign
(`results/campaign/`, `results/campaign_kmh/`, dataset with duplicates) are in git history (commit d668673 and
earlier). See `.specs/experiments/campaign.md` and `COLAB_RUNBOOK.md` section 6.

- `manifest.json` -- the single source of truth for the campaign: run groups (one distinct computation each), aliases
  (same computation, different spelling), per-paper tables, the GPU part of every group (`gpu_part`, `gpu_stage`), the
  per-leaf required files (`required_leaf_files`: `round_predictions.json`, `round_queries.json`) and the GPU-time
  estimate table it uses (`gpu_time_estimates`). Results under `results/campaign_fixed/`. 294 runs / 98 groups:
  paper 1 = 20 strategies (12 classical + 8 DeepAL+ of ADR 0015) + KMH x nq {10,50,100} + the upper bound; papers 2/3 =
  `rnhal`, `texhal`, `texhal_q17`, `hier_transfer`. **Generated**: edit `scripts/campaign/build_manifest.py` and run
  `make campaign-manifest` (`tests/test_campaign.py` and `build_manifest.py --check` fail if a committed file drifts).
- `manifest_kmh_sweep.json` -- the paper-1 KMH sweep (ADR 0012), 165 runs / 55 groups, results under
  `results/campaign_kmh_fixed/`, same GPU-part fields.
- `manifest_micro.json` / `manifest_kmh_sweep_micro.json` -- the same structures on `DATA/daninhas_micro/` (n_query
  3/4/5, n_init 10, n_round 1, upper bound n_init 806) for the CPU smoke; results under
  `results/smoke_campaign_fixed/` and `results/smoke_campaign_kmh_fixed/`.
- `manifest_control_ki43.json` -- KI-43 control (prepared, not run): RandomSampling nq100 into
  `results/campaign_control_fixed/`.
- `gpu_time_estimates.json` -- estimated GPU seconds per run of every group of the two full-scale manifests
  (`measured` on the first campaign, or `estimate` with its rule for the eight DeepAL+ strategies); built by
  `scripts/campaign/build_gpu_time_estimates.py` (`make gpu-time-estimates`, read-only on the old trees). Used by
  `campaign list` (hours per part / GPU part) and by the balance test. No GPU model is named.

Params (full scale on `DATA/daninhas_full_fixed/`; the `_micro` mirrors on `DATA/daninhas_micro/`, `n_drop` 2 in the
paper-1 micro files for smoke speed):

- `params_paper1.json` -- DANINHAS reference params (n_epoch 10, batch 256, lr 0.05, momentum 0.3, `config_kmh`
  reference hierarchy); used by every paper-1 strategy except the dropout family and VAAL, by the upper bound and by
  `make lab-check`/`colab-check` (SSRAEKmeansHCSampling reads `config_kmh`; the other strategies ignore it).
- `params_paper1_mc_dropout.json` -- `params_paper1.json` plus `"model_args": {"dropout_p": 0.5}` (a dropout layer
  before the ResNet-50 classifier, KI-41, ADR 0014); used by the dropout family: `LeastConfidenceDropout`,
  `MarginSamplingDropout`, `EntropySamplingDropout`, `BALDDropout`, `VarRatio`, `MeanSTD`.
- `params_paper1_vaal.json` -- `params_paper1.json` plus `"strategy_args": {"VAAL": {...}}`: the `VAALArgs` defaults
  except `image_size: 32` (author's cost decision 2026-11-11, about 4x cheaper than 64; the micro file also uses 1
  VAE/discriminator epoch instead of 30). Since ADR 0016 the campaign signature reads only the strategy's own
  `strategy_args` block, so this file is the same computation as `params_paper1.json` for every other strategy.
- `params_kmh.json` -- ResNet-ImageNet features + hierarchical k-means, reference hierarchy; used by the KMH groups
  (paper 1), which are also papers 2/3's "w/o representation module" row.

GPU parts (ADR 0016): `part_1_gpu` (account A: paper 1 at nq100, the upper bound, every ablation part, KMH sweep nq100
and nq50; 282 runs, about 47.5 estimated GPU h) and `part_2_gpu` (account B: paper 1 at nq50 and nq10, KMH sweep nq10;
177 runs, about 48.6 h). `make campaign-run-gpu-part GPU_PART=...` runs both manifests for one part in stage order
(cheap groups, ablations, KMH sweep, then `AdversarialBIM`, `VAAL`/`WAAL`, `AdversarialDeepFool`).

Run `python -m dalmax.campaign list` for the job table (add `--exclude-strategy A,B` to drop strategies from
`list`/`run`/`verify`; `--manifest A,B` merges several manifests into one plan). `make campaign-smoke` passes
`AdversarialBIM,AdversarialDeepFool` by default: those two baselines are far too slow on CPU; the real GPU campaign
runs them (`SMOKE_EXCLUDE=` includes them in the smoke too). KMH at n_query 10/50 lives under `paper1/nq{N}/` like the
other paper-1 strategies. Each group: `id`, `part`, `gpu_part`, `gpu_stage`, `params_json`, `strategy_name`, `n_query`,
`n_init_labeled`, `n_round`, `seeds`, `dir_results`, `used_by`, `note`. Groups consumed by more than one paper live
under `shared/` (one exception kept from the first campaign for stable ids: `texhal/6_1/rep_q17`, TexHAL's canonical
config since ADR 0011, whose reference/full rows in `texhal_q17` are aliases of it). The Q=[5,17] 6.2/6.3 groups stay
in part `texhal` and are tabulated as `6_2_q5_17`/`6_3_q5_17`.
