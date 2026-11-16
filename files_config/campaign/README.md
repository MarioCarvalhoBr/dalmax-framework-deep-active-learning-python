# Campaign manifests

Since 2026-11-11 (ADR 0016) every manifest here describes the **clean re-run on `DATA/daninhas_full_fixed`**
(the duplicate-free dataset, KI-46) into NEW results roots; the manifests of the first campaign
(`results/campaign/`, `results/campaign_kmh/`, dataset with duplicates) are in git history (commit d668673 and
earlier; those result trees are archived under `local_assets/results_old/` since 2026-11-15). See `.specs/experiments/campaign.md` and `COLAB_RUNBOOK.md` section 6.

- `manifest.json` -- the single source of truth for the campaign: run groups (one distinct computation each), aliases
  (same computation, different spelling), per-paper tables, the GPU part of every group (`gpu_part`, `gpu_stage`), the
  per-leaf required files (`required_leaf_files`: `round_predictions.json`, `round_queries.json`) and the GPU-time
  estimate table it uses (`gpu_time_estimates`). Results under `results/campaign_fixed/`. 384 runs / 128 groups
  (297 / 99 before the `texhal_q13` part of ADR 0017, 2026-11-15; 321 / 107 before the `descriptors_nq` and
  `seeds_ext` parts of ADR 0020, 2026-11-16; 357 / 119 before the `kmh_confirm` part of ADR 0021, 2026-11-16):
  paper 1 = 20 strategies (12 classical + 8 DeepAL+ of ADR 0015) + KMH x nq {10,50,100} + the upper bound; the paper-1
  control part `paper1_control` = `WAALRandomQuery` at nq100 (3 runs, `p1_control/WAALRandomQuery/nq100`, WAAL's
  training with a uniform random query, its own table `paper1_waal_control_nq100` with WAAL and RandomSampling, never a
  row or `best_of` candidate of the benchmark tables; ADR 0015 amendment of 2026-11-12); papers 2/3 =
  `rnhal`, `texhal`, `texhal_q17`, `hier_transfer`, `texhal_q13` (ADR 0017: TexHAL's 6.2/6.3 ablations with its
  canonical VCTex Q=13, 8 groups / 24 runs in `part_1_gpu`, appended after every executed group); then (ADR 0020,
  both in `part_2_gpu`, appended after every group of tag v5) `descriptors_nq` (RNHAL and TexHAL with the params
  files of their executed nq100 runs at n_query 50 and 10, 4 groups / 12 runs, ids `descriptors_nq/<method>/nq{N}`,
  results under `results/campaign_fixed/descriptors_nq/`, tables `descriptors_nq{50,10}` in paper 1) and `seeds_ext`
  (the 8 headline methods at nq100 on seeds 4, 5, 6, exact copies of their seeds-1-3 groups, 8 groups / 24 runs, ids
  `seeds_ext/<method>`, results under `results/campaign_fixed/seeds_ext/`, table `seeds_ext_nq100` in paper 3; a
  group's `seeds` may differ from the manifest's default `[1, 2, 3]`); then (ADR 0021, `part_1_gpu`, appended after
  every group of tag v6) `kmh_confirm` (KMH with l2-normalised features at n_query 100/50/10, the KMH reference,
  RandomSampling and WAAL at 50/10, on seeds 4, 5, 6, exact copies of `kmh_norm/nq{N}/l2` (sweep) and
  `p1/{KMH,RandomSampling,WAAL}/nq{50,10}`, 9 groups / 27 runs, ids `kmh_confirm/nq{N}/<method>`, results under
  `results/campaign_fixed/kmh_confirm/`, tables `kmh_confirm_nq{100,50,10}` in paper 3, whose nq100 reference,
  RandomSampling and WAAL rows are aliases of the `seeds_ext` runs). **Generated**: edit `scripts/campaign/build_manifest.py` and run
  `make campaign-manifest` (`tests/test_campaign.py` and `build_manifest.py --check` fail if a committed file drifts).
- `manifest_kmh_sweep.json` -- the paper-1 KMH sweep (ADR 0012), 231 runs / 77 groups (165 / 55 before the
  `kmh_wide` part of ADR 0018, 2026-11-15: the KMH wide-top exploration, 13 groups / 39 runs in `part_2_gpu`, ids
  `kmh_wide/nq{N}/<config>`, appended after every executed group, results under `results/campaign_kmh_fixed/wide/`;
  then the `kmh_norm` part of ADR 0019, 2026-11-15: the KMH feature post-processing round, 9 groups / 27 runs in
  `part_1_gpu`, ids `kmh_norm/nq{N}/{l2,zscore,pca64_l2}`, appended after `kmh_wide`, results under
  `results/campaign_kmh_fixed/norm/`, plus the aliases `kmh_norm/nq{N}/none` of the executed references),
  results under `results/campaign_kmh_fixed/`, same GPU-part fields.
- `manifest_micro.json` / `manifest_kmh_sweep_micro.json` -- the same structures on `DATA/daninhas_micro/` (n_query
  3/4/5, n_init 10, n_round 1, upper bound n_init 806) for the CPU smoke; results under
  `results/smoke_campaign_fixed/` and `results/smoke_campaign_kmh_fixed/`.
- `manifest_control_ki43.json` -- KI-43 control (prepared, not run): RandomSampling nq100 into
  `results/campaign_control_fixed/`.
- `gpu_time_estimates.json` -- estimated GPU seconds per run of every group of the two full-scale manifests
  (`measured` on the first campaign, or `estimate` with its rule for the eight DeepAL+ strategies and the WAAL
  control, which is estimated equal to `WAAL`, and for the `texhal_q13` groups, each equal to its measured
  `texhal_q17` twin, `TWIN_PREFIXES`, and for the `kmh_wide` groups, a least-squares line in the total number of
  clusters over the measured sweep groups of the same query size and number of levels, `FIT_PREFIXES`, and for the
  `kmh_norm` groups, the measured reference `kmh/nq{N}/hier_L3_ref` of the same budget, `REFERENCE_PREFIXES`, and
  for the `descriptors_nq` groups, the measured nq100 run x the measured KMH ratio nq{N}/nq100, `SCALED_PREFIXES`, and
  for the `seeds_ext` groups, the entry of the seeds-1-3 counterpart, `SEEDS_EXT_COUNTERPARTS`, and for the
  `kmh_confirm` groups, the counterpart's wall time measured on the clean campaign, `KMH_CONFIRM_COUNTERPARTS`); built by
  `scripts/campaign/build_gpu_time_estimates.py` (`make gpu-time-estimates`, read-only on the old trees). Used by
  `campaign list` (hours per part / GPU part) and by the balance test. No GPU model is named.

Params (full scale on `DATA/daninhas_full_fixed/`; the `_micro` mirrors on `DATA/daninhas_micro/`, `n_drop` 2 in the
paper-1 micro files for smoke speed):

- `params_paper1.json` -- DANINHAS reference params (n_epoch 10, batch 256, lr 0.05, momentum 0.3, `config_kmh`
  reference hierarchy); used by every paper-1 strategy except the dropout family and VAAL, by the WAAL control
  (the same file as `WAAL`, whose `strategy_args.WAAL` block it reads), by the upper bound and by
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
and nq50, the WAAL control, `texhal_q13`, `kmh_norm`, `kmh_confirm`; 363 runs / 121 groups, about 64.0 estimated GPU h; 285 / 95 before ADR 0017, 309 / 103 before ADR 0019, 336 / 112 and 55.5 h before ADR 0021) and `part_2_gpu` (account B: paper 1 at nq50 and
nq10, KMH sweep nq10, `kmh_wide`, `descriptors_nq`, `seeds_ext`;
252 runs / 84 groups, about 64.0 h; 177 / 59 and 48.8 h before ADR 0018, 216 / 72 and 58.0 h before ADR 0020). `make campaign-run-gpu-part GPU_PART=...` runs both manifests for one part in stage order
(cheap groups, ablations, KMH sweep, then `AdversarialBIM`, `VAAL`/`WAAL`/the WAAL control, `AdversarialDeepFool`);
run `make campaign-pilot GPU_PART=...` first (seed 1 of the part's VAAL/WAAL groups + RandomSampling; not the
control) and read
`make campaign-pilot-report GPU_PART=...`. The three groups `shared/kmh_nq100`, `p1/KMH/nq50`, `p1/KMH/nq10` of the
main manifest repeat the sweep's `kmh/nq{100,50,10}/hier_L3_ref` on purpose (ADR 0016 amendment).

Run `python -m dalmax.campaign list` for the job table (add `--exclude-strategy A,B` to drop strategies from
`list`/`run`/`verify`; `--manifest A,B` merges several manifests into one plan). `make campaign-smoke` passes
`AdversarialBIM,AdversarialDeepFool` by default: those two baselines are far too slow on CPU; the real GPU campaign
runs them (`SMOKE_EXCLUDE=` includes them in the smoke too). KMH at n_query 10/50 lives under `paper1/nq{N}/` like the
other paper-1 strategies. Each group: `id`, `part`, `gpu_part`, `gpu_stage`, `params_json`, `strategy_name`, `n_query`,
`n_init_labeled`, `n_round`, `seeds`, `dir_results`, `used_by`, `note`. Groups consumed by more than one paper live
under `shared/` (one exception, kept for stable ids: `texhal/6_1/rep_q13`, TexHAL's canonical config since ADR 0017,
consumed by papers 2 and 3, whose reference/full rows in `texhal_q13` are aliases of it; until 2026-11-15 the
exception was `texhal/6_1/rep_q17`, ADR 0011, now consumed by paper 2 only, with `texhal_q17`'s aliases and the
supplementary tables `6_2_q17`/`6_3_q17`). The Q=[5,17] 6.2/6.3 groups stay
in part `texhal` and are tabulated as `6_2_q5_17`/`6_3_q5_17`.
