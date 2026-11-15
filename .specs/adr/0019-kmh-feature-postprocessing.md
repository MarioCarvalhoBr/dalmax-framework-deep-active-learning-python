# ADR 0019: Post-process the frozen KMH features before clustering (part `kmh_norm` of the KMH sweep, account A)

- **Status:** Accepted (author decision 2026-11-15); prepared, not executed
- **Date:** 2026-11-15

## Context

Paper 1's KMH is hierarchical k-means selection on frozen ImageNet ResNet-50 features (`RepresentationStrategy`
with `embedding.extractor = "resnet_imagenet"` and `selection.method = "hierarchical"`). The 2,048-d pooled vectors
enter the Euclidean k-means of the vendored pipeline (`dalmax/tools/SSL/src/`) raw: no normalisation of any kind.
Pooled ReLU activations have non-negative coordinates of very different scales, so the Euclidean distance between
two images is dominated by a few high-variance dimensions and by the overall activation magnitude.

On the clean campaign (ADR 0016, final-round weighted F1, mean of seeds 1-3, nq100, reference hierarchy
`[600,200,100]`/`[30,15,2]`), replacing these features by another descriptor under the same hierarchy moved the
score by several points (TexHAL, VCTex `Q=13`: 0.8341, +4.2 points; RNHAL, SSRAE: 0.8608, +6.9 points; KMH, ImageNet:
0.7921), while the hierarchy and its knobs (ADR 0012 sweep, 55 executed groups) moved it inside the seed noise (5 to
10 points of standard deviation at nq50/nq10). Paper 1 names the feature space as the open lever of KMH. The wider
hierarchies of ADR 0018 (part `kmh_wide`, 39 runs, account B, tag `campaign-fixed-v4`) are running in parallel.

## Decision

We will add an optional, config-driven post-processing of the embeddings before the selection, and run an
exploration round that changes ONLY the geometry of the frozen ImageNet features, as a new part **`kmh_norm`** of the
KMH sweep manifest on Colab account A.

### The post-processing step

- **Params key** `embedding.postprocess = {"method": "none" | "l2" | "zscore" | "pca_l2", "pca_dim": <int>}`
  (`dalmax/config/schema.py::PostprocessConfig`, parsed by `dalmax/config/loader.py`). Absent = `{"method": "none"}`,
  the identity. Fail fast (`ConfigError`): unknown method, a present block without `method`, an unknown key,
  `pca_dim` missing for `pca_l2` or set for any other method (a value silently ignored would be a reproducibility
  hazard), a non-integer or non-positive `pca_dim`.
- **Module** `dalmax/embeddings/postprocess.py`: `FeaturePostprocessor` (fit once, transform many), with
  `IdentityPostprocessor` (`none`), `L2Postprocessor` (`l2`), `ZScorePostprocessor` (`zscore`) and
  `PCAL2Postprocessor` (`pca_l2`); registry `POSTPROCESS_REGISTRY` + `build_postprocessor` in
  `dalmax/embeddings/registry.py` (no if/elif chain; keys = `schema.VALID_POSTPROCESS_METHODS`, tested).
- **Semantics**, fitted on the embeddings of the INITIAL unlabeled pool the strategy already holds (no labels, no
  test images, nothing from the classifier), float64 for every fit and transform, float32 result:
  - `l2`: each vector divided by its Euclidean norm; a zero vector stays zero (zero-norm guard).
  - `zscore`: each dimension minus the pool mean, divided by the pool's population standard deviation; a dimension
    constant over the pool (every value equal, an exact test) is centred on that value and not scaled, so it maps
    to exactly 0 (zero-variance guard).
  - `pca_l2`: centre with the pool mean, project on the first `pca_dim` principal components of the pool (full,
    deterministic SVD of the centred pool, `numpy.linalg.svd`, no randomised solver; sign convention: the
    largest-magnitude loading of every component positive), then `l2`. `pca_dim` larger than
    `min(pool size, feature dimension)` is a `ConfigError` at the fit. The explained variance ratio of the kept
    components is reported.
- **Where it is applied**: `dalmax/query_strategies/representation.py::RepresentationStrategy.query`, after the
  embeddings are loaded from (or written to) the `EmbeddingCache` and after the SSRAE `variant` slice
  (`variants.slice_embedding`), right before `selection.select`. The post-processor is injected by
  `registry.build_strategy` through the constructor (presets and the generic name, exactly like `embedding.variant`),
  fitted at the first query on that query's matrix (= the initial unlabeled pool) and reused in every later round
  (the pool only shrinks). The cache keeps holding the raw `"full"` vectors: its key and files are unchanged, so
  account A's Drive cache of the ImageNet features of seeds 1-3 (written by `shared/kmh_nq100` and the sweep groups
  of nq100/nq50) is loaded, not recomputed.
- **Reporting**: `run_metadata.json`'s config snapshot carries `dataset.embedding.postprocess` (it follows from
  `to_dict(config)`; `{"method": "none", "pca_dim": null}` for every run without the key). A non-identity method adds
  one log line at the fit (`Embedding post-processing (ADR 0019): method=..., fitted on the initial unlabeled pool
  of N images, dim D -> D', guarded=G[, explained variance ...]`) and per-round `strategy_diagnostics` in
  `results.json` (`postprocess_method`, `postprocess_fit_pool_size`, `postprocess_input_dim`,
  `postprocess_output_dim`, `postprocess_guarded`, `postprocess_explained_variance`). The identity adds neither.
- **Campaign signature**: `dalmax/campaign.py::effective_signature` adds the resolved block to a representation
  strategy's behavior ONLY when the method is not `none`; absent and `{"method": "none"}` keep every signature.

### The identity guarantee, and how it is tested

With the key absent or `"none"` the code path is behaviour-identical to tag `campaign-fixed-v4`: the identity's
`fit` reads only the matrix shape (it keeps a shape summary that nothing downstream reads), its `transform` returns
the array itself (no copy, no cast), no log line and no diagnostics are added, no random number is drawn anywhere
(construction, fit, transform), and the cache key is untouched. Per file of `git diff campaign-fixed-v4 -- dalmax/`:

| file | change | default (absent / `"none"`) path |
|---|---|---|
| `config/schema.py` | `PostprocessConfig`; `EmbeddingConfig.postprocess` (default identity) | validation only; a params file without the key resolves to the default; `to_dict` gains one key, so the `run_metadata.json` snapshot gains `embedding.postprocess` (bookkeeping, nothing reads it back) |
| `config/loader.py` | `_build_postprocess_config`; `_build_embedding_config` takes a context string | absent key -> `PostprocessConfig()`; `extractor`, `q`, `variant` parsed exactly as before |
| `embeddings/postprocess.py` | new module | only `IdentityPostprocessor` is reached |
| `embeddings/registry.py` | `POSTPROCESS_REGISTRY`, `build_postprocessor` | builds one object; no RNG, no I/O |
| `query_strategies/registry.py` | `postprocessor=build_postprocessor(...)` passed to `RepresentationStrategy` | built after the provider, the selection and its generator, in the same order as before; no draw |
| `query_strategies/representation.py` | constructor argument, `first_query` flag, fit/transform calls, `diagnostics()` override | the same stacked, sliced float32 array object reaches `select`; the same log lines; `diagnostics()` returns `{}` like the base class; the cache-key code is untouched |
| `campaign.py` | `_postprocess_behavior` in `effective_signature`; `kmh_norm` in `PARTS`/`PHASE_LABELS` | not executed by `tools/trainer.py`; identity adds nothing to a signature (pinned digests) |

Evidence:

1. The golden-run tests (`tests/test_golden_run.py`, fixtures unchanged) pass.
2. `tests/test_postprocess_identity.py::test_micro_run_with_and_without_explicit_none_is_identical` (`dataset`,
   `slow`): two `tools/trainer.py` runs on `DATA/daninhas_micro/` with `params_kmh_micro.json`, key absent vs
   `{"method": "none"}`, give identical logged queries, `round_queries.json`, `round_predictions.json`,
   `results.json` (every metric, exactly; no `strategy_diagnostics`) and `run_metadata.json` config (apart from the
   params path and results dir); a third run with `l2` has the same round 0 (metrics, predictions, initial set).
3. `test_selection_receives_the_pre_change_array`: a spy selection records every array handed to it over three
   rounds, for SSRAE (full, spatial, spectral), VCTex and the ImageNet extractor (stub in the fast suite, the real
   provider in a `slow` variant), with the default and with the `"none"`-built post-processor, and compares it
   exactly (values, dtype, shape) with the v4 query path; `test_matches_the_campaign_fixed_v4_class` runs the v4
   `RepresentationStrategy` class itself (read from the tag with `git show`) side by side: same ids, same arrays,
   same selection-generator state, same global NumPy stream, same cache file names.
4. `test_build_strategy_draws_no_random_number_for_any_method` and
   `tests/test_embedding_postprocess.py::test_fit_and_transform_draw_no_random_number`: building the strategy with
   any method, fitting and transforming leave the NumPy legacy, Python `random` and torch streams where `"none"`
   leaves them.
5. `tests/test_campaign.py::test_v4_groups_keep_their_identity`: the 107 + 68 groups of tag `campaign-fixed-v4`
   (486 executed + 39 `kmh_wide`) keep id, parts, stage, params, budgets, seeds, results dir and effective signature
   under the new signature code (digest computed with the v4 code on a checkout of the tag, for the full and micro
   manifests); `test_postprocess_none_keeps_the_signature_and_other_methods_change_it`.

`tools/` is unchanged between the tags.

### KI-50 is not fixed here

The ImageNet provider still consumes the global torch stream at construction (KI-50). That is deliberate: the new
runs must start from the same round-0 classifier per seed as every executed KMH run, so that a difference can be
attributed to the feature geometry. The post-processing is constructed, fitted and applied without any draw, and only
after round 0. **Round 0 of every `kmh_norm` run is expected to equal that of the executed KMH runs seed by seed**
(round-0 weighted F1 0.3839 / 0.4783 / 0.2469 for seeds 1 / 2 / 3, identical across all executed
`resnet_imagenet` groups on both accounts); `COLAB_RUNBOOK.md` section 6g lists it as a post-run check.

### The round

Part `kmh_norm`: 9 run groups x seeds 1-3 = 27 runs, the REFERENCE hierarchy `k=[600,200,100]`,
`sample_sizes=[30,15,2]`, every other knob at the default of the executed `kmh/nq{N}/hier_L3_ref` groups (the
hierarchy fixed beforehand, so this round depends on no choice made on the test set), ImageNet features, NIL 100,
8 rounds, `gpu_part = part_1_gpu` (account A), `gpu_stage` 3 (the sweep's stage), appended after `kmh_wide`.

| group id | `embedding.postprocess` | params | results |
|---|---|---|---|
| `kmh_norm/nq{N}/l2` | `{"method": "l2"}` | `files_config/ablations/kmh_sweep/nq{N}/hier_L3_ref__norm_l2.json` | `results/campaign_kmh_fixed/norm/nq{N}/l2/` |
| `kmh_norm/nq{N}/zscore` | `{"method": "zscore"}` | `.../hier_L3_ref__norm_zscore.json` | `.../norm/nq{N}/zscore/` |
| `kmh_norm/nq{N}/pca64_l2` | `{"method": "pca_l2", "pca_dim": 64}` | `.../hier_L3_ref__norm_pca64_l2.json` | `.../norm/nq{N}/pca64_l2/` |

for `N` in 100, 50, 10. Each group's resolved config equals that of the executed `kmh/nq{N}/hier_L3_ref` except the
post-processing block (`tests/test_kmh_sweep_configs.py`, `tests/test_campaign.py::test_kmh_norm_part_layout`).
Report tables `kmh_norm_nq100`, `kmh_norm_nq50`, `kmh_norm_nq10`: first row the executed raw-feature reference,
through the alias `kmh_norm/nq{N}/none` (params `hier_L3_ref__norm_none.json`, the explicit identity; the manifest
validation checks that its effective signature equals the reference's), then the three variants. Existing table
ids and rows are unchanged. `pca_dim = 64` keeps 64 of 2,048 dimensions (a 32x reduction); it is one point, not a
sweep of the dimension.

GPU time estimates (`basis: estimate`, `build_gpu_time_estimates.py::REFERENCE_PREFIXES`): each group equals the
measured `kmh/nq{N}/hier_L3_ref` of the same budget (571 / 564 / 553 s per run at nq100 / 50 / 10), 4.2 GPU h in
total. `pca_l2` clusters 64-d vectors and may be faster; no speed-up is assumed.

The nq10 table spans both GPU parts: its reference row was executed on account B (part `part_2_gpu`), the three
variants run on account A. Both accounts produced the same round-0 model for every executed KMH run (the values above),
which is the check that the comparison is made from the same starting point; the GPU of each leaf is in its
`run_metadata.json`.

## Selection on the test set

As in ADR 0018: there is no validation split, every configuration is compared by its final-round TEST weighted F1,
and three more configurations per query size make the selected KMH value more optimistic, with a seed spread of 5 to
10 points. A configuration taken from this round is therefore **confirmed on new seeds (4, 5, 6)**, together with
`RandomSampling` and `WAAL` at the same query size, before any claim that KMH matches or beats WAAL (the confirmation
part planned by ADR 0018, not built here). Until then the thesis reports the `kmh_norm` numbers as exploration,
disclosed as selected on the test set.

## What this round can and cannot show

- It can show whether the Euclidean geometry of the frozen ImageNet features (vector magnitude, per-dimension scale,
  the directions of low pool variance) limits KMH under the reference hierarchy, at each query size.
- It cannot show that ImageNet features are as good as a descriptor: it varies the geometry of the same frozen
  features, not the features. A gain does not transfer to SSRAE/VCTex without running them (their vectors have
  other scales), and a null result does not rule out other feature spaces.
- `pca_l2` changes two things at once (dimension and normalisation); `l2` alone separates the normalisation.

## Pre-flight (2026-11-15, CPU, local)

Recorded in `.specs/experiments/campaign.md` ("KMH feature post-processing"). In short: golden runs, the identity
tests (including the end-to-end micro run with and without the explicit `"none"`) and the pinned v4 digests pass; the
9 micro mirrors run 9/9 and verify OK=9 with audits PASS; on the real ImageNet features of the seed-1 initial pool
(7,130 images) raw norms 14.0 to 30.7 (median 19.2) become 1 after `l2`/`pca_l2` and 24.2 to 170.7 (median 39.3)
after `zscore` (no constant dimension, no zero-norm vector), the 64 principal components explain 0.8956 of the pool
variance, and for every variant and budget (10, 50, 100) the reference hierarchy returns exactly `n_query` distinct
images with no empty cluster and no top-up. The local tree verifies OK=486, MISSING=66 (39 `kmh_wide` + 27
`kmh_norm`), and a dry run of `part_1_gpu` would start only the 27 new jobs.

## Consequences

- KMH sweep manifest 231 runs / 77 groups (was 204 / 68); `part_1_gpu` 336 runs / 112 groups (was 309 / 103,
  about 55.5 estimated GPU h); `part_2_gpu` unchanged (216 / 72); both manifests 552 runs (was 525). The main
  manifest is byte-identical; the sweep only appends (the three references gain the `used_by` entry of their new
  table row, which is not part of a group's identity).
- New tag **`campaign-fixed-v5`** (created by the author after review and audit) is the notebook's
  `DEFAULT_GIT_REF`. **Unlike v3 and v4, this tag DOES change code that `tools/trainer.py` executes**
  (`RepresentationStrategy`, the config schema/loader and the strategy registry); the default path is covered by
  the tests of "The identity guarantee" above, so the 486 executed runs and the 39 `kmh_wide` runs (v4, account B)
  stay comparable with the new ones. The code-version audit lists a further commit for the new part only: on account
  A three commits after `kmh_norm` (v2 for 285 leaves, v3 for the 24 `texhal_q13` leaves, v5 for the 27 new ones),
  and the merged `make campaign-verify-all` up to four (v2, v3, v4, v5), all explained by ADR 0017/0018/0019.
- Account B keeps running `kmh_wide` with the v4 notebook: nothing for B changes, and B must NOT switch to v5 in the
  middle of the part (that would put a third code version inside one part). Only account A uses v5.
- The amended code-version rule (`.claude/rules/reproducibility.md`, `.specs/research-rules/reproducibility.md`):
  a new tag for a later part normally requires that nothing a trainer run executes changed; v5 is the one exception,
  allowed because the change is behind a new key whose default path is tested identical to the previous tag.
- The notebook's `ZIP_ONLY_PART` gains `"kmh_norm"` (folder `results/campaign_kmh_fixed/norm`, 27 checkpoints, the
  sweep root's logs renamed `campaign_part_1_gpu__kmh_norm.log` / `failures_part_1_gpu__kmh_norm.log`); the two
  earlier values keep working. Operator steps: `COLAB_RUNBOOK.md` section 6g.
- Follow-up after execution: record the outcome here and in `.specs/experiments/campaign.md`; if a variant becomes
  paper 1's KMH at some query size, it goes through the seeds 4-6 confirmation first.
