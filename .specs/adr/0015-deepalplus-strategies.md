# ADR 0015: Port the DeepAL+ strategies DalMax lacks (eight new query strategies)

- **Status:** Accepted (implemented 2026-11-11 on branch `feat/deepalplus-fixed-dataset`); wired into paper 1 of the
  campaign by ADR 0016 (all eight at `n_query` 10/50/100; `VAAL` with `image_size` 32; the signature follow-up of
  Consequences is done)
- **Date:** 2026-11-11

## Context

Paper 1 benchmarks the classical deep active learning strategies. DalMax's 12 baselines come from DeepAL
(Huang, 2021). DeepAL+ (Zhan et al., "A Comparative Survey of Deep Active Learning", arXiv 2203.13450; code
`SineZHAN/deepALplus`, MIT licence, read at commit `93c2884`, never executed) extends that toolkit. The author
decided to include in a full re-run of the campaign every DeepAL+ method DalMax lacks: `VarRatio`, `MeanSTD`,
`KCenterGreedyPCA`, `BadgeSampling`, `CEALSampling`, `LossPredictionLoss`, `VAAL`, `WAAL` (DeepAL+'s
`KMeansSamplingGPU` is the remaining one, see Decision 9).

Reading the upstream code (and, where it disagreed with the papers, the original authors' code: `sinhasam/vaal`,
`cjshui/WAAL`, `Mephisto405/Learning-Loss-for-Active-Learning`, `JordanAsh/badge`) showed several defects and
undocumented choices that would change results if copied. Constraints: every existing strategy stays bit-identical
for the same seed (RNG stream, parameter order, logits; the golden fixtures pass unchanged); no hardcoded
`.cuda()`; strategy hyperparameters through the typed config layer; no if/elif on strategy names in
`dalmax/models/base.py`.

## Decision

We will add the eight strategies as `Strategy` subclasses registered in
`dalmax/query_strategies/registry.py::DEEPALPLUS_STRATEGY_REGISTRY`, each built as
`Class(dataset, net, logger, **DEEPALPLUS_STRATEGY_KWARGS[name](config, dataset, net, logger))`, with
hyperparameters in a new optional params-JSON block `"strategy_args": {"<StrategyName>": {...}}`
(`dalmax/config/schema.py::StrategyArgs`, syntax in `.specs/experiments/experimental-protocol.md`). Shared pieces:

- `dalmax/query_strategies/ranking.py`: `rank_ascending`/`rank_descending`, the seeded tie-break of KI-45
  generalised (`adversarial_base.rank_by_distance` delegates to it; same result).
- `dalmax/seeding.py::isolated_torch_seed(seed)`: auxiliary networks are initialised inside it, so they never
  consume the run's global torch stream; `torch_generator(seed)` for explicit draws. Every auxiliary DataLoader gets
  an explicit `generator=`.
- Training hook: `DeepLearning.build_model()` is the single task-model constructor (`DeepLearning.train` now calls
  it; same RNG consumption). A strategy that changes how the task model is trained overrides `Strategy.train()` and
  delegates to an injected trainer object (`dalmax/models/loss_prediction.py::LossPredictionTrainer`,
  `dalmax/models/waal.py::WAALTrainer`); `CEALSampling.train` passes another dataset to `net.train`.
  `DeepLearning.train` itself is untouched.
- `Strategy.diagnostics()` (default `{}`) feeds an optional `strategy_diagnostics` key of `results.json`; absent
  for every strategy that reports nothing, so their `results.json` schema is unchanged.

Per method, with every deviation from DeepAL+:

1. **VarRatio** (`var_ratio.py`). DeepAL+ computes `1 - max softmax` of one deterministic pass, which ranks exactly
   like LeastConfidence. We implement the cited definition (Freeman, 1965; Gal, Islam & Ghahramani, 2017): `T =
   n_drop` MC-dropout passes, score `1 - f_m / T` (`f_m` = passes predicting the modal class), highest first. Scores
   take at most `T + 1` values, so exact ties are broken at random by `rank_descending` with
   `derive_seed(seed, "var_ratio_ties")`. Uses the KI-41 MC-dropout machinery: needs `model_args.dropout_p`
   (`params_paper1_mc_dropout.json`, 0.5), raises `MCDropoutError` otherwise.
2. **MeanSTD** (`mean_std.py`). Mean over classes of the per-class population std (`ddof = 0`, as `np.std`) of the
   MC-dropout softmax, highest first; needs dropout as above. Stable sort instead of torch's unstable descending sort.
3. **KCenterGreedyPCA** (`kcenter_greedy_pca.py`). PCA to 50 dimensions when the embedding is wider (2048 here),
   then the greedy loop, now shared with `KCenterGreedy` as `kcenter_greedy.py::k_center_greedy` (moved verbatim;
   `KCenterGreedy`'s result is unchanged, tested against the old body). PCA `random_state =
   derive_seed(seed, "kcenter_pca")` (sklearn uses the randomized solver for this shape; DeepAL+ left it unseeded).
   Float32 instead of DeepAL+'s float16 cast (precision loss, spurious ties); squared distances clamped at 0 before
   the square root (ADR 0014).
4. **BadgeSampling** (`badge_sampling.py`). Gradient embedding of the last linear layer with the predicted class as
   pseudo-label, `g_c = (p_c - [c = y_hat]) h`, class-major (5 x 2048 = 10,240-d), then k-means++ seeding (first
   point: largest gradient norm; next: probability proportional to `D^2`). Computed batched in torch
   (`gradient_embeddings`), and the selection uses the rank-one factorisation `<g_x, g_y> = <a_x, a_y><h_x, h_y>`
   in float64 (the BADGE authors' later speed-up; no 10,240-d matrix, no pairwise matrix; tested equal to the
   explicit embeddings). Draws from `np.random.default_rng(derive_seed(seed, "badge"))` instead of scipy's global
   `rv_discrete`; chosen points get probability 0 (DeepAL+ re-draws); all-zero distances (duplicate images,
   KI-46) fall back to a uniform draw among unchosen points (DeepAL+ stops in `pdb.set_trace()`).
5. **CEALSampling** (`ceal.py`). The base strategy (params key, default `EntropySampling`, restricted to
   `LeastConfidence`/`MarginSampling`/`EntropySampling`, built through the registry) picks the queried samples;
   the remaining unlabeled samples with predictive entropy `< delta_t` get their predicted class as pseudo-label for
   the NEXT training only (never written into `labeled_idxs`, recomputed every round). Threshold values and schedule
   **revised by the amendment below** (paper's CACD values `delta_0 = 0.05`, `dr = 0.0033`, schedule as a params key,
   clamp at 0); the first implementation used DeepAL+'s `5e-5` / `0.033e-5`, which pseudo-label nothing here. Entropy via
   `torch.xlogy` (DeepAL+'s `p log p` gives NaN for `p = 0`, silently excluded). The count and the accuracy of the
   pseudo-labels per round are logged and written to `results.json` (`strategy_diagnostics`), reporting only.
6. **LossPredictionLoss** (`loss_prediction.py`, `dalmax/models/loss_prediction.py`). Loss-prediction module on
   the four ResNet-50 stage maps (`DaninhasModelResNet50.forward_with_stages`, same modules and order as
   `forward`; adaptive GAP + FC 128 + ReLU per stage, concatenation, FC to 1), trained jointly with the task model
   on `CE + 1.0 * ranking_loss` (margin 1.0). Task-model budget = `n_epoch` like every strategy (DeepAL+ adds 20
   epochs); the module's gradient reaches the backbone during the first `floor(detach_fraction * n_epoch)` epochs
   (default 0.6 = the paper's 120/200; DeepAL+ detaches from epoch 20 on whatever the budget). One forward pass per
   batch (DeepAL+ runs the backbone twice per batch, updating BatchNorm statistics twice). Odd batch: the last
   sample is left out of the ranking loss only; a batch of one has no pair. Module rebuilt with every task model
   (DeepAL+ reuses one module instance across rounds while rebuilding the backbone), initialised from
   `derive_seed(seed, "loss_prediction")`. Module optimizer Adam lr 0.01 (DeepAL+; the paper uses the backbone's
   SGD). Query: highest predicted loss.
7. **VAAL** (`vaal.py`, `dalmax/models/vaal.py`). Task model trained exactly as for the other strategies. At every
   query a fresh VAE + latent discriminator are trained (`epochs` 30, `latent_dim` 32, `beta` 1, `adversary_param`
   1, `num_vae_steps` 2, `num_adv_steps` 1) on the joint labeled/unlabeled loader (labeled set cycled), with
   **discriminator targets labeled = 1, unlabeled = 0** (original VAAL; DeepAL+ uses 1 for both, so its
   discriminator learns nothing) and the VAE fooling it (both = 1); query = the LOWEST discriminator outputs (most
   likely unlabeled; DeepAL+ sorts descending). DeepAL+ runs `num_vae_steps` discriminator steps (its loop uses the
   wrong counter); we run `num_adv_steps`. Optimizer Adam lr 5e-4 for both networks (original VAAL; DeepAL+ reuses
   the task optimizer, SGD lr 0.05 here). Network: the 64 x 64 `VAE_ImageNet`-style VAE; DANINHAS images (128 x
   128) are area-resized to 64 x 64 for it. DeepAL+'s `VAE_ImageNet` cannot run as written (first convolution stride
   1, so `View((-1, 1024*4*4))` turns each image into four rows); our first convolution has stride 2, the shape its
   `1024*4*4`/`1024*8*8` sizes imply. Initialisation as both codes (Kaiming on encoder/decoder Conv2d/Linear,
   PyTorch defaults on `fc_mu`/`fc_logvar` and transposed convolutions, which their `weight_init` skips). All
   randomness (weights, reparameterisation noise, loader order) from `derive_seed(seed, "vaal")`; nothing touches
   the global torch stream. Networks in eval mode for scoring.
8. **WAAL** (`waal.py`, `dalmax/models/waal.py`). Replaces the task training (every round, round 0 included): feature
   extractor = our ResNet-50 backbone, classifier = its final linear layer (one `model(x)` gives both), Wasserstein
   critic `2048 -> 512 -> 512 -> 1` with sigmoid (DeepAL+/original `Discriminator`), gradient penalty on random
   interpolates; joint loader; task step `CE + alpha W + alpha * 5 * gp`, critic step `-alpha W - alpha * 2 * gp`;
   `alpha = 1e-3`, `gamma_ratio = 1` (DeepAL+; the original uses `|L|/|U|`), SGD with the task optimizer args for
   both (as both codes). Query `0.5 ||log p||_2 + 0.5 ||log p||_1 - 10 * critic(z)`, lowest first, with `log p` =
   `log_softmax` in float64 (no `-inf` on a saturated softmax). The evaluated model is the ordinary task model, so
   `saved_model.pth` and the inference tools work unchanged (tested end to end); the critic is not saved. Critic
   init, GP coefficients and loader order from `derive_seed(seed, "waal")`; the task model from `build_model()`.
9. **Not ported: `KMeansSamplingGPU`**. Same method as `KMeansSampling` with faiss GPU k-means: a GPU-only
   dependency and no new method. (Its upstream query is also broken: `index.search(X, 1)` returns, for every
   pool point, the id of its nearest *centroid*, and those centroid ids are then used as pool positions, so it
   returns one id per pool point, all drawn from the first `n` unlabeled samples.)

Kept as upstream and flagged (open, would change results if changed): WAAL's critic step **subtracts** the gradient
penalty (`- alpha * gp * 2`, the critic is pushed to increase it, the opposite of WGAN-GP), as in the original WAAL
code too; the gradient penalty of WAAL's task step is computed on detached features and therefore contributes no
gradient (a constant), also as in both codes (its recomputation is kept: in train mode it updates BatchNorm
statistics); VAAL reuses the same batch for both VAE steps (the original draws a new one between steps).

## Consequences

- Positive: paper 1 can cover every DeepAL+ method with the same protocol, seeds, model and budget; two upstream
  defects that would have produced a meaningless VAAL and a duplicate of LeastConfidence are avoided.
- Behaviour preservation: the 20 pre-existing strategy names are untouched (the golden fixtures pass unchanged;
  `KCenterGreedy` tested against its old body; `DeepLearning.train`'s RNG consumption unchanged by `build_model`;
  `DaninhasModelResNet50` gained only methods, no parameter or state).
- `run_metadata.json` of every run written from now on carries `config.dataset.strategy_args` (all five sub-blocks,
  defaults included). `dalmax.campaign.effective_signature` hashes the whole dataset config, so a params file that
  sets a `strategy_args` block makes every strategy's group signature differ from the same params without it; the
  campaign-wiring task should reduce the signature to the block the strategy reads.
- Cost: `VAAL` and `WAAL` are far more expensive per round than a plain strategy (VAAL trains a VAE for 30 epochs over
  the pool at every query; WAAL's epochs run over the unlabeled pool, not the labeled set). `LossPredictionLoss`
  costs about one plain run. Budget before scheduling (see the implementation report of 2026-11-11).
- Follow-up (not done here, by request): add the eight strategies to the campaign manifest(s), runbooks and
  notebook; update `CLAUDE.md`'s strategy list and `.claude/rules/reproducibility.md` with the
  `isolated_torch_seed` convention (recorded in `.specs/research-rules/reproducibility.md`).

## Amendment (2026-11-11, review follow-up before any GPU time)

An independent review of the first implementation found that three of the ports could silently degenerate and
that their behaviour was not observable. Changes (method semantics unchanged except CEAL's threshold values):

1. **CEAL threshold.** Reviewer's pre-flight measurement on a trained nq100 checkpoint: the minimum predictive
   entropy over 1,000 pool images is 5.4e-5 nats, 9.2% of them are below 5e-3 and 24.5% below 5e-2. With DeepAL+'s
   `delta_0 = 5e-5` nothing is ever pseudo-labelled, so CEAL would be a copy of EntropySampling. The paper (Wang et
   al., arXiv 1701.03551, Sec. IV-A.2) reads: "For CACD, [...] We resize all the faces into 200 x 150, then we set the
   parameters: δ0 = 0.05, dr = 0.0033 and K = 2000. For Caltech-256, we resize all the images to 256 x 256 and we set
   δ0 = 0.005, dr = 0.00033 and K = 1000." Both settings give a non-empty pseudo-label set here (24.5% and 9.2% at
   round 0). **The defaults are now the CACD setting (`delta_0 = 0.05`, `dr = 0.0033`)**: it is the setting of the
   paper's own sensitivity analysis (Sec. IV-D: δ in 0.045–0.1 with dr = 0.0033; dr in 0.001–0.0035 with δ = 0.05),
   which states that this range "leads to [...] about 10% 60% samples [being] pseudo-labeled", and 24.5% falls inside
   it; the Caltech setting (9.2%) is at its lower edge. DeepAL+'s values are the deviation. The threshold is clamped
   at 0. **Schedule (evidence found during the follow-up)**: the paper's Eq. (9) is `δ = δ0` for `t = 0` and
   `δ = δ − dr ∗ t` for `t > 0`, a recursive update whose literal reading is DeepAL+'s cumulative schedule
   `δ0 − dr t(t+1)/2`, not the linear `δ0 − dr t` stated in item 5 of the first version of this ADR (that statement
   was wrong). Both are now a params key (`strategy_args.CEALSampling.delta_schedule`, `"linear"` | `"cumulative"`).
   The default stays `"linear"` (the coordinator's earlier decision); with the CACD values and 8 rounds the
   cumulative reading reaches 0 at round 6 (0.05 − 0.0033·21 < 0), so rounds 6–8 would pseudo-label nothing, while
   the linear one ends at 0.0236. **Open: confirm the schedule.** Training cost grows with the pseudo-labelled set:
   with the same `n_epoch`, a CEAL training takes `ceil((|L| + |P|) / batch)` SGD steps per epoch, so CEAL also
   takes more SGD steps per round than the plain strategies (`ceal_train_size` is reported per training). The
   probabilities CEAL thresholds are now sized by `n_classes` (one eval pass), not by the classes present among the
   unlabeled samples' true labels.
2. **WAAL observability (method unchanged: alpha, penalty sign, gamma as DeepAL+/original).** Per training:
   `waal_n_steps`, `waal_wasserstein_last_epoch`; per query: critic mean/std on the labeled set and on the pool,
   `waal_critic_init_corr` (Pearson between the pool scores of the trained critic and of a frozen copy of the critic
   at initialisation), std of the uncertainty term and of the `selection * critic` term. Recorded findings: (a) the
   reviewer found that at `alpha = 1e-3` with the task SGD settings the critic stayed 0.99-correlated with its
   initialisation on a synthetic problem, so the critic term may carry little information here; reproduced in this
   follow-up on a toy (correlation 0.998–1.000 at `alpha = 1e-3`, with and without trainable features); at
   `alpha = 1` with parameter-free features the critic separates labeled from unlabeled (AUC 1.0,
   `tests/test_deepalplus_strategies.py`), while with trainable features the task step's alignment term wins and the
   critic's ordering ends inverted (AUC 0.0–0.08); (b) training-budget confound: the joint loader iterates over the
   pool (labeled set cycled), so WAAL takes about 280 SGD steps per round (≈ 7,100 / 256 × 10 epochs) against 10 to
   40 for every other strategy (|L| from 100 to 900 at batch 256), round 0 included.
3. **Seeded tie-break everywhere.** `LossPredictionLoss`, `VAAL`, `WAAL` and `MeanSTD` ranked with a stable argsort,
   which falls back to pool order if their scores collapse (the KI-45 failure). They now rank with
   `ranking.rank_descending`/`rank_ascending` and a per-run generator: tags `"loss_prediction_ties"`,
   `"vaal_ties"`, `"waal_ties"`, `"mean_std_ties"` (distinct from the trainers' own generators).
4. **Degeneracy diagnostics** in `results.json["strategy_diagnostics"]`, per round, reporting only:
   VAAL `vaal_discriminator_auc` (labeled vs unlabeled at query time), `vaal_score_std`, `vaal_score_n_unique`,
   `vaal_mean_kl`, `vaal_mu_dim_std`; LossPredictionLoss `lpl_labeled_spearman` (predicted loss vs true
   cross-entropy on the labeled set after training), `lpl_pool_score_std`, `lpl_pool_score_n_unique`; VarRatio
   `var_ratio_n_nonzero`; MeanSTD `mean_std_score_mean`/`_std`; and for all eight `pool_order_fraction`, the fraction
   of the query among the first `2 n` unlabeled pool positions (KI-45 signature; ~`2n/|U|` when informative, 1.0 in
   pool-order failure). Implemented per strategy (`dalmax/query_strategies/diagnostics.py`), not in the runner, so
   the `results.json` of the other strategies is unchanged; every DeepAL+ port now writes `strategy_diagnostics`.
5. **Tests**: discriminator learnability (VAAL discriminator on fixed separable latents; WAAL critic with
   parameter-free features at `alpha = 1`), CEAL with the new defaults pseudo-labels a confident toy model without
   touching `labeled_idxs`, constant scores use the seeded tie-break (not pool order) for MeanSTD, VarRatio,
   LossPredictionLoss, VAAL and WAAL, and the micro e2e runs assert `pool_order_fraction < 0.8` and that no query is
   the pool-order head.
6. **Explicit DataLoader generators** in `WAALTrainer.score_components` and `LossPredictionTrainer.predict_losses`
   (the convention of `.specs/research-rules/reproducibility.md`; the first version left them out).

Observation recorded for VAAL (open, not changed): on a tiny toy, end to end, the adversarial VAE drives the
discriminator's AUC to about 0 within a few epochs, and the AUC differs between eval-mode scoring (BatchNorm
running statistics, our choice) and train-mode scoring (per-batch statistics, as the original sampler implicitly
does): the VAE normalizes the labeled and the unlabeled batch separately during training, as DeepAL+ and the
original do, so the discriminator is trained on per-batch-normalized latents. `vaal_discriminator_auc` will show
whether this matters on DANINHAS; if it stays near or below 0.5, scoring in train mode is the first thing to try.


## Amendment (2026-11-12): the WAAL training-budget control `WAALRandomQuery` (author decision, before any run)

Finding (b) of the first amendment is a confound: WAAL takes about 280 SGD steps per round (the joint loader iterates
over the pool) against 10 to 40 for every other strategy, round 0 included, so a gain of WAAL over `RandomSampling`
could come from the training and not from which images WAAL queries. Decision: add a control that keeps WAAL's training
and replaces only the query by a uniform random draw from the unlabeled pool.

1. **Code** (`dalmax/query_strategies/waal.py`). A shared, unregistered base `WAALTrainedStrategy(Strategy)` holds
   everything that is not the query rule: `train()` (the injected `WAALTrainer.fit`), the reporting-only critic
   evaluation at query time (`_score_pool`: `score_components` on the pool, then `_record_critic`: on the labeled set)
   and `diagnostics()`. `WAAL(WAALTrainedStrategy)` keeps its query (`uncertainty - selection * critic`, lowest first,
   `waal_ties` tie-break) and its diagnostics, unchanged (same keys, values and order). `WAALRandomQuery
   (WAALTrainedStrategy)` draws `rng.choice(unlabeled_idxs, n, replace=False)`; its scores never reach the pick.
2. **What is shared, precisely**: the trainer (built by the same `registry._waal_trainer`: `strategy_args.WAAL` block,
   generator `derive_seed(seed, "waal")`), hence the task-model construction from the run's torch stream
   (`DeepLearning.build_model`), the critic built inside `isolated_torch_seed`, the joint loader order, the
   gradient-penalty draws and the optimizers; the hyperparameters (the control reads the `WAAL` block:
   `dalmax/config/schema.py::STRATEGY_ARGS_READ_BY["WAALRandomQuery"] = "waal"`; there is no `"WAALRandomQuery"` key in
   `STRATEGY_ARGS_KEYS`, a params JSON that sets one is rejected, so the two cannot drift); the critic evaluation of
   every query, which also draws one loader seed per call from the trainer generator, so the generator state handed to
   the next training is identical; the params file (`params_paper1.json`). **Not shared**: the query rule, its
   generator (`derive_seed(seed, "waal_random_query")`, created once per run by
   `registry._waal_random_query_kwargs`, advancing across rounds) and the query diagnostics. Consequence: for the same
   seed the round-0 model, round-0 metrics and round-0 diagnostics are identical (tested:
   `tests/test_waal_random_query.py` on a tiny model, `tests/test_deepalplus_e2e.py::test_waal_control_reproduces_waal_round0`
   through `tools/trainer.py` on the micro dataset); later rounds differ only through the labeled set.
3. **Registry**: a new family `CONTROL_STRATEGY_REGISTRY` / `CONTROL_STRATEGY_KWARGS` (controls are not benchmark
   strategies, so `DEEPALPLUS_STRATEGY_REGISTRY` keeps exactly the eight ports); `build_strategy` dispatches both
   families through `KWARGS_STRATEGY_REGISTRY` / `KWARGS_STRATEGY_FACTORIES` (no new branch); `dalmax/cli.py` choice
   `WAALRandomQuery`. The campaign signature reads `STRATEGY_ARGS_READ_BY`, so the control's signature depends on the
   WAAL block only (and differs from WAAL's by the strategy name).
4. **Diagnostics** (`strategy_diagnostics`): per training `waal_n_steps`, `waal_wasserstein_last_epoch` (as WAAL); per
   query `waal_critic_labeled_mean/std`, `waal_critic_unlabeled_mean/std`, `waal_critic_init_corr`,
   `pool_order_fraction` and `query_rule = "uniform_random"`; not `waal_uncertainty_std` / `waal_critic_term_std`
   (terms of a score the control does not use).
5. **Campaign** (ADR 0016 amendment of the same day): group `p1_control/WAALRandomQuery/nq100`, part `paper1_control`,
   its own table, nq100 only. Reading: WAAL vs control isolates the selection at equal training; control vs
   `RandomSampling` isolates the training at equal (random) selection.

Rejected: a flag on WAAL (`strategy_args.WAAL.random_query`): it would make the control a WAAL configuration that the
benchmark rows and the signature would have to special-case, and a WAAL block with the flag set could leak into the
benchmark by a params mistake; a copy of WAAL's class (duplicated training code that could drift); giving the control
its own `strategy_args` block (two blocks to keep equal by hand).
