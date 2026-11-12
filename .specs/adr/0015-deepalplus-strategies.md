# ADR 0015: Port the DeepAL+ strategies DalMax lacks (eight new query strategies)

- **Status:** Accepted (implemented 2026-11-11 on branch `feat/deepalplus-fixed-dataset`; not yet wired into any
  campaign manifest, see Consequences)
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
   the NEXT training only (never written into `labeled_idxs`, recomputed every round). `delta_t = delta_0 - dr * t`
   (linear, `delta_0 = 5e-5`, `dr = 0.033e-5`): DeepAL+'s code subtracts `dr * t` at every round, i.e. a cumulative
   `delta_0 - dr t(t+1)/2`, while its own documentation and the paper describe the linear schedule. Entropy via
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
