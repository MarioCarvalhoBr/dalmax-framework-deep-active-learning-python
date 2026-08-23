# Reproducibility

Every experiment must be fully determined by: **(params JSON + CLI args + seed +
git commit)**. Given those four things, the same results.json must be reproducible
(modulo hardware nondeterminism already tracked in known-issues).

## Seed policy

- `demo.py` seeds `np.random.seed(args.seed)` and `torch.manual_seed(args.seed)`
  from the `--seed` CLI argument (see `run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh`,
  which sweep `SEEDS=(1 2 3)`). Any new source of randomness (sampling, clustering,
  augmentation) must derive from this same seed — never a literal.
- **Known violation to fix, not imitate**: `core/query_strategies/ssrae_kmeans_sampling.py`
  instantiates `KMeans(n_clusters=n, random_state=3, n_init=10)` — the `3` is a
  literal, completely ignoring `--seed`. Two runs with different `--seed` values
  produce identical cluster assignments. New/edited strategy code must pass the
  experiment seed into any stochastic estimator (`KMeans(random_state=self.dataset.seed)`
  or equivalent, once the seed is threaded through — see refactor Phase 2, "seed
  propagation audit").
- `torch.backends.cudnn.enabled = False` is currently used as a determinism
  shortcut in `demo.py:59`; it globally disables cuDNN and costs performance on the
  10 GB lab GPUs. Prefer `torch.backends.cudnn.deterministic = True` +
  `torch.backends.cudnn.benchmark = False` instead, once changed as part of a
  tracked task (do not silently swap this in an unrelated change).

## Config snapshot

- Each run's results directory (`{dir_results}/{dataset}/SEED_{seed}/NQ_{n_query}_NIL_{n_init}_NR_{n_round}_NE_{n_epoch}/{strategy}/`)
  must contain enough to reconstruct the run: the params JSON content used, the
  CLI args, and the git commit hash active at run time. Today only `results.json`,
  `predictions.csv`, plots, and `log-dalmax.log` are written — commit hash and full
  config snapshot are a gap (see `.specs/quality/known-issues.md`); any new
  reporting code should add this rather than assume it already exists.

## Embedding cache discipline

- `utils/data.py` caches full-dataset embeddings at fixed paths:
  `results/features_dict_ssrae.pkl`, `results/features_dict_vctex.pkl`,
  `results/Y_train.pkl` — with **no cache key** for dataset name, `Q`
  (SSRAE `Q=13`, VCTex `Q=[5,17]`), or embedding variant (`full`/`spatial`/
  `spectral`, per the representation ablation). **Never reuse an embedding cache
  across a different `(dataset, extractor, Q, variant)` combination** — if you are
  about to run an ablation variant, first verify (or force-recompute) that the
  cache file was produced for that exact combination. This is the single highest-
  risk reproducibility hazard in the ablation study (`.specs/experiments/ablation-study.md`).
- Longer term, the fix is a keyed cache path (e.g.
  `results/features_dict_{dataset}_{extractor}_{Q}_{variant}.pkl`) — part of
  refactor Phase 2 (embedding provider abstraction).

## Enforcement

- The `experiment-auditor` agent (`.claude/agents/experiment-auditor.md`) checks
  seed propagation, params/spec consistency, and cache validity before any
  experiment handoff to the lab machine or Colab.
