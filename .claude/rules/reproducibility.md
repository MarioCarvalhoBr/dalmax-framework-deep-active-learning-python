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

- `utils/data.py`'s `cache_file_path(name, dataset_folder, q=None, pool_hash=None)`
  keys SSRAE/VCTex feature caches and `Y_train` under `results/cache/` as
  `{name}_{dataset_folder}[_Q{q}[_pool{pool_hash}]].pkl` (e.g.
  `results/cache/features_ssrae_daninhas_micro_Q13_pool<12-hex>.pkl`). SSRAE/VCTex
  feature extraction runs only over the **unlabeled pool** at the moment
  `create_feature_maps_ssrae`/`create_feature_maps_vctex` is called, and that pool
  depends on `--seed` and `--n_init_labeled` — so `pool_hash` is
  `hashlib.sha256(np.where(self.labeled_idxs==0)[0].astype(np.int64).tobytes()).hexdigest()[:12]`,
  computed before the cache path is built. This closes the seed/`n_init_labeled`
  collision hazard: two runs with the same `(dataset_folder, Q)` but different
  seeds/`n_init_labeled` never silently share a cache file. `Y_train` has no pool
  concept (`q=None`) and never carries a `pool_hash` segment, even if one is passed.
  The old fixed-path caches (`results/features_dict_ssrae.pkl`,
  `results/features_dict_vctex.pkl`, `results/Y_train.pkl`) are orphaned — no
  longer read or written by this code path — and must never be resurrected.
- **Still missing**: an `embedding_variant` (`full`/`spatial`/`spectral`, per the
  representation ablation) key component, and `Q` itself is still a hardcoded
  literal (`13` for SSRAE, `[5,17]` for VCTex) rather than config-driven. **Never
  reuse an embedding cache across a different `(dataset, extractor, Q, pool,
  variant)` combination** — if you are about to run an ablation variant, first
  verify (or force-recompute) that the cache file was produced for that exact
  combination. This remaining gap is the highest-risk reproducibility hazard left
  in the ablation study (`.specs/experiments/ablation-study.md`).
- Longer term, the fix is the fuller `EmbeddingProvider`/`EmbeddingCache`
  abstraction (ADR 0003) that supersedes `cache_file_path` with a
  `(dataset, extractor, Q, variant, split)`-keyed cache — part of refactor Phase 2.
  See `.specs/quality/known-issues.md` KI-3.

## Enforcement

- The `experiment-auditor` agent (`.claude/agents/experiment-auditor.md`) checks
  seed propagation, params/spec consistency, and cache validity before any
  experiment handoff to the lab machine or Colab.
