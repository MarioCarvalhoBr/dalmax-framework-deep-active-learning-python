# Reproducibility rules

Expanded version of the standing rule (mirrored in `.claude/rules/reproducibility.md`,
owned by a different batch): every experiment must be fully determined by
**(params JSON + CLI args + seed + git commit)**.

## Seed policy

- `--seed` (CLI, default 1) is the single source of randomness for a run:
  `np.random.seed(args.seed)` and `torch.manual_seed(args.seed)` are set at
  the top of `main()` in `demo.py`, before `dataset.initialize_labels(...)`
  (which itself uses `np.random.shuffle` to pick the initial labeled pool —
  correctly seed-derived).
- **Violation found**: `core/query_strategies/ssrae_kmeans_sampling.py:23`
  and `core/query_strategies/vctex_kmeans_sampling.py:26` both call
  `KMeans(n_clusters=n, random_state=3, n_init=10)` — a **literal hardcoded
  seed**, independent of `--seed`. Any experiment using
  `SSRAEKmeansSampling` or `VCTexKmeansSampling` across "different seeds"
   1/2/3 is **not actually varying the k-means randomness** between those
  runs — only the initial labeled pool and network initialization vary.
  This must be fixed (derive `random_state` from `args.seed`) before these
  strategies are used in any ablation row that claims seed-independent
  variance (§6.3 "without hierarchical module" explicitly requires a new/
  fixed variant of this class — see `experiments/ablation-study.md`).
- `SSRAEKmeansHCSampling`/`VCTexKmeansHCSampling`
  (`core/query_strategies/ssl_ssrae_sampling.py`) delegate clustering to
  `core/tools/SSL/src/hierarchical_kmeans_gpu.py` — **TBD** whether that
  module's k-means calls are seeded from `args.seed` or also hardcoded; not
  verified in this batch, flag for the `experiment-auditor` agent.
- `torch.backends.cudnn.enabled = False` (`demo.py`) trades GPU performance
  for determinism; note it disables cuDNN entirely rather than using
  `torch.backends.cudnn.deterministic = True` +
  `torch.backends.cudnn.benchmark = False`, which would keep cuDNN's faster
  kernels while still being deterministic. Flagged as a known issue
  (performance, not correctness) elsewhere.

## Cache policy

`utils/data.py` maintains three pickle caches, all under `results/`, with
**no cache key** encoding what was used to produce them:

| Cache file | Written by | Keys it should have but doesn't |
|---|---|---|
| `results/features_dict_ssrae.pkl` | `Data.create_feature_maps_ssrae` | dataset, `Q` (currently 13), embedding_variant |
| `results/features_dict_vctex.pkl` | `Data.create_feature_maps_vctex` | dataset, `Q` (currently `[5,17]`) |
| `results/Y_train.pkl` | `Data.__init__` | dataset |

Each is written **only if the file does not already exist**
(`if not os.path.exists(path_pkl): ...`) — meaning:

- Changing `Q` (e.g. per the advisor's "fix Q per the SSRAE article" request
  — see `experiments/ablation-study.md` §6.1) **silently reuses the stale
  cache** computed at the old `Q` unless the `.pkl` file is deleted by hand
  first. There is no code-level guard against this.
- Switching dataset (`DANINHAS` ↔ `CIFAR10`) with the SSRAE/VCTex strategies
  selected would similarly reuse whatever `features_dict_*.pkl` happens to
  exist, if it does — a stale-cache hazard confirmed by reading the code
  (not merely inferred).
- **Rule going forward**: any change to `Q`, the SSRAE/VCTex extractor
  implementation, the dataset, or (once implemented) the
  `embedding_variant` must either (a) delete the corresponding `.pkl`
  before running, or (b) — the correct fix, required before the ablation
  study can run its three representation variants without manual
  intervention between them — key the cache filename/dict by
  `(dataset, extractor, Q, embedding_variant)` and never silently reuse a
  cache computed under a different key.
- `results/Y_train.pkl` has no dataset key either; same rule applies when
  switching datasets.

## Config snapshot per run

**Not currently implemented.** `demo.py` logs the full `args` and the
selected dataset's `params` block to `log-dalmax.log` via
`logger.warning(json.dumps(...))` (visible in `dir_results/log-dalmax.log`
after the run), and `results.json` records `dataset_name, strategy_name,
n_init_labeled, n_query, n_round, seed`, but:

- `n_epoch` and the rest of the params JSON (optimizer args, `config_kmh`,
  etc.) are **not** duplicated into `results.json` itself — they are only
  recoverable from the run's log file or by knowing which params JSON was
  used for that batch.
- **No git commit hash is recorded anywhere in the results.** This is
  required by the standing rule and is currently a gap: add
  `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` (or equivalent) to
  `dados_config_results` in `demo.py` before any ablation runs are executed,
  so every `results.json` is traceable to the exact code state that produced
  it. Track this as a refactor-plan Phase 1 item (owned by another batch).

## What "fully determined" means in practice today

To reproduce a given `results.json` leaf directory exactly, you currently
need: the params JSON file used (identified only by filename convention,
e.g. `params_df_gpu_0.json` vs `_1.json` — not embedded in the results), the
exact CLI invocation (seed, n_query, n_round, dataset_name, strategy_name,
dir_results — n_init_labeled and n_epoch are NOT in `results.json`, must be
inferred from the directory name / params file), and — until the fix above
lands — a best-effort guess at which git commit produced it (check file
timestamps against `git log` as a fallback).
