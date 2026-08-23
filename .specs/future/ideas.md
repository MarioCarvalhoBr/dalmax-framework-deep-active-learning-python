# Future ideas / backlog

Not committed to any timeline; captured here so they aren't lost between
sessions. Cross-reference the relevant spec file where a fuller rationale
already exists rather than duplicating it.

- **Package rename execution**: consolidate `core/` + `utils/` into a single
  `dalmax/` package (ADR 0002, ref. `architecture/refactor-plan.md` Phase 4
  — both owned by another batch). Blocked on the Phase 2/3 refactor landing
  first so the rename touches a smaller, already-cleaned surface.
- **Pydantic/dataclass config layer** replacing raw params-JSON dict access
  (`self.params[dataset_name][key]` scattered across strategies) — would
  also fix the hardcoded `'DANINHAS'` key issue
  (`core/query_strategies/ssl_ssrae_sampling.py:66`) structurally, by making
  the active dataset's config the thing passed in, not a dict requiring a
  manual key lookup. See `experiments/ablation-study.md`'s "Code
  capabilities" section — this is a prerequisite for §6.2's hierarchy
  ablation to run cleanly.
- **Experiment tracking (W&B or MLflow)**: today, run provenance is
  reconstructed from directory-naming convention + log file contents (see
  `research-rules/reproducibility.md`) — no run-level dashboard, no
  git-commit-hash logging. Either tool would also solve the "config
  snapshot per run" gap identified there.
- **Dockerfile for the lab machine**: would pin the CUDA 12.4 /
  torch 2.5.0 / torchvision 0.20.0 stack (see
  `infrastructure/environment-setup.md`) reproducibly instead of relying on
  a manually-maintained conda/pip environment on shared lab hardware.
- **Dataset versioning (DVC or similar)**: `DATA/daninhas_full/` is
  gitignored and exists only as a local/lab-machine copy; there is no
  content hash or version tag tying a given `results/` run to the exact
  dataset snapshot that produced it. Low risk today (dataset appears
  static) but worth doing before any dataset revision (e.g. added classes,
  relabeled patches) to avoid silently invalidating past comparisons.
- **CIFAR10 parity for all strategies**: `SSRAEKmeansHCSampling`/
  `VCTexKmeansHCSampling` cannot currently run against CIFAR10 due to the
  hardcoded `'DANINHAS'` params key (see above) — fixing this would let
  CIFAR10 serve as a true secondary-benchmark sanity check for RNHAL, not
  just for the uncertainty-based baselines.
- **Consolidate `utils/report/2_`, `3_`, `4_` scripts**: their nearly
  identical `MetricsType`/`create_csv_tables` definitions (see
  `use-cases/generate-report.md`) suggest either a copy-paste history or
  a not-yet-understood division of responsibility; worth reading in full
  and either documenting the real distinction or merging them into one
  parameterized script.
- **Macro-F1 reporting**: add `average='macro'` alongside the existing
  `average='weighted'` in `Data.calc_metrics_sklearn` (see
  `research-rules/metrics.md`) — needed for the ablation study but also
  generally useful given `daninhas_full`'s class imbalance
  (`research-rules/dataset-protocol.md`), where weighted metrics can mask
  poor minority-class performance.
- **`ExperimentNotifier` documentation**: its README (if any) was not read
  in this batch; worth folding its actual usage/config into
  `infrastructure/execution-environments.md` once reviewed, since it is
  currently only described via its CLI invocation in the run scripts.
- **Reconcile `results/dalmax1/` provenance**: it contains more strategies
  than `run_pipe_gpu_0.sh` alone would produce (see
  `experiments/baseline-results.md`) — worth an audit of `log-dalmax.log`
  timestamps/commit hashes (once the git-commit-hash logging fix lands) to
  document exactly which script/commit produced each strategy's results,
  for the paper's methods section.
