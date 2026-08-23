# MASTER PROMPT — Project Scaffolding for DalMax

## 1. Context (read carefully, do not invent beyond it)

This repository is a PhD research laboratory for **Deep Active Learning applied to
UAV weed recognition in precision agriculture** (UFMS, Brazil). It was born as
"DalMax" and is being professionalized under the new project name:

- **Project name: `DalMax` — Deep Active Learning Laboratory for UAV Weed Recognition.**
  Keep "DalMax" mentioned once in the README as the historical/legacy name and keep
  the existing citation blocks. Do NOT rename Python packages/folders yet (that is
  refactor work); record the package rename (`core`/`utils` → `dalmax/`) as a decision
  in an ADR and in the refactor plan spec.

Scientific context:
- The main contribution is **RNHAL (Randomized Network-guided Hierarchical Active
  Learning)**: (1) a randomized-network spatio-spectral representation module based on
  **SSRAE** (see `phd_files/artigo-original-tecnica-ssrae-manuscript.pdf` and the
  implementation in `core/tools/SSRAE/`), and (2) hierarchical k-means batch selection
  (`core/tools/SSL/`, strategies `SSRAEKmeansHCSampling` / `SSRAEKmeansSampling` in
  `core/query_strategies/`).
- The paper under revision is `phd_files/Artigo_melhorias_Active_Learning_Mario.pdf`,
  with LaTeX sources in `phd_files/Active_Learning_Mario/` (read `method_full.tex` for
  the formal RNHAL definition).
- Dataset: `DATA/daninhas_full/` — ~10,000 RGB image patches, 5 classes
  (BRACHIARIA, COLONIAO, GRAMINEA, MAMONA, OUTRAS_FOLHAS_LARGAS), `train/` + `test/`
  folders, ~47 MB total. CIFAR10 is a secondary benchmark dataset.
- Classifier under AL: ResNet50 (`core/daninhas_model.py`), orchestrated by
  `utils/orchestrator.py`, entry point `demo.py`, params in `params_df_gpu_*.json`.
- SSRAE embedding layout (**verified 2026-08-23** in `core/tools/SSRAE/extractor.py`):
  each `rnn.beta` has shape `(9, Q+1)` (9 = 3×3 patch dims); the final vector is
  `torch.hstack([beta_R, beta_G, beta_B, beta_S_R, beta_S_G, beta_S_B]).reshape((1,-1))`,
  length `54*(Q+1)`. Because `hstack` concatenates along columns before the row-major
  `reshape`, the result is **row-interleaved**, NOT six contiguous blocks — `emb[:len//2]`
  is not "spatial" and `emb[len//2:]` is not "spectral". The correct slicing reshapes to
  `(9, 6*(Q+1))` and takes column-groups: columns `[0, 3*(Q+1))` = spatial (R,G,B),
  columns `[3*(Q+1), 6*(Q+1))` = spectral (RG,GB,BR). See the corrected code in §6.1.

Execution environments (this constrains all tooling you generate):
1. **Local dev notebook (this machine)**: 16 GB RAM, **no GPU**, Python 3.12,
   Poetry 2.2.1 installed. Used for coding, smoke tests on tiny subsets, reports.
2. **Lab machine (primary training)**: 2× NVIDIA GPUs with 10 GB each. Workflow:
   `git push` from the notebook → `git pull` + run on the lab machine (see
   `run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh`). Results come back via git or copy.
3. **Google Colab Pro (secondary/burst)**: used for one-off runs. Dataset is small
   enough to upload as a zip and extract into the runtime (never read thousands of
   small files directly from Drive).

## 2. Ground rules for this session

- **Read before writing.** Before generating anything, read: `README.md`, `demo.py`,
  `requirements.txt`, `utils/orchestrator.py`, `utils/data.py`, `utils/dataset.py`,
  `core/deep_learning.py`, `core/daninhas_model.py`, `core/query_strategies/strategy.py`,
  `core/query_strategies/ssrae_kmeans_sampling.py`, `core/query_strategies/ssl_ssrae_sampling.py`,
  `core/tools/SSRAE/extractor.py`, `core/tools/SSL/src/hierarchical_kmeans_gpu.py`,
  `core/tools/SSL/src/hierarchical_sampling.py`, `params_df_gpu_0.json`,
  `scripts/run_pipline.sh`, `phd_files/Active_Learning_Mario/method_full.tex`.
- **Do not invent facts.** Every claim in README/specs must be traceable to the code,
  the params files, or the LaTeX sources. If something is unknown, write `TBD` with a
  note instead of guessing.
- **Never touch** `DATA/`, `results/`, `phd_files/`, `ExperimentNotifier/` (read-only
  context). Never delete or rewrite experiment outputs.
- **Do not run training.** The only executions allowed here: `poetry` commands,
  `ruff`, `pytest` (fast tests only), `python -c` import checks.
- Documentation language: **English** for all generated docs (README, CLAUDE.md,
  specs, agents). Code comments in future work: English.
- Use subagents per the model-delegation rule (§4.3): delegate file authoring to
  `sonnet` subagents in parallel where possible; you (the main session) orchestrate
  and review.

## 3. Deliverable A — Poetry environment

1. Read `requirements.txt` and create `pyproject.toml` (Poetry) with:
   - `[tool.poetry]` metadata: name `dalmax`, version `1.0.0`, description, author
     `Mário de Araújo Carvalho <mariodearaujocarvalho@gmail.com>`, license MIT,
     readme, repository URL from the README citation block.
   - `python = ">=3.10,<3.13"` (torch 2.5.0 upper bound).
   - Main dependencies from `requirements.txt` **plus `pandas`** (it is imported by
     `demo.py` and `utils/report/*` but missing from requirements — fix this).
   - Keep the pinned versions from requirements.txt as minimum-compatible constraints
     (`~=` or exact pins — prefer exact pins for reproducibility, this is a research
     lab).
   - A `[tool.poetry.group.dev.dependencies]` group: `ruff`, `pytest`, `pytest-cov`,
     `pre-commit`.
   - `[tool.ruff]` config in `pyproject.toml` (line-length 100, target py310, sensible
     ignore list; do NOT auto-fix existing code in this session).
2. Create `poetry.toml` with `[virtualenvs] in-project = true` so the venv lives in
   `./.venv`.
3. Run `poetry lock` and `poetry install`. If torch installation is too heavy/slow for
   this machine, install with `poetry install --no-root` and report status honestly.
4. Keep `requirements.txt` as a **generated export** for the lab machine and Colab:
   add a Makefile target `export-reqs` running
   `poetry export -f requirements.txt --output requirements.txt --without-hashes`
   (add the `poetry-plugin-export` note in docs if needed).
5. Add `.venv/` to `.gitignore` if missing.

## 4. Deliverable B — `.claude/` (prompt engineering)

Create this exact structure:

```
.claude/
├── settings.json
├── settings.local.json
├── agents/
│   ├── code-reviewer.md
│   ├── implementer.md
│   ├── mechanic.md
│   ├── experiment-auditor.md
│   ├── spec-keeper.md
│   └── paper-liaison.md
├── commands/
│   ├── sync-specs.md
│   ├── new-strategy.md
│   ├── review.md
│   ├── smoke-test.md
│   ├── ablation-status.md
│   ├── results-report.md
│   └── handoff-lab.md
├── rules/
│   ├── model-delegation.md
│   ├── spec-sync.md
│   ├── code-quality.md
│   ├── reproducibility.md
│   ├── data-safety.md
│   └── git-workflow.md
└── skills/
    ├── adding-query-strategy/SKILL.md
    ├── running-experiments/SKILL.md
    ├── ablation-study/SKILL.md
    └── results-reporting/SKILL.md
```

### 4.1 Agents (`.claude/agents/*.md`, with YAML frontmatter: name, description, tools, model)

- **code-reviewer** (`model: sonnet`) — reviews diffs for correctness, reproducibility
  (seed handling, cache invalidation), performance on 10 GB GPUs, and adherence to
  `.claude/rules/`. Read-only tools. Must check that any change to strategies keeps
  `demo.py --strategy_name` choices, `utils/orchestrator.py` registry, and `.specs/`
  in sync.
- **implementer** (`model: sonnet`) — routine implementation: new features, refactors
  with a written plan, new query strategies, config changes. Must follow
  `.specs/architecture/refactor-plan.md` and update specs afterward.
- **mechanic** (`model: haiku`) — trivial mechanical work: renames, docstring/comment
  translation to English, config tweaks, running `ruff`/`pytest` and reporting.
- **experiment-auditor** (`model: sonnet`) — before any experiment handoff, verifies:
  seeds propagated everywhere (flags hardcoded `random_state`), params JSON consistent
  with the spec, results directory naming, embedding cache validity (Q, dataset,
  variant), and that the run script matches `.specs/experiments/`.
- **spec-keeper** (`model: haiku`) — after merged changes, updates `.specs/` and
  `CLAUDE.md` to match reality; maintains the ADR index.
- **paper-liaison** (`model: sonnet`) — cross-checks code vs the LaTeX paper
  (`phd_files/Active_Learning_Mario/`): notation (Q, L, k_i, n_query), reported
  configurations, and drafts LaTeX table/text skeletons from `results/*/results.json`.
  Read-only on code; may write only LaTeX drafts into a `paper_drafts/` folder.

### 4.2 Commands (`.claude/commands/*.md`)

- **/sync-specs** — scan recent git changes, update affected `.specs/` files, report drift.
- **/new-strategy `<ClassName>`** — guided flow to add a query strategy: create module in
  `core/query_strategies/`, register in `__init__.py` + `utils/orchestrator.py` +
  `demo.py` choices, add smoke test, update `.specs/use-cases/add-new-strategy.md`.
- **/review** — run the code-reviewer agent on the current diff.
- **/smoke-test** — run `make smoke` (fast CPU pipeline on a tiny subset) and interpret failures.
- **/ablation-status** — read `.specs/experiments/ablation-study.md` and `results/`,
  print a checklist of which ablation runs are implemented / executed / pending.
- **/results-report `<results_dir>`** — run the `utils/report/` pipeline for a results
  directory and summarize metrics (mean F1 across seeds etc.).
- **/handoff-lab** — pre-flight for the lab machine: run experiment-auditor, ensure
  `requirements.txt` export is fresh, ensure no uncommitted changes, print the exact
  commands to run on each GPU (based on `run_pipe_gpu_*.sh`).

### 4.3 Rules (`.claude/rules/*.md`)

Create `rules/model-delegation.md` with EXACTLY this content:

```markdown
# Model delegation for coding tasks

For all coding tasks, the main session (running a high-power model, e.g. Fable) acts
strictly as an orchestrator — it delegates, provides context, and reviews — and
participates directly only when a case is genuinely complex enough to need the top
model. Use judgement to pick an appropriate lower-power model and run the actual
coding in a subagent via the Agent tool:

- **`sonnet`** — routine implementation work: new features, refactors, survey JSON
  authoring, bug fixes with a known cause.
- **`haiku`** — trivial/mechanical changes: renames, copy tweaks, small config edits,
  running validations.
- **Main session directly** — only when delegation would clearly cost more than it
  saves (one-line edits mid-conversation) or when the task genuinely needs the top
  model (subtle debugging, architecture decisions).

Give the subagent full context in its prompt (relevant files, conventions from
`.claude/rules/`, the spec-sync requirement), then review its output in the main
session before considering the task done.

This is a standing user instruction: it counts as the user having asked for subagent use.
```

- **spec-sync.md** — "Sync between code, `.specs/` and `.claude/`": any change to
  behavior, CLI, params schema, strategy registry, or experiment protocol MUST update
  the corresponding `.specs/` file in the same task; new architectural decisions get
  an ADR in `.specs/adr/`; whenever new rules/conventions emerge in conversation, they
  must be persisted into `.claude/rules/` and `.specs/`.
- **code-quality.md** — the soul of the project: the target is a **new, elegant,
  scalable, fast, and organized codebase**. Principles: single responsibility; no
  duplicated strategy boilerplate; registry pattern over if/elif chains; explicit
  dependency injection (no `setattr` magic); type hints on all new/edited code; no
  hardcoded dataset names, paths, or seeds inside logic; caches must have explicit
  keys (dataset, Q, variant) and invalidation; English-only code and comments;
  fail-fast errors over silent fallbacks; every module importable without side effects.
- **reproducibility.md** — every experiment fully determined by (params JSON + CLI
  args + seed + git commit); the run's config snapshot and commit hash must be saved
  into the results dir; never reuse an embedding cache across different Q/dataset/
  ablation-variant; all randomness must derive from the experiment seed (no literal
  `random_state=3`).
- **data-safety.md** — `DATA/` is immutable; `results/` is append-only (never edit or
  delete past runs); no dataset images, no secrets, no absolute personal paths in
  committed files; large artifacts (>5 MB, `.pkl`, `.pth`) never committed.
- **git-workflow.md** — conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`,
  `exp:` for experiment configs/results metadata); small focused commits; never push
  without user confirmation; branch for anything touching `core/` behavior.

### 4.4 Skills (`.claude/skills/*/SKILL.md`)

- **adding-query-strategy** — step-by-step checklist of every file that must change to
  add a strategy (derive it from how `SSRAEKmeansSampling` is wired today), including
  tests and specs.
- **running-experiments** — how to run locally (tiny smoke subset, CPU), on the lab
  machine (2 GPUs, one params JSON per GPU, `nohup`/`tmux`, where results land), and
  on Colab (upload zip, pip install from exported requirements, copy results back);
  include the ExperimentNotifier usage if evident from that folder's README.
- **ablation-study** — operational guide for the three ablations (mirror of
  `.specs/experiments/ablation-study.md`, see §6): which code changes each ablation
  requires, which configs to run, expected outputs (F1 on the validation/test split),
  and how results feed the paper's `\subsection{Ablation study}`.
- **results-reporting** — how to use `utils/report/` scripts (`1_...` → `4_...`) to go
  from raw results dirs to averaged metrics and LaTeX-ready tables.

### 4.5 Settings

- `settings.json` (committed, team-level): permissions allowlist for `poetry *`,
  `ruff *`, `pytest *`, `make *`, `python -c *`, `git status/diff/log/add/commit`;
  deny `rm -rf`, any write under `DATA/` and `phd_files/`; sensible defaults (no
  dangerous auto-approve).
- `settings.local.json` (gitignored — add to `.gitignore`): minimal valid JSON `{}`
  with a comment-free structure, as a placeholder for machine-local overrides.

## 5. Deliverable C — `.specs/` (specifications)

Create this structure with real content (not lorem ipsum). Where information must
come from deeper reading of the code, read the code; where it is genuinely unknown,
mark `TBD`.

```
.specs/
├── 00-overview.md            # what DalMax is, scientific goal, method summary (RNHAL), status
├── README.md                 # index of all specs + how/when to update them (spec-sync contract)
├── architecture/
│   ├── current-state.md      # honest map of today's code: modules, data flow (demo.py →
│   │                         # orchestrator → dataset/net/strategy → results), coupling points
│   ├── target-architecture.md# the refactored design: dalmax/ package, strategy registry,
│   │                         # config layer (dataclasses/pydantic over params JSON), embedding
│   │                         # provider abstraction (SSRAE | VCTex | ResNet-ImageNet) with
│   │                         # cache keyed by (dataset, extractor, Q, variant), selection
│   │                         # module (flat kmeans | hierarchical), experiment runner, CLI
│   └── refactor-plan.md      # phased plan (see §7) with acceptance criteria per phase
├── adr/
│   ├── README.md             # ADR index
│   ├── 0001-adopt-poetry.md
│   ├── 0002-project-rename-dalmax.md
│   ├── 0003-embedding-provider-abstraction.md
│   └── template.md           # standard ADR template (Context/Decision/Consequences)
├── experiments/
│   ├── experimental-protocol.md  # datasets, splits, seeds (1..3), n_init_labeled, n_query,
│   │                             # n_round, metrics (acc, precision, recall, macro F1),
│   │                             # results dir naming convention (from demo.py)
│   ├── ablation-study.md         # THE ablation spec — full content in §6 below
│   └── baseline-results.md       # where existing results live in results/, which strategy
│                                 # won (RNHAL/SSRAE-based), pointer to paper tables; TBD-fill
├── research-rules/
│   ├── reproducibility.md    # expanded version of the rule: seed policy, cache policy,
│   │                         # config snapshot per run
│   ├── metrics.md            # exact metric definitions used (sklearn macro averages, per
│   │                         # utils/dataset.py calc_metrics_sklearn), rounding, aggregation
│   │                         # across seeds
│   └── dataset-protocol.md   # daninhas_full: 5 classes, train/test layout, patch origin
│                             # (UAV imagery), class imbalance notes (TBD if not verifiable)
├── infrastructure/
│   ├── execution-environments.md # the 3 environments (local CPU / lab 2×10GB / Colab Pro),
│   │                             # decision matrix: what runs where, handoff workflow via git,
│   │                             # Colab checklist (zip dataset to runtime, pin deps, save
│   │                             # results back), GPU memory guidance for batch sizes
│   └── environment-setup.md      # Poetry install, .venv, lab machine setup from exported
│                                 # requirements.txt, CUDA notes
├── quality/
│   ├── code-standards.md     # mirrors rules/code-quality.md with concrete examples from
│   │                         # this repo (before/after style)
│   ├── testing-strategy.md   # smoke tests now; unit tests for embedding slicing, hierarchy
│   │                         # configs, registry; golden-run regression test (tiny subset,
│   │                         # fixed seed → expected selected indices)
│   └── known-issues.md       # verified issues list — START from §8 below and extend with
│                             # what you find while reading
├── templates/
│   ├── spec-template.md
│   ├── experiment-report-template.md   # per-run report: config, commit, seeds, metrics table
│   └── adr → (use adr/template.md; just reference it)
├── use-cases/
│   ├── add-new-strategy.md
│   ├── run-full-benchmark.md          # all strategies × seeds on lab machine
│   ├── run-ablation.md
│   └── generate-report.md
└── future/
    └── ideas.md               # backlog: package rename execution, pydantic configs, W&B or
                                # MLflow tracking, Dockerfile for lab machine, dataset
                                # versioning (DVC), CIFAR10 parity for all strategies
```

Note: the classic `api/`, `frontend/`, `business-rules/` folders were deliberately
replaced by `experiments/`, `research-rules/`, and `adr/` because this is a research
laboratory, not a product web app. Record that reasoning in `.specs/README.md`.

## 6. Ablation study spec (content for `.specs/experiments/ablation-study.md`)

This is the advisor-requested ablation section for the paper
(`\subsection{Ablation study}`). Write the spec so a future session can implement it
without re-deriving anything. Three sub-studies, all evaluated with **macro F1** on
the held-out split, dataset `daninhas_full`, seeds and budgets consistent with
`experimental-protocol.md`:

### 6.1 Representation ablation
- Fix **Q** at the value used in the SSRAE reference article / current experiments
  (verify the current Q in the codebase — check where `ColorFeatureExtractor(Q=...)`
  is instantiated; record the actual value in the spec).
- SSRAE embedding layout (verified in `core/tools/SSRAE/extractor.py`):
  `emb = hstack[β_R, β_G, β_B, β_S_RG, β_S_GB, β_S_BR]`, but this is a **row-interleaved**
  layout, not six contiguous blocks (**Layout caveat, verified 2026-08-23** — see below).
- **Layout caveat (verified 2026-08-23)**: each `beta` block has shape `(9, Q+1)`; the
  `torch.hstack(...).reshape((1,-1))` in `extractor.py` interleaves the six blocks per
  row rather than concatenating them contiguously. `emb[:len(emb)//2]` is therefore
  **not** spatial-only and `emb[len(emb)//2:]` is **not** spectral-only. Correct slicing:
  ```python
  M = emb.reshape(9, 6*(Q+1))          # rows = patch dims, column groups = [R,G,B,RG,GB,BR]
  emb_spatial  = M[:, :3*(Q+1)].reshape(-1)   # Θ_R, Θ_G, Θ_B
  emb_spectral = M[:, 3*(Q+1):].reshape(-1)   # Ω_RG, Ω_GB, Ω_BR
  emb_full     = emb
  ```
- Variants to run:
  - `emb_spatial`  (as computed above) → spatial-only (R, G, B blocks) → F1
  - `emb_spectral` (as computed above) → spectral-only (RG, GB, BR blocks) → F1
  - `emb_full     = emb`                    → full (already implemented) → F1
- Implementation requirement: an `embedding_variant` config option
  (`full | spatial | spectral`) that slices the cached full embedding — never
  recompute SSRAE three times; the cache must be keyed so variants cannot collide.

### 6.2 Hierarchy ablation
All with `n_query = 100`, SSRAE full embeddings, varying the hierarchy config
(`config_kmh` in the params JSON):
- `L=1`, k = [50]
- `L=2`, k = [300, 100]  (alternative: [100, 50] — record both, run per advisor's choice)
- `L=3`, k = [300, 100, 50]
- `L=4`, k = [300, 100, 50, 25]
Implementation requirement: hierarchy depth/cluster counts must come entirely from
config (no hardcoded `'DANINHAS'` key lookup — note this is currently broken, see
known-issues), and `sample_sizes` handling per level must be specified explicitly.

### 6.3 Contribution of the two RNHAL stages
- **RNHAL (full)**: F1 taken from the already-executed experiments (reference runs).
- **Without representation module**: keep hierarchical selection, replace SSRAE
  embeddings with **ImageNet-pretrained ResNet embeddings** (penultimate layer of the
  existing ResNet50; new embedding provider).
- **Without hierarchical module**: SSRAE embeddings + **flat k-means**, selecting
  **random images from each cluster proportionally to the budget** (note: this is
  different from the current `SSRAEKmeansSampling`, which uses k=n and picks the
  closest-to-centroid image — the spec must make this distinction explicit).

Also list in the spec the code capabilities the refactor must provide to make these
ablations one-config-line runs (embedding provider abstraction, embedding_variant
slicing, configurable hierarchy, proportional-cluster-sampling strategy).

## 7. Deliverable D — root documents

- **README.md** — rewrite professionally in English: project name DalMax (legacy:
  DalMax), badges placeholders, method overview (RNHAL, one paragraph + the strategy
  list that actually exists in `demo.py` choices), installation via Poetry (+ pip
  fallback via exported requirements.txt), dataset section (daninhas_full structure +
  CIFAR10), usage examples taken from the real CLI, execution environments summary,
  repository layout tree, links to `.specs/` and `CLAUDE.md`, license, both citation
  blocks (repo + qualification), contact. Keep all factual content from the current
  README that is still true; drop what is stale (e.g., Python 3.9 claim vs 3.10–3.12
  Poetry constraint — reconcile honestly).
- **CLAUDE.md** — operational guide for Claude Code sessions: what the project is
  (3 lines), pointers to `.specs/00-overview.md`; how to set up (`poetry install`);
  commands (make targets); the non-negotiables (summaries of the rules with links to
  `.claude/rules/*`); the multi-agent working mode (model-delegation is standing
  policy); current phase and next milestone (refactor → ablations); what never to do
  (touch DATA/, delete results/, run full trainings locally).
- **AGENTS.md** — tool-agnostic mirror of CLAUDE.md's essential content (for other
  agentic tools), pointing to `.claude/agents/` for role definitions and `.specs/`
  for truth.
- **LICENSE** — verify the existing MIT license file; update copyright year to 2026
  and holder to "Mário de Araújo Carvalho" if not already correct.

## 8. Deliverable E — quality harness

1. **`tests/`** (pytest):
   - `test_imports.py` — every module under `core/` and `utils/` imports cleanly.
   - `test_registry.py` — every `strategy_name` choice in `demo.py` resolves via
     `utils/orchestrator.get_strategy` without error.
   - `test_ssrae_embedding_layout.py` — on a tiny random RGB array, assert the
     extractor output length is `6*(Q+1)*9` and that the layout is row-interleaved
     (reshape to `(9, 6*(Q+1))` column-groups), NOT contiguous halves — the
     corrected spatial/spectral column-group slicing must match the recomputed
     beta blocks (CPU, small Q, must run in seconds; see the Layout caveat in
     `.specs/experiments/ablation-study.md` §6.1).
   - Mark anything needing the real dataset or GPU with `@pytest.mark.skipif`.
2. **`Makefile`** — targets: `setup` (poetry install), `lint` (ruff check), `format`
   (ruff format), `test` (pytest fast), `smoke` (tiny CPU end-to-end run of demo.py on
   a generated micro-dataset or a 2-class/50-image subset — design it, but if a true
   smoke run is not feasible without touching code, create the target as a documented
   stub and record it in known-issues), `export-reqs`.
3. **`.github/workflows/ci.yml`** — on push/PR: setup Python 3.12 + Poetry (cache),
   `ruff check`, `pytest -m "not gpu and not dataset"`. Check `.github/` first — if a
   workflow already exists, extend rather than replace.
4. **`.gitignore`** — ensure: `.venv/`, `__pycache__/`, `.ruff_cache/`, `*.pkl`,
   `*.pth`, `.claude/settings.local.json`, `paper_drafts/`. Do NOT ignore `results/`
   wholesale (history is already tracked); instead document the artifacts policy in
   `.specs/research-rules/reproducibility.md`.
5. **Known issues seed list** for `.specs/quality/known-issues.md` (all verified —
   confirm each while reading, then extend):
   - `demo.py` is a ~290-line monolith mixing CLI, training loop, plotting, and persistence.
   - `pandas` imported in `demo.py` and `utils/report/` but absent from `requirements.txt`.
   - Pickle caches hardcoded in `utils/data.py` (`results/features_dict_ssrae.pkl`,
     `results/features_dict_vctex.pkl`, `results/Y_train.pkl`) with no cache key for
     dataset/Q/variant → stale-cache hazard for ablations.
   - `core/query_strategies/ssl_ssrae_sampling.py` hardcodes `self.params['DANINHAS']`
     → breaks any other dataset.
   - `SSRAEKmeansSampling` hardcodes `KMeans(random_state=3)` → ignores experiment seed.
   - Strategies mutate `dataset.features_dict` (deleting selected ids) as a side effect.
   - `demo.py` injects params via `setattr(strategy, "params", params)` instead of the
     constructor.
   - `torch.backends.cudnn.enabled = False` globally disables cuDNN (performance loss
     on the lab GPUs); should use deterministic-mode flags instead.
   - `utils/orchestrator.py` uses if/elif chains → replace with registries.
   - Dead/experimental files at root: `temp_teste.py`, `test.py` (not a pytest file),
     `TRASH_TEXT.md`, `sampled_data.pdf`, `core/query_strategies/old_functions.py`.
   - Logging misuse: informational messages emitted at `warning` level.
   - Mixed Portuguese/English comments; no type hints; no tests; heavy result
     artifacts inside the git repo.

## 9. Refactor plan skeleton (content for `.specs/architecture/refactor-plan.md`)

Phase the plan so ablations become trivial after Phase 2:
- **Phase 1 — Safety net**: tests in §8, golden-run capture (tiny subset, fixed seed,
  record selected indices + metrics), CI green.
- **Phase 2 — Core refactor** (enables ablations): config layer replacing raw params
  dict (validation, no hardcoded dataset keys); embedding provider abstraction
  (SSRAE / VCTex / ResNet-ImageNet) with keyed cache; selection module abstraction
  (flat k-means closest-to-centroid | flat k-means proportional-random | hierarchical
  k-means) fully config-driven; strategy/dataset/model registries; seed propagation
  audit; split `demo.py` into `runner` + `reporting` + thin CLI.
- **Phase 3 — Ablation implementation**: the three studies of §6 as configs + minimal
  new code, run scripts for the lab machine (one per GPU), report generation.
- **Phase 4 — Polish**: package rename to `dalmax/`, docs refresh, dead-code removal.
Each phase lists acceptance criteria and which agent (from `.claude/agents/`) drives it.

## 10. Execution order and wrap-up

1. Read the files in §2, then create everything in parallel subagent batches:
   (A) Poetry, (B) `.claude/`, (C) `.specs/`, (D) root docs, (E) harness.
2. Verify: `poetry check`, `ruff check` (report-only), `pytest` (fast tests must pass),
   every command/agent/skill file has valid frontmatter, every `.specs` cross-link resolves.
3. Produce logical commits (do not push): `docs: ...`, `chore(poetry): ...`,
   `chore(claude): ...`, `docs(specs): ...`, `test: ...`.
4. End with a summary: what was created, what is `TBD`, verified known-issues count,
   and the recommended next session ("start Phase 1 of refactor-plan.md").
