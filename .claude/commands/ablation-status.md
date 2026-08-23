---
description: Read the ablation study spec and results/, and print a checklist of which ablation runs are implemented, executed, or pending.
---

Read `.specs/experiments/ablation-study.md` (mirrored by
`.claude/skills/ablation-study/SKILL.md`) and cross-reference against
`results/` to report ablation progress.

Steps:

1. Load `.specs/experiments/ablation-study.md` for the three sub-studies:
   - **6.1 Representation ablation**: `emb_spatial` / `emb_spectral` / `emb_full`
     (SSRAE, Q=13).
   - **6.2 Hierarchy ablation**: `L=1..4` configs (`k=[50]`, `k=[300,100]` or
     `[100,50]`, `k=[300,100,50]`, `k=[300,100,50,25]`), `n_query=100`.
   - **6.3 RNHAL stage contribution**: full RNHAL vs. no-representation (SSRAE
     replaced with ImageNet-ResNet embeddings) vs. no-hierarchy (flat k-means,
     proportional random sampling per cluster).
2. For each variant, check two things:
   - **Implemented**: does the code support it today? (e.g. does an
     `embedding_variant` config option exist and slice the cached SSRAE
     embedding per `.specs/experiments/ablation-study.md`'s implementation
     requirement, or is this still `TBD`/pending per
     `.specs/architecture/refactor-plan.md` Phase 3?)
   - **Executed**: search `results/` (e.g. `find results -name results.json`)
     for a results directory whose config matches the variant (dataset
     `daninhas_full`, matching `config_kmh` or `embedding_variant`, seeds
     1-3). Use `$ARGUMENTS` to scope the search to a specific results root
     (e.g. `results/dalmax1/`) if given.
3. Print a checklist:

```
## 6.1 Representation ablation
- [x/ /?] emb_full     — implemented: yes | executed: <path or "none found">
- [ ] emb_spatial      — implemented: <status> | executed: <status>
- [ ] emb_spectral     — implemented: <status> | executed: <status>

## 6.2 Hierarchy ablation
- [ ] L=1 k=[50]           — implemented: <status> | executed: <status>
- [ ] L=2 k=[300,100]      — implemented: <status> | executed: <status>
- [ ] L=3 k=[300,100,50]   — implemented: yes (config_kmh in params_df_gpu_*.json) | executed: <status>
- [ ] L=4 k=[300,100,50,25]— implemented: <status> | executed: <status>

## 6.3 RNHAL stage contribution
- [ ] Full RNHAL             — implemented: yes | executed: <status>
- [ ] No representation module — implemented: <status> | executed: <status>
- [ ] No hierarchical module   — implemented: <status> | executed: <status>
```

4. End with a one-line summary of what to run next on the lab machine, per
   `.claude/skills/running-experiments/SKILL.md`.
