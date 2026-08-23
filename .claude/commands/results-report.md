---
description: Run the utils/report/ pipeline for a results directory and summarize metrics (mean F1 across seeds etc.).
argument-hint: <results_dir>
---

Generate a results report for `$ARGUMENTS` (a results directory, e.g.
`results/dalmax1/daninhas_full/`). Follow
`.claude/skills/results-reporting/SKILL.md` for the full pipeline description;
this command is the short operational version.

Steps:

1. Validate `$ARGUMENTS` is a real directory under `results/` and was not
   passed empty — if empty, ask which results directory to summarize or default
   to the most recently modified subdirectory of `results/`.
2. Run the `utils/report/` scripts in order, matching each script's own
   `Example usage` header comment for the exact flags:
   - `poetry run python utils/report/2_report_build_chunk_results.py --input_dir $ARGUMENTS --pattern "SEED*"`
     — builds per-seed CSV tables and plots from `results.json` files.
   - `poetry run python utils/report/3_cm_build_average.py --input_dir $ARGUMENTS --pattern "SEED*"`
     — averages confusion matrices across seeds.
   - `poetry run python utils/report/4_report_build_average_results.py --input_dir $ARGUMENTS --pattern "SEED*"`
     — builds the final averaged metrics table/plots across seeds.
   - Optionally, `poetry run python utils/report/build_method_metrics.py --method
     <StrategyName> --round <N> --nq <n_query>` for a single method/round/query
     summary (mean of `all_f1_score` etc. across seeds).
   - `1_cm_extract_from_pdf.py` is only needed if confusion matrices must be
     re-extracted from a PDF rather than read from `results.json` directly —
     skip it unless the JSON-based pipeline is insufficient.
3. Summarize the output: mean (and std, if computed by the script) of accuracy,
   precision, recall, macro F1 across seeds, per strategy and per `n_query`
   configuration found under `$ARGUMENTS`.
4. Never write into `$ARGUMENTS` itself if it lives under `results/` in a way
   that would overwrite prior run artifacts — write derived report outputs to
   wherever the script defaults to (check each script's `argparse` defaults) and
   flag if that default would violate `.claude/rules/data-safety.md`
   (results/ is append-only).

Report back: the command(s) actually run, and the summarized metrics table.
