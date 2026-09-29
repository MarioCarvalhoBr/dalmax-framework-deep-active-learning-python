---
description: Pre-flight for the lab machine - run experiment-auditor, ensure the lab-machine Poetry environment is in sync with poetry.lock, ensure no uncommitted changes, and print the exact commands to run on each GPU.
---

Prepare a clean handoff to a GPU machine (lab machine or Colab; hardware-agnostic, no GPU model is
assumed anywhere). See [`LAB_RUNBOOK.md`](../../LAB_RUNBOOK.md) for the full step-by-step operator
guide this pre-flight feeds into (one-time setup, sanity checks, campaign launch/monitoring, results
collection) -- this command only produces the GO/NO-GO verdict and the commands to paste, not the
guide itself.

Steps:

1. Run the `experiment-auditor` agent (`.claude/agents/experiment-auditor.md`) against the run about to
   be handed off (use `$ARGUMENTS` to tell it which campaign part is intended, e.g. `PART=all`). Do not
   proceed past a NO-GO verdict without the user's explicit override.
2. Ensure the machine's Poetry environment matches `poetry.lock`: run `poetry install` after the
   `git pull` in step 4 below (or confirm it was already run for this commit) -- DalMax is Poetry-only.
3. Run `git status` -- there must be no uncommitted changes to anything under `dalmax/`, `tools/`,
   `scripts/` or `files_config/` that the machine needs. List any uncommitted files and ask the user to
   commit (`.claude/rules/git-workflow.md` -- conventional commits, no push without confirmation).
4. Print the exact commands to run:
   ```
   git pull
   make lab-setup                 # poetry install + torch/CUDA visibility check
   make lab-check GPU=0           # one short real-data sanity run before committing to the campaign
   make campaign-list             # expect 192 runs / 64 groups
   make campaign-run PART=all     # or one process per GPU: CUDA_VISIBLE_DEVICES=0 ... PART=paper1,upper_bound
                                  #                          CUDA_VISIBLE_DEVICES=1 ... PART=rnhal,texhal
   make campaign-verify && make campaign-report
   ```
   After the first job finishes, check `environment.gpus[0].name` in its `run_metadata.json`.
5. `ExperimentNotifier` (gitignored, lab-local `.env`) is not called by the campaign runner; if an email
   is wanted, wrap the launch command yourself.
6. If this is a relaunch (e.g. after a crash), `make campaign-run` skips every job whose leaf is
   complete (`results.json` is written last), so re-launching the same command is safe; add
   `RUN_ARGS=--no-skip-existing` only to force a full re-run.

Report: GO/NO-GO from the auditor, Poetry environment sync status, uncommitted files (if any), and the
final commands to paste on the machine.
