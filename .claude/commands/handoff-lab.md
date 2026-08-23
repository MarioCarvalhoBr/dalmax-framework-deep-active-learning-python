---
description: Pre-flight for the lab machine - run experiment-auditor, ensure requirements.txt export is fresh, ensure no uncommitted changes, and print the exact commands to run on each GPU.
---

Prepare a clean handoff to the lab machine (2x NVIDIA GPUs, 10 GB each).

Steps:

1. Run the `experiment-auditor` agent (`.claude/agents/experiment-auditor.md`)
   against the run about to be handed off (use `$ARGUMENTS` to tell it which
   strategy/params file/results root is intended, e.g. "SSRAEKmeansHCSampling,
   params_df_gpu_0.json and params_df_gpu_1.json, results/dalmax1 and
   results/dalmax2"). Do not proceed past a NO-GO verdict without the user's
   explicit override.
2. Ensure `requirements.txt` is a fresh export from Poetry (per Deliverable A):
   run `make export-reqs` (`poetry export -f requirements.txt --output
   requirements.txt --without-hashes`) and check `git diff requirements.txt`
   is empty after — if not, the lockfile and the exported file had drifted.
3. Run `git status` — there must be no uncommitted changes to anything under
   `core/`, `utils/`, `demo.py`, or `params_df_gpu_*.json` that the lab machine
   needs. List any uncommitted files and ask the user to commit
   (`.claude/rules/git-workflow.md` — conventional commits, no push without
   confirmation) before proceeding.
4. Print the exact commands to run on each GPU, based on the current
   `run_pipe_gpu_0.sh` / `run_pipe_gpu_1.sh`:
   ```
   # GPU 0 (results/dalmax1/)
   git pull
   bash run_pipe_gpu_0.sh
   # GPU 1 (results/dalmax2/)
   bash run_pipe_gpu_1.sh
   ```
   Adjust the printed commands if `$ARGUMENTS` specifies different scripts,
   queries, or seeds than the current `QUERIES=(10 50 100)` / `SEEDS=(1 2 3)`
   / `n_round 8` defaults.
5. Remind that each script emails a notification via
   `ExperimentNotifier/main.py` on completion (see
   `.claude/skills/running-experiments/SKILL.md`) — confirm
   `ExperimentNotifier/.env` is configured on the lab machine (it is
   gitignored, so it must be set up there independently, not pulled from git).

Report: GO/NO-GO from the auditor, requirements.txt freshness, uncommitted
files (if any), and the final commands to paste on the lab machine.
