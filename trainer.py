"""Thin CLI shim: kept at the repo root so every existing invocation
(`python trainer.py ...`, `run_pipe_gpu_0.sh`/`run_pipe_gpu_1.sh`,
`Makefile`'s `smoke` target, `tests/test_golden_run.py`) keeps working
unchanged. All actual argparse/config/run/report logic lives in `dalmax.cli`
(`.specs/architecture/target-architecture.md` §10, `.claude/rules/code-quality.md`
"single responsibility" — the old 289-line `main()` used to mix CLI parsing,
the training loop, plotting, and persistence in this one file).

Renamed from the historical `demo.py` to `trainer.py` on 2026-08-23 (`git mv`, see ADR
0006) — a pure rename, no behavior change: the historical name `demo.py` no
longer exists anywhere in this repository. This file trains a model and
writes `saved_model.pth`; the companion inference tools that consume that
checkpoint (`predict.py`, `loader.py`, `gui.py`, `dalmax/inference/`) live
alongside it at the repo root.
"""

from dalmax.cli import main

if __name__ == "__main__":
    main()
