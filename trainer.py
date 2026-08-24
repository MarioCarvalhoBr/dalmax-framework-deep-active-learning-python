"""Thin CLI shim: kept at the repo root so every existing invocation
(`python demo.py ...`, `run_pipe_gpu_0.sh`/`run_pipe_gpu_1.sh`,
`Makefile`'s `smoke` target, `tests/test_golden_run.py`) keeps working
unchanged. All actual argparse/config/run/report logic now lives in
`dalmax.cli` (`.specs/architecture/target-architecture.md` §10,
`.claude/rules/code-quality.md` "single responsibility" — the old 289-line
`main()` used to mix CLI parsing, the training loop, plotting, and
persistence in this one file).
"""

from dalmax.cli import main

if __name__ == "__main__":
    main()
