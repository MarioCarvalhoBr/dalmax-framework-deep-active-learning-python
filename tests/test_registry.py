"""`dalmax.cli`'s `--strategy_name` `choices=[...]` must exactly match
`dalmax.query_strategies.registry.STRATEGY_REGISTRY`'s keys.

`dalmax.cli` (historically `demo.py` itself, before Phase 2's split; the repo
root shim is now `trainer.py`, renamed from the historical `demo.py` 2026-08-23 — see
`.specs/architecture/refactor-plan.md` Phase 2 and ADR 0006) is the single
source of truth for which query strategies the CLI accepts. We parse its
`--strategy_name` `choices=[...]` out of the source via `ast` (never
importing `dalmax.cli`, since importing it has side effects: it eagerly calls
`dalmax.logging_utils.get_logger()`, which creates a log file handler and a
`results/logs/` directory as an import-time side effect — same reasoning that
previously applied to the historical `demo.py` before it became this thin
shim).

Historically this file also tested the legacy `utils.orchestrator.get_strategy`
`if/elif` registry directly, against `core.query_strategies.strategy.Strategy`.
Both `utils/orchestrator.py` and `core/query_strategies/` were deleted in
refactor Phase 4 (`.specs/architecture/refactor-plan.md`) once the `dalmax`
registries fully replaced them (see `.specs/architecture/current-state.md`
§0/§11) — `build_strategy`'s own behavior (presets, `ConfigError`s, seed
derivation, constructing every legacy class) is covered by
`tests/test_strategy_registry.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

from dalmax.query_strategies.registry import STRATEGY_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PY = REPO_ROOT / "dalmax" / "cli.py"


def _extract_strategy_choices() -> list[str]:
    """Statically parse the `choices=[...]` list of the `--strategy_name`
    `add_argument` call in `dalmax/cli.py`, without executing it."""
    tree = ast.parse(CLI_PY.read_text(), filename=str(CLI_PY))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not any(
            isinstance(arg, ast.Constant) and arg.value == "--strategy_name"
            for arg in node.args
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                choices = ast.literal_eval(keyword.value)
                return list(choices)

    raise AssertionError(
        "Could not find `--strategy_name` add_argument(..., choices=[...]) in dalmax/cli.py"
    )


STRATEGY_CHOICES = _extract_strategy_choices()


def test_strategy_choices_found_at_least_one():
    assert len(STRATEGY_CHOICES) > 0


def test_strategy_choices_match_new_registry():
    assert set(STRATEGY_CHOICES) == set(STRATEGY_REGISTRY)
