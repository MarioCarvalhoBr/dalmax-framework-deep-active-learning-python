"""Every `--strategy_name` choice in `dalmax.cli` must resolve to a
`core.query_strategies.strategy.Strategy` subclass, through either the
legacy `utils.orchestrator.get_strategy` (for the 12 legacy names it has
always supported) or the new `dalmax.query_strategies.registry.build_strategy`
(for all 17 CLI choices, including the four `*Kmeans*Sampling` presets and
the new generic `RepresentationStrategy`).

`dalmax.cli` (formerly `demo.py`, see `.specs/architecture/refactor-plan.md`
Phase 2) is the single source of truth for which query strategies the CLI
accepts. We parse its `--strategy_name` `choices=[...]` out of the source via
`ast` (never importing `dalmax.cli`, since importing it has side effects: it
eagerly calls `utils.LOGGER.get_logger()`, which creates a log file handler
and a `results/logs/` directory as an import-time side effect — same
reasoning that previously applied to `demo.py` before it became this thin
shim).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from core.query_strategies.strategy import Strategy
from dalmax.query_strategies.registry import (
    GENERIC_REPRESENTATION_NAME,
    STRATEGY_REGISTRY,
    build_strategy,
)
from utils.orchestrator import get_strategy

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PY = REPO_ROOT / "dalmax" / "cli.py"

# `utils.orchestrator.get_strategy` never learned about the new generic
# `RepresentationStrategy` CLI choice (it is a `dalmax`-only concept, built
# via `dalmax.query_strategies.registry.build_strategy` with a config/rng
# `utils.orchestrator.get_strategy(name)` does not accept) — it is covered
# separately by `tests/test_strategy_registry.py`.
LEGACY_UNSUPPORTED_NAMES = {GENERIC_REPRESENTATION_NAME}


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


@pytest.mark.parametrize(
    "strategy_name",
    [name for name in STRATEGY_CHOICES if name not in LEGACY_UNSUPPORTED_NAMES],
)
def test_strategy_name_resolves_to_strategy_subclass_via_legacy_orchestrator(strategy_name):
    strategy_cls = get_strategy(strategy_name)
    assert isinstance(strategy_cls, type), (
        f"get_strategy({strategy_name!r}) did not return a class: {strategy_cls!r}"
    )
    assert issubclass(strategy_cls, Strategy), (
        f"get_strategy({strategy_name!r}) returned {strategy_cls!r}, "
        f"which is not a subclass of core.query_strategies.strategy.Strategy"
    )


def test_get_strategy_rejects_unknown_name():
    with pytest.raises(NotImplementedError):
        get_strategy("__not_a_real_strategy__")


def test_build_strategy_rejects_unknown_name():
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError):
        build_strategy("__not_a_real_strategy__", None, None, None, None, rng)
