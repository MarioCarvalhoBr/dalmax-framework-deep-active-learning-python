"""Every `--strategy_name` choice in demo.py must resolve through the registry.

The list of valid strategy names is the single source of truth for which
query strategies `demo.py`'s CLI accepts. We parse it out of the source via
`ast` (never importing/executing `demo.py`, since importing it has side
effects: it eagerly calls `utils.LOGGER.get_logger()`, which creates a log
file handler and a `results/logs/` directory as an import-time side effect).

For each name, `utils.orchestrator.get_strategy(name)` must return a class
that is a subclass of `core.query_strategies.strategy.Strategy`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.query_strategies.strategy import Strategy
from utils.orchestrator import get_strategy

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PY = REPO_ROOT / "demo.py"


def _extract_strategy_choices() -> list[str]:
    """Statically parse the `choices=[...]` list of the `--strategy_name`
    `add_argument` call in demo.py, without executing demo.py."""
    tree = ast.parse(DEMO_PY.read_text(), filename=str(DEMO_PY))

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
        "Could not find `--strategy_name` add_argument(..., choices=[...]) in demo.py"
    )


STRATEGY_CHOICES = _extract_strategy_choices()


def test_strategy_choices_found_at_least_one():
    assert len(STRATEGY_CHOICES) > 0


@pytest.mark.parametrize("strategy_name", STRATEGY_CHOICES)
def test_strategy_name_resolves_to_strategy_subclass(strategy_name):
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
