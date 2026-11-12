"""The WAAL training-budget control `WAALRandomQuery` (ADR 0015 amendment of 2026-11-12).

WAAL's training unchanged, query = a uniform random draw from the unlabeled pool. Fast and
CPU-only: the tiny staged net and stub pool of `tests/test_deepalplus_strategies.py`. The
end-to-end comparison of round 0 through `tools/trainer.py` on the micro dataset lives in
`tests/test_deepalplus_e2e.py` (dataset + slow).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from dalmax.campaign import effective_signature
from dalmax.config.loader import load_experiment_config
from dalmax.config.schema import STRATEGY_ARGS_KEYS, STRATEGY_ARGS_READ_BY, ConfigError, WAALArgs
from dalmax.models.waal import WAALTrainer
from dalmax.query_strategies.registry import build_strategy
from dalmax.query_strategies.waal import (
    UNIFORM_RANDOM_QUERY_RULE,
    WAAL,
    WAAL_RANDOM_QUERY_SEED_TAG,
    WAALRandomQuery,
    WAALTrainedStrategy,
)
from dalmax.seeding import derive_seed
from tests.test_deepalplus_strategies import (
    _assert_same_state,
    _PoolDataset,
    _state,
    _tiny_data,
    _tiny_net,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_MICRO = REPO_ROOT / "files_config/params_micro.json"
PAPER1 = REPO_ROOT / "files_config/campaign/params_paper1.json"
PAPER1_VAAL = REPO_ROOT / "files_config/campaign/params_paper1_vaal.json"
LOGGER = logging.getLogger("test_waal_random_query")


def _config(strategy: str, params: Path = PARAMS_MICRO, seed: int = 1):
    return load_experiment_config(
        params, "DANINHAS", strategy_name=strategy, seed=seed, n_init_labeled=2, n_query=3, n_round=1,
        dir_results="results/unused/", device="cpu",
    )


def _pool(n: int = 14, n_labeled: int = 4) -> _PoolDataset:
    data = _tiny_data(n, seed=7)
    return _PoolDataset(n, n_labeled=n_labeled, xs=data.xs.numpy(), ys=data.Y.numpy())


def _round0(name: str, seed: int = 1):
    """Build `name` through the registry (as the runner does) on a fresh tiny net, train it once
    (round 0) and query once; return the strategy, the round-0 model state and test predictions."""
    torch.manual_seed(seed)  # the run's global torch stream (`seed_everything`)
    net = _tiny_net()
    strategy = build_strategy(name, _pool(), net, _config(name, seed=seed), LOGGER, np.random.default_rng(0))
    strategy.train()
    state = _state(net.clf)
    probs = net.predict_prob(_tiny_data(6, seed=9))
    picked = strategy.query(3)
    return strategy, state, probs, picked


# --- what is shared with WAAL ------------------------------------------------------------------------


def test_the_control_reads_the_waal_block_and_has_none_of_its_own() -> None:
    assert STRATEGY_ARGS_READ_BY["WAALRandomQuery"] == STRATEGY_ARGS_KEYS["WAAL"] == "waal"
    assert "WAALRandomQuery" not in STRATEGY_ARGS_KEYS  # a params JSON has no "WAALRandomQuery" block


def test_a_waal_random_query_block_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(PARAMS_MICRO.read_text())
    raw["DANINHAS"]["strategy_args"] = {"WAALRandomQuery": {"alpha": 0.1}}
    params = tmp_path / "params.json"
    params.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="unknown strategy name"):
        _config("WAALRandomQuery", params)


def test_build_strategy_gives_the_control_waals_trainer_and_its_own_query_generator() -> None:
    waal = build_strategy("WAAL", _pool(), _tiny_net(), _config("WAAL"), LOGGER, None)
    control = build_strategy("WAALRandomQuery", _pool(), _tiny_net(), _config("WAALRandomQuery"), LOGGER, None)
    assert type(control) is WAALRandomQuery and isinstance(control, WAALTrainedStrategy)
    assert not isinstance(control, WAAL) and not hasattr(control, "selection")
    assert isinstance(control.trainer, WAALTrainer) and control.trainer.args == waal.trainer.args == WAALArgs()
    assert control.trainer.rng.bit_generator.state == waal.trainer.rng.bit_generator.state
    expected = np.random.default_rng(derive_seed(1, WAAL_RANDOM_QUERY_SEED_TAG)).permutation(50)
    assert control.rng.permutation(50).tolist() == expected.tolist()
    assert WAAL_RANDOM_QUERY_SEED_TAG not in ("waal", "waal_ties")


def test_same_seed_gives_the_same_round0_model_metrics_and_trainer_state() -> None:
    waal, waal_state, waal_probs, waal_pick = _round0("WAAL")
    control, control_state, control_probs, control_pick = _round0("WAALRandomQuery")
    _assert_same_state(waal_state, control_state)  # identical round-0 model
    assert torch.equal(waal_probs, control_probs)  # hence identical round-0 metrics
    for key in ("waal_n_steps", "waal_wasserstein_last_epoch", "waal_critic_labeled_mean",
                "waal_critic_unlabeled_mean", "waal_critic_init_corr"):
        assert waal.diagnostics()[key] == control.diagnostics()[key], key
    # The query-time critic evaluation advances the trainer generator equally, so round 1 trains
    # from the same generator state: the runs differ only through the labeled set.
    assert waal.trainer.rng.bit_generator.state == control.trainer.rng.bit_generator.state
    assert torch.equal(torch.as_tensor(waal.net.clf.classifier.weight),
                       torch.as_tensor(control.net.clf.classifier.weight))
    assert len(waal_pick) == len(control_pick) == 3


# --- what differs: the query -------------------------------------------------------------------------


class _ScoreTrainer:
    """Stub trainer: fixed score components, records the calls."""

    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.calls: list[int] = []

    def score_components(self, net, fit, data):
        self.calls.append(len(data))
        values = torch.arange(len(data), dtype=torch.float64) * self.scale
        return values, -values, values


def _control(seed: int, dataset: _PoolDataset, scale: float = 1.0) -> WAALRandomQuery:
    strategy = WAALRandomQuery(dataset, None, LOGGER, trainer=_ScoreTrainer(scale),
                               rng=np.random.default_rng(seed))
    strategy.fit_result = object()  # type: ignore[assignment]
    return strategy


def test_the_control_query_is_a_seeded_uniform_draw_independent_of_the_scores() -> None:
    dataset = _pool(40, n_labeled=5)
    unlabeled = np.arange(40)[~dataset.labeled_idxs]
    picks = [_control(seed, _pool(40, 5)).query(6) for seed in range(6)]
    for seed, pick in enumerate(picks):
        expected = np.random.default_rng(seed).choice(unlabeled, size=6, replace=False)
        assert pick.tolist() == expected.tolist()
        assert len(set(pick.tolist())) == 6 and not set(pick.tolist()) & set(range(5))
    assert len({tuple(p) for p in picks}) > 1  # the seed matters
    # Other scores, same generator: the same pick (the scores never reach the selection).
    assert _control(0, _pool(40, 5), scale=-3.0).query(6).tolist() == picks[0].tolist()


def test_the_control_query_reports_a_random_rule_and_the_critic_statistics() -> None:
    strategy = _control(0, _pool(40, 5))
    strategy.query(6)
    assert strategy.trainer.calls == [35, 5]  # pool first, then the labeled set, as WAAL
    diag = strategy.diagnostics()
    assert diag["query_rule"] == [UNIFORM_RANDOM_QUERY_RULE] == ["uniform_random"]
    for key in ("waal_critic_labeled_mean", "waal_critic_labeled_std", "waal_critic_unlabeled_mean",
                "waal_critic_unlabeled_std", "waal_critic_init_corr", "pool_order_fraction"):
        assert len(diag[key]) == 1, key
    # The WAAL score terms are not part of the control's query, so they are not reported.
    assert "waal_uncertainty_std" not in diag and "waal_critic_term_std" not in diag


def test_the_control_fails_fast() -> None:
    strategy = WAALRandomQuery(_pool(10, 2), None, LOGGER, trainer=_ScoreTrainer(1.0),
                               rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError, match="before train"):
        strategy.query(2)
    strategy.fit_result = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="cannot query 9"):
        strategy.query(9)


# --- campaign signature ------------------------------------------------------------------------------


def _signature(params: Path, strategy: str) -> str:
    return effective_signature(str(params), "DANINHAS", strategy, 100, 100, 8)


def test_the_control_signature_reads_the_waal_block_only(tmp_path: Path) -> None:
    # Another strategy's block does not matter (params_paper1_vaal.json differs only by the VAAL block) ...
    assert _signature(PAPER1, "WAALRandomQuery") == _signature(PAPER1_VAAL, "WAALRandomQuery")
    # ... the WAAL block does, and the control is never the same computation as WAAL.
    raw = json.loads(PAPER1.read_text())
    raw["DANINHAS"]["strategy_args"] = {"WAAL": {"alpha": 0.5}}
    changed = tmp_path / "params.json"
    changed.write_text(json.dumps(raw))
    assert _signature(changed, "WAALRandomQuery") != _signature(PAPER1, "WAALRandomQuery")
    assert _signature(PAPER1, "WAALRandomQuery") != _signature(PAPER1, "WAAL")
