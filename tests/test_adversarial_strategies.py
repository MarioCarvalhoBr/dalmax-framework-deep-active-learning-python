"""Fast CPU tests for the per-sample AdversarialBIM / AdversarialDeepFool ports,
including the seeded tie-break of the ranking (KI-45, ADR 0014)."""

import logging

import numpy as np
import pytest
import torch
from torch import nn

from dalmax.config.loader import load_experiment_config
from dalmax.query_strategies.adversarial_base import (
    TIE_BREAK_SEED_TAG,
    PerSampleAdversarialStrategy,
    rank_by_distance,
)
from dalmax.query_strategies.adversarial_bim import AdversarialBIM
from dalmax.query_strategies.adversarial_deepfool import AdversarialDeepFool
from dalmax.query_strategies.registry import build_strategy
from dalmax.seeding import derive_seed


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class _Clf(nn.Module):
    """Linear classifier returning `(logits, embedding)` like the project models."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        e = self.drop(x.flatten(1))
        return self.fc(e), e


class _Net:
    def __init__(self, clf: nn.Module) -> None:
        self.clf = clf
        self.device = torch.device("cpu")


class _Data:
    def __init__(self, xs: torch.Tensor) -> None:
        self.xs = xs

    def __getitem__(self, i):
        return self.xs[i], 0, i

    def __len__(self):
        return len(self.xs)


class _Dataset:
    def __init__(self, xs: torch.Tensor, unlabeled_ids: np.ndarray) -> None:
        self.xs, self.ids = xs, unlabeled_ids

    def get_unlabeled_data(self):
        return self.ids, _Data(self.xs[self.ids])


def _tiny_setup(cls, **kw):
    torch.manual_seed(0)
    clf = _Clf(3 * 8 * 8, 3)
    ids = np.array([0, 2, 3, 5, 7, 9])
    xs_full = torch.randn(10, 3, 8, 8)
    strat = cls(_Dataset(xs_full, ids), _Net(clf), logging.getLogger("t"), rng=_rng(), **kw)
    return strat, ids


def _one_dim_strategy(cls, **kw):
    clf = _Clf(1, 2)
    with torch.no_grad():
        clf.fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        clf.fc.bias.zero_()
    return cls(None, _Net(clf), logging.getLogger("t"), rng=_rng(), **kw)


@pytest.mark.parametrize("cls,kw", [(AdversarialBIM, {"eps": 0.5}), (AdversarialDeepFool, {})])
def test_query_returns_unique_unlabeled_ids(cls, kw):
    strat, ids = _tiny_setup(cls, **kw)
    sel = strat.query(2)
    assert len(sel) == 2 and len(set(sel.tolist())) == 2
    assert set(sel.tolist()) <= set(ids.tolist())


@pytest.mark.parametrize("cls,kw", [(AdversarialBIM, {"eps": 0.5}), (AdversarialDeepFool, {})])
def test_query_preserves_device_and_mode(cls, kw):
    for training in (True, False):
        strat, _ = _tiny_setup(cls, **kw)
        strat.net.clf.train(training)
        strat.net.clf.drop.train(True)
        strat.query(2)
        assert strat.net.clf.training is training
        assert strat.net.clf.drop.training is True
        assert next(strat.net.clf.parameters()).device.type == "cpu"


def test_bim_distance_hand_computed():
    # logits = (x, -x), x=0.9 -> class 0; each step moves eta by -0.25;
    # after 4 steps x+eta = -0.1 < 0 flips, eta = -1.0 -> distance 1.0.
    strat = _one_dim_strategy(AdversarialBIM, eps=0.25)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.tensor([0.9])) == pytest.approx(1.0)


def test_bim_max_iter_cap_returns_inf():
    # The cap (2 steps, eta = -0.5, x+eta = 0.4 > 0) is reached before the
    # prediction flips, so the sample did not reach the boundary: distance inf
    # (before KI-39 this returned the partial 0.5**2 = 0.25 -- not comparable).
    strat = _one_dim_strategy(AdversarialBIM, eps=0.25, max_iter=2)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.tensor([0.9])) == float("inf")


def test_bim_default_cap_is_50():
    strat = _one_dim_strategy(AdversarialBIM)
    assert strat.max_iter == 50


class _SaturatedClf(nn.Module):
    """Logits independent of the input value: the input gradient is exactly 0."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.copy_(torch.tensor([50.0, 0.0, 0.0]))

    def forward(self, x):
        e = x.flatten(1) * 0.0
        return self.fc(e), e


def _saturated_strategy(cls, n_pool: int = 4, **kw):
    xs = torch.randn(n_pool, 4)
    ids = np.arange(n_pool)
    return cls(_Dataset(xs, ids), _Net(_SaturatedClf()), logging.getLogger("t"), rng=_rng(), **kw), ids


def test_bim_zero_gradient_returns_inf_without_hanging():
    strat, _ = _saturated_strategy(AdversarialBIM)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.randn(4)) == float("inf")


def test_deepfool_zero_gradient_returns_inf():
    strat, _ = _saturated_strategy(AdversarialDeepFool)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.randn(4)) == float("inf")


@pytest.mark.parametrize("cls", [AdversarialBIM, AdversarialDeepFool])
def test_query_all_inf_still_returns_n_unique_ids(cls):
    strat, ids = _saturated_strategy(cls, n_pool=5)
    sel = strat.query(3)
    assert len(sel) == 3 and len(set(sel.tolist())) == 3
    assert set(sel.tolist()) <= set(ids.tolist())


def test_query_ranks_inf_after_finite_and_fills_from_inf():
    # Two 1-D samples: x=0.9 flips within the cap; x=100 needs > max_iter steps (inf).
    clf = _Clf(1, 2)
    with torch.no_grad():
        clf.fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        clf.fc.bias.zero_()
    xs = torch.tensor([[100.0], [0.9], [200.0]])
    ids = np.array([0, 1, 2])
    strat = AdversarialBIM(_Dataset(xs, ids), _Net(clf), logging.getLogger("t"), eps=0.25, max_iter=10, rng=_rng())
    assert strat.query(1).tolist() == [1]
    sel = strat.query(3)
    assert sel[0] == 1 and sorted(sel.tolist()) == [0, 1, 2]


def test_deepfool_distance_hand_computed():
    # Minimal step to the boundary is 0.9 -> distance ~0.81.
    strat = _one_dim_strategy(AdversarialDeepFool, max_iter=5)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.tensor([0.9])) == pytest.approx(0.81, abs=1e-3)


# --- tie-break of the ranking (KI-45) ------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_rank_without_ties_equals_the_stable_argsort(seed: int) -> None:
    dis = np.random.default_rng(100 + seed).random(500)
    dis[::7] = np.inf  # several inf (they tie with each other, and rank last)
    order = rank_by_distance(dis, _rng(seed))
    finite = np.isfinite(dis)
    expected = dis.argsort(kind="stable")
    n_finite = int(finite.sum())
    assert order[:n_finite].tolist() == expected[:n_finite].tolist()
    assert set(order[n_finite:].tolist()) == set(np.flatnonzero(~finite).tolist())


def test_rank_puts_inf_last_even_with_ties() -> None:
    dis = np.array([np.inf, 0.5, np.inf, 0.25, 0.5, 0.25, np.inf])
    order = rank_by_distance(dis, _rng(3))
    assert sorted(order[:2].tolist()) == [3, 5] and sorted(order[2:4].tolist()) == [1, 4]
    assert sorted(order[4:].tolist()) == [0, 2, 6]
    assert sorted(order.tolist()) == list(range(len(dis)))


def _class_ordered_bim_like_distances(n_per_class: int = 400, n_classes: int = 5) -> np.ndarray:
    """A pool stored class by class whose distances take a few discrete values
    (BIM: every step moves each element by exactly eps), so most samples tie."""
    rng = np.random.default_rng(7)
    steps = rng.integers(1, 4, size=n_per_class * n_classes)  # 1, 2 or 3 BIM steps
    dis = (0.05 ** 2) * steps.astype(float) ** 2
    dis[rng.random(dis.shape) < 0.05] = np.inf  # a few never flip (KI-39)
    return dis


def test_ties_are_not_broken_in_pool_order() -> None:
    dis = _class_ordered_bim_like_distances()
    n = 100
    tied = np.flatnonzero(dis == dis.min())
    assert len(tied) > 3 * n  # the n picks all come from the tied minimum
    picked = rank_by_distance(dis, _rng(1))[:n]
    assert set(picked.tolist()) <= set(tied.tolist())
    # The old stable argsort took exactly the n lowest tied indices (first class only).
    old = dis.argsort(kind="stable")[:n]
    assert old.tolist() == tied[:n].tolist() and old.max() < 400
    assert set(picked.tolist()) != set(old.tolist())
    # Uniform over the tied set: every class block is represented.
    assert len({int(i) // 400 for i in picked}) == 5


def test_tie_break_is_uniform_over_the_tied_samples() -> None:
    dis = np.zeros(50)
    counts = np.zeros(50)
    rng = _rng(11)
    for _ in range(2000):
        counts[rank_by_distance(dis, rng)[:10]] += 1
    freq = counts / 2000
    assert np.allclose(freq, 10 / 50, atol=0.05)  # each sample picked ~20% of the time


def test_tie_break_is_reproducible_and_advances_across_rounds() -> None:
    dis = _class_ordered_bim_like_distances()
    first, second = _rng(5), _rng(5)
    round1_a, round1_b = rank_by_distance(dis, first), rank_by_distance(dis, second)
    assert round1_a.tolist() == round1_b.tolist()  # same seed, same order
    round2 = rank_by_distance(dis, first)
    assert round2.tolist() != round1_a.tolist()  # the generator advanced
    assert rank_by_distance(dis, _rng(6)).tolist() != round1_a.tolist()


class _ConstantDistance(PerSampleAdversarialStrategy):
    """Every sample at the same distance: the whole ranking is a tie."""

    def cal_dis(self, x: torch.Tensor) -> float:
        return 0.01


def test_strategy_query_uses_the_injected_generator() -> None:
    torch.manual_seed(0)
    ids = np.arange(0, 400, 2)
    xs = torch.randn(400, 4)

    def strategy(seed: int) -> _ConstantDistance:
        return _ConstantDistance(_Dataset(xs, ids), _Net(nn.Linear(4, 2)), logging.getLogger("t"), rng=_rng(seed))

    a, b = strategy(9), strategy(9)
    first = a.query(20)
    assert first.tolist() == b.query(20).tolist()
    assert sorted(first.tolist()) != ids[:20].tolist()
    assert set(first.tolist()) <= set(ids.tolist()) and len(set(first.tolist())) == 20
    assert a.query(20).tolist() != first.tolist()  # next round: new draw


def test_adversarial_strategies_require_a_generator() -> None:
    with pytest.raises(TypeError):
        AdversarialBIM(None, _Net(_Clf(1, 2)), logging.getLogger("t"))  # no rng: must not build


@pytest.mark.parametrize("name", ["AdversarialBIM", "AdversarialDeepFool"])
def test_registry_injects_a_seed_derived_tie_break_generator(name: str) -> None:
    def build(seed: int):
        config = load_experiment_config(
            "files_config/params_micro.json", "DANINHAS", strategy_name=name, seed=seed,
            n_init_labeled=10, n_query=5, n_round=1, dir_results="results/unused/", device="cpu",
        )
        return build_strategy(name, None, _Net(_Clf(1, 2)), config, logger=None, rng=_rng(123))

    expected = np.random.default_rng(derive_seed(1, TIE_BREAK_SEED_TAG)).permutation(100)
    assert TIE_BREAK_SEED_TAG == "adversarial_ties"
    assert build(1).rng.permutation(100).tolist() == expected.tolist()
    assert build(2).rng.permutation(100).tolist() != expected.tolist()
