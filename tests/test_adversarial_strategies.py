"""Fast CPU tests for the per-sample AdversarialBIM / AdversarialDeepFool ports."""

import logging

import numpy as np
import pytest
import torch
from torch import nn

from dalmax.query_strategies.adversarial_bim import AdversarialBIM
from dalmax.query_strategies.adversarial_deepfool import AdversarialDeepFool


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
    strat = cls(_Dataset(xs_full, ids), _Net(clf), logging.getLogger("t"), **kw)
    return strat, ids


def _one_dim_strategy(cls, **kw):
    clf = _Clf(1, 2)
    with torch.no_grad():
        clf.fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        clf.fc.bias.zero_()
    return cls(None, _Net(clf), logging.getLogger("t"), **kw)


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


def test_bim_max_iter_caps_loop():
    strat = _one_dim_strategy(AdversarialBIM, eps=0.25, max_iter=2)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.tensor([0.9])) == pytest.approx(0.25)


def test_deepfool_distance_hand_computed():
    # Minimal step to the boundary is 0.9 -> distance ~0.81.
    strat = _one_dim_strategy(AdversarialDeepFool, max_iter=5)
    strat.net.clf.eval()
    assert strat.cal_dis(torch.tensor([0.9])) == pytest.approx(0.81, abs=1e-3)
