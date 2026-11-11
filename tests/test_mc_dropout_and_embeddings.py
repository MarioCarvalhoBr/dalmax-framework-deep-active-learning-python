"""KI-41 / KI-42 fixes (ADR 0014): optional dropout layer of the DANINHAS ResNet-50,
MC-dropout inference mode, NaN-free BALD/entropy scores, trained 2048-d embeddings,
the `model_args` params block and its checkpoint round trip.

The ResNet-50 is built from the local torch hub cache of the ImageNet weights
(as in `tests/test_checkpoint.py`), on tiny inputs; nothing is trained except a
two-layer stub.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.config.schema import ConfigError, ModelArgs
from dalmax.models.base import DeepLearning
from dalmax.models.checkpoint import load_checkpoint, save_checkpoint
from dalmax.models.cifar10_cnn import CIFAR10Model
from dalmax.models.daninhas_resnet50 import DaninhasModelResNet50
from dalmax.models.mc_dropout import MCDropoutError, active_dropout_modules, mc_dropout_mode
from dalmax.models.registry import get_network
from dalmax.query_strategies.bayesian_active_learning_disagreement_dropout import bald_scores
from dalmax.query_strategies.entropy_sampling_dropout import EntropySamplingDropout

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PAPER1 = REPO_ROOT / "files_config/campaign/params_paper1.json"
PARAMS_MC_DROPOUT = REPO_ROOT / "files_config/campaign/params_paper1_mc_dropout.json"
PARAMS_PAPER1_MICRO = REPO_ROOT / "files_config/campaign/params_paper1_micro.json"
PARAMS_MC_DROPOUT_MICRO = REPO_ROOT / "files_config/campaign/params_paper1_mc_dropout_micro.json"

# sha256 of "|".join(state_dict keys) of `DaninhasModelResNet50(5)` on `main` before the
# fix (commit 919a896): 322 keys, `embedding_layer.*` included. The dropout layer adds none.
PRE_FIX_STATE_DICT_KEYS_SHA256 = "37b3e89f08df321573bc7d25186ee2678780dcb6f2c22c93d9488681d4cb6dcb"


class _PreFixDaninhasModelResNet50(nn.Module):
    """Verbatim copy (comments dropped) of `DaninhasModelResNet50` before ADR 0014."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        self.embedding_dim = 50
        self.embedding_layer = nn.Linear(num_features, self.embedding_dim)
        self.classifier = nn.Linear(num_features, self.n_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        e1 = self.embedding_layer(features)
        e1 = torch.relu(e1)
        x = self.classifier(features)
        return x, e1


def _fixed_input(n: int = 2, size: int = 32) -> torch.Tensor:
    return torch.randn(n, 3, size, size, generator=torch.Generator().manual_seed(1))


def _build_seeded(factory, seed: int = 0) -> tuple[nn.Module, torch.Tensor]:
    """Build a model right after `torch.manual_seed(seed)`; also return the next RNG
    draws, which prove the construction consumed the same number of random numbers."""
    torch.manual_seed(seed)
    model = factory()
    return model, torch.rand(4)


# --- (a) dropout_p = 0 builds exactly the pre-fix model -------------------------------


@pytest.fixture(scope="module")
def pre_fix_model() -> tuple[nn.Module, torch.Tensor]:
    return _build_seeded(lambda: _PreFixDaninhasModelResNet50(5))


@pytest.mark.parametrize(
    "kwargs", [{}, {"dropout_p": 0.0}, {"dropout_p": 0.5}], ids=["default", "p0", "p05"]
)
def test_construction_matches_the_pre_fix_model(pre_fix_model, kwargs: dict) -> None:
    reference, reference_next_draws = pre_fix_model
    model, next_draws = _build_seeded(lambda: DaninhasModelResNet50(5, **kwargs))

    ref_state, state = reference.state_dict(), model.state_dict()
    assert list(state) == list(ref_state)
    assert hashlib.sha256("|".join(state).encode()).hexdigest() == PRE_FIX_STATE_DICT_KEYS_SHA256
    for key in ref_state:
        assert torch.equal(state[key], ref_state[key]), key
    # Same RNG consumption at construction: the next draws coincide.
    assert torch.equal(next_draws, reference_next_draws)
    assert [n for n, _ in model.named_parameters()] == [n for n, _ in reference.named_parameters()]

    # Same logits in eval mode (dropout is the identity there).
    reference.eval()
    model.eval()
    x = _fixed_input()
    with torch.no_grad():
        ref_logits, _ = reference(x)
        logits, _ = model(x)
    assert torch.equal(logits, ref_logits)


def test_dropout_p_zero_has_no_dropout_module_and_p_half_has_one() -> None:
    assert active_dropout_modules(DaninhasModelResNet50(5)) == []
    modules = active_dropout_modules(DaninhasModelResNet50(5, dropout_p=0.5))
    assert len(modules) == 1 and modules[0].p == 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_dropout_p_out_of_range_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="dropout_p"):
        DaninhasModelResNet50(5, dropout_p=bad)


def test_cifar10_model_keeps_its_builtin_dropout() -> None:
    torch.manual_seed(0)
    model = CIFAR10Model(10)
    assert [m.p for m in active_dropout_modules(model)] == [0.5]
    assert not any("dropout" in key for key in model.state_dict())


# --- (e) embeddings are the 2048-d features that feed the classifier (KI-42) ----------


class _ImageData(torch.utils.data.Dataset):
    """Minimal `(x, y, idx)` dataset with a `.Y` attribute, as `DeepLearning` expects."""

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        self.xs, self.Y = xs, ys

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, i: int):
        return self.xs[i], self.Y[i], i


_TEST_PARAMS = {
    "n_epoch": 1,
    "n_classes": 3,
    "n_drop": 4,
    "train_args": {"batch_size": 4, "num_workers": 0},
    "test_args": {"batch_size": 4, "num_workers": 0},
    "optimizer_args": {"lr": 0.05, "momentum": 0.3},
}


def test_get_embeddings_returns_the_pooled_features_feeding_the_classifier() -> None:
    torch.manual_seed(0)
    model = DaninhasModelResNet50(3, dropout_p=0.5)
    net = DeepLearning(DaninhasModelResNet50, dict(_TEST_PARAMS), torch.device("cpu"))
    net.clf = model
    x = _fixed_input(3)
    data = _ImageData(x, torch.tensor([0, 1, 2]))

    embeddings = net.get_embeddings(data)
    assert model.get_embedding_dim() == 2048 and embeddings.shape == (3, 2048)
    model.eval()
    with torch.no_grad():
        pooled = model.feature_extractor(x).flatten(1)
        logits, _ = model(x)
    torch.testing.assert_close(embeddings, pooled)
    with torch.no_grad():
        torch.testing.assert_close(model.classifier(embeddings), logits)


# --- (b)/(c) MC-dropout mode -------------------------------------------------------------


class _BNDropNet(nn.Module):
    """Tiny classifier with BatchNorm and an optional dropout before the last layer."""

    def __init__(self, n_classes: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = torch.relu(self.bn(self.fc1(x)))
        return self.fc2(self.dropout(e1)), e1


def _trained_stub_net(dropout_p: float) -> tuple[DeepLearning, _ImageData]:
    torch.manual_seed(0)
    xs = torch.randn(12, 8)
    data = _ImageData(xs, torch.tensor([0, 1, 2] * 4))
    net = DeepLearning(
        _BNDropNet, dict(_TEST_PARAMS), torch.device("cpu"), model_kwargs={"dropout_p": dropout_p}
    )
    net.train(data)
    return net, data


def _bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items() if k.startswith("bn.")}


def test_train_builds_the_model_with_the_injected_kwargs() -> None:
    net, _ = _trained_stub_net(0.5)
    assert isinstance(net.clf.dropout, nn.Dropout) and net.clf.dropout.p == 0.5


@pytest.mark.parametrize("method", ["predict_prob_dropout", "predict_prob_dropout_split"])
def test_mc_passes_differ_and_batchnorm_is_untouched(method: str) -> None:
    net, data = _trained_stub_net(0.5)
    net.clf.train()  # any prior mode must be restored afterwards
    before = _bn_buffers(net.clf)
    probs = getattr(net, method)(data, n_drop=4)
    after = _bn_buffers(net.clf)
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} changed during MC prediction"
    assert net.clf.training and net.clf.bn.training and net.clf.dropout.training

    if method == "predict_prob_dropout_split":
        assert probs.shape == (4, len(data), 3)
        assert not torch.allclose(probs[0], probs[1])  # dropout makes the passes differ
        torch.testing.assert_close(probs.sum(2), torch.ones(4, len(data)))
    else:
        assert probs.shape == (len(data), 3)
        torch.testing.assert_close(probs.sum(1), torch.ones(len(data)))


def test_mc_dropout_mode_runs_batchnorm_in_eval_and_dropout_in_train() -> None:
    model = _BNDropNet(3, dropout_p=0.5)
    model.train()
    with mc_dropout_mode(model):
        assert not model.training and not model.bn.training and model.dropout.training
    assert model.training and model.bn.training and model.dropout.training


@pytest.mark.parametrize("method", ["predict_prob_dropout", "predict_prob_dropout_split"])
def test_mc_prediction_without_dropout_fails_fast(method: str) -> None:
    net, data = _trained_stub_net(0.0)
    net.clf.eval()
    with pytest.raises(MCDropoutError, match="KI-41"):
        getattr(net, method)(data, n_drop=4)
    assert not net.clf.training  # modes untouched by the failed call


def test_mc_dropout_mode_rejects_the_default_daninhas_model() -> None:
    with pytest.raises(MCDropoutError, match="KI-41"):
        with mc_dropout_mode(DaninhasModelResNet50(5)):
            pass


@pytest.mark.parametrize("n_drop", [0, -1])
def test_mc_prediction_rejects_a_pass_count_below_one(n_drop: int) -> None:
    net, data = _trained_stub_net(0.5)
    with pytest.raises(ValueError, match="n_drop"):
        net.predict_prob_dropout(data, n_drop=n_drop)


# --- (d) NaN-free scores with saturated probabilities --------------------------------------


def test_bald_scores_are_finite_with_saturated_probabilities() -> None:
    probs = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]],
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.3, 0.3, 0.4]],
        ]
    )
    scores = bald_scores(probs)
    assert torch.isfinite(scores).all()
    # The pre-fix formula gives NaN wherever a probability is exactly 0 ...
    pb = probs.mean(0)
    old = (-probs * torch.log(probs)).sum(2).mean(0) - (-pb * torch.log(pb)).sum(1)
    assert torch.isnan(old[:2]).all()
    # ... and the same value where it is defined.
    torch.testing.assert_close(scores[2], old[2])
    # A sample predicted identically by every pass carries no mutual information.
    assert scores[0].item() == pytest.approx(0.0, abs=1e-7)
    assert scores[1].item() < 0  # disagreement between passes = positive mutual information


class _FixedProbsNet:
    def __init__(self, probs: torch.Tensor) -> None:
        self.params = {"n_drop": 3}
        self._probs = probs

    def predict_prob_dropout(self, data, n_drop: int) -> torch.Tensor:
        assert n_drop == 3
        return self._probs


class _PoolDataset:
    def __init__(self, n: int) -> None:
        self.n = n

    def get_unlabeled_data(self):
        return np.arange(10, 10 + self.n), None


def test_entropy_sampling_dropout_handles_zero_probabilities() -> None:
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.9, 0.1, 0.0]])
    strategy = EntropySamplingDropout(
        _PoolDataset(4), _FixedProbsNet(probs), logging.getLogger("t")
    )
    assert strategy.query(2).tolist() == [12, 11]  # highest entropy first, no NaN involved


# --- params JSON `model_args` --------------------------------------------------------------


def _load(params: Path, strategy: str = "LeastConfidenceDropout"):
    return load_experiment_config(
        params,
        "DANINHAS",
        strategy_name=strategy,
        seed=1,
        n_init_labeled=100,
        n_query=100,
        n_round=8,
        dir_results="results/unused/",
        device="cpu",
    )


def test_model_args_absent_means_unset_and_builds_the_default_model() -> None:
    config = _load(PARAMS_PAPER1)
    assert config.dataset.model_args == ModelArgs() and config.dataset.model_args.as_kwargs() == {}
    assert to_dict(config)["dataset"]["model_args"] == {"dropout_p": None}
    assert get_network(config, "cpu").model_kwargs == {}


@pytest.mark.parametrize(
    "pair", [(PARAMS_PAPER1, PARAMS_MC_DROPOUT), (PARAMS_PAPER1_MICRO, PARAMS_MC_DROPOUT_MICRO)]
)
def test_mc_dropout_params_differ_from_paper1_only_by_the_dropout_layer(
    pair: tuple[Path, Path],
) -> None:
    base, mc = (json.loads(p.read_text()) for p in pair)
    assert set(base) == set(mc) == {"_comment", "DANINHAS"}
    assert "KI-41" in mc["_comment"]
    mc_block = dict(mc["DANINHAS"])
    assert mc_block.pop("model_args") == {"dropout_p": 0.5}
    assert mc_block == base["DANINHAS"]

    config = _load(pair[1])
    assert config.dataset.model_args.dropout_p == 0.5
    assert to_dict(config)["dataset"]["model_args"] == {"dropout_p": 0.5}
    assert get_network(config, "cpu").model_kwargs == {"dropout_p": 0.5}


def _params_with_model_args(tmp_path: Path, model_args) -> Path:
    raw = json.loads(PARAMS_PAPER1.read_text())
    raw["DANINHAS"]["model_args"] = model_args
    path = tmp_path / "params.json"
    path.write_text(json.dumps(raw))
    return path


@pytest.mark.parametrize(
    "model_args,match",
    [
        ({"dropout": 0.5}, "unknown key"),
        ({"dropout_p": 1.0}, r"\[0, 1\)"),
        ({"dropout_p": -0.2}, r"\[0, 1\)"),
        ({"dropout_p": True}, "must be a number"),
        ({"dropout_p": "0.5"}, "must be a number"),
        ([0.5], "JSON object"),
    ],
)
def test_invalid_model_args_fail_fast(tmp_path: Path, model_args, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _load(_params_with_model_args(tmp_path, model_args))


def test_model_args_comment_keys_are_allowed(tmp_path: Path) -> None:
    config = _load(_params_with_model_args(tmp_path, {"_comment": "x", "dropout_p": 0}))
    assert config.dataset.model_args.dropout_p == 0.0


# --- checkpoint round trip -------------------------------------------------------------------


def test_checkpoint_records_and_restores_model_kwargs(tmp_path: Path) -> None:
    clf = DaninhasModelResNet50(3, dropout_p=0.5).eval()
    path = str(tmp_path / "saved_model.pth")
    save_checkpoint(
        path,
        clf,
        model_name="DANINHAS",
        n_classes=3,
        class_names=["a", "b", "c"],
        img_size=32,
        model_kwargs={"dropout_p": 0.5},
    )
    loaded, meta = load_checkpoint(path)
    assert meta["model_kwargs"] == {"dropout_p": 0.5}
    assert [m.p for m in active_dropout_modules(loaded)] == [0.5]
    x = _fixed_input()
    with torch.no_grad():
        assert torch.equal(loaded(x)[0], clf(x)[0])


def test_checkpoint_without_model_kwargs_key_loads_the_default_model(tmp_path: Path) -> None:
    clf = DaninhasModelResNet50(3).eval()
    path = str(tmp_path / "saved_model.pth")
    save_checkpoint(
        path, clf, model_name="DANINHAS", n_classes=3, class_names=["a", "b", "c"], img_size=32
    )
    payload = torch.load(path, weights_only=True)
    del payload["model_kwargs"]  # what every checkpoint written before ADR 0014 looks like
    torch.save(payload, path)
    loaded, meta = load_checkpoint(path)
    assert meta["model_kwargs"] == {} and active_dropout_modules(loaded) == []
