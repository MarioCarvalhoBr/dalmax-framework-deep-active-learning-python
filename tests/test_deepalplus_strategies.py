"""DeepAL+ ports (ADR 0015): scoring rules against hand-computed cases, determinism,
fail-fast behaviour, the `strategy_args` config block and registry wiring.

Fast and CPU-only: synthetic tensors, stub datasets/nets and tiny models; the only
ResNet-50 built here comes from the local torch hub cache of the ImageNet weights
(as in `tests/test_mc_dropout_and_embeddings.py`). The end-to-end runs through
`tools/trainer.py` live in `tests/test_deepalplus_e2e.py` (dataset + slow).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.config.schema import (
    CEALArgs,
    ConfigError,
    LossPredictionArgs,
    StrategyArgs,
    VAALArgs,
    WAALArgs,
)
from dalmax.models.base import DeepLearning
from dalmax.models.daninhas_resnet50 import DaninhasModelResNet50
from dalmax.models.joint_dataset import JointLabeledUnlabeled
from dalmax.models.loss_prediction import (
    LossPredictionTrainer,
    attached_epochs,
    loss_prediction_ranking_loss,
)
from dalmax.models.mc_dropout import MCDropoutError
from dalmax.models.vaal import (
    VAE,
    VAALSampler,
    adversarial_loss,
    discriminator_loss,
    discriminator_targets,
    resize_for_vae,
)
from dalmax.models.waal import WAALTrainer, gradient_penalty, waal_scores
from dalmax.query_strategies.badge_sampling import (
    BadgeSampling,
    gradient_embeddings,
    kmeans_plus_plus,
)
from dalmax.query_strategies.ceal import CEALSampling, ceal_threshold, predictive_entropy
from dalmax.query_strategies.entropy_sampling import EntropySampling
from dalmax.query_strategies.kcenter_greedy import KCenterGreedy, k_center_greedy
from dalmax.query_strategies.kcenter_greedy_pca import KCenterGreedyPCA, reduce_embeddings
from dalmax.query_strategies.loss_prediction import LossPredictionLoss
from dalmax.query_strategies.mean_std import MeanSTD, mean_std_scores
from dalmax.query_strategies.ranking import rank_ascending, rank_descending
from dalmax.query_strategies.registry import (
    DEEPALPLUS_STRATEGY_REGISTRY,
    STRATEGY_REGISTRY,
    build_strategy,
)
from dalmax.query_strategies.vaal import VAAL
from dalmax.query_strategies.var_ratio import VarRatio, variation_ratios
from dalmax.query_strategies.waal import WAAL
from dalmax.seeding import derive_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_MICRO = REPO_ROOT / "files_config/params_micro.json"
LOGGER = logging.getLogger("test_deepalplus")


# --- shared stubs ---------------------------------------------------------------------


class _ImageData(torch.utils.data.Dataset):
    """`(x, y, idx)` dataset with a `.Y` attribute, as `DeepLearning` expects."""

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        self.xs, self.Y = xs, ys

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, i: int):
        return self.xs[i], self.Y[i], i


class _PoolDataset:
    """Stub `Data`: a pool of `n` samples, the first `n_labeled` labeled."""

    def __init__(self, n: int, n_labeled: int = 0, xs=None, ys=None) -> None:
        self.n_pool = n
        self.labeled_idxs = np.zeros(n, dtype=bool)
        self.labeled_idxs[:n_labeled] = True
        self.X_train = xs if xs is not None else np.zeros((n, 1))
        self.Y_train = ys if ys is not None else np.zeros(n, dtype=np.int64)

    def handler(self, x, y):
        return _ImageData(torch.as_tensor(x), torch.as_tensor(y))

    def get_unlabeled_data(self):
        idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return idxs, self.handler(self.X_train[idxs], self.Y_train[idxs])

    def get_labeled_data(self):
        idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return idxs, self.handler(self.X_train[idxs], self.Y_train[idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)


class _SplitProbsNet:
    """Returns fixed MC-dropout passes `[T, n_unlabeled, C]` (no real model)."""

    def __init__(self, probs: torch.Tensor) -> None:
        self.params = {"n_drop": probs.shape[0]}
        self._probs = probs

    def predict_prob_dropout_split(self, data, n_drop: int) -> torch.Tensor:
        assert n_drop == self._probs.shape[0]
        return self._probs.clone()


def _onehot_passes(preds: list[list[int]], n_classes: int) -> torch.Tensor:
    """`[T, N, C]` softmax-like passes whose argmax is `preds[t][i]`."""
    t = torch.tensor(preds)
    return F.one_hot(t, n_classes).float() * 0.8 + 0.2 / n_classes


_TEST_PARAMS = {
    "n_epoch": 2,
    "n_classes": 3,
    "n_drop": 4,
    "train_args": {"batch_size": 5, "num_workers": 0},
    "test_args": {"batch_size": 4, "num_workers": 0},
    "optimizer_args": {"lr": 0.05, "momentum": 0.3},
}


class _TinyStagedNet(nn.Module):
    """Four "stages" + pooled features + classifier, the interface
    `LossPredictionTrainer`/`WAALTrainer` need (`DaninhasModelResNet50`-like)."""

    def __init__(self, n_classes: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(3, 4, 3, padding=1),
                nn.Conv2d(4, 4, 3, stride=2, padding=1),
                nn.Conv2d(4, 6, 3, stride=2, padding=1),
                nn.Conv2d(6, 6, 3, stride=2, padding=1),
            ]
        )
        self.bn = nn.BatchNorm2d(4)
        self.classifier = nn.Linear(6, n_classes)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward_with_stages(self, x: torch.Tensor):
        stages = []
        h = x
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(h))
            if i == 0:
                h = self.bn(h)
            stages.append(h)
        features = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.classifier(self.dropout(features)), features, stages

    def forward(self, x: torch.Tensor):
        logits, features, _ = self.forward_with_stages(x)
        return logits, features

    def get_embedding_dim(self) -> int:
        return 6

    def stage_channels(self) -> tuple[int, ...]:
        return (4, 4, 6, 6)


def _tiny_data(n: int = 12, seed: int = 0) -> _ImageData:
    g = torch.Generator().manual_seed(seed)
    return _ImageData(torch.randn(n, 3, 8, 8, generator=g), torch.arange(n) % 3)


def _tiny_net(dropout_p: float = 0.0) -> DeepLearning:
    return DeepLearning(
        _TinyStagedNet, dict(_TEST_PARAMS), torch.device("cpu"), {"dropout_p": dropout_p}
    )


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def _assert_same_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


# --- ranking ------------------------------------------------------------------------


def test_rank_descending_without_ties_is_the_stable_descending_order() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert rank_descending(scores, np.random.default_rng(0)).tolist() == [1, 2, 3, 0]


def test_rank_descending_breaks_ties_at_random_and_reproducibly() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
    orders = {tuple(rank_descending(scores, np.random.default_rng(s))[:4]) for s in range(20)}
    assert len(orders) > 1  # ties broken differently for different seeds
    a = rank_descending(scores, np.random.default_rng(7))
    b = rank_descending(scores, np.random.default_rng(7))
    assert a.tolist() == b.tolist() and a[-1] == 4


def test_rank_ascending_is_the_adversarial_rule() -> None:
    dis = np.array([2.0, 1.0, 1.0, np.inf])
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    expected = np.lexsort((rng_b.permutation(4), dis))
    assert rank_ascending(dis, rng_a).tolist() == expected.tolist()


# --- VarRatio -----------------------------------------------------------------------


def test_variation_ratios_hand_computed() -> None:
    # 4 passes x 3 samples: votes (0,0,0,0) -> 0; (0,1,0,2) -> 1 - 2/4; (1,1,2,1) -> 1 - 3/4.
    probs = _onehot_passes([[0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 2, 1]], 3)
    torch.testing.assert_close(
        variation_ratios(probs), torch.tensor([0.0, 0.5, 0.25], dtype=torch.float64)
    )


def test_var_ratio_queries_highest_first_and_is_deterministic() -> None:
    probs = _onehot_passes([[0, 0, 1, 2], [0, 1, 1, 2], [0, 0, 2, 0], [0, 2, 1, 1]], 3)
    # scores: 0, 0.5, 0.25, 0.5 -> samples 1 and 3 tie at the top.
    dataset = _PoolDataset(4)

    def query(seed: int) -> list[int]:
        strategy = VarRatio(
            dataset, _SplitProbsNet(probs), LOGGER, rng=np.random.default_rng(seed)
        )
        return strategy.query(3).tolist()

    assert query(1) == query(1)
    assert all(sorted(q[:2]) == [1, 3] and q[2] == 2 for q in (query(s) for s in range(10)))
    assert len({tuple(query(s)[:2]) for s in range(20)}) == 2  # both tie orders occur


def test_var_ratio_and_mean_std_fail_fast_without_dropout() -> None:
    net = _tiny_net(dropout_p=0.0)
    data = _tiny_data()
    net.train(data)
    dataset = _PoolDataset(len(data), xs=data.xs.numpy(), ys=data.Y.numpy())
    with pytest.raises(MCDropoutError):
        VarRatio(dataset, net, LOGGER, rng=np.random.default_rng(0)).query(2)
    with pytest.raises(MCDropoutError):
        MeanSTD(dataset, net, LOGGER).query(2)


def test_var_ratio_and_mean_std_run_with_dropout() -> None:
    net = _tiny_net(dropout_p=0.5)
    data = _tiny_data()
    torch.manual_seed(0)
    net.train(data)
    dataset = _PoolDataset(len(data), xs=data.xs.numpy(), ys=data.Y.numpy())
    assert len(VarRatio(dataset, net, LOGGER, rng=np.random.default_rng(0)).query(3)) == 3
    assert len(MeanSTD(dataset, net, LOGGER).query(3)) == 3


# --- MeanSTD ------------------------------------------------------------------------


def test_mean_std_scores_hand_computed_population_std() -> None:
    probs = torch.tensor(
        [
            [[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]],
            [[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]],
        ]
    )
    # sample 0: no spread -> 0; sample 1: |0.9-0.5|/2 = 0.2 per class; sample 2: 0.1.
    torch.testing.assert_close(
        mean_std_scores(probs), torch.tensor([0.0, 0.2, 0.1], dtype=torch.float64)
    )
    strategy = MeanSTD(_PoolDataset(3), _SplitProbsNet(probs), LOGGER)
    assert strategy.query(3).tolist() == [1, 2, 0]


# --- KCenterGreedy (refactor) and KCenterGreedyPCA ----------------------------------


def _reference_kcenter(embeddings: np.ndarray, labeled: np.ndarray, n: int) -> np.ndarray:
    """Verbatim pre-ADR-0015 body of `KCenterGreedy.query` (after the ADR 0014 clamp)."""
    labeled_idxs = labeled.copy()
    n_pool = len(labeled_idxs)
    dist_mat = np.matmul(embeddings, embeddings.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    np.maximum(dist_mat, 0, out=dist_mat)
    dist_mat = np.sqrt(dist_mat)
    mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]
    for _ in range(n):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n_pool)[~labeled_idxs][q_idx_]
        labeled_idxs[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
    return np.arange(n_pool)[(labeled ^ labeled_idxs)]


class _EmbeddingNet:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = torch.from_numpy(embeddings)

    def get_embeddings(self, data) -> torch.Tensor:
        return self._embeddings.clone()


def test_k_center_greedy_is_the_unchanged_loop() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 16)).astype(np.float32)
    labeled = np.zeros(60, dtype=bool)
    labeled[rng.choice(60, 10, replace=False)] = True
    expected = _reference_kcenter(embeddings.copy(), labeled, 7)
    assert k_center_greedy(embeddings.copy(), labeled, 7).tolist() == expected.tolist()
    dataset = _PoolDataset(60)
    dataset.labeled_idxs = labeled.copy()
    picked = KCenterGreedy(dataset, _EmbeddingNet(embeddings), LOGGER).query(7)
    assert picked.tolist() == expected.tolist()


def test_reduce_embeddings_applies_seeded_pca_only_when_wider() -> None:
    rng = np.random.default_rng(0)
    wide = rng.normal(size=(80, 64)).astype(np.float32)
    reduced = reduce_embeddings(wide, 50, random_state=11)
    assert reduced.shape == (80, 50) and reduced.dtype == np.float32
    expected = PCA(n_components=50, random_state=11).fit_transform(wide).astype(np.float32)
    np.testing.assert_array_equal(reduced, expected)
    narrow = wide[:, :50]
    np.testing.assert_array_equal(reduce_embeddings(narrow, 50, random_state=11), narrow)


def test_kcenter_greedy_pca_runs_the_shared_loop_on_the_reduced_embeddings() -> None:
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(70, 64)).astype(np.float32)
    dataset = _PoolDataset(70, n_labeled=10)
    strategy = KCenterGreedyPCA(
        dataset, _EmbeddingNet(embeddings), LOGGER, n_components=50, random_state=5
    )
    picked = strategy.query(6)
    reduced = reduce_embeddings(embeddings, 50, 5)
    assert picked.tolist() == k_center_greedy(reduced, dataset.labeled_idxs, 6).tolist()
    assert picked.tolist() == strategy.query(6).tolist()  # deterministic
    assert not dataset.labeled_idxs[picked].any() and len(set(picked.tolist())) == 6


# --- BADGE --------------------------------------------------------------------------


def test_gradient_embeddings_equal_the_last_layer_gradient_with_the_predicted_label() -> None:
    torch.manual_seed(0)
    layer = nn.Linear(4, 3)
    features = torch.randn(5, 4)
    logits = layer(features)
    probs = F.softmax(logits, dim=1).detach()
    embeddings = gradient_embeddings(probs, features)
    assert embeddings.shape == (5, 12)
    for i in range(5):
        layer.zero_grad()
        F.cross_entropy(layer(features[i : i + 1]), probs[i : i + 1].argmax(1)).backward()
        torch.testing.assert_close(embeddings[i], layer.weight.grad.reshape(-1))


def _reference_kmeans_pp(emb: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """k-means++ on explicit embeddings (float64), drawing exactly like `kmeans_plus_plus`."""
    chosen = [int(np.argmax((emb**2).sum(1)))]
    min_d2 = np.full(len(emb), np.inf)
    while len(chosen) < n:
        d2 = ((emb - emb[chosen[-1]]) ** 2).sum(1)
        min_d2 = np.minimum(min_d2, d2)
        weights = min_d2.copy()
        weights[chosen] = 0
        chosen.append(int(rng.choice(len(emb), p=weights / weights.sum())))
    return chosen


def test_kmeans_plus_plus_matches_explicit_gradient_embeddings() -> None:
    gen = torch.Generator().manual_seed(3)
    probs = F.softmax(torch.randn(40, 5, generator=gen, dtype=torch.float64), dim=1)
    features = torch.randn(40, 8, generator=gen, dtype=torch.float64)
    residuals = probs.clone()
    residuals[torch.arange(40), probs.argmax(1)] -= 1
    explicit = gradient_embeddings(probs, features).numpy()
    picked = kmeans_plus_plus(residuals.numpy(), features.numpy(), 10, np.random.default_rng(4))
    assert picked.tolist() == _reference_kmeans_pp(explicit, 10, np.random.default_rng(4))
    assert len(set(picked.tolist())) == 10


def test_kmeans_plus_plus_falls_back_to_uniform_when_all_distances_vanish() -> None:
    residuals = np.tile([[0.5, -0.5]], (6, 1))
    features = np.tile([[1.0, 2.0]], (6, 1))  # six identical embeddings
    picked = kmeans_plus_plus(residuals, features, 4, np.random.default_rng(0))
    assert len(set(picked.tolist())) == 4
    again = kmeans_plus_plus(residuals, features, 4, np.random.default_rng(0))
    assert picked.tolist() == again.tolist()


class _LogitsNet:
    def __init__(self, logits: torch.Tensor, features: torch.Tensor) -> None:
        self.logits, self.features = logits, features

    def predict_logits_and_embeddings(self, data):
        return self.logits.clone(), self.features.clone()


def test_badge_sampling_query_is_deterministic_per_seed() -> None:
    gen = torch.Generator().manual_seed(0)
    logits, features = torch.randn(30, 5, generator=gen), torch.randn(30, 16, generator=gen)
    dataset = _PoolDataset(35, n_labeled=5)

    def query(seed: int) -> list[int]:
        net = _LogitsNet(logits, features)
        return BadgeSampling(dataset, net, LOGGER, rng=np.random.default_rng(seed)).query(8).tolist()

    assert query(1) == query(1)
    assert query(1) != query(2)
    assert all(i >= 5 for i in query(1))  # unlabeled pool ids only


def test_predict_logits_and_embeddings_matches_forward() -> None:
    net = _tiny_net()
    data = _tiny_data()
    net.train(data)
    logits, embeddings = net.predict_logits_and_embeddings(data)
    net.clf.eval()
    with torch.no_grad():
        ref_logits, ref_features = net.clf(data.xs)
    torch.testing.assert_close(logits, ref_logits)
    torch.testing.assert_close(embeddings, ref_features)


# --- CEAL ---------------------------------------------------------------------------


def test_ceal_threshold_decays_linearly() -> None:
    assert ceal_threshold(5e-5, 0.033e-5, 1) == pytest.approx(5e-5 - 0.033e-5)
    assert ceal_threshold(5e-5, 0.033e-5, 8) == pytest.approx(5e-5 - 8 * 0.033e-5)


def test_predictive_entropy_has_no_nan_on_zero_probabilities() -> None:
    probs = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    torch.testing.assert_close(predictive_entropy(probs), torch.tensor([0.0, math.log(2)]))


class _RecordingNet:
    """Fixed probabilities over the CURRENT unlabeled pool; records training sets."""

    def __init__(self, probs_by_id: np.ndarray, dataset: _PoolDataset) -> None:
        self.probs_by_id = torch.as_tensor(probs_by_id, dtype=torch.float32)
        self.dataset = dataset
        self.trained_on: list[tuple[np.ndarray, np.ndarray]] = []

    def predict_prob(self, data) -> torch.Tensor:
        idxs = np.arange(self.dataset.n_pool)[~self.dataset.labeled_idxs]
        assert len(idxs) == len(data)
        return self.probs_by_id[idxs].clone()

    def train(self, data) -> None:
        self.trained_on.append((data.xs.numpy().copy(), data.Y.numpy().copy()))


def _ceal_setup() -> tuple[_PoolDataset, _RecordingNet, CEALSampling]:
    # Pool of 8, ids 0-1 labeled. Ids 2,3: uncertain (queried by entropy);
    # ids 4,5: confident (entropy ~1e-7), predicted classes 1 and 2; id 5's true
    # label is 0 (a wrong pseudo-label); ids 6,7: moderately confident (above delta).
    xs = np.arange(8, dtype=np.float32)[:, None]
    ys = np.array([0, 1, 0, 1, 1, 0, 2, 2])
    dataset = _PoolDataset(8, n_labeled=2, xs=xs, ys=ys)
    eps = 1e-8
    probs = np.array(
        [
            [0.4, 0.3, 0.3],
            [0.4, 0.3, 0.3],
            [1 / 3, 1 / 3, 1 / 3],
            [0.34, 0.33, 0.33],
            [eps, 1 - 2 * eps, eps],
            [eps, eps, 1 - 2 * eps],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ]
    )
    net = _RecordingNet(probs, dataset)
    base = EntropySampling(dataset, net, LOGGER)
    strategy = CEALSampling(
        dataset, net, LOGGER, base_strategy=base, delta=5e-5, delta_decay=0.033e-5
    )
    return dataset, net, strategy


def test_ceal_pseudo_labels_feed_only_the_next_training() -> None:
    dataset, net, strategy = _ceal_setup()
    strategy.train()  # round 0: labeled only
    assert net.trained_on[-1][0].ravel().tolist() == [0.0, 1.0]

    q = strategy.query(2)
    assert sorted(q.tolist()) == [2, 3]
    assert dataset.labeled_idxs.sum() == 2  # query never labels anything itself
    strategy.update(q)
    strategy.train()
    x, y = net.trained_on[-1]
    assert x.ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert y.tolist() == [0, 1, 0, 1, 1, 2]  # ids 4, 5 carry their PREDICTED class
    assert dataset.labeled_idxs.tolist() == [True] * 4 + [False] * 4

    diag = strategy.diagnostics()
    assert diag["ceal_round"] == [1]
    assert diag["ceal_n_pseudo_labels"] == [2]
    assert diag["ceal_pseudo_label_accuracy"] == [0.5]
    assert diag["ceal_delta"][0] == pytest.approx(5e-5 - 0.033e-5)


def test_ceal_recomputes_pseudo_labels_every_round() -> None:
    dataset, net, strategy = _ceal_setup()
    strategy.update(strategy.query(2))
    strategy.train()
    q2 = strategy.query(2)  # ids 4, 5 have the lowest entropy -> not picked; 6, 7 are
    assert sorted(q2.tolist()) == [6, 7]
    strategy.update(q2)
    strategy.train()
    x, _ = net.trained_on[-1]
    assert sorted(x.ravel().tolist()) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert strategy.diagnostics()["ceal_n_pseudo_labels"] == [2, 2]


def test_ceal_without_pseudo_labels_trains_on_the_labeled_set() -> None:
    dataset, net, strategy = _ceal_setup()
    strategy.delta = 1e-12  # nothing passes
    strategy.update(strategy.query(2))
    strategy.train()
    assert net.trained_on[-1][0].ravel().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert strategy.diagnostics()["ceal_pseudo_label_accuracy"] == [None]


# --- LossPredictionLoss -------------------------------------------------------------


def test_ranking_loss_hand_computed() -> None:
    pred = torch.tensor([3.0, 1.0, 2.0, 0.0])
    true = torch.tensor([0.5, 2.0, 1.0, 0.1])
    # pairs (0,3): d_pred 3, d_true 0.4 > 0 -> max(0, 1 - 3) = 0
    # pairs (1,2): d_pred -1, d_true 1.0 > 0 -> max(0, 1 + 1) = 2;  (0 + 2) / 2 = 1
    assert loss_prediction_ranking_loss(pred, true, margin=1.0).item() == pytest.approx(1.0)
    # d_true == 0 counts as "-1": pair (0,1) d_pred 0.25 -> max(0, 1 + 0.25) = 1.25
    flat = loss_prediction_ranking_loss(torch.tensor([0.25, 0.0]), torch.tensor([1.0, 1.0]))
    assert flat.item() == pytest.approx(1.25)
    with pytest.raises(ValueError, match="even"):
        loss_prediction_ranking_loss(pred[:3], true[:3])


@pytest.mark.parametrize(
    "fraction,n_epoch,expected", [(0.6, 10, 6), (0.6, 200, 120), (0.0, 10, 0), (1.0, 10, 10)]
)
def test_attached_epochs(fraction: float, n_epoch: int, expected: int) -> None:
    assert attached_epochs(fraction, n_epoch) == expected


def _fit_lpl(weight: float, detach_fraction: float) -> tuple[DeepLearning, nn.Module]:
    torch.manual_seed(0)
    net = _tiny_net()
    trainer = LossPredictionTrainer(
        LossPredictionArgs(weight=weight, detach_fraction=detach_fraction),
        np.random.default_rng(1),
    )
    module = trainer.fit(net, _tiny_data())
    return net, module


def test_module_gradient_reaches_the_backbone_only_before_the_detach_epoch() -> None:
    reference, _ = _fit_lpl(weight=0.0, detach_fraction=1.0)
    always_detached, _ = _fit_lpl(weight=1.0, detach_fraction=0.0)
    attached, _ = _fit_lpl(weight=1.0, detach_fraction=1.0)
    # Detached from epoch 1: the backbone sees the cross-entropy only.
    _assert_same_state(_state(always_detached.clf), _state(reference.clf))
    assert not torch.equal(attached.clf.convs[0].weight, reference.clf.convs[0].weight)


def test_loss_prediction_training_is_deterministic_and_leaves_the_global_stream() -> None:
    net_a, module_a = _fit_lpl(1.0, 0.6)
    after_a = torch.rand(3)
    net_b, module_b = _fit_lpl(1.0, 0.6)
    after_b = torch.rand(3)
    _assert_same_state(_state(net_a.clf), _state(net_b.clf))
    _assert_same_state(_state(module_a), _state(module_b))
    assert torch.equal(after_a, after_b)

    # Building the module does not consume the global torch stream.
    trainer = LossPredictionTrainer(LossPredictionArgs(), np.random.default_rng(0))
    torch.manual_seed(5)
    trainer.build_module((4, 4, 6, 6), torch.device("cpu"))
    with_module = torch.rand(3)
    torch.manual_seed(5)
    assert torch.equal(with_module, torch.rand(3))


def test_loss_prediction_strategy_queries_highest_predicted_loss() -> None:
    net, module = _fit_lpl(1.0, 0.6)
    data = _tiny_data()
    dataset = _PoolDataset(len(data), n_labeled=3, xs=data.xs.numpy(), ys=data.Y.numpy())
    trainer = LossPredictionTrainer(LossPredictionArgs(), np.random.default_rng(0))
    strategy = LossPredictionLoss(dataset, net, LOGGER, trainer=trainer)
    with pytest.raises(RuntimeError, match="before train"):
        strategy.query(2)
    strategy.module = module
    _, unlabeled = dataset.get_unlabeled_data()
    scores = trainer.predict_losses(net, module, unlabeled).numpy()
    expected = (np.argsort(-scores, kind="stable")[:4] + 3).tolist()
    assert strategy.query(4).tolist() == expected


def test_resnet50_forward_with_stages_matches_forward() -> None:
    torch.manual_seed(0)
    model = DaninhasModelResNet50(5).eval()
    x = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, features = model(x)
        s_logits, s_features, stages = model.forward_with_stages(x)
    assert torch.equal(logits, s_logits) and torch.equal(features, s_features)
    assert model.stage_channels() == (256, 512, 1024, 2048)
    assert [s.shape[1] for s in stages] == [256, 512, 1024, 2048]


# --- WAAL ---------------------------------------------------------------------------


def test_waal_scores_hand_computed_and_ranked_lowest_first() -> None:
    probs = torch.tensor([[0.5, 0.5], [0.9, 0.1], [0.5, 0.5]], dtype=torch.float64)
    log_p = probs.log()
    critic = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64)
    ln2 = math.log(2)
    expected_0 = 0.5 * math.sqrt(2) * ln2 + 0.5 * 2 * ln2
    l2_1 = math.hypot(math.log(0.9), math.log(0.1))
    expected_1 = 0.5 * l2_1 + 0.5 * -(math.log(0.9) + math.log(0.1))
    scores = waal_scores(log_p, critic, selection=10.0)
    torch.testing.assert_close(
        scores,
        torch.tensor([expected_0, expected_1, expected_0 - 2.0], dtype=torch.float64),
    )
    assert scores.argsort().tolist() == [2, 0, 1]  # uncertain + "unlabeled-looking" first


def test_gradient_penalty_of_a_linear_critic() -> None:
    critic = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 0.0, 4.0]]))  # ||w|| = 5
    h_s, h_t = torch.randn(4, 3), torch.randn(4, 3)
    gp = gradient_penalty(lambda z: critic(z).reshape(-1), h_s, h_t, torch.Generator())
    assert gp.item() == pytest.approx((5.0 - 1.0) ** 2)


def _fit_waal(seed: int = 0) -> tuple[DeepLearning, nn.Module]:
    torch.manual_seed(seed)
    net = _tiny_net()
    trainer = WAALTrainer(WAALArgs(), np.random.default_rng(seed))
    critic = trainer.fit(net, _tiny_data(5, seed=1), _tiny_data(9, seed=2))
    return net, critic


def test_waal_training_is_deterministic_and_keeps_the_task_model_interface() -> None:
    net_a, critic_a = _fit_waal()
    net_b, critic_b = _fit_waal()
    _assert_same_state(_state(net_a.clf), _state(net_b.clf))
    _assert_same_state(_state(critic_a), _state(critic_b))
    assert all(p.requires_grad for p in net_a.clf.parameters())
    # The evaluated model is the plain task model: ordinary prediction works.
    preds = net_a.predict(_tiny_data(4, seed=3))
    assert preds.shape == (4,)


def test_waal_task_model_starts_from_the_same_weights_as_plain_training() -> None:
    torch.manual_seed(0)
    plain_init = _state(_tiny_net().build_model())
    torch.manual_seed(0)
    net = _tiny_net()
    first: dict[str, torch.Tensor] = {}
    original_build = net.build_model

    def spy() -> nn.Module:
        model = original_build()
        first.update(_state(model))
        return model

    net.build_model = spy  # type: ignore[method-assign]
    WAALTrainer(WAALArgs(), np.random.default_rng(0)).fit(net, _tiny_data(5), _tiny_data(9))
    _assert_same_state(first, plain_init)


def test_waal_strategy_queries_lowest_score() -> None:
    net, critic = _fit_waal()
    data = _tiny_data(10, seed=4)
    dataset = _PoolDataset(10, n_labeled=2, xs=data.xs.numpy(), ys=data.Y.numpy())
    trainer = WAALTrainer(WAALArgs(), np.random.default_rng(0))
    strategy = WAAL(dataset, net, LOGGER, trainer=trainer, selection=10.0)
    with pytest.raises(RuntimeError, match="before train"):
        strategy.query(2)
    strategy.critic = critic
    _, unlabeled = dataset.get_unlabeled_data()
    scores = trainer.score_pool(net, critic, unlabeled, 10.0).numpy()
    assert strategy.query(3).tolist() == (np.argsort(scores, kind="stable")[:3] + 2).tolist()


def test_joint_dataset_cycles_the_shorter_set() -> None:
    labeled = _ImageData(torch.arange(2).float(), torch.tensor([7, 8]))
    unlabeled = _ImageData(torch.arange(5).float() + 10, torch.zeros(5, dtype=torch.long))
    joint = JointLabeledUnlabeled(labeled, unlabeled)
    assert len(joint) == 5
    assert [joint[i][1].item() for i in range(5)] == [0, 1, 0, 1, 0]
    assert [joint[i][3].item() for i in range(5)] == [10, 11, 12, 13, 14]


# --- VAAL ---------------------------------------------------------------------------


def test_discriminator_targets_differ_between_labeled_and_unlabeled() -> None:
    lab, unlab = discriminator_targets(3, 2)
    assert lab.tolist() == [1.0, 1.0, 1.0] and unlab.tolist() == [0.0, 0.0]
    lab_preds, unlab_preds = torch.tensor([0.9]), torch.tensor([0.1])
    # Discriminator: labeled -> 1, unlabeled -> 0, so a correct discriminator has a small loss.
    assert discriminator_loss(lab_preds, unlab_preds).item() == pytest.approx(-2 * math.log(0.9))
    # VAE (fooling): both -> 1, so the same predictions cost a lot.
    assert adversarial_loss(lab_preds, unlab_preds).item() == pytest.approx(
        -math.log(0.9) - math.log(0.1)
    )


@pytest.mark.parametrize("image_size", [32, 64])
def test_vae_reconstructs_the_input_shape(image_size: int) -> None:
    torch.manual_seed(0)
    vae = VAE(image_size, latent_dim=8).eval()
    x = torch.randn(2, 3, image_size, image_size)
    recon, z, mu, logvar = vae(x, torch.Generator().manual_seed(0))
    assert recon.shape == x.shape  # DeepAL+'s VAE_ImageNet turns a batch of 2 into 8 rows
    assert z.shape == mu.shape == logvar.shape == (2, 8)


def test_resize_for_vae_area_downsamples() -> None:
    x = torch.arange(16.0).view(1, 1, 4, 4)
    torch.testing.assert_close(
        resize_for_vae(x, 2), torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]])
    )
    assert resize_for_vae(x, 4) is x


def _vaal_sampler(seed: int) -> VAALSampler:
    args = VAALArgs(epochs=1, latent_dim=8, image_size=32)
    loader_args = {"batch_size": 4, "num_workers": 0}
    return VAALSampler(args, np.random.default_rng(seed), torch.device("cpu"), loader_args, loader_args)


def _image_set(n: int, seed: int, size: int = 64) -> _ImageData:
    g = torch.Generator().manual_seed(seed)
    return _ImageData(torch.randn(n, 3, size, size, generator=g), torch.zeros(n, dtype=torch.long))


def test_vaal_sampler_is_deterministic_and_leaves_the_global_stream() -> None:
    labeled, unlabeled = _image_set(4, 0), _image_set(6, 1)
    torch.manual_seed(9)
    vae, dis = _vaal_sampler(3).fit(labeled, unlabeled)
    scores_a = _vaal_sampler(3).score(vae, dis, unlabeled)
    after_fit = torch.rand(3)
    torch.manual_seed(9)
    vae_b, dis_b = _vaal_sampler(3).fit(labeled, unlabeled)
    scores_b = _vaal_sampler(3).score(vae_b, dis_b, unlabeled)
    torch.manual_seed(9)
    assert torch.equal(after_fit, torch.rand(3))  # fit consumed nothing global
    assert torch.equal(scores_a, scores_b)
    assert ((scores_a >= 0) & (scores_a <= 1)).all()


class _FixedScoreSampler:
    args = VAALArgs()

    def __init__(self, scores: torch.Tensor) -> None:
        self.scores = scores

    def fit(self, labeled, unlabeled):
        return None, None

    def score(self, vae, discriminator, data) -> torch.Tensor:
        return self.scores.clone()


def test_vaal_queries_the_lowest_discriminator_scores() -> None:
    dataset = _PoolDataset(6, n_labeled=2)
    sampler = _FixedScoreSampler(torch.tensor([0.9, 0.1, 0.5, 0.05]))
    assert VAAL(dataset, None, LOGGER, sampler=sampler).query(2).tolist() == [5, 3]


# --- config: the `strategy_args` block ----------------------------------------------


def _params_with(tmp_path: Path, strategy_args) -> Path:
    raw = json.loads(PARAMS_MICRO.read_text())
    raw["DANINHAS"]["strategy_args"] = strategy_args
    path = tmp_path / "params.json"
    path.write_text(json.dumps(raw))
    return path


def _load(params: Path, strategy: str = "VAAL"):
    return load_experiment_config(
        params,
        "DANINHAS",
        strategy_name=strategy,
        seed=1,
        n_init_labeled=10,
        n_query=5,
        n_round=2,
        dir_results="results/unused/",
        device="cpu",
    )


def test_strategy_args_absent_means_defaults() -> None:
    config = _load(PARAMS_MICRO)
    assert config.dataset.strategy_args == StrategyArgs()
    resolved = to_dict(config)["dataset"]["strategy_args"]
    assert resolved["vaal"]["epochs"] == 30 and resolved["waal"]["alpha"] == 1e-3
    assert resolved["ceal"] == {
        "base_strategy": "EntropySampling",
        "delta": 5e-5,
        "delta_decay": 0.033e-5,
    }
    assert resolved["loss_prediction"]["detach_fraction"] == 0.6
    assert resolved["kcenter_greedy_pca"]["n_components"] == 50


def test_strategy_args_override_and_comment_keys(tmp_path: Path) -> None:
    config = _load(
        _params_with(
            tmp_path,
            {
                "_comment": "x",
                "VAAL": {"epochs": 2, "lr": 1, "_comment": "y"},
                "CEALSampling": {"base_strategy": "MarginSampling"},
            },
        )
    )
    args = config.dataset.strategy_args
    assert args.vaal.epochs == 2 and args.vaal.lr == 1.0 and isinstance(args.vaal.lr, float)
    assert args.vaal.latent_dim == 32
    assert args.ceal == CEALArgs(base_strategy="MarginSampling")


@pytest.mark.parametrize(
    "block,match",
    [
        ({"Nope": {}}, "unknown strategy name"),
        ({"VAAL": {"epoch": 3}}, "unknown key"),
        ({"VAAL": {"epochs": 2.5}}, "integer"),
        ({"VAAL": {"epochs": True}}, "must be a int"),
        ({"WAAL": {"alpha": "0.1"}}, "number"),
        ({"VAAL": {"image_size": 40}}, "multiple of 16"),
        ({"VAAL": {"epochs": 0}}, "> 0"),
        ({"LossPredictionLoss": {"detach_fraction": 1.5}}, r"\[0, 1\]"),
        ({"CEALSampling": {"base_strategy": "BALDDropout"}}, "base_strategy"),
        ({"CEALSampling": {"base_strategy": 3}}, "string"),
        ({"VAAL": [1]}, "JSON object"),
        ([1], "JSON object"),
    ],
)
def test_invalid_strategy_args_fail_fast(tmp_path: Path, block, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _load(_params_with(tmp_path, block))


# --- registry -----------------------------------------------------------------------


class _RegistryNet:
    def __init__(self) -> None:
        self.params = {"n_drop": 3}


def test_registry_contains_the_eight_ports() -> None:
    assert set(DEEPALPLUS_STRATEGY_REGISTRY) == {
        "VarRatio",
        "MeanSTD",
        "KCenterGreedyPCA",
        "BadgeSampling",
        "CEALSampling",
        "LossPredictionLoss",
        "VAAL",
        "WAAL",
    }
    assert set(DEEPALPLUS_STRATEGY_REGISTRY) <= set(STRATEGY_REGISTRY)


@pytest.mark.parametrize("name", sorted(DEEPALPLUS_STRATEGY_REGISTRY))
def test_build_strategy_constructs_each_port_without_touching_any_stream(name: str) -> None:
    config = _load(PARAMS_MICRO, strategy=name)
    main_rng = np.random.default_rng(0)
    torch.manual_seed(3)
    strategy = build_strategy(name, _PoolDataset(10, 2), _RegistryNet(), config, LOGGER, main_rng)
    assert type(strategy) is DEEPALPLUS_STRATEGY_REGISTRY[name]
    assert main_rng.integers(0, 10**9) == np.random.default_rng(0).integers(0, 10**9)
    after = torch.rand(2)
    torch.manual_seed(3)
    assert torch.equal(after, torch.rand(2))


def test_build_strategy_derives_each_generator_from_the_seed() -> None:
    def build(name: str):
        config = _load(PARAMS_MICRO, strategy=name)
        return build_strategy(name, _PoolDataset(10, 2), _RegistryNet(), config, LOGGER, None)

    expected = np.random.default_rng(derive_seed(1, "var_ratio_ties")).permutation(50)
    assert build("VarRatio").rng.permutation(50).tolist() == expected.tolist()
    expected = np.random.default_rng(derive_seed(1, "badge")).permutation(50)
    assert build("BadgeSampling").rng.permutation(50).tolist() == expected.tolist()
    assert build("KCenterGreedyPCA").random_state == derive_seed(1, "kcenter_pca")
    assert build("KCenterGreedyPCA").n_components == 50
    ceal = build("CEALSampling")
    assert type(ceal.base_strategy) is EntropySampling
    assert (ceal.delta, ceal.delta_decay) == (5e-5, 0.033e-5)
    assert build("WAAL").selection == 10.0
    assert build("VAAL").sampler.args == VAALArgs()


# --- params files ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "vaal_file,base_file,epochs",
    [
        ("params_paper1_vaal.json", "params_paper1.json", 30),
        ("params_paper1_vaal_micro.json", "params_paper1_micro.json", 1),
    ],
)
def test_vaal_params_files_differ_from_paper1_only_by_the_vaal_block(
    vaal_file: str, base_file: str, epochs: int
) -> None:
    campaign = REPO_ROOT / "files_config/campaign"
    vaal, base = (json.loads((campaign / f).read_text()) for f in (vaal_file, base_file))
    block = dict(vaal["DANINHAS"])
    strategy_args = block.pop("strategy_args")
    assert block == base["DANINHAS"] and list(strategy_args) == ["VAAL"]
    config = _load(campaign / vaal_file)
    assert config.dataset.strategy_args.vaal == VAALArgs(epochs=epochs)
    assert config.dataset.strategy_args.waal == WAALArgs()
