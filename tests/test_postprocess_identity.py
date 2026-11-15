"""Identity guarantee of the feature post-processing (ADR 0019) for everything executed before it.

With `embedding.postprocess` absent or `{"method": "none"}`, `RepresentationStrategy` must behave
exactly as the code of tag `campaign-fixed-v4` (the code of the 486 executed runs and of the
`kmh_wide` runs in progress): the same arrays handed to the selection, the same ids, the same RNG
consumption, the same cache key. Covered here:

- `test_selection_receives_the_pre_change_array`: a spy selection records every array the strategy
  hands it over three rounds, for the three extractors (SSRAE full/spatial/spectral, VCTex,
  ImageNet), and compares it exactly with a verbatim copy of the v4 query path (stack, float32,
  SSRAE variant slice), with the default post-processor and with the one built from `"none"`;
- `test_matches_the_campaign_fixed_v4_class`: the same comparison against the v4
  `RepresentationStrategy` class itself, read from the tag with `git show` (skipped when the tag is
  not in the checkout): ids, arrays, selection RNG, global NumPy state, cache file names;
- `test_build_strategy_draws_no_random_number_for_any_method`: building the strategy with any
  post-processing leaves the global streams where `"none"` leaves them (round 0 is trained after the
  build and before the first query, so it is unaffected);
- `test_micro_run_with_and_without_explicit_none_is_identical` (`dataset`, `slow`): two
  `tools/trainer.py` runs on `DATA/daninhas_micro/` (key absent vs `"none"`) give identical queries,
  per-round predictions and metrics, and an `l2` run has the same round 0.

Real providers where they are cheap on 8x8 images (SSRAE, VCTex); the ImageNet case uses a stub
named `resnet_imagenet` in the fast suite and the real provider in a `slow` variant.
"""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dalmax.config.loader import load_experiment_config
from dalmax.config.schema import PostprocessConfig
from dalmax.embeddings.base import EmbeddingProvider
from dalmax.embeddings.cache import EmbeddingCache
from dalmax.embeddings.postprocess import IdentityPostprocessor
from dalmax.embeddings.registry import build_postprocessor
from dalmax.embeddings.ssrae_provider import SSRAEProvider
from dalmax.embeddings.variants import slice_embedding
from dalmax.embeddings.vctex_provider import VCTexProvider
from dalmax.query_strategies.registry import GENERIC_REPRESENTATION_NAME, build_strategy
from dalmax.query_strategies.representation import RepresentationStrategy
from dalmax.selection.base import SelectionStrategy
from dalmax.selection.hierarchical_kmeans import HierarchicalKMeansSelection

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_TAG = "campaign-fixed-v4"
N_POOL, N_INIT, N_QUERY, N_ROUNDS = 40, 5, 4, 3
HIERARCHY = {"n_clusters": [6, 3], "n_levels": 2, "sample_sizes": [2, 2]}


class _Dataset:
    """Duck-typed `dalmax.data.datasets.Data`: what `RepresentationStrategy` reads."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.X_train = rng.integers(0, 256, size=(N_POOL, 8, 8, 3), dtype=np.uint8)
        self.Z_train_paths = [f"DATA/fake/train/c{i % 3}/img{i:03d}.png" for i in range(N_POOL)]
        self.labeled_idxs = np.zeros(N_POOL, dtype=bool)
        self.labeled_idxs[:N_INIT] = True
        self.dataset_folder = "fake_identity_dataset"


class _StubImageNetProvider(EmbeddingProvider):
    """2048-d non-negative features (like pooled ResNet-50 activations), deterministic, no global RNG."""

    name = "resnet_imagenet"

    def __init__(self, q: None = None, device: str = "cpu") -> None:
        super().__init__(q=q, device=device)
        self._weights = np.random.default_rng(2048).normal(size=(8 * 8 * 3, 2048))

    def embed(self, images: np.ndarray) -> np.ndarray:
        flat = np.asarray(images, dtype=np.float64).reshape(len(images), -1) / 255.0
        return np.maximum(flat @ self._weights, 0.0).astype(np.float32)


class _Spy(SelectionStrategy):
    """Records (a copy of) every array and id vector handed to the wrapped selection."""

    def __init__(self) -> None:
        self.inner = HierarchicalKMeansSelection(hierarchy=dict(HIERARCHY), device="cpu")
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def select(self, embeddings: np.ndarray, ids: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
        self.calls.append((np.array(embeddings, copy=True), np.array(ids, copy=True)))
        return self.inner.select(embeddings, ids, budget, rng)


# (extractor, variant, provider factory)
CASES: dict[str, tuple[str, Any]] = {
    "ssrae-full": ("full", lambda: SSRAEProvider(q=2, device="cpu")),
    "ssrae-spatial": ("spatial", lambda: SSRAEProvider(q=2, device="cpu")),
    "ssrae-spectral": ("spectral", lambda: SSRAEProvider(q=2, device="cpu")),
    "vctex-full": ("full", lambda: VCTexProvider(q=(5, 17), device="cpu")),
    "resnet_imagenet-full": ("full", lambda: _StubImageNetProvider()),
}


def _config(variant: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(dataset=types.SimpleNamespace(
        name="DANINHAS", embedding=types.SimpleNamespace(variant=variant)))


def _run(cls: type, variant: str, provider: EmbeddingProvider, cache_root: Path, **extra: Any):
    """Three query rounds of `cls`; returns (strategy, spy, selected ids per round)."""
    dataset = _Dataset()
    spy = _Spy()
    strategy = cls(dataset, None, _config(variant), None, embedding_provider=provider, selection=spy,
                   rng=np.random.default_rng(11), cache=EmbeddingCache(root=cache_root), **extra)
    selected = []
    for _ in range(N_ROUNDS):
        ids = strategy.query(N_QUERY)
        dataset.labeled_idxs[ids] = True
        selected.append(ids)
    return strategy, spy, selected


def _v4_reference_arrays(provider: EmbeddingProvider, variant: str, calls: list) -> list[np.ndarray]:
    """The arrays the v4 query hands the selection, recomputed independently for the recorded id sets:
    embed the initial pool once, then per round stack the current ids' vectors as float32 and slice the
    SSRAE variant (verbatim logic of `campaign-fixed-v4:dalmax/query_strategies/representation.py`)."""
    dataset = _Dataset()
    initial = np.where(~dataset.labeled_idxs)[0]
    vectors = provider.embed(dataset.X_train[initial])
    pool = {int(i): vectors[row] for row, i in enumerate(initial)}
    out = []
    for _, ids in calls:
        embeddings = np.stack([pool[int(i)] for i in ids]).astype(np.float32)
        if provider.name == "ssrae" and variant != "full":
            embeddings = slice_embedding(embeddings, variant, provider.q)
        out.append(embeddings)
    return out


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("postprocessor", ["default", "explicit_none"])
def test_selection_receives_the_pre_change_array(case: str, postprocessor: str, tmp_path: Path) -> None:
    variant, factory = CASES[case]
    extra = {} if postprocessor == "default" else {"postprocessor": build_postprocessor(PostprocessConfig())}
    strategy, spy, _ = _run(RepresentationStrategy, variant, factory(), tmp_path / "cache", **extra)
    assert isinstance(strategy.postprocessor, IdentityPostprocessor)
    expected = _v4_reference_arrays(factory(), variant, spy.calls)
    assert len(spy.calls) == N_ROUNDS
    for (got, _), want in zip(spy.calls, expected, strict=True):
        assert got.dtype == want.dtype == np.float32 and got.shape == want.shape
        np.testing.assert_array_equal(got, want)
    assert strategy.diagnostics() == {}  # results.json keeps its schema: no strategy_diagnostics


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["full"])
def test_selection_receives_the_pre_change_array_real_imagenet_provider(variant: str, tmp_path: Path) -> None:
    from dalmax.embeddings.resnet_imagenet_provider import ResNetImageNetProvider

    provider = ResNetImageNetProvider(device="cpu")
    strategy, spy, _ = _run(RepresentationStrategy, variant, provider, tmp_path / "cache",
                            postprocessor=build_postprocessor(PostprocessConfig()))
    for (got, _), want in zip(spy.calls, _v4_reference_arrays(provider, variant, spy.calls), strict=True):
        assert got.shape[1] == 2048
        np.testing.assert_array_equal(got, want)


def _v4_representation_class() -> type:
    """`RepresentationStrategy` as committed at tag campaign-fixed-v4 (its imports are unchanged since)."""
    proc = subprocess.run(["git", "show", f"{V4_TAG}:dalmax/query_strategies/representation.py"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.skip(f"tag {V4_TAG} is not available in this checkout: {proc.stderr.strip()}")
    module = types.ModuleType("representation_campaign_fixed_v4")
    exec(compile(proc.stdout, "representation_campaign_fixed_v4.py", "exec"), module.__dict__)  # noqa: S102
    return module.RepresentationStrategy


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_the_campaign_fixed_v4_class(case: str, tmp_path: Path) -> None:
    v4_class = _v4_representation_class()
    variant, factory = CASES[case]
    np.random.seed(5)
    old, old_spy, old_ids = _run(v4_class, variant, factory(), tmp_path / "v4")
    old_global = np.random.get_state()[1].copy()
    np.random.seed(5)
    new, new_spy, new_ids = _run(RepresentationStrategy, variant, factory(), tmp_path / "new",
                                 postprocessor=build_postprocessor(PostprocessConfig()))
    for a, b in zip(old_ids, new_ids, strict=True):
        np.testing.assert_array_equal(a, b)
    for (old_emb, old_seen), (new_emb, new_seen) in zip(old_spy.calls, new_spy.calls, strict=True):
        np.testing.assert_array_equal(old_seen, new_seen)
        assert old_emb.dtype == new_emb.dtype
        np.testing.assert_array_equal(old_emb, new_emb)
    assert old.rng.bit_generator.state == new.rng.bit_generator.state  # same selection RNG consumption
    np.testing.assert_array_equal(np.random.get_state()[1], old_global)  # same global NumPy stream
    assert sorted(p.name for p in (tmp_path / "v4").iterdir()) == sorted(p.name for p in (tmp_path / "new").iterdir())
    assert old.diagnostics() == new.diagnostics() == {}


@pytest.mark.parametrize(("method", "pca_dim"), [("l2", None), ("zscore", None), ("pca_l2", 3)])
def test_non_identity_is_fitted_once_on_the_initial_pool(method: str, pca_dim: int | None, tmp_path: Path) -> None:
    """The fit happens at the first query on the initial pool's matrix; later rounds reuse it. The
    diagnostics carry one entry per query round."""
    config = PostprocessConfig(method=method, pca_dim=pca_dim)
    strategy, spy, _ = _run(RepresentationStrategy, "full", SSRAEProvider(q=2, device="cpu"), tmp_path / "cache",
                            postprocessor=build_postprocessor(config))
    raw = _v4_reference_arrays(SSRAEProvider(q=2, device="cpu"), "full", spy.calls)
    reference = build_postprocessor(config)
    reference.fit(raw[0])  # the initial unlabeled pool
    for (got, _), raw_round in zip(spy.calls, raw, strict=True):
        np.testing.assert_allclose(got, reference.transform(raw_round), rtol=1e-6, atol=1e-7)
    assert strategy.postprocessor.summary.n_fit == N_POOL - N_INIT
    diag = strategy.diagnostics()
    assert diag["postprocess_method"] == [method] * N_ROUNDS
    assert diag["postprocess_fit_pool_size"] == [N_POOL - N_INIT] * N_ROUNDS
    assert diag["postprocess_output_dim"][0] == (pca_dim or raw[0].shape[1])
    assert (diag["postprocess_explained_variance"][0] is None) == (method != "pca_l2")


def _micro_config(postprocess: PostprocessConfig):
    config = load_experiment_config(REPO_ROOT / "files_config/params_micro.json", "DANINHAS",
                                    strategy_name=GENERIC_REPRESENTATION_NAME, seed=3, n_init_labeled=5,
                                    n_query=4, n_round=1, dir_results="results/_unused/", device="cpu")
    embedding = dataclasses.replace(config.dataset.embedding, postprocess=postprocess)
    return dataclasses.replace(config, dataset=dataclasses.replace(config.dataset, embedding=embedding))


@pytest.mark.parametrize(
    "postprocess",
    [PostprocessConfig(method="l2"), PostprocessConfig(method="zscore"), PostprocessConfig(method="pca_l2", pca_dim=8)],
    ids=lambda p: p.method,
)
def test_build_strategy_draws_no_random_number_for_any_method(postprocess: PostprocessConfig) -> None:
    def build_and_capture(cfg: PostprocessConfig):
        np.random.seed(17)
        random.seed(17)
        torch.manual_seed(17)
        dataset = _Dataset()
        strategy = build_strategy(GENERIC_REPRESENTATION_NAME, dataset, None, _micro_config(cfg), None,
                                  np.random.default_rng(0))
        return strategy, (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone(),
                          strategy.rng.bit_generator.state)

    none_strategy, none_state = build_and_capture(PostprocessConfig())
    strategy, state = build_and_capture(postprocess)
    assert isinstance(none_strategy.postprocessor, IdentityPostprocessor)
    assert strategy.postprocessor.method == postprocess.method and not strategy.postprocessor.is_identity
    np.testing.assert_array_equal(state[0], none_state[0])
    assert state[1] == none_state[1]
    assert torch.equal(state[2], none_state[2])
    assert state[3] == none_state[3]


# --- end to end on the micro dataset ---------------------------------------------------------------

MICRO_ROOT = REPO_ROOT / "DATA" / "daninhas_micro"
PARAMS_KMH_MICRO = REPO_ROOT / "files_config/campaign/params_kmh_micro.json"
E2E_NIL, E2E_NQ, E2E_NR, E2E_SEED = 10, 5, 2, 1
QUERY_RE = r"Round (\d+) query_idxs \(sorted\): (\[[^\]]*\])"


def _write_params(path: Path, postprocess: dict | None) -> Path:
    params = json.loads(PARAMS_KMH_MICRO.read_text())
    if postprocess is not None:
        params["DANINHAS"]["embedding"]["postprocess"] = postprocess
    path.write_text(json.dumps(params, indent=4))
    return path


def _trainer(cwd: Path, params: Path, dir_results: str) -> tuple[Path, str]:
    """One `tools/trainer.py` run with `cwd` as working directory (its `results/` side effects, logs,
    embedding cache and `original_indices.txt`, stay under the test's tmp directory)."""
    cmd = [sys.executable, str(REPO_ROOT / "tools/trainer.py"), "--params_json", str(params), "--dataset_name",
           "DANINHAS", "--strategy_name", GENERIC_REPRESENTATION_NAME, "--n_init_labeled", str(E2E_NIL),
           "--n_query", str(E2E_NQ), "--n_round", str(E2E_NR), "--seed", str(E2E_SEED), "--device", "cpu",
           "--dir_results", dir_results]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=3600)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"trainer.py exited with {proc.returncode}:\n{output[-5000:]}"
    leaf = (cwd / dir_results / "daninhas_micro" / f"SEED_{E2E_SEED}"
            / f"NQ_{E2E_NQ}_NIL_{E2E_NIL}_NR_{E2E_NR}_NE_1" / GENERIC_REPRESENTATION_NAME)
    return leaf, output


def _read(leaf: Path, name: str) -> Any:
    return json.loads((leaf / name).read_text())


@pytest.mark.dataset
@pytest.mark.slow
def test_micro_run_with_and_without_explicit_none_is_identical(tmp_path: Path) -> None:
    import re

    if not (MICRO_ROOT / "train").is_dir():
        pytest.skip(f"{MICRO_ROOT} not present (run `make micro-dataset`)")
    (tmp_path / "DATA").symlink_to(REPO_ROOT / "DATA", target_is_directory=True)
    (tmp_path / "results").mkdir()
    absent = _write_params(tmp_path / "params_absent.json", None)
    explicit = _write_params(tmp_path / "params_none.json", {"method": "none"})
    l2 = _write_params(tmp_path / "params_l2.json", {"method": "l2"})

    leaf_a, out_a = _trainer(tmp_path, absent, "results/absent")
    leaf_n, out_n = _trainer(tmp_path, explicit, "results/none")
    leaf_l, out_l = _trainer(tmp_path, l2, "results/l2")

    # Absent vs explicit "none": identical queries, predictions and metrics, no diagnostics.
    assert re.findall(QUERY_RE, out_a) == re.findall(QUERY_RE, out_n) and len(re.findall(QUERY_RE, out_a)) == E2E_NR
    for name in ("round_queries.json", "round_predictions.json"):
        assert _read(leaf_a, name) == _read(leaf_n, name), name
    results_a, results_n = _read(leaf_a, "results.json"), _read(leaf_n, "results.json")
    assert results_a == results_n  # every metric list, exactly (and no strategy_diagnostics key)
    assert "strategy_diagnostics" not in results_a
    meta_a, meta_n = _read(leaf_a, "run_metadata.json")["config"], _read(leaf_n, "run_metadata.json")["config"]
    for meta in (meta_a, meta_n):
        assert meta["dataset"]["embedding"]["postprocess"] == {"method": "none", "pca_dim": None}
    assert {k: v for k, v in meta_a.items() if k not in ("dir_results", "params_json_path")} == {
        k: v for k, v in meta_n.items() if k not in ("dir_results", "params_json_path")}
    assert "Embedding post-processing" not in out_a + out_n

    # l2: round 0 (trained before the first query) is the same; the post-processing is reported.
    results_l = _read(leaf_l, "results.json")
    for key in ("all_acc", "all_f1_score", "all_f1_macro"):
        assert results_l[key][0] == results_a[key][0], key
    assert _read(leaf_l, "round_predictions.json")["predictions"][0] == _read(leaf_a, "round_predictions.json")[
        "predictions"][0]
    assert _read(leaf_l, "round_queries.json")["rounds"][0] == _read(leaf_a, "round_queries.json")["rounds"][0]
    assert results_l["strategy_diagnostics"]["postprocess_method"] == ["l2"] * E2E_NR
    assert "Embedding post-processing (ADR 0019): method='l2'" in out_l
