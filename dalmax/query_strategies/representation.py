"""`RepresentationStrategy`: the single class backing every embedding-based
query strategy (`SSRAEKmeansSampling`, `VCTexKmeansSampling`,
`SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`, and the new generic
`RepresentationStrategy` CLI choice used by the Phase 3 ablations).

It factors out what used to be duplicated across
`core/query_strategies/ssrae_kmeans_sampling.py` (flat k-means, closest to
centroid) and `core/query_strategies/ssl_ssrae_sampling.py` (`SSLStrategy`:
hierarchical k-means) into "compute/cache embeddings for the current pool"
(`dalmax.embeddings`) + "select a batch given embeddings" (`dalmax.selection`),
composed via dependency injection (`.claude/rules/code-quality.md`: "no
`setattr` magic ... pass dependencies through constructors").

Subclasses `core.query_strategies.strategy.Strategy` (unmodified, vendored)
so `train`/`predict`/`update`/`info`/`save_model` keep working exactly as for
every other strategy; only `query` is overridden here.

## Embedding lifecycle (replicates legacy semantics, cleaner)

The legacy code extracts SSRAE/VCTex features exactly once, over the
*initial* unlabeled pool (`Data.initialize_labels` calls
`create_feature_maps_ssrae`/`_vctex` right after the initial split), then
mutates `self.dataset.features_dict` in place every round, deleting the
ids it just selected (`ssrae_kmeans_sampling.py:47-50`,
`ssl_ssrae_sampling.py:99-102`) so the next round's k-means only sees what
remains unlabeled.

This class achieves the same effect without any mutation of
`dataset.features_dict` (which it never touches) and without needing a
special one-time "extract now" call: on its *first* `query()` call, it
computes (or loads from the keyed `EmbeddingCache`) embeddings for the pool
that is unlabeled *at that moment* — identical to the initial unlabeled
pool, since `initialize_labels` has already run by the time `query()` is
first called — and caches that `{id: embedding}` mapping for the lifetime
of the strategy instance. Every subsequent call simply restricts to
`dataset.labeled_idxs`'s current unlabeled ids (a shrinking subset of that
same initial pool, since ids only ever move from unlabeled to labeled) and
looks up their already-computed embeddings — equivalent to the legacy
"extract once, delete selected ids" dance, cheaper, and with no shared
mutable state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.query_strategies.strategy import Strategy
from dalmax.embeddings.base import EmbeddingProvider
from dalmax.embeddings.cache import EmbeddingCache, compute_pool_embeddings, pool_hash
from dalmax.embeddings.variants import slice_embedding
from dalmax.selection.base import SelectionStrategy

if TYPE_CHECKING:
    from dalmax.config.schema import ExperimentConfig


class RepresentationStrategy(Strategy):
    """Embedding provider + selection strategy, composed behind `Strategy.query`.

    Parameters
    ----------
    dataset, net, logger:
        Same as every other `Strategy` subclass.
    config:
        The resolved `ExperimentConfig` for this run (read-only: used for
        `config.dataset.name`/`embedding.variant`, never mutated).
    embedding_provider:
        The `EmbeddingProvider` instance to use (already constructed with
        its `q`/`device`); see `dalmax.query_strategies.registry.build_strategy`
        for how this is selected per `--strategy_name`.
    selection:
        The `SelectionStrategy` instance to use.
    rng:
        `np.random.Generator` forwarded to `selection.select(...)` on every
        call — see `.claude/rules/reproducibility.md`.
    cache:
        Optional `EmbeddingCache` override (defaults to the standard
        `results/cache/embeddings` root); tests inject a `tmp_path`-scoped
        cache here.
    """

    def __init__(
        self,
        dataset,
        net,
        config: ExperimentConfig,
        logger,
        *,
        embedding_provider: EmbeddingProvider,
        selection: SelectionStrategy,
        rng: np.random.Generator,
        cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(dataset, net, logger)
        self.config = config
        self.embedding_provider = embedding_provider
        self.selection = selection
        self.rng = rng
        self.cache = cache if cache is not None else EmbeddingCache()
        # Populated lazily on the first `query()` call (see module docstring).
        self._pool_embeddings: dict[int, np.ndarray] | None = None

    def query(self, n: int) -> np.ndarray:
        current_unlabeled_ids = np.where(~self.dataset.labeled_idxs)[0]

        if self._pool_embeddings is None:
            self._pool_embeddings = self._compute_initial_pool_embeddings(
                current_unlabeled_ids
            )

        embeddings = np.stack(
            [self._pool_embeddings[int(img_id)] for img_id in current_unlabeled_ids]
        ).astype(np.float32)

        if self.embedding_provider.name == "ssrae" and self.config.dataset.embedding.variant != "full":
            embeddings = slice_embedding(
                embeddings, self.config.dataset.embedding.variant, self.embedding_provider.q
            )

        # Observability: the embedding matrix shape actually handed to
        # selection depends on extractor/q/variant in ways that are easy to
        # get wrong (see the HIGH finding in `dalmax.config.loader`/
        # `dalmax.query_strategies.registry` about `q` defaulting silently);
        # logging it every query makes a wrong shape visible in the run log
        # instead of only surfacing as a downstream metric anomaly. `logger`
        # may be `None` in lightweight tests that construct
        # `RepresentationStrategy` directly (see `tests/test_representation_
        # strategy.py`), so this is best-effort, matching `Strategy`'s own
        # `logger.warning(...)` style elsewhere.
        if self.logger is not None:
            self.logger.warning(
                "Embedding matrix shape after slicing: "
                f"extractor={self.embedding_provider.name!r}, "
                f"q={self.embedding_provider.q!r}, "
                f"variant={self.config.dataset.embedding.variant!r}, "
                f"shape={embeddings.shape}"
            )

        selected_ids = self.selection.select(
            embeddings, current_unlabeled_ids, n, self.rng
        )
        return np.asarray(selected_ids, dtype=np.int64)

    def _compute_initial_pool_embeddings(
        self, initial_unlabeled_ids: np.ndarray
    ) -> dict[int, np.ndarray]:
        images = self.dataset.X_train[initial_unlabeled_ids]
        # NOTE: not `getattr(self.dataset, "dataset_folder", self.config.dataset.name)` —
        # getattr's default argument is evaluated eagerly regardless of
        # whether the attribute is found, so that form would require every
        # caller's `config` to have a `dataset.name` even when `dataset`
        # already has `dataset_folder` (as `utils.data.Data` always does).
        if hasattr(self.dataset, "dataset_folder"):
            dataset_name = self.dataset.dataset_folder
        else:
            dataset_name = self.config.dataset.name
        # Always cache/load the "full" variant (see dalmax.embeddings.cache
        # module docstring); a non-"full" config variant is applied by
        # slicing the loaded "full" embeddings in `query`, never by caching
        # a sliced variant directly.
        key = self.embedding_provider.key(
            dataset=dataset_name,
            split="train",
            pool_hash=pool_hash(initial_unlabeled_ids),
            variant="full",
        )
        return compute_pool_embeddings(
            self.embedding_provider, images, initial_unlabeled_ids, self.cache, key
        )
