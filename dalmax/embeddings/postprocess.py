"""Feature post-processing of pool embeddings before selection (ADR 0019).

`RepresentationStrategy` hands the selection module (e.g. the Euclidean
hierarchical k-means of KMH) the raw vectors of its embedding provider. This
module changes ONLY their geometry, never the features themselves: a
`FeaturePostprocessor` is fitted once per run on the embeddings of the INITIAL
unlabeled pool (no labels, no test images, nothing from the classifier) and then
transforms the embeddings of the current unlabeled pool in every round (the pool
only shrinks, so the statistics stay those of the initial pool).

Methods (`dalmax.config.schema.PostprocessConfig.method`, registry
`dalmax.embeddings.registry.POSTPROCESS_REGISTRY`):

- `"none"`: `IdentityPostprocessor`, the default. `fit` stores nothing and
  `transform` returns its argument itself (same object, no copy, no cast), so a
  run without the params key hands the selection exactly the array it handed it
  before this module existed.
- `"l2"`: `L2Postprocessor`, every vector divided by its Euclidean norm; a
  zero vector stays zero (zero-norm guard). Nothing is fitted (the fit only
  records the dimensions and how many pool vectors had zero norm).
- `"zscore"`: `ZScorePostprocessor`, every dimension standardised with the mean
  and the population standard deviation (`ddof=0`) of the initial pool. A
  dimension that is constant over the pool (every value equal, an exact test) is
  centred on that value and not scaled, so it maps to exactly 0 (zero-variance
  guard).
- `"pca_l2"`: `PCAL2Postprocessor`, centred on the pool mean, projected on the
  first `pca_dim` principal components of the pool, then `"l2"`. The
  components come from a full, deterministic SVD of the centred pool
  (`numpy.linalg.svd`, no randomised solver), with a fixed sign convention (the
  largest-magnitude loading of every component is positive). The explained
  variance ratio of the kept components is reported.

Numerics: every fit and transform is computed in float64; the transformed
vectors are returned as float32 (the dtype the selection modules work in).

RNG: nothing here draws a random number (no NumPy legacy or `Generator` draw,
no torch call), so a post-processed run consumes exactly the random streams of
an identity run and its round 0 (trained before the first query) is unchanged
(`tests/test_postprocess_identity.py`).

Errors: a `pca_dim` larger than the feature dimension or than the initial pool
raises `dalmax.config.schema.ConfigError` at the fit (the first query); a
malformed input (not a 2-D finite matrix, another dimension than the fitted one,
`transform` before `fit`) raises `PostprocessError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from dalmax.config.schema import POSTPROCESS_IDENTITY, ConfigError


class PostprocessError(ValueError):
    """A feature post-processor received input it cannot transform (shape, non-finite values, unfitted)."""


@dataclass(frozen=True)
class PostprocessSummary:
    """What a fit computed, for the run log and `strategy_diagnostics` (reporting only).

    `guarded` counts the inputs a guard handled at the fit: zero-norm pool vectors
    (`l2`, and the projected vectors of `pca_l2`) or constant dimensions (`zscore`).
    `explained_variance_ratio` (per kept component) and its sum are set for `pca_l2`
    only.
    """

    method: str
    n_fit: int
    input_dim: int
    output_dim: int
    guarded: int = 0
    explained_variance_ratio: tuple[float, ...] | None = None

    @property
    def explained_variance(self) -> float | None:
        """Total explained variance ratio of the kept components (`pca_l2`), else None."""
        if self.explained_variance_ratio is None:
            return None
        return float(sum(self.explained_variance_ratio))

    def describe(self) -> str:
        """One log line."""
        text = (
            f"method={self.method!r}, fitted on the initial unlabeled pool of {self.n_fit} images, "
            f"dim {self.input_dim} -> {self.output_dim}, guarded={self.guarded}"
        )
        if self.explained_variance_ratio is not None:
            head = ", ".join(f"{r:.4f}" for r in self.explained_variance_ratio[:5])
            text += (
                f", explained variance of the {len(self.explained_variance_ratio)} kept components "
                f"{self.explained_variance:.4f} (first: {head})"
            )
        return text


def _as_float64_matrix(vectors: np.ndarray, where: str) -> np.ndarray:
    """`vectors` as a 2-D finite float64 array (a copy), or `PostprocessError`."""
    matrix = np.asarray(vectors)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise PostprocessError(f"{where}: expected a non-empty (N, D) matrix, got shape {matrix.shape}")
    matrix = matrix.astype(np.float64, copy=True)
    if not np.all(np.isfinite(matrix)):
        raise PostprocessError(f"{where}: the embeddings contain NaN or infinite values")
    return matrix


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    """Every row divided by its Euclidean norm; a zero row stays zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


class FeaturePostprocessor(ABC):
    """Fit on the initial unlabeled pool once, then transform any subset of it.

    Subclasses implement `_fit` (statistics from a float64 pool matrix, returning
    the summary) and `_transform` (float64 in, float64 out); the base class
    validates the input, enforces fit-before-transform and the fitted dimension,
    and casts the result to float32.
    """

    method: ClassVar[str]

    def __init__(self) -> None:
        self._summary: PostprocessSummary | None = None

    @classmethod
    def from_config(cls, config: Any) -> FeaturePostprocessor:
        """Build from a `dalmax.config.schema.PostprocessConfig` (or anything with its fields)."""
        return cls()

    @property
    def is_identity(self) -> bool:
        return False

    @property
    def summary(self) -> PostprocessSummary:
        if self._summary is None:
            raise PostprocessError(f"{type(self).__name__}: fit() was not called")
        return self._summary

    def fit(self, pool: np.ndarray) -> PostprocessSummary:
        """Fit on the embeddings of the initial unlabeled pool (rows = images)."""
        self._summary = self._fit(_as_float64_matrix(pool, f"{type(self).__name__}.fit"))
        return self._summary

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Transform embeddings of (a subset of) the fitted pool; float32 result, one row per input row."""
        summary = self.summary
        matrix = _as_float64_matrix(vectors, f"{type(self).__name__}.transform")
        if matrix.shape[1] != summary.input_dim:
            raise PostprocessError(
                f"{type(self).__name__}.transform: dimension {matrix.shape[1]} != fitted {summary.input_dim}"
            )
        return self._transform(matrix).astype(np.float32)

    @abstractmethod
    def _fit(self, pool: np.ndarray) -> PostprocessSummary:
        raise NotImplementedError

    @abstractmethod
    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityPostprocessor(FeaturePostprocessor):
    """`"none"`: no post-processing. `transform` returns its argument itself (no copy, no cast)."""

    method = POSTPROCESS_IDENTITY

    @property
    def is_identity(self) -> bool:
        return True

    def fit(self, pool: np.ndarray) -> PostprocessSummary:
        # Reads only the shape: no validation, no copy, nothing stored that transform would use.
        n_fit, dim = np.shape(pool)
        self._summary = PostprocessSummary(self.method, n_fit=n_fit, input_dim=dim, output_dim=dim)
        return self._summary

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return vectors

    def _fit(self, pool: np.ndarray) -> PostprocessSummary:  # pragma: no cover - fit is overridden
        raise NotImplementedError

    def _transform(self, matrix: np.ndarray) -> np.ndarray:  # pragma: no cover - transform is overridden
        raise NotImplementedError


class L2Postprocessor(FeaturePostprocessor):
    """`"l2"`: unit Euclidean norm per vector (zero vectors stay zero)."""

    method = "l2"

    def _fit(self, pool: np.ndarray) -> PostprocessSummary:
        zero_norm = int(np.count_nonzero(np.linalg.norm(pool, axis=1) == 0))
        n_fit, dim = pool.shape
        return PostprocessSummary(self.method, n_fit=n_fit, input_dim=dim, output_dim=dim, guarded=zero_norm)

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        return _l2_rows(matrix)


class ZScorePostprocessor(FeaturePostprocessor):
    """`"zscore"`: per-dimension standardisation with the initial pool's mean and std (constant dims -> 0)."""

    method = "zscore"

    def __init__(self) -> None:
        super().__init__()
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def _fit(self, pool: np.ndarray) -> PostprocessSummary:
        constant = np.ptp(pool, axis=0) == 0  # exact: every pool value of the dimension is equal
        # A constant dimension is centred on its own value (exactly 0 afterwards, not the rounding of a mean)
        # and left unscaled; every other dimension gets the pool mean and population std (> 0 there).
        self._center = np.where(constant, pool[0], pool.mean(axis=0))
        self._scale = np.where(constant, 1.0, pool.std(axis=0))
        n_fit, dim = pool.shape
        return PostprocessSummary(self.method, n_fit=n_fit, input_dim=dim, output_dim=dim,
                                  guarded=int(np.count_nonzero(constant)))

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self._center) / self._scale


class PCAL2Postprocessor(FeaturePostprocessor):
    """`"pca_l2"`: centre on the pool mean, project on the first `pca_dim` principal components, then L2."""

    method = "pca_l2"

    def __init__(self, pca_dim: int) -> None:
        super().__init__()
        if isinstance(pca_dim, bool) or not isinstance(pca_dim, int) or pca_dim <= 0:
            raise ConfigError(f"pca_l2: pca_dim must be an int > 0, got {pca_dim!r}")
        self.pca_dim = pca_dim
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: Any) -> PCAL2Postprocessor:
        return cls(pca_dim=config.pca_dim)

    def _fit(self, pool: np.ndarray) -> PostprocessSummary:
        n_fit, dim = pool.shape
        if self.pca_dim > min(n_fit, dim):
            raise ConfigError(
                f"pca_l2: pca_dim={self.pca_dim} exceeds min(pool size, feature dimension) = "
                f"min({n_fit}, {dim}) of the initial unlabeled pool"
            )
        self._mean = pool.mean(axis=0)
        centred = pool - self._mean
        # Full deterministic SVD (no randomised solver): rows of vt are the principal axes, by decreasing
        # singular value.
        _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
        components = vt[: self.pca_dim]
        # Sign convention (an SVD fixes each axis only up to sign): largest-magnitude loading positive.
        pivots = components[np.arange(self.pca_dim), np.argmax(np.abs(components), axis=1)]
        self._components = components * np.where(pivots < 0, -1.0, 1.0)[:, None]
        variances = singular_values**2
        total = float(variances.sum())
        ratios = variances[: self.pca_dim] / total if total > 0 else np.zeros(self.pca_dim)
        projected_norms = np.linalg.norm(centred @ self._components.T, axis=1)
        return PostprocessSummary(
            self.method, n_fit=n_fit, input_dim=dim, output_dim=self.pca_dim,
            guarded=int(np.count_nonzero(projected_norms == 0)),
            explained_variance_ratio=tuple(float(r) for r in ratios),
        )

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        return _l2_rows((matrix - self._mean) @ self._components.T)
