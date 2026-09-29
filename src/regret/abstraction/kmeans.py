"""Weighted k-means (k-means++ seeding, Lloyd iterations) on dense float32 features.

Distances use ||x||^2 - 2 x.c + ||c||^2, so the heavy lifting is one BLAS matrix product per
chunk of rows. Deterministic for a given seed on a given machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

F32 = NDArray[np.float32]


@dataclass
class KMeansResult:
    centroids: F32
    labels: NDArray[np.int32]
    inertia: float  # weighted sum of squared distances to the assigned centroid
    iterations: int


def _assign(x: F32, c: F32, chunk: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    c_sq = (c * c).sum(1)
    labels = np.empty(len(x), np.int32)
    dist = np.empty(len(x), np.float32)
    for s in range(0, len(x), chunk):
        xs = x[s : s + chunk]
        d = c_sq[None, :] - 2.0 * (xs @ c.T)  # + ||x||^2, which doesn't change the argmin
        lab = d.argmin(1)
        labels[s : s + chunk] = lab
        dist[s : s + chunk] = np.maximum(d[np.arange(len(xs)), lab] + (xs * xs).sum(1), 0.0)
    return labels, dist


def _plus_plus(x: F32, w: NDArray[np.float64], k: int, rng: np.random.Generator) -> F32:
    """k-means++ seeding: each new center is drawn with probability ∝ weight * distance²."""
    centers = [x[rng.choice(len(x), p=w / w.sum())]]
    d2 = ((x - centers[0]) ** 2).sum(1).astype(np.float64)
    for _ in range(1, k):
        p = w * d2
        total = p.sum()
        i = rng.choice(len(x), p=p / total) if total > 0 else rng.integers(len(x))
        centers.append(x[i])
        d2 = np.minimum(d2, ((x - x[i]) ** 2).sum(1))
    return np.stack(centers).astype(np.float32)


def kmeans(
    x: F32,
    k: int,
    *,
    weights: NDArray[np.floating] | None = None,
    seed: int = 0,
    max_iter: int = 50,
    tol: float = 1e-4,
    init_sample: int = 100_000,
    chunk: int = 1 << 16,
) -> KMeansResult:
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(x)
    if n < k:
        raise ValueError(f"need at least k={k} rows, got {n}")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)

    sample = rng.choice(n, size=min(n, init_sample), replace=False, p=w / w.sum())
    centroids = _plus_plus(x[np.sort(sample)], np.ones(len(sample)), k, rng)

    prev = np.inf
    labels = np.zeros(n, np.int32)
    inertia = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        labels, dist = _assign(x, centroids, chunk)
        inertia = float((w * dist).sum())
        # Weighted means per cluster.
        mass = np.bincount(labels, weights=w, minlength=k)
        sums = np.stack(
            [np.bincount(labels, weights=w * x[:, j], minlength=k) for j in range(x.shape[1])],
            axis=1,
        )
        empty = mass == 0
        centroids = np.where(
            empty[:, None], centroids, sums / np.maximum(mass, 1e-30)[:, None]
        ).astype(np.float32)
        # Re-seed empty clusters at the worst-fit points.
        for c, i in zip(np.flatnonzero(empty), np.argsort(-w * dist), strict=False):
            centroids[c] = x[i]
        log.debug("k-means iter %d inertia %.6g", it, inertia)
        if np.isfinite(prev) and prev - inertia <= tol * prev and not empty.any():
            break
        prev = inertia
    labels, dist = _assign(x, centroids, chunk)
    return KMeansResult(centroids, labels, float((w * dist).sum()), it)
