"""Persistent-homology test for a ring (H1) in population activity."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Ring-test density filter frozen for the pipelines by the pre-registered checks in
# `scripts/validate.py decisions` (results/validation/decisions.json): real Mouse32 wake is a
# ring in 8/9 (k, keep) settings, a synthetic ring in 9/9, a no-ring control in 1/9, and the
# 200-shuffle p-value at this setting is 1/201. The functions below stay unfiltered by default.
FROZEN_DENSITY = {"density_keep": 0.5, "density_k": 15}


def density_filter(X, keep, k=15):
    """Keep the ``keep`` fraction of points with the smallest distance to their k-th neighbour.

    This is the codensity filtration of Carlsson et al. (2008, Int J Comput Vis 76:1–12):
    sparse points (transient, off-manifold states) are dropped before building a Rips complex,
    where a few of them can otherwise fill in the hole of a ring.
    """
    d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]  # [:, 0] is self
    return X[d <= np.quantile(d, keep)]


def h1_persistence(X, n_landmarks=800, n_pcs=10, seed=0, density_keep=None, density_k=15):
    """H1 persistence diagram of a point cloud.

    Parameters
    ----------
    X : array_like or nap.TsdFrame, shape (n, d)
        Points (e.g. population rate vectors).
    n_landmarks : int
        Farthest-point (greedy permutation) subsample size, done by ripser's ``n_perm``.
    n_pcs : int
        If ``d > n_pcs``, project onto the top ``n_pcs`` principal components first (denoising).
    seed : int
        Picks the farthest-point start index.
    density_keep : float, optional
        If given, apply :func:`density_filter` (after PCA) keeping this fraction of points.
    density_k : int
        Neighbour rank used by the density filter.

    Returns
    -------
    dict
        ``diagram`` (k, 2) birth/death pairs and ``lifetimes`` (k,), both sorted by lifetime
        descending, and ``ring_score`` (top lifetime / second; inf if only one H1 feature,
        0 if none).
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] > n_pcs:
        X = PCA(n_pcs, random_state=seed).fit_transform(X)
    if density_keep is not None:
        X = density_filter(X, density_keep, density_k)
    X = np.roll(X, -int(np.random.default_rng(seed).integers(len(X))), axis=0)
    kw = {"n_perm": n_landmarks} if len(X) > n_landmarks else {}
    dgm = ripser(X, maxdim=1, **kw)["dgms"][1]
    life = dgm[:, 1] - dgm[:, 0]
    order = np.argsort(life)[::-1]
    life = life[order]
    if len(life) == 0:
        score = 0.0
    elif len(life) == 1 or life[1] == 0:
        score = float("inf")
    else:
        score = float(life[0] / life[1])
    return {"diagram": dgm[order], "lifetimes": life, "ring_score": score}


def ring_score(X, n_landmarks=800, n_pcs=10, seed=0, density_keep=None, density_k=15):
    """Top H1 lifetime divided by the second (inf if only one H1 feature, 0 if none)."""
    return h1_persistence(X, n_landmarks, n_pcs, seed, density_keep, density_k)["ring_score"]


def _shuffled_score(X, shifts, kw):
    Xs = np.stack([np.roll(X[:, j], s) for j, s in enumerate(shifts)], axis=1)
    return ring_score(Xs, **kw)


_WORKER = {}


def _init_worker(X, kw):
    _WORKER.update(X=X, kw=kw)


def _worker_score(shifts):
    return _shuffled_score(_WORKER["X"], shifts, _WORKER["kw"])


def shuffle_null(
    rates,
    n_shuffles=20,
    seed=0,
    n_landmarks=800,
    n_pcs=10,
    density_keep=None,
    density_k=15,
    n_jobs=1,
):
    """Null distribution of :func:`ring_score` with each unit circularly shifted independently.

    Shifting keeps every unit's rate statistics but breaks the co-activation that forms a ring.
    Every shuffle is scored exactly like the data (same PCA, density filter and landmarks).

    Parameters
    ----------
    rates : array_like or nap.TsdFrame, shape (time, units)
    n_shuffles : int
    seed : int
        Seeds the shifts (drawn up front, so the result does not depend on ``n_jobs``).
    density_keep, density_k
        As in :func:`h1_persistence`.
    n_jobs : int
        Worker processes (``spawn`` start method: JAX may be loaded in the parent, and forking
        a multithreaded process can deadlock).

    Returns
    -------
    ndarray, shape (n_shuffles,)
    """
    X = np.asarray(rates, dtype=float)
    rng = np.random.default_rng(seed)
    shifts = [rng.integers(len(X), size=X.shape[1]) for _ in range(n_shuffles)]
    kw = {
        "n_landmarks": n_landmarks,
        "n_pcs": n_pcs,
        "seed": seed,
        "density_keep": density_keep,
        "density_k": density_k,
    }
    if n_jobs == 1:
        return np.array([_shuffled_score(X, s, kw) for s in shifts])
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        n_jobs, mp_context=ctx, initializer=_init_worker, initargs=(X, kw)
    ) as ex:
        return np.array(list(ex.map(_worker_score, shifts)))


def is_ring(
    X, null, alpha=0.05, n_landmarks=800, n_pcs=10, seed=0, density_keep=None, density_k=15
):
    """Decide whether ``X`` carries a ring relative to a shuffle null.

    Parameters
    ----------
    X : array_like, nap.TsdFrame or float
        The unshuffled data given to :func:`shuffle_null`, or its precomputed ring score.
    null : ndarray
        Output of :func:`shuffle_null` (computed with the same settings).
    alpha : float

    Returns
    -------
    ring : bool
        ``ring_score(X)`` exceeds the ``1 − alpha`` quantile of the null, taken as an actual
        null value (``method="higher"``, never interpolated, so with ``n ≤ 1/alpha`` shuffles
        the score must beat every shuffle). The decision uses the quantile, not
        ``p < alpha``, because with few shuffles the p-value floor ``1 / (n + 1)`` can exceed
        ``alpha``.
    p : float
        Permutation p-value ``(1 + #{null ≥ score}) / (n + 1)``.
    """
    null = np.asarray(null, dtype=float)
    if np.ndim(X) == 0:
        score = float(X)
    else:
        score = ring_score(X, n_landmarks, n_pcs, seed, density_keep, density_k)
    p = (1 + np.sum(null >= score)) / (len(null) + 1)
    return bool(score > np.quantile(null, 1 - alpha, method="higher")), float(p)
