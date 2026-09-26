"""Persistent-homology test for a ring (H1) in population activity."""

import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA


def h1_persistence(X, n_landmarks=800, n_pcs=10, seed=0):
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


def ring_score(X, n_landmarks=800, n_pcs=10, seed=0):
    """Top H1 lifetime divided by the second (inf if only one H1 feature, 0 if none)."""
    return h1_persistence(X, n_landmarks, n_pcs, seed)["ring_score"]


def shuffle_null(rates, n_shuffles=20, seed=0, n_landmarks=800, n_pcs=10):
    """Null distribution of :func:`ring_score` with each unit circularly shifted independently.

    Shifting keeps every unit's rate statistics but breaks the co-activation that forms a ring.

    Parameters
    ----------
    rates : array_like or nap.TsdFrame, shape (time, units)
    n_shuffles : int
    seed : int

    Returns
    -------
    ndarray, shape (n_shuffles,)
    """
    X = np.asarray(rates, dtype=float)
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shifts = rng.integers(len(X), size=X.shape[1])
        Xs = np.stack([np.roll(X[:, j], s) for j, s in enumerate(shifts)], axis=1)
        null[i] = ring_score(Xs, n_landmarks, n_pcs, seed)
    return null


def is_ring(X, null, alpha=0.05, n_landmarks=800, n_pcs=10, seed=0):
    """Decide whether ``X`` carries a ring relative to a shuffle null.

    Parameters
    ----------
    X : array_like, nap.TsdFrame or float
        The unshuffled data given to :func:`shuffle_null`, or its precomputed ring score.
    null : ndarray
        Output of :func:`shuffle_null`.
    alpha : float

    Returns
    -------
    ring : bool
        ``ring_score(X)`` exceeds the ``1 − alpha`` quantile of the null. The decision uses the
        quantile, not ``p < alpha``, because with few shuffles the p-value floor
        ``1 / (n + 1)`` can exceed ``alpha``.
    p : float
        Permutation p-value ``(1 + #{null ≥ score}) / (n + 1)``.
    """
    null = np.asarray(null, dtype=float)
    score = float(X) if np.isscalar(X) else ring_score(X, n_landmarks, n_pcs, seed)
    p = (1 + np.sum(null >= score)) / (len(null) + 1)
    return bool(score > np.quantile(null, 1 - alpha)), float(p)
