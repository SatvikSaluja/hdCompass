"""Low-dimensional embedding of population activity and the angle around its ring."""

import numpy as np
import pynapple as nap
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from .circular import wrap


def embed(rates, method="isomap", n_components=2, n_neighbors=15, max_points=4000, seed=0):
    """Embed population rates: PCA to 10 dims, then Isomap fitted on a subsample.

    Parameters
    ----------
    rates : nap.TsdFrame
        Output of :func:`hdcompass.features.population_rates`.
    method : {"isomap", "pca"}
        ``"pca"`` returns the leading principal components (no Isomap).
    n_components : int
    n_neighbors : int
        Isomap neighbourhood size.
    max_points : int
        Isomap is fitted on at most this many random points, then applied to all.
    seed : int

    Returns
    -------
    nap.TsdFrame
        ``(time, n_components)`` on the same index and support as ``rates``.
    """
    if method not in ("isomap", "pca"):
        raise ValueError(f"unknown method {method!r}")
    X = np.asarray(rates, dtype=float)
    X = PCA(min(10, X.shape[1]), random_state=seed).fit_transform(X)
    if method == "pca":
        Y = X[:, :n_components]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), min(max_points, len(X)), replace=False)
        iso = Isomap(n_neighbors=n_neighbors, n_components=n_components).fit(X[idx])
        Y = iso.transform(X)
    return nap.TsdFrame(t=rates.t, d=Y, time_support=rates.time_support)


def ring_angle(embedding):
    """Angle of each point around a circle fitted to the first two embedding dimensions.

    The circle is the algebraic least-squares (Kåsa) fit of ``x² + y² + a·x + b·y + c = 0``
    on mean-centred coordinates; the angle is ``atan2`` about the fitted centre.

    Parameters
    ----------
    embedding : nap.TsdFrame

    Returns
    -------
    nap.Tsd
        Angles in [0, 2π), same index and support as ``embedding``.
    """
    xy = np.asarray(embedding, dtype=float)[:, :2]
    xy = xy - xy.mean(axis=0)
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    (a, b, _), *_ = np.linalg.lstsq(A, -(x**2 + y**2), rcond=None)
    theta = wrap(np.arctan2(y + b / 2, x + a / 2))
    return nap.Tsd(t=embedding.t, d=theta, time_support=embedding.time_support)
