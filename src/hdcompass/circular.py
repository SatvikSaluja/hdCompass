"""Circular statistics on angles in radians."""

import numpy as np
import pynapple as nap

TWO_PI = 2 * np.pi


def wrap(a):
    """Wrap angles to [0, 2π).

    Parameters
    ----------
    a : array_like
        Angles in radians.

    Returns
    -------
    ndarray
        Angles in [0, 2π). ``np.mod`` can round tiny negatives up to exactly 2π; those map to 0.
    """
    r = np.mod(np.asarray(a, dtype=float), TWO_PI)
    return np.where(r >= TWO_PI, 0.0, r)


def circ_diff(a, b):
    """Signed circular difference ``a - b`` in (−π, π].

    Parameters
    ----------
    a, b : array_like
        Angles in radians.

    Returns
    -------
    ndarray
    """
    d = wrap(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.where(d > np.pi, d - TWO_PI, d)


def circ_mean(a, w=None):
    """Weighted circular mean in [0, 2π).

    Parameters
    ----------
    a : array_like
        Angles in radians.
    w : array_like, optional
        Non-negative weights, same shape as ``a``.

    Returns
    -------
    float
    """
    z = np.average(np.exp(1j * np.asarray(a, dtype=float)), weights=w)
    return float(wrap(np.angle(z)))


def circ_mae(a, b):
    """Mean absolute circular error between two angle arrays (radians)."""
    return float(np.mean(np.abs(circ_diff(a, b))))


def circ_corr(a, b):
    """Fisher–Lee circular–circular correlation.

    Uses the O(n) expansion of the pairwise definition
    ``Σ_{i<j} sin(a_i−a_j) sin(b_i−b_j) / sqrt(Σ sin²(a_i−a_j) Σ sin²(b_i−b_j))``.

    Parameters
    ----------
    a, b : array_like
        Paired angles in radians.

    Returns
    -------
    float
        In [−1, 1]; +1 for ``b = a + c``, −1 for ``b = −a + c``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    num = np.sum(sa * sb) * np.sum(ca * cb) - np.sum(sa * cb) * np.sum(ca * sb)
    da = np.sum(sa**2) * np.sum(ca**2) - np.sum(sa * ca) ** 2
    db = np.sum(sb**2) * np.sum(cb**2) - np.sum(sb * cb) ** 2
    return float(num / np.sqrt(da * db))


def epoch_diffs(angle_tsd):
    """Consecutive circular differences (rad) within each epoch — never across epoch gaps.

    Parameters
    ----------
    angle_tsd : nap.Tsd

    Returns
    -------
    ndarray
        Concatenated over epochs; ``len(angle_tsd) − n_epochs`` values when every epoch is
        non-empty.
    """
    out = [np.array([])]
    for i in range(len(angle_tsd.time_support)):
        v = angle_tsd.restrict(angle_tsd.time_support[i]).values
        out.append(circ_diff(v[1:], v[:-1]))
    return np.concatenate(out)


def unwrap_speed(angle_tsd):
    """Signed angular velocity (rad/s), computed per epoch — never across epoch gaps.

    Parameters
    ----------
    angle_tsd : nap.Tsd
        Angles in radians.

    Returns
    -------
    nap.Tsd
        Velocity at each sample (``np.gradient`` of the unwrapped angle within its epoch);
        take ``abs`` for speed. Epochs with fewer than two samples are dropped.
    """
    ts, vs = [], []
    for i in range(len(angle_tsd.time_support)):
        seg = angle_tsd.restrict(angle_tsd.time_support[i])
        if len(seg) < 2:
            continue
        ts.append(seg.t)
        vs.append(np.gradient(np.unwrap(seg.values), seg.t))
    if not ts:
        return nap.Tsd(t=np.array([]), d=np.array([]))
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vs), time_support=angle_tsd.time_support)
