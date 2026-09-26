"""Angular tuning curves, from tracked head angle or from the unsupervised ring angle."""

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

from .circular import TWO_PI


def fit_tuning(spikes, angle, ep, n_bins=60, smooth_bins=2):
    """Tuning curves on ``n_bins`` equal bins of [0, 2π).

    Parameters
    ----------
    spikes : nap.TsGroup
    angle : nap.Tsd
        Angle (rad) in [0, 2π).
    ep : nap.IntervalSet
        Epochs to use; intersected with ``angle.time_support``.
    n_bins : int
        Bin ``i`` is centred on ``(i + 0.5)·2π/n_bins``.
    smooth_bins : float
        Std (in bins) of the circular Gaussian smoothing; 0 disables it.

    Returns
    -------
    ndarray, shape (units, n_bins)
        Rates in Hz, rows in ``spikes.keys()`` order, unvisited bins filled with the unit's mean,
        floored at 0.05 Hz.
    """
    tc = nap.compute_tuning_curves(
        data=spikes,
        features=angle,
        bins=n_bins,
        epochs=ep.intersect(angle.time_support),
        range=(0.0, TWO_PI),
        feature_names=["angle"],
    )
    tc = np.asarray(tc, dtype=float)
    tc = np.where(np.isnan(tc), np.nanmean(tc, axis=1, keepdims=True), tc)
    if smooth_bins > 0:
        tc = gaussian_filter1d(tc, smooth_bins, axis=1, mode="wrap")
    return np.maximum(tc, 0.05)


def tuning_from_ring(spikes, ring_angle, ep, n_bins=60, smooth_bins=2):
    """:func:`fit_tuning` with the unsupervised ring angle in place of tracked head angle.

    This closes the loop with no behaviour at all: ring angle → tuning → decoder.
    """
    return fit_tuning(spikes, ring_angle, ep, n_bins, smooth_bins)
