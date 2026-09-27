"""Population rate features, binned and smoothed strictly within epochs."""

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d


def bin_counts(spikes, ep, bin_size):
    """Spike counts in bins that never cross an epoch boundary.

    Each epoch ``[s, e)`` gets ``floor((e − s) / bin_size)`` left-closed bins starting at ``s``;
    the trailing partial bin is dropped. Epochs shorter than one bin yield empty arrays, so the
    output stays aligned with ``ep``.

    Parameters
    ----------
    spikes : nap.TsGroup
    ep : nap.IntervalSet
    bin_size : float
        Seconds.

    Returns
    -------
    list of (t, counts)
        One per epoch: bin centres ``(n,)`` and counts ``(n, units)`` in ``spikes.keys()`` order.
    """
    times = [spikes[k].t for k in spikes.keys()]
    out = []
    for s, e in zip(ep.start, ep.end, strict=True):
        n = max(int(np.floor((e - s) / bin_size + 1e-9)), 0)
        edges = s + np.arange(n + 1) * bin_size
        counts = np.stack([np.diff(np.searchsorted(st, edges)) for st in times], axis=1)
        counts = counts.reshape(n, len(times))
        out.append((edges[:-1] + bin_size / 2, counts))
    return out


def units_above(spikes, ep, min_rate):
    """Keys of units whose mean rate on ``ep`` is at least ``min_rate`` Hz."""
    tot = ep.tot_length()
    return [k for k in spikes.keys() if len(spikes[k].restrict(ep)) / tot >= min_rate]


def population_rates(spikes, ep, bin_size=0.1, smooth_std=0.1, min_rate=1.0):
    """Z-scored, sqrt-transformed, smoothed population rates.

    Steps: drop units whose rate on ``ep`` is below ``min_rate``; count per epoch
    (:func:`bin_counts`); ``sqrt``; Gaussian smooth each epoch separately
    (window ``6·smooth_std``, nearest-value padding); drop bins whose centre lies within
    ``2·smooth_std`` of an epoch edge; z-score each unit over all kept bins.

    The edge drop is the guard for pynapple#623: pynapple's ``smooth`` zero-pads at epoch edges,
    and even with better padding the edge bins' smoothing windows are mostly extrapolated.

    Parameters
    ----------
    spikes : nap.TsGroup
    ep : nap.IntervalSet
    bin_size, smooth_std : float
        Seconds. ``smooth_std=0`` disables smoothing and the edge drop.
    min_rate : float
        Hz.

    Returns
    -------
    nap.TsdFrame
        Bins × kept units (columns are unit ids); time support is the trimmed epochs.
    """
    keep = units_above(spikes, ep, min_rate)
    if not keep:
        raise ValueError(f"no unit reaches min_rate={min_rate} Hz on this epoch set")
    spikes = spikes[keep]

    guard = 2 * smooth_std
    ts, xs, starts, ends = [], [], [], []
    for (t, c), s, e in zip(bin_counts(spikes, ep, bin_size), ep.start, ep.end, strict=True):
        if len(t) == 0:
            continue
        x = np.sqrt(c.astype(float))
        if smooth_std > 0:
            x = gaussian_filter1d(x, smooth_std / bin_size, axis=0, mode="nearest", truncate=3.0)
        ok = (t - s >= guard) & (e - t >= guard)
        if not ok.any():
            continue
        ts.append(t[ok])
        xs.append(x[ok])
        starts.append(t[ok][0] - bin_size / 2)
        ends.append(t[ok][-1] + bin_size / 2)
    if not ts:
        raise ValueError("no bins left after the epoch-edge guard; epochs too short")

    x = np.concatenate(xs)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    # constant units: smoothing round-off leaves sd ~1e-16, which would amplify noise to ±1
    x = (x - mu) / np.where(sd > 1e-9 * (np.abs(mu) + 1), sd, 1.0)
    return nap.TsdFrame(
        t=np.concatenate(ts),
        d=x,
        columns=np.asarray(keep),
        time_support=nap.IntervalSet(starts, ends),
    )
