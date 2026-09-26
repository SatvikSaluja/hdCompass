"""Synthetic head-direction populations with a known latent angle."""

import numpy as np
import pynapple as nap

from .circular import TWO_PI, wrap


def tuning_curves(pref_angle, angles, kappa=4.0, peak_rate=30.0, base_rate=0.5):
    """Von Mises tuning ``base + (peak − base)·exp(κ(cos(θ − φ) − 1))``.

    Parameters
    ----------
    pref_angle : array_like, shape (units,)
        Preferred directions φ (rad).
    angles : array_like, shape (n,)
        Angles θ at which to evaluate (rad).
    kappa, peak_rate, base_rate : float
        Concentration, peak and baseline rates (Hz).

    Returns
    -------
    ndarray, shape (units, n)
        Firing rates in Hz.
    """
    d = np.asarray(angles)[None, :] - np.asarray(pref_angle)[:, None]
    return base_rate + (peak_rate - base_rate) * np.exp(kappa * (np.cos(d) - 1.0))


def simulate_hd(
    n_cells,
    duration,
    dt=0.001,
    kappa=4.0,
    peak_rate=30.0,
    base_rate=0.5,
    sigma=1.0,
    jump_rate=0.0,
    seed=0,
):
    """Simulate a head-direction cell population driven by a diffusing latent angle.

    Parameters
    ----------
    n_cells : int
        Number of cells; preferred directions are evenly spaced plus jitter.
    duration : float
        Length in seconds.
    dt : float
        Simulation step (s). Spikes are Poisson counts per step, times uniform within the step.
    kappa, peak_rate, base_rate : float
        Von Mises tuning parameters (see :func:`tuning_curves`).
    sigma : float
        Diffusion of the wrapped Gaussian random walk (rad/√s).
    jump_rate : float
        Rate (per s) of Poisson jumps to a uniformly drawn new angle (SWS-like discontinuities).
    seed : int
        Seed; output is deterministic given it.

    Returns
    -------
    spikes : nap.TsGroup
        One unit per cell, metadata ``pref_angle``.
    angle : nap.Tsd
        True latent angle at step centres, in [0, 2π).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    t = (np.arange(n) + 0.5) * dt

    base = rng.uniform(0, TWO_PI) + np.cumsum(rng.normal(0.0, sigma * np.sqrt(dt), n))
    jumps = rng.random(n) < jump_rate * dt
    segment = np.cumsum(jumps)  # 0 before the first jump
    starts = np.flatnonzero(jumps)
    offsets = np.concatenate([[0.0], rng.uniform(0, TWO_PI, starts.size) - base[starts]])
    theta = wrap(base + offsets[segment])

    spacing = TWO_PI / n_cells
    pref = wrap(np.arange(n_cells) * spacing + rng.uniform(-0.25, 0.25, n_cells) * spacing)

    support = nap.IntervalSet(0.0, n * dt)
    units = {}
    for i in range(n_cells):
        rate = tuning_curves(pref[i : i + 1], theta, kappa, peak_rate, base_rate)[0]
        counts = rng.poisson(rate * dt)
        st = np.repeat(t - dt / 2, counts) + rng.random(counts.sum()) * dt
        units[i] = nap.Ts(np.sort(st), time_support=support)
    spikes = nap.TsGroup(units, time_support=support, metadata={"pref_angle": pref})
    return spikes, nap.Tsd(t=t, d=theta, time_support=support)
