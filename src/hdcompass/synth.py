"""Synthetic head-direction populations with a known latent angle."""

import numpy as np
import pynapple as nap

from .circular import TWO_PI, wrap
from .datasets import Session


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
    pref_angle=None,
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
    pref_angle : array_like, shape (n_cells,), optional
        Fixed preferred directions (e.g. to keep the same cells across simulated epochs).

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

    if pref_angle is None:
        spacing = TWO_PI / n_cells
        pref = wrap(np.arange(n_cells) * spacing + rng.uniform(-0.25, 0.25, n_cells) * spacing)
    else:
        pref = wrap(pref_angle)
        if pref.shape != (n_cells,):
            raise ValueError("pref_angle must have shape (n_cells,)")

    support = nap.IntervalSet(0.0, n * dt)
    units = {}
    for i in range(n_cells):
        rate = tuning_curves(pref[i : i + 1], theta, kappa, peak_rate, base_rate)[0]
        counts = rng.poisson(rate * dt)
        st = np.repeat(t - dt / 2, counts) + rng.random(counts.sum()) * dt
        units[i] = nap.Ts(np.sort(st), time_support=support)
    spikes = nap.TsGroup(units, time_support=support, metadata={"pref_angle": pref})
    return spikes, nap.Tsd(t=t, d=theta, time_support=support)


STATE_PARAMS = {
    "wake": {"sigma": 1.0, "jump_rate": 0.0, "peak_rate": 30.0},
    "rem": {"sigma": 1.0, "jump_rate": 0.0, "peak_rate": 30.0},
    "sws": {"sigma": 4.0, "jump_rate": 0.3, "peak_rate": 26.0},
}


def simulate_session(
    n_cells=20,
    wake=2000.0,
    rem=2500.0,
    sws=12000.0,
    rem_epoch=90.0,
    sws_epoch=300.0,
    gap=10.0,
    kappa=4.0,
    params=None,
    seed=0,
):
    """A Mouse32-shaped synthetic session with known per-state compass dynamics.

    One wake block, then alternating SWS and REM epochs (separated by ``gap`` seconds of
    unlabelled, spike-free time) until the requested totals are reached. The same cells
    (preferred directions) fire in every state; each state has its own diffusion, jump rate and
    peak rate (``params``, default :data:`STATE_PARAMS`: SWS is faster, jumpier and has ~14%
    lower gain, matching the measured Mouse32 population-rate drop).

    Parameters
    ----------
    n_cells : int
    wake, rem, sws : float
        Total seconds per state (defaults match Mouse32: 1958 s, 2565 s, 12402 s, rounded).
    rem_epoch, sws_epoch : float
        Maximum length of one sleep epoch.
    gap : float
        Seconds between consecutive epochs.
    kappa : float
    params : dict[str, dict], optional
        Per-state ``sigma``, ``jump_rate``, ``peak_rate``; merged over the defaults.
    seed : int

    Returns
    -------
    session : Session
        ``angle`` holds the true angle during wake only, like real tracking.
    truth : nap.Tsd
        True angle in every state (for validating sleep decoding).
    """
    params = {k: {**v, **(params or {}).get(k, {})} for k, v in STATE_PARAMS.items()}
    rng = np.random.default_rng(seed)
    spacing = TWO_PI / n_cells
    pref = wrap(np.arange(n_cells) * spacing + rng.uniform(-0.25, 0.25, n_cells) * spacing)

    plan = [("wake", wake)]
    left = {"rem": rem, "sws": sws}
    while left["rem"] > 0 or left["sws"] > 0:
        for state, size in (("sws", sws_epoch), ("rem", rem_epoch)):
            if left[state] > 0:
                d = min(size, left[state])
                plan.append((state, d))
                left[state] -= d

    t0 = 0.0
    spikes = [[] for _ in range(n_cells)]
    ts, vs, eps = [], [], {"wake": [], "rem": [], "sws": []}
    for i, (state, d) in enumerate(plan):
        sp, ang = simulate_hd(
            n_cells, d, kappa=kappa, seed=seed * 100_003 + i, pref_angle=pref, **params[state]
        )
        for k in range(n_cells):
            spikes[k].append(sp[k].t + t0)
        ts.append(ang.t + t0)
        vs.append(ang.values)
        eps[state].append((t0, t0 + d))
        t0 += d + gap

    iset = {k: nap.IntervalSet(*zip(*v, strict=True)) for k, v in eps.items()}
    support = nap.IntervalSet(0.0, t0)
    group = nap.TsGroup(
        {k: nap.Ts(np.concatenate(v), time_support=support) for k, v in enumerate(spikes)},
        time_support=support,
        metadata={"pref_angle": pref},
    )
    all_ep = iset["wake"].union(iset["rem"]).union(iset["sws"])
    truth = nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vs), time_support=all_ep)
    session = Session(
        spikes=group,
        angle=truth.restrict(iset["wake"]),
        wake=iset["wake"],
        rem=iset["rem"],
        sws=iset["sws"],
    )
    return session, truth
