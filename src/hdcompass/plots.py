"""Figures. Every function returns a ``matplotlib.figure.Figure`` and never calls ``show``.

Figures are built with ``Figure()`` directly (no pyplot), so no GUI backend is ever needed.
"""

import numpy as np
from matplotlib.figure import Figure

STATES_ORDER = ("wake", "rem", "sws")


def plot_ring(embedding, color_angle):
    """Scatter of the first two embedding dimensions, coloured by an angle (rad)."""
    xy = np.asarray(embedding)
    c = np.asarray(color_angle)
    if len(c) != len(xy):
        raise ValueError("color_angle must have one value per embedding point")
    fig = Figure(figsize=(4.5, 4))
    ax = fig.add_subplot()
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="twilight", vmin=0, vmax=2 * np.pi, s=3)
    fig.colorbar(sc, ax=ax, label="angle (rad)")
    ax.set(xlabel="dim 1", ylabel="dim 2", title="Population activity embedding")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_barcode(h1, top=30):
    """H1 barcode (longest ``top`` bars) from :func:`hdcompass.topology.h1_persistence`."""
    dgm = np.asarray(h1["diagram"])[:top]
    fig = Figure(figsize=(5, 3.5))
    ax = fig.add_subplot()
    for i, (b, d) in enumerate(dgm):
        ax.plot([b, d], [i, i], color="C0" if i == 0 else "0.5", lw=2)
    ax.invert_yaxis()
    ax.set(xlabel="filtration radius", ylabel="H1 feature", title="H1 barcode")
    fig.tight_layout()
    return fig


def plot_decode(true, decoded, var):
    """Decoded angle (with true angle if given) and posterior circular variance over time.

    Parameters
    ----------
    true : nap.Tsd or None
    decoded, var : nap.Tsd
    """
    fig = Figure(figsize=(8, 4))
    ax1, ax2 = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    if true is not None:
        ax1.plot(true.t, true.values, ".", ms=1.5, color="0.6", label="head direction")
    ax1.plot(decoded.t, decoded.values, ".", ms=1.5, color="C3", label="decoded")
    ax1.set(ylabel="angle (rad)", ylim=(0, 2 * np.pi))
    ax1.legend(loc="upper right", markerscale=6, fontsize=8)
    ax2.plot(var.t, var.values, color="C0", lw=0.8)
    ax2.set(xlabel="time (s)", ylabel="circ. var.", ylim=(0, 1))
    fig.tight_layout()
    return fig


def plot_state_dynamics(summary):
    """Bar charts of :func:`hdcompass.dynamics.state_dynamics` metrics per state."""
    states = [s for s in STATES_ORDER if s in summary] + [
        s for s in summary if s not in STATES_ORDER
    ]
    metrics = [
        ("sigma", "σ (rad/√s)"),
        ("median_abs_speed", "median |speed| (rad/s)"),
        ("jump_fraction", "jump fraction (>π/2)"),
        ("mean_circ_variance", "mean circ. variance"),
    ]
    fig = Figure(figsize=(10, 2.8))
    for ax, (key, label) in zip(fig.subplots(1, len(metrics)), metrics, strict=True):
        ax.bar(states, [summary[s][key] for s in states], color=["C0", "C1", "C2"][: len(states)])
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    return fig


def plot_sweep(results):
    """Recovery error vs number of cells, one line per κ (mean ± sd over seeds)."""
    fig = Figure(figsize=(8, 3.2))
    axes = fig.subplots(1, 2, sharey=True)
    kappas = sorted({r["kappa"] for r in results})
    for ax, key, title in zip(
        axes, ("manifold_mae", "decode_mae"), ("unsupervised ring", "HMM, true tuning"), strict=True
    ):
        for i, k in enumerate(kappas):
            ns = sorted({r["n_cells"] for r in results if r["kappa"] == k})
            vals = [[r[key] for r in results if r["kappa"] == k and r["n_cells"] == n] for n in ns]
            ax.errorbar(
                ns,
                [np.mean(v) for v in vals],
                [np.std(v) for v in vals],
                marker="o",
                color=f"C{i}",
                label=f"κ={k}",
                capsize=3,
            )
        ax.set(xscale="log", xlabel="cells", title=title)
        ax.set_xticks(sorted({r["n_cells"] for r in results}))
        ax.set_xticklabels(sorted({r["n_cells"] for r in results}))
    axes[0].set_ylabel("circular MAE (rad)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
