"""Summary of decoded compass dynamics per brain state."""

import numpy as np

from .circular import epoch_diffs, unwrap_speed


def state_dynamics(result_by_state):
    """Summarise decoded dynamics for each state.

    Parameters
    ----------
    result_by_state : dict[str, DecodeResult]
        E.g. ``{"wake": ..., "rem": ..., "sws": ...}``.

    Returns
    -------
    dict[str, dict]
        Per state: ``sigma``, ``eps``, ``median_abs_speed`` (rad/s, per-epoch gradient),
        ``jump_fraction`` (fraction of consecutive within-epoch bins whose decoded angle moves
        more than π/2), ``mean_circ_variance``, ``n_bins``.
    """
    out = {}
    for state, r in result_by_state.items():
        steps = epoch_diffs(r.angle)
        speed = np.abs(unwrap_speed(r.angle).values)
        out[state] = {
            "sigma": r.sigma,
            "eps": r.eps,
            "median_abs_speed": float(np.median(speed)) if speed.size else float("nan"),
            "jump_fraction": float(np.mean(np.abs(steps) > np.pi / 2)) if steps.size else 0.0,
            "mean_circ_variance": float(np.mean(r.variance.values)),
            "n_bins": int(len(r.angle)),
        }
    return out
