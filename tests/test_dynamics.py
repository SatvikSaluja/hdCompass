import numpy as np

from hdcompass.decode import decode
from hdcompass.dynamics import state_dynamics
from hdcompass.synth import simulate_hd, tuning_curves


def test_jumps_raise_jump_fraction():
    results = {}
    for name, jump_rate in (("smooth", 0.0), ("jumpy", 2.0)):
        spikes, _ = simulate_hd(40, 60, jump_rate=jump_rate, seed=5)
        tc = tuning_curves(spikes["pref_angle"].values, (np.arange(60) + 0.5) * 2 * np.pi / 60)
        results[name] = decode(spikes, tc, spikes.time_support, sigma=1.0, eps=0.01)
    summary = state_dynamics(results)
    assert summary["jumpy"]["jump_fraction"] > summary["smooth"]["jump_fraction"]
    assert summary["smooth"]["n_bins"] == 3000
    assert set(summary["smooth"]) >= {"sigma", "eps", "median_abs_speed", "mean_circ_variance"}
