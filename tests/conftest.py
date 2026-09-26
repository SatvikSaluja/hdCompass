import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hdcompass.synth import simulate_hd, tuning_curves  # noqa: E402

GRID = (np.arange(60) + 0.5) * 2 * np.pi / 60


@pytest.fixture(scope="session")
def hd40():
    """40-cell synthetic session, 300 s, sigma=1 rad/√s, kappa=4."""
    spikes, angle = simulate_hd(40, 300, seed=0)
    return spikes, angle, tuning_curves(spikes["pref_angle"].values, GRID)
