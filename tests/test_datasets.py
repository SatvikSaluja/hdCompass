from datetime import UTC
from pathlib import Path

import pytest

from hdcompass.datasets import MOUSE32_FILE, load_mouse32

PATH = Path("~/.cache/hdcompass").expanduser() / MOUSE32_FILE


@pytest.mark.skipif(not PATH.exists(), reason="Mouse32 NWB not cached")
def test_mouse32_session():
    s = load_mouse32(PATH)
    assert len(s.spikes) == 31
    assert set(s.spikes["location"]) == {"adn"}
    for ep in (s.wake, s.rem, s.sws):
        assert ep.tot_length() > 0
    assert s.wake.intersect(s.angle.time_support).tot_length() == pytest.approx(s.wake.tot_length())


def test_dandi000939_loader_on_a_local_nwb(tmp_path):
    from datetime import datetime

    import numpy as np
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.behavior import CompassDirection, SpatialSeries
    from pynwb.epoch import TimeIntervals

    from hdcompass.datasets import load_dandi000939

    nwb = NWBFile("Open field and sleep recording", "test", datetime.now(UTC))
    for col in ("is_head_direction", "is_excitatory", "is_fast_spiking"):
        nwb.add_unit_column(col, col)
    for i in range(3):
        nwb.add_unit(
            spike_times=np.sort(np.random.default_rng(i).uniform(0, 300, 200)),
            is_head_direction=i < 2,
            is_excitatory=True,
            is_fast_spiking=False,
        )
    t = np.arange(100.0, 200.0, 0.01)
    v = np.mod(0.3 * t, 2 * np.pi)
    v[:50] = np.nan  # tracking loss
    cd = CompassDirection()
    cd.add_spatial_series(
        SpatialSeries(name="head-direction", data=v, timestamps=t, reference_frame="Clockwise")
    )
    nwb.create_processing_module("behavior", "tracking").add(cd)
    nwb.add_epoch(0.0, 100.0, ["home_cage"])
    nwb.add_epoch(100.0, 200.0, ["wake_square"])
    nwb.add_epoch(200.0, 300.0, ["home_cage"])
    ss = TimeIntervals(name="sleep_states", description="wake, nrem, rem")
    ss.add_column("state", "state")
    for a, b, st in ((0, 60, "nrem"), (60, 80, "rem"), (200, 250, "nrem"), (250, 270, "rem")):
        ss.add_row(start_time=float(a), stop_time=float(b), state=st)
    nwb.add_time_intervals(ss)
    path = tmp_path / "sub-X_ses-1.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)

    s = load_dandi000939(path, cache_dir=tmp_path)
    assert len(s.spikes) == 3 and list(s.spikes["is_head_direction"]) == [True, True, False]
    assert s.wake.start[0] >= 100.5 - 1e-9 and s.wake.end[0] <= 200.0  # NaN-led start dropped
    assert not np.isnan(s.angle.values).any()
    assert s.rem.tot_length() == 40 and s.sws.tot_length() == 110
    path.unlink()  # second load must come from the cache
    s2 = load_dandi000939(path, cache_dir=tmp_path)
    np.testing.assert_array_equal(s2.spikes[0].t, s.spikes[0].t)
