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
