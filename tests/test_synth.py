import numpy as np

from hdcompass.synth import simulate_hd, tuning_curves


def test_shape_and_determinism():
    s1, a1 = simulate_hd(10, 20, seed=3)
    s2, a2 = simulate_hd(10, 20, seed=3)
    s3, _ = simulate_hd(10, 20, seed=4)
    assert len(s1) == 10 and len(a1) == 20_000
    assert np.all((a1.values >= 0) & (a1.values < 2 * np.pi))
    assert all(np.array_equal(s1[k].t, s2[k].t) for k in s1.keys())
    np.testing.assert_array_equal(a1.values, a2.values)
    assert not np.array_equal(s1[0].t, s3[0].t)


def test_empirical_rates_match_tuning():
    spikes, angle = simulate_hd(12, 200, sigma=2.0, seed=0)
    edges = np.linspace(0, 2 * np.pi, 13)
    occ_bin = np.digitize(angle.values, edges) - 1
    occupancy = np.bincount(occ_bin, minlength=12) * 0.001
    rates = tuning_curves(spikes["pref_angle"].values, angle.values)  # (units, T)
    for i, k in enumerate(spikes.keys()):
        idx = np.clip(np.searchsorted(angle.t, spikes[k].t), 0, len(angle) - 1)
        emp = np.bincount(occ_bin[idx], minlength=12) / occupancy
        expected = np.bincount(occ_bin, weights=rates[i], minlength=12) / (occupancy / 0.001)
        assert np.abs(emp - expected).sum() / expected.sum() < 0.2


def test_simulate_session_structure():
    from hdcompass.synth import simulate_session

    s, truth = simulate_session(n_cells=8, wake=30, rem=40, sws=70, rem_epoch=20, sws_epoch=30)
    assert s.wake.tot_length() == 30 and s.rem.tot_length() == 40 and s.sws.tot_length() == 70
    assert len(s.rem) == 2 and len(s.sws) == 3
    assert s.wake.intersect(s.sws).tot_length() == 0 and s.rem.intersect(s.sws).tot_length() == 0
    assert s.angle.time_support.tot_length() == 30  # tracked angle only in wake, like real data
    assert truth.time_support.tot_length() == 140
    for ep in (s.wake, s.rem, s.sws):  # every state has spikes from the same cells
        assert all(len(s.spikes[k].restrict(ep)) > 0 for k in s.spikes.keys())
