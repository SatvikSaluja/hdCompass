import numpy as np
import pynapple as nap
import pytest

from hdcompass.features import bin_counts, population_rates


def _regular(rate, ep):
    """Perfectly regular spikes, offset half an interval so none sits on a bin edge."""
    t = [np.arange(s + 0.5 / rate, e, 1 / rate) for s, e in zip(ep.start, ep.end, strict=True)]
    return nap.Ts(np.concatenate(t))


def test_constant_rate_stays_constant_and_no_gap_crossing():
    ep = nap.IntervalSet([0.0, 50.0], [20.0, 80.0])
    spikes = nap.TsGroup({0: _regular(50.0, ep), 1: _regular(20.0, ep)}, time_support=ep)
    r = population_rates(spikes, ep, bin_size=0.1, smooth_std=0.2)
    assert np.allclose(r.values, 0.0)  # constant input -> zero after z-scoring, edges included
    guard = 0.4
    for s, e in zip(ep.start, ep.end, strict=True):
        t = r.restrict(nap.IntervalSet(s, e)).t
        assert t.min() - s >= guard - 1e-9 and e - t.max() >= guard - 1e-9
    assert not np.any((r.t > 20.0) & (r.t < 50.0))


def test_bin_counts_stay_inside_epochs():
    ep = nap.IntervalSet([0.0, 50.0], [20.0, 80.0])
    spikes = nap.TsGroup({0: _regular(50.0, ep)}, time_support=ep)
    (t0, c0), (t1, c1) = bin_counts(spikes, ep, 0.1)
    assert len(t0) == 200 and len(t1) == 300
    assert t0.max() < 20.0 and t1.min() > 50.0
    np.testing.assert_array_equal(np.concatenate([c0, c1]).ravel(), 5)


def test_low_rate_units_dropped():
    ep = nap.IntervalSet(0.0, 100.0)
    rng = np.random.default_rng(0)
    spikes = nap.TsGroup(
        {
            0: nap.Ts(np.sort(rng.uniform(0, 100, 1000))),  # 10 Hz
            1: nap.Ts(np.sort(rng.uniform(0, 100, 50))),  # 0.5 Hz
        },
        time_support=ep,
    )
    r = population_rates(spikes, ep, min_rate=1.0)
    assert list(r.columns) == [0]
    with pytest.raises(ValueError):
        population_rates(spikes, ep, min_rate=100.0)
