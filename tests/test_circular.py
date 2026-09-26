import numpy as np
import pynapple as nap

from hdcompass.circular import (
    TWO_PI,
    circ_corr,
    circ_diff,
    circ_mae,
    circ_mean,
    epoch_diffs,
    unwrap_speed,
    wrap,
)


def test_wrap_edges():
    out = wrap([0.0, TWO_PI, -1e-17, -np.pi, 3 * TWO_PI + 1.0])
    assert np.all((out >= 0) & (out < TWO_PI))
    np.testing.assert_allclose(out, [0, 0, 0, np.pi, 1.0], atol=1e-12)


def test_circ_diff_antisymmetric_and_range():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-10, 10, (2, 1000))
    d = circ_diff(a, b)
    assert np.all((d > -np.pi) & (d <= np.pi))
    np.testing.assert_allclose(d, -circ_diff(b, a), atol=1e-12)
    assert circ_diff(np.pi, 0.0) == np.pi


def test_mae_mean_corr():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, TWO_PI, 500)
    assert circ_mae(a, wrap(a + TWO_PI)) < 1e-12
    assert abs(circ_diff(circ_mean(wrap(1.0 + 0.1 * rng.standard_normal(500))), 1.0)) < 0.02
    assert circ_corr(a, wrap(a + 2.0)) > 0.999
    assert circ_corr(a, wrap(-a + 2.0)) < -0.999


def test_speed_and_diffs_never_cross_gaps():
    t = np.concatenate([np.arange(0, 10, 0.1), np.arange(20, 30, 0.1)])
    v = wrap(np.concatenate([0.5 * np.arange(100) * 0.1, 3.0 + 0.5 * np.arange(100) * 0.1]))
    ang = nap.Tsd(t=t, d=v, time_support=nap.IntervalSet([0, 20], [10, 30]))
    np.testing.assert_allclose(unwrap_speed(ang).values, 0.5, atol=1e-9)
    assert len(epoch_diffs(ang)) == len(t) - 2
