import numpy as np
import pynapple as nap

from hdcompass.align import align
from hdcompass.circular import circ_diff, wrap


def test_recovers_offset_and_reflection_exactly():
    t = np.arange(0, 100, 0.1)
    ref = nap.Tsd(t=t, d=wrap(np.cumsum(np.random.default_rng(0).normal(0, 0.2, len(t)))))
    for sign, offset in ((1, 0.7), (-1, 1.3)):
        rec = nap.Tsd(t=t, d=wrap(sign * (ref.values - offset)))
        aligned, tf = align(rec, ref)
        assert tf["sign"] == sign
        assert abs(circ_diff(tf["offset"], offset)) < 1e-9
        assert np.max(np.abs(circ_diff(aligned.values, ref.values))) < 1e-9


def test_fit_on_reference_support_only_applied_everywhere():
    t = np.arange(0, 200, 0.1)
    truth = wrap(np.cumsum(np.random.default_rng(1).normal(0, 0.2, len(t))))
    ref = nap.Tsd(t=t[t < 100], d=truth[t < 100], time_support=nap.IntervalSet(0, 99.95))
    rec = nap.Tsd(t=t, d=wrap(-truth + 2.0))
    aligned, _ = align(rec, ref)
    assert len(aligned) == len(t)
    assert np.max(np.abs(circ_diff(aligned.values, truth))) < 1e-9
