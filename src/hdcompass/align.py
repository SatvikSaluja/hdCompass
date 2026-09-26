"""Align an unsupervised angle to a reference angle (sign and offset)."""

import pynapple as nap

from .circular import circ_diff, circ_mae, circ_mean, wrap


def align(recovered, reference):
    """Fit ``aligned = sign·recovered + offset`` to a reference angle.

    A ring angle recovered without labels has an arbitrary origin and direction. For each sign
    the offset is the circular mean of ``reference − sign·recovered``; the sign with the lower
    circular MAE wins. The fit uses only samples inside ``reference.time_support`` (wake); the
    same transform is applied to every sample of ``recovered``.

    Parameters
    ----------
    recovered : nap.Tsd
        Angles (rad), may extend beyond the reference support (e.g. sleep).
    reference : nap.Tsd
        Ground-truth angle (rad), e.g. tracked head direction.

    Returns
    -------
    aligned : nap.Tsd
        Same index and support as ``recovered``, in [0, 2π).
    transform : dict
        ``{"sign": ±1, "offset": float}``.
    """
    fit = recovered.restrict(reference.time_support)
    rec = fit.values
    ref = fit.value_from(reference).values
    best = None
    for sign in (1, -1):
        offset = circ_mean(circ_diff(ref, sign * rec))
        err = circ_mae(ref, sign * rec + offset)
        if best is None or err < best[0]:
            best = (err, sign, offset)
    _, sign, offset = best
    aligned = nap.Tsd(
        t=recovered.t, d=wrap(sign * recovered.values + offset), time_support=recovered.time_support
    )
    return aligned, {"sign": sign, "offset": offset}


def angle_error(estimate, reference):
    """Circular MAE of ``estimate`` against ``reference`` sampled at the estimate's timestamps.

    Only samples inside ``reference.time_support`` count. Returns NaN if there are none.
    """
    est = estimate.restrict(reference.time_support)
    if len(est) == 0:
        return float("nan")
    return circ_mae(est.values, est.value_from(reference).values)
