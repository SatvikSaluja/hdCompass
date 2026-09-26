"""Command line: ``hdcompass mouse32`` and ``hdcompass sweep``."""

import argparse
from pathlib import Path

import numpy as np
import pynapple as nap

from .align import align, angle_error
from .benchmark import recovery_sweep, save_json
from .datasets import load_mouse32
from .decode import SIGMA_GRID, decode
from .dynamics import state_dynamics
from .features import population_rates
from .manifold import embed, ring_angle
from .plots import plot_barcode, plot_decode, plot_ring, plot_state_dynamics, plot_sweep
from .topology import h1_persistence, is_ring, shuffle_null
from .tuning import fit_tuning, tuning_from_ring

QUICK_SECONDS = 120.0


def first_seconds(ep, seconds):
    """The first ``seconds`` of an IntervalSet, counted cumulatively across its intervals."""
    starts, ends, left = [], [], seconds
    for s, e in zip(ep.start, ep.end, strict=True):
        if left <= 0:
            break
        d = min(e - s, left)
        starts.append(s)
        ends.append(s + d)
        left -= d
    return nap.IntervalSet(starts, ends)


def _save_figs(figs, out):
    for name, fig in figs.items():
        fig.savefig(Path(out) / f"{name}.png", dpi=120)


def run_mouse32(out, quick=False, data=None, overwrite=False):
    """Full label-free pipeline on Mouse32; writes ``report.json`` and PNGs to ``out``.

    Head angle is used only to validate (alignment error) and to colour figures; the ring,
    tuning curves and decoders are built from spikes alone.

    Returns
    -------
    report : dict
    figs : dict[str, Figure]
    """
    out = Path(out)
    if (out / "report.json").exists() and not overwrite:
        raise FileExistsError(f"{out / 'report.json'} exists; use --overwrite")
    s = load_mouse32(data)
    states = {"wake": s.wake, "rem": s.rem, "sws": s.sws}
    if quick:
        states = {k: first_seconds(v, QUICK_SECONDS) for k, v in states.items()}
    n_shuffles = 5 if quick else 20
    sigma_grid = np.geomspace(0.1, 50.0, 4) if quick else SIGMA_GRID

    # 1. label-free ring: rates -> topology -> embedding -> ring angle
    rates = population_rates(s.spikes, states["wake"])
    spikes = s.spikes[list(rates.columns)]
    h1 = h1_persistence(rates)
    null = shuffle_null(rates, n_shuffles)
    ring, p = is_ring(h1["ring_score"], null)
    emb = embed(rates)
    ring_ang = ring_angle(emb)
    ring_aligned, ring_tf = align(ring_ang, s.angle)

    # 2. tuning from the ring angle, HMM decoding per state
    tc_ring = tuning_from_ring(spikes, ring_ang, states["wake"])
    results = {k: decode(spikes, tc_ring, ep, sigma_grid=sigma_grid) for k, ep in states.items()}
    dyn = state_dynamics(results)

    # 3. validation against tracked head direction (wake only)
    wake_aligned, dec_tf = align(results["wake"].angle, s.angle)
    tc_head = fit_tuning(spikes, s.angle, states["wake"])
    head = decode(spikes, tc_head, states["wake"], sigma_grid=sigma_grid)
    indep = decode(spikes, tc_ring, states["wake"], sigma=np.inf)
    indep_aligned, _ = align(indep.angle, s.angle)

    report = {
        "quick": quick,
        "n_adn_units": len(s.spikes),
        "n_units_used": len(spikes),
        "durations_s": {k: float(ep.tot_length()) for k, ep in states.items()},
        "topology": {
            "ring_score": h1["ring_score"],
            "null": null,
            "p": p,
            "is_ring": ring,
            "n_shuffles": n_shuffles,
        },
        "manifold": {"align": ring_tf, "ring_angle_mae_rad": angle_error(ring_aligned, s.angle)},
        "decode_wake_validation": {
            "hmm_ring_tuning_mae_rad": angle_error(wake_aligned, s.angle),
            "independent_ring_tuning_mae_rad": angle_error(indep_aligned, s.angle),
            "hmm_head_tuning_mae_rad": angle_error(head.angle, s.angle),
            "align": dec_tf,
        },
        "loglik": {k: r.loglik for k, r in results.items()},
        "dynamics": dyn,
    }
    window = first_seconds(results["wake"].angle.time_support, 60.0)
    figs = {
        "ring": plot_ring(emb, emb.value_from(s.angle).values),
        "barcode": plot_barcode(h1),
        "decode_wake": plot_decode(
            s.angle.restrict(window),
            wake_aligned.restrict(window),
            results["wake"].variance.restrict(window),
        ),
        "dynamics": plot_state_dynamics(dyn),
    }
    save_json(report, out / "report.json", overwrite)
    _save_figs(figs, out)
    return report, figs


def run_sweep(out, quick=False, overwrite=False):
    """Synthetic recovery sweep; writes ``report.json`` and ``sweep.png`` to ``out``."""
    out = Path(out)
    results = recovery_sweep(quick=quick, out_path=out / "report.json", overwrite=overwrite)
    figs = {"sweep": plot_sweep(results)}
    _save_figs(figs, out)
    return results, figs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hdcompass", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("mouse32", "real-data pipeline"), ("sweep", "synthetic benchmark")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--out", required=True, help="output directory")
        p.add_argument("--quick", action="store_true", help="small smoke-test run")
        p.add_argument("--overwrite", action="store_true", help="replace an existing report")
        if name == "mouse32":
            p.add_argument("--data", default=None, help="path to Mouse32-140822.nwb")
    args = parser.parse_args(argv)

    if args.command == "mouse32":
        report, _ = run_mouse32(args.out, args.quick, args.data, args.overwrite)
        topo = report["topology"]
        print(f"ring_score={topo['ring_score']:.2f} is_ring={topo['is_ring']}")
    else:
        results, _ = run_sweep(args.out, args.quick, args.overwrite)
        print(f"{len(results)} configurations")
    print(f"wrote {Path(args.out) / 'report.json'}")


if __name__ == "__main__":
    main()
