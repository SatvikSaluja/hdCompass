"""Validate hdcompass: long synthetic ground truth, held-out checks on Mouse32, sweep summary.

Usage:
  .venv/bin/python scripts/validate.py synthetic [--seed 0] [--out results/validation]
  .venv/bin/python scripts/validate.py real      [--out results/validation]
  .venv/bin/python scripts/validate.py sweep     [--out results/validation]

synthetic  Mouse32-shaped session (wake 2000 s, REM 2500 s, SWS 12000 s, 20 cells) with known
           per-state sigma, jumps and SWS gain drop, run through the full label-free pipeline;
           sleep decoding error, recovered sigma, jump fraction and posterior calibration are
           checked against the ground truth. A well-specified run (true tuning, true sigma)
           checks the decoder's calibration itself.
real       Mouse32: held-out 2-fold wake validation (60 s alternating blocks), gain-corrected
           sleep decoding, held-out posterior coverage, sigma grid-edge flags, embedding
           robustness across Isomap neighbourhoods and seeds.
sweep      Summary table and trend checks for results/sweep/report.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pynapple as nap
from matplotlib.figure import Figure

from hdcompass.align import align, angle_error
from hdcompass.benchmark import save_json
from hdcompass.circular import TWO_PI, circ_mae, epoch_diffs, wrap
from hdcompass.cli import run_pipeline
from hdcompass.datasets import load_mouse32
from hdcompass.decode import SIGMA_GRID, _counts, decode, fit_sigma, forward_backward
from hdcompass.features import population_rates
from hdcompass.manifold import embed, ring_angle
from hdcompass.synth import STATE_PARAMS, simulate_session, tuning_curves
from hdcompass.tuning import fit_tuning, tuning_from_ring

LEVEL = 0.9
MAX_COVERAGE_BINS = 150_000
GRID = (np.arange(60) + 0.5) * TWO_PI / 60


def pop_rate(spikes, ep):
    return sum(len(spikes[k].restrict(ep)) for k in spikes.keys()) / ep.tot_length()


def apply_tf(tsd, tf):
    return nap.Tsd(
        t=tsd.t, d=wrap(tf["sign"] * tsd.values + tf["offset"]), time_support=tsd.time_support
    )


def values_at(t, angle):
    """``angle`` sampled at times ``t`` (nearest sample)."""
    return nap.Tsd(t=t, d=np.zeros(len(t))).value_from(angle).values


def hpd_coverage(post, true_angle, level=LEVEL):
    """Fraction of bins whose true angle lies in the posterior's highest-density set of mass
    ``level``, and the mean width of that set (rad). Evaluated on at most MAX_COVERAGE_BINS
    evenly strided bins to bound memory."""
    stride = max(1, len(post) // MAX_COVERAGE_BINS)
    post, true_angle = post[::stride], np.asarray(true_angle)[::stride]
    n = post.shape[1]
    idx = np.minimum((wrap(true_angle) / TWO_PI * n).astype(int), n - 1)
    order = np.argsort(-post, axis=1)
    cum = np.cumsum(np.take_along_axis(post, order, axis=1), axis=1)
    rank = np.argmax(order == idx[:, None], axis=1)
    before = np.where(rank > 0, cum[np.arange(len(post)), np.maximum(rank - 1, 0)], 0.0)
    width = ((cum < level).sum(axis=1) + 1) * TWO_PI / n
    return float(np.mean(before < level)), float(np.mean(width))


def posterior(spikes, tuning, ep, r):
    t, counts, resets, _ = _counts(spikes, tuning, ep, r.bin_size)
    return t, forward_backward(counts, tuning, r.bin_size, r.sigma, r.eps, resets)[0]


def on_edge(sigma, grid=SIGMA_GRID):
    return bool(np.isclose(sigma, grid[0]) or np.isclose(sigma, grid[-1]))


def jump_fraction(tsd):
    d = epoch_diffs(tsd)
    return float(np.mean(np.abs(d) > np.pi / 2)) if d.size else 0.0


# --------------------------------------------------------------------------- synthetic


def run_synthetic(out, seed):
    t0 = time.perf_counter()
    s, truth = simulate_session(seed=seed)
    sim_s = time.perf_counter() - t0
    report, _, det = run_pipeline(s, out / f"synthetic_seed{seed}", quick=False, overwrite=True)
    pipe_s = time.perf_counter() - t0 - sim_s
    spikes, states, res, tc = det["spikes"], det["states"], det["results"], det["tuning_ring"]
    pref = spikes["pref_angle"].values
    _, tf = align(res["wake"].angle, s.angle)  # ring frame -> true frame, fitted on wake only
    wake_rate = pop_rate(spikes, states["wake"])

    per_state = {}
    for st, r in res.items():
        p = STATE_PARAMS[st]
        true_b = values_at(r.angle.t, truth)
        true_tsd = nap.Tsd(t=r.angle.t, d=true_b, time_support=r.angle.time_support)
        # coverage in the ring frame: map truth into it with the inverse wake transform
        true_ring = wrap(tf["sign"] * (true_b - tf["offset"]))
        _, post = posterior(spikes, tc, states[st], r)
        cov, width = hpd_coverage(post, true_ring)
        del post
        row = {
            "true_sigma": p["sigma"],
            "true_jump_rate": p["jump_rate"],
            "true_peak_rate": p["peak_rate"],
            "true_jump_fraction_20ms": jump_fraction(true_tsd),
            "decoded_sigma": r.sigma,
            "sigma_at_grid_edge": on_edge(r.sigma),
            "decoded_jump_fraction": report["dynamics"][st]["jump_fraction"],
            "mean_circ_variance": report["dynamics"][st]["mean_circ_variance"],
            "mae_rad": circ_mae(apply_tf(r.angle, tf).values, true_b),
            "hpd90_coverage": cov,
            "hpd90_width_rad": width,
            "pop_rate_hz": pop_rate(spikes, states[st]),
            "gain_vs_wake": pop_rate(spikes, states[st]) / wake_rate,
        }
        if st != "wake":  # R2: rescale wake tuning by the state's population-rate ratio
            g = row["gain_vs_wake"]
            rg = decode(spikes, tc * g, states[st])
            _, post = posterior(spikes, tc * g, states[st], rg)
            cov_g, width_g = hpd_coverage(post, true_ring)
            del post
            row["gain_corrected"] = {
                "decoded_sigma": rg.sigma,
                "sigma_at_grid_edge": on_edge(rg.sigma),
                "decoded_jump_fraction": jump_fraction(rg.angle),
                "mean_circ_variance": float(np.mean(rg.variance.values)),
                "mae_rad": circ_mae(apply_tf(rg.angle, tf).values, values_at(rg.angle.t, truth)),
                "hpd90_coverage": cov_g,
                "hpd90_width_rad": width_g,
            }
        # well-specified model: true tuning (state gain), true sigma, true jump prob
        eps = 1 - np.exp(-p["jump_rate"] * r.bin_size)
        true_tc = tuning_curves(pref, GRID, kappa=4.0, peak_rate=p["peak_rate"])
        rt = decode(spikes, true_tc, states[st], sigma=p["sigma"], eps=eps)
        _, post = posterior(spikes, true_tc, states[st], rt)
        cov_t, width_t = hpd_coverage(post, values_at(rt.angle.t, truth))
        del post
        _, ll = fit_sigma(spikes, true_tc, states[st], eps=eps)
        row["true_model"] = {
            "mae_rad": circ_mae(rt.angle.values, values_at(rt.angle.t, truth)),
            "hpd90_coverage": cov_t,
            "hpd90_width_rad": width_t,
            "ml_sigma_true_tuning": float(SIGMA_GRID[np.argmax(ll)]),
        }
        per_state[st] = row
        print(f"[synthetic] {st}: {json.dumps(row, default=float)[:300]}", flush=True)

    out_report = {
        "seed": seed,
        "durations_s": report["durations_s"],
        "n_units_used": report["n_units_used"],
        "topology": report["topology"],
        "ring_angle_mae_rad": report["manifold"]["ring_angle_mae_rad"],
        "wake_align": tf,
        "per_state": per_state,
        "runtime_s": {"simulate": sim_s, "pipeline": pipe_s, "total": time.perf_counter() - t0},
    }
    save_json(out_report, out / f"synthetic_seed{seed}.json", overwrite=True)
    fig_synthetic(per_state).savefig(out / f"synthetic_seed{seed}.png", dpi=120)
    return out_report


def fig_synthetic(ps):
    states = list(ps)
    x = np.arange(len(states))
    fig = Figure(figsize=(12, 3.2))
    ax = fig.subplots(1, 4)
    ax[0].bar(x - 0.27, [ps[s]["true_sigma"] for s in states], 0.27, label="true", color="0.6")
    ax[0].bar(x, [ps[s]["decoded_sigma"] for s in states], 0.27, label="decoded")
    ax[0].bar(
        x + 0.27,
        [ps[s].get("gain_corrected", ps[s])["decoded_sigma"] for s in states],
        0.27,
        label="gain-corrected",
    )
    ax[0].set(yscale="log", title="σ (rad/√s)")
    ax[1].bar(x - 0.2, [ps[s]["true_jump_fraction_20ms"] for s in states], 0.4, color="0.6")
    ax[1].bar(x + 0.2, [ps[s]["decoded_jump_fraction"] for s in states], 0.4)
    ax[1].set(title="jump fraction (>π/2)")
    ax[2].bar(x - 0.27, [ps[s]["true_model"]["mae_rad"] for s in states], 0.27, color="0.6")
    ax[2].bar(x, [ps[s]["mae_rad"] for s in states], 0.27)
    ax[2].bar(x + 0.27, [ps[s].get("gain_corrected", ps[s])["mae_rad"] for s in states], 0.27)
    ax[2].set(title="decoding MAE vs truth (rad)")
    ax[3].bar(x - 0.27, [ps[s]["true_model"]["hpd90_coverage"] for s in states], 0.27, color="0.6")
    ax[3].bar(x, [ps[s]["hpd90_coverage"] for s in states], 0.27)
    ax[3].bar(
        x + 0.27, [ps[s].get("gain_corrected", ps[s])["hpd90_coverage"] for s in states], 0.27
    )
    ax[3].axhline(LEVEL, color="k", lw=0.8, ls="--")
    ax[3].set(title="90% HPD coverage", ylim=(0, 1))
    for a in ax:
        a.set_xticks(x, states)
    ax[0].legend(fontsize=7)
    fig.text(0.5, 0.01, "grey = truth / well-specified model", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------- real


def blocks(ep, size=60.0):
    """Split each interval into ``size``-second blocks; return (even, odd) IntervalSets."""
    starts, ends = [], []
    for s, e in zip(ep.start, ep.end, strict=True):
        st = np.arange(s, e, size)
        starts.append(st)
        ends.append(np.minimum(st + size, e))
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    return nap.IntervalSet(starts[::2], ends[::2]), nap.IntervalSet(starts[1::2], ends[1::2])


def run_real(out):
    t0 = time.perf_counter()
    s = load_mouse32()
    in_sample = json.loads(Path("results/mouse32/report.json").read_text())
    rates = population_rates(s.spikes, s.wake)
    spikes = s.spikes[list(rates.columns)]
    ring = ring_angle(embed(rates))

    # R1: held-out wake validation
    a, b = blocks(s.wake)
    folds = []
    for train, test in ((a, b), (b, a)):
        f = {}
        ring_al, _ = align(ring, s.angle.restrict(train))
        f["ring_angle_mae_rad"] = angle_error(ring_al.restrict(test), s.angle)
        tc_r = tuning_from_ring(spikes, ring, train)
        sig, _ = fit_sigma(spikes, tc_r, train)
        _, tf = align(decode(spikes, tc_r, train, sigma=sig).angle, s.angle)
        f["hmm_ring_tuning_mae_rad"] = angle_error(
            apply_tf(decode(spikes, tc_r, test, sigma=sig).angle, tf), s.angle
        )
        _, tfi = align(decode(spikes, tc_r, train, sigma=np.inf).angle, s.angle)
        f["independent_ring_tuning_mae_rad"] = angle_error(
            apply_tf(decode(spikes, tc_r, test, sigma=np.inf).angle, tfi), s.angle
        )
        tc_h = fit_tuning(spikes, s.angle, train)
        sig_h, _ = fit_sigma(spikes, tc_h, train)
        rh = decode(spikes, tc_h, test, sigma=sig_h)
        f["hmm_head_tuning_mae_rad"] = angle_error(rh.angle, s.angle)
        t, post = posterior(spikes, tc_h, test, rh)
        f["hmm_head_tuning_hpd90_coverage"], f["hmm_head_tuning_hpd90_width_rad"] = hpd_coverage(
            post, values_at(t, s.angle)
        )
        f["hmm_head_tuning_mean_circ_variance"] = float(np.mean(rh.variance.values))
        f["sigma_ring_train"], f["sigma_head_train"] = sig, sig_h
        folds.append(f)
        print(f"[real] fold: {f}", flush=True)
    held_out = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}

    # R2: gain-corrected sleep decoding (tuning from the full-wake ring, as in the pipeline)
    tc = tuning_from_ring(spikes, ring, s.wake)
    wake_rate = pop_rate(spikes, s.wake)
    gain = {}
    for st, ep in (("rem", s.rem), ("sws", s.sws)):
        g = pop_rate(spikes, ep) / wake_rate
        rg = decode(spikes, tc * g, ep)
        gain[st] = {
            "gain_vs_wake": g,
            "uncorrected": in_sample["dynamics"][st],
            "corrected": {
                "sigma": rg.sigma,
                "sigma_at_grid_edge": on_edge(rg.sigma),
                "jump_fraction": jump_fraction(rg.angle),
                "mean_circ_variance": float(np.mean(rg.variance.values)),
                "loglik": rg.loglik,
            },
            "uncorrected_loglik": in_sample["loglik"][st],
        }
        print(f"[real] gain {st}: {gain[st]}", flush=True)

    # R3: sigma grid edges in the pipeline's own fits
    edges = {st: on_edge(d["sigma"]) for st, d in in_sample["dynamics"].items()}

    # robustness of the unsupervised ring angle to Isomap settings
    robust = []
    for nn, seed in ((10, 0), (15, 0), (30, 0), (15, 1), (15, 2)):
        al, _ = align(ring_angle(embed(rates, n_neighbors=nn, seed=seed)), s.angle)
        robust.append(
            {"n_neighbors": nn, "seed": seed, "ring_angle_mae_rad": angle_error(al, s.angle)}
        )
        print(f"[real] robustness {robust[-1]}", flush=True)

    rep = {
        "held_out_folds": folds,
        "held_out_mean": held_out,
        "in_sample": in_sample["decode_wake_validation"]
        | {"ring_angle_mae_rad": in_sample["manifold"]["ring_angle_mae_rad"]},
        "gain": gain,
        "sigma_at_grid_edge": edges,
        "isomap_robustness": robust,
        "runtime_s": time.perf_counter() - t0,
    }
    save_json(rep, out / "real.json", overwrite=True)
    return rep


# --------------------------------------------------------------------------- sweep


def run_sweep_summary(out):
    rep = json.loads(Path("results/sweep/report.json").read_text())
    rows = rep["results"]
    table = []
    for k in sorted({r["kappa"] for r in rows}):
        for n in sorted({r["n_cells"] for r in rows}):
            g = [r for r in rows if r["kappa"] == k and r["n_cells"] == n]
            table.append(
                {
                    "kappa": k,
                    "n_cells": n,
                    "n_seeds": len(g),
                    "manifold_mae_mean": float(np.mean([r["manifold_mae"] for r in g])),
                    "manifold_mae_sd": float(np.std([r["manifold_mae"] for r in g])),
                    "decode_mae_mean": float(np.mean([r["decode_mae"] for r in g])),
                    "decode_mae_sd": float(np.std([r["decode_mae"] for r in g])),
                    "is_ring_fraction": float(np.mean([r["is_ring"] for r in g])),
                    "ring_score_median": float(
                        np.median(
                            [r["ring_score"] if r["ring_score"] is not None else np.inf for r in g]
                        )
                    ),
                    "decode_sigma": [r["decode_sigma"] for r in g],
                    "runtime_s_mean": float(np.mean([r["runtime_s"] for r in g])),
                }
            )
    checks = {}
    for k in sorted({r["kappa"] for r in table}):
        for key in ("manifold_mae_mean", "decode_mae_mean"):
            v = [r[key] for r in table if r["kappa"] == k]
            checks[f"kappa={k} {key} non-increasing with cells"] = bool(np.all(np.diff(v) <= 1e-9))
    sig = np.array([s for r in table for s in r["decode_sigma"]])
    checks["decode_sigma within one grid step of true 1.0 (fraction)"] = float(
        np.mean(np.abs(np.log(sig)) <= np.log(SIGMA_GRID[1] / SIGMA_GRID[0]) + 1e-9)
    )
    out_rep = {
        "config_hash": rep["config_hash"],
        "config": rep["config"],
        "table": table,
        "checks": checks,
    }
    save_json(out_rep, out / "sweep_summary.json", overwrite=True)
    for r in table:
        print(
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in r.items()
                if k != "decode_sigma"
            }
        )
    print(checks)
    return out_rep


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("what", choices=["synthetic", "real", "sweep"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/validation")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    {
        "synthetic": lambda: run_synthetic(out, a.seed),
        "real": lambda: run_real(out),
        "sweep": lambda: run_sweep_summary(out),
    }[a.what]()


if __name__ == "__main__":
    main()
