"""Run the frozen hdcompass pipeline on DANDI:000939 (Duszkiewicz et al. 2024, postsubiculum).

Pre-registered before any of these sessions was analysed:
- sessions: the 22 without optogenetics ("ogen" not in the file name), dandiset version VERSION;
- wake = the first open-field epoch ("wake_square"); REM / SWS = "rem" / "nrem" sleep states;
- pipeline defaults exactly as frozen on Mouse32 by `scripts/validate.py decisions`;
- REM enters cross-session statistics only for sessions with >= MIN_REM_S seconds of REM;
- hypotheses, paired over sessions, one-sided Wilcoxon signed-rank: sigma_SWS > sigma_wake,
  sigma_SWS > sigma_REM, mean circular variance SWS > wake.

Usage: .venv/bin/python scripts/run_dandi000939.py [--out results/dandi000939] [--jobs 8]
       [--overwrite]   (resumable: sessions with a report are not rerun without --overwrite)
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import wilcoxon

from hdcompass.benchmark import save_json
from hdcompass.cli import run_pipeline
from hdcompass.datasets import DANDI_API, load_dandi000939

VERSION = "0.260512.1701"
MIN_REM_S = 60.0
STATES = ("wake", "rem", "sws")


def assets():
    url = f"{DANDI_API}/dandisets/000939/versions/{VERSION}/assets/?page_size=200"
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = json.load(r)["results"]
    return sorted((a["path"], a["asset_id"]) for a in rows if "ogen" not in a["path"])


def summarize(name, s, report):
    dyn, val, topo = report["dynamics"], report["decode_wake_validation"], report["topology"]
    row = {
        "session": name,
        "n_units_total": len(s.spikes),
        "n_hd_cells_flagged": int(np.sum(s.spikes["is_head_direction"])),
        "n_units_embed": report["n_units_used"],
        "n_units_decode": report["n_units_decode"],
        "durations_s": report["durations_s"],
        "ring_score": topo["ring_score"],
        "ring_p": topo["p"],
        "is_ring": topo["is_ring"],
        "ring_angle_mae_rad": report["manifold"]["ring_angle_mae_rad"],
        "hmm_ring_tuning_mae_rad": val["hmm_ring_tuning_mae_rad"],
        "independent_ring_tuning_mae_rad": val["independent_ring_tuning_mae_rad"],
        "hmm_head_tuning_mae_rad": val["hmm_head_tuning_mae_rad"],
        "rem_ok": report["durations_s"]["rem"] >= MIN_REM_S,
    }
    for st in STATES:
        for k in ("sigma", "mean_circ_variance", "jump_fraction", "median_abs_speed"):
            row[f"{k}_{st}"] = dyn[st][k]
    return row


def paired(rows, a, b, key, need_rem=False):
    use = [r for r in rows if r["rem_ok"] or not need_rem]
    x = np.array([r[f"{key}_{a}"] for r in use])
    y = np.array([r[f"{key}_{b}"] for r in use])
    d = x - y
    res = {
        "n": len(use),
        "greater": int((d > 0).sum()),
        "equal": int((d == 0).sum()),
        "less": int((d < 0).sum()),
        "median_ratio": float(np.median(x / y)),
    }
    res["wilcoxon_p_one_sided"] = (
        float(wilcoxon(x, y, alternative="greater").pvalue) if np.any(d != 0) else None
    )
    return res


def figure(rows):
    fig = Figure(figsize=(12, 3.4))
    ax = fig.subplots(1, 3)
    for r in rows:
        states = [s for s in STATES if s != "rem" or r["rem_ok"]]
        xs = [STATES.index(s) for s in states]
        ax[0].plot(xs, [r[f"sigma_{s}"] for s in states], "-o", color="0.5", alpha=0.6, ms=3)
        ax[1].plot(
            xs, [r[f"mean_circ_variance_{s}"] for s in states], "-o", color="0.5", alpha=0.6, ms=3
        )
    ax[0].set(yscale="log", title="σ per session (rad/√s)")
    ax[1].set(yscale="log", title="mean circular variance")
    for a in ax[:2]:
        a.set_xticks(range(3), STATES)
    keys = (
        "ring_angle_mae_rad",
        "hmm_ring_tuning_mae_rad",
        "independent_ring_tuning_mae_rad",
        "hmm_head_tuning_mae_rad",
    )
    for i, k in enumerate(keys):
        v = [r[k] for r in rows]
        ax[2].plot(
            np.full(len(v), i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(v)),
            v,
            "o",
            ms=3,
            alpha=0.7,
        )
    ax[2].set_xticks(range(4), ["ring", "HMM\nring", "indep.\nring", "HMM\nhead"])
    ax[2].set(title="wake MAE vs head angle (rad)")
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="results/dandi000939")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    rows = []
    for path, aid in assets():
        name = Path(path).stem.split("_behavior")[0]
        t0 = time.perf_counter()
        s = load_dandi000939(aid)
        rep_path = out / name / "report.json"
        if rep_path.exists() and not a.overwrite:
            report = json.loads(rep_path.read_text())
        else:
            report, _, _ = run_pipeline(s, out / name, overwrite=a.overwrite, n_jobs=a.jobs)
        rows.append(summarize(name, s, report))
        r = rows[-1]
        print(
            f"[000939] {name}: units {r['n_units_total']} ring {r['is_ring']} "
            f"({r['ring_score']:.2f}) MAE ring/hmm/head {r['ring_angle_mae_rad']:.3f}/"
            f"{r['hmm_ring_tuning_mae_rad']:.3f}/{r['hmm_head_tuning_mae_rad']:.3f} "
            f"sigma w/r/s {r['sigma_wake']:.2f}/{r['sigma_rem']:.2f}/{r['sigma_sws']:.2f} "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )

    stats = {
        "sessions": len(rows),
        "ring_detected": int(sum(r["is_ring"] for r in rows)),
        "median_mae_rad": {
            k: float(np.median([r[k] for r in rows]))
            for k in (
                "ring_angle_mae_rad",
                "hmm_ring_tuning_mae_rad",
                "independent_ring_tuning_mae_rad",
                "hmm_head_tuning_mae_rad",
            )
        },
        "sigma_sws_gt_wake": paired(rows, "sws", "wake", "sigma"),
        "sigma_sws_gt_rem": paired(rows, "sws", "rem", "sigma", need_rem=True),
        "sigma_rem_vs_wake": paired(rows, "rem", "wake", "sigma", need_rem=True),
        "circvar_sws_gt_wake": paired(rows, "sws", "wake", "mean_circ_variance"),
    }
    print(json.dumps(stats, indent=1))
    save_json(
        {
            "dandiset": "000939",
            "version": VERSION,
            "pre_registration": __doc__,
            "sessions": rows,
            "stats": stats,
        },
        out / "summary.json",
        overwrite=True,
    )
    figure(rows).savefig(out / "summary.png", dpi=120)


if __name__ == "__main__":
    main()
