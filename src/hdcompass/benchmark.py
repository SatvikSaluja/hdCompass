"""Synthetic recovery benchmark: how well is the compass recovered vs population size?"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .align import align, angle_error
from .decode import SIGMA_GRID, decode
from .features import population_rates
from .manifold import embed, ring_angle
from .synth import simulate_hd, tuning_curves
from .topology import h1_persistence, is_ring, shuffle_null


def jsonable(x):
    """Convert numpy types to Python and non-finite floats to None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    return x


def save_json(obj, path, overwrite=False):
    """Write ``obj`` as strict JSON; refuse to replace an existing file unless ``overwrite``."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True (CLI: --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False))


def recovery_sweep(
    n_cells=(8, 16, 32, 64),
    kappas=(2, 4, 8),
    duration=600,
    seeds=(0, 1, 2),
    out_path=None,
    overwrite=False,
    quick=False,
    n_shuffles=20,
    n_landmarks=800,
    sigma_grid=SIGMA_GRID,
    n_jobs=1,
):
    """Run synth → features → topology → manifold → align, plus the HMM with true tuning.

    Parameters
    ----------
    n_cells, kappas, seeds : sequence
        Grid of configurations (full product).
    duration : float
        Seconds simulated per configuration.
    out_path : str or Path, optional
        If given, write ``{"config", "config_hash", "results"}`` as JSON there.
    overwrite : bool
        Replace an existing ``out_path``. Checked before any computation.
    quick : bool
        One configuration (32 cells, κ=4, seed 0), 60 s, 5 shuffles, 4-value sigma grid.
    n_shuffles, n_landmarks : int
        Topology settings.
    sigma_grid : array_like
        Grid for the decoder's maximum-likelihood ``sigma``.
    n_jobs : int
        Worker processes for the shuffle null (does not change results).

    Returns
    -------
    list of dict
        Per configuration: ``n_cells``, ``kappa``, ``seed``, ``ring_score``, ``ring_p``,
        ``is_ring``, ``manifold_mae``, ``decode_mae``, ``decode_sigma``, ``runtime_s``.
    """
    if quick:
        n_cells, kappas, seeds, duration = (32,), (4,), (0,), 60
        n_shuffles, sigma_grid = 5, np.geomspace(0.1, 50.0, 4)
    config = {
        "n_cells": list(n_cells),
        "kappas": list(kappas),
        "duration": duration,
        "seeds": list(seeds),
        "n_shuffles": n_shuffles,
        "n_landmarks": n_landmarks,
        "sigma_grid": [float(s) for s in sigma_grid],
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    if out_path is not None and Path(out_path).exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass overwrite=True (CLI: --overwrite)")

    results = []
    for n in n_cells:
        for kappa in kappas:
            for seed in seeds:
                t0 = time.perf_counter()
                spikes, angle = simulate_hd(n, duration, kappa=kappa, seed=seed)
                rates = population_rates(spikes, spikes.time_support)
                null = shuffle_null(
                    rates, n_shuffles, seed=seed, n_landmarks=n_landmarks, n_jobs=n_jobs
                )
                score = h1_persistence(rates, n_landmarks, seed=seed)["ring_score"]
                ring, p = is_ring(score, null)
                recovered = ring_angle(embed(rates, seed=seed))
                aligned, _ = align(recovered, angle)
                grid = (np.arange(60) + 0.5) * 2 * np.pi / 60
                true_tc = tuning_curves(spikes["pref_angle"].values, grid, kappa=kappa)
                res = decode(spikes, true_tc, spikes.time_support, sigma_grid=sigma_grid)
                results.append(
                    {
                        "n_cells": n,
                        "kappa": kappa,
                        "seed": seed,
                        "ring_score": score,
                        "ring_p": p,
                        "is_ring": ring,
                        "manifold_mae": angle_error(aligned, angle),
                        "decode_mae": angle_error(res.angle, angle),
                        "decode_sigma": res.sigma,
                        "runtime_s": time.perf_counter() - t0,
                    }
                )
    if out_path is not None:
        save_json(
            {"config": config, "config_hash": config_hash, "results": results}, out_path, overwrite
        )
    return results
