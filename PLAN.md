# hdcompass — build plan (single phase)

**Goal:** a pip-installable, tested Python package that recovers the head-direction (HD)
"internal compass" from thalamic spikes **without behavioural labels**, decodes it with
uncertainty, and compares its dynamics across wake, REM and slow-wave sleep (SWS).

**Instruction to Claude Code:** build *everything* in this file in one pass. Do **not**
run long analyses, full synthetic sweeps or the full Mouse32 pipeline. Only run the fast
test suite and the `--quick` smoke commands listed under *Acceptance*. Heavy runs are the
user's to launch later. Never write result numbers into docs that were not produced by a
command you actually ran.

---

## 0. Verified facts (checked 2026-09-26, do not re-derive)

- Dataset: `Mouse32-140822.nwb` (Peyrache et al. 2015), already cached at
  `/home/satvik/side_project/cache/public/Mouse32-140822.nwb`. Public fallback:
  `https://osf.io/jb2gd/download` (~37 MB).
- `nap.load_file(path)` returns keys: `units` (TsGroup, 49 units, metadata `location`:
  31 `adn`, 18 `thalamus`), `ry` (Tsd, head angle, wake only, support 8812.4–10771.3 s),
  `epochs` (IntervalSet tagged `sleep`/`wake`/`sleep`), `sws` and `rem` (IntervalSets).
- `nap.compute_tuning_curves(data=, features=, bins=, epochs=, range=, feature_names=)`
  returns an `xarray.DataArray` (units × bins).
- **Pitfall:** `/home/satvik/contri/pynapple` is a source clone. Never run Python with
  cwd `/home/satvik/contri` — it shadows the installed pynapple. Work in `/home/satvik/hdcompass`.
- pynapple `smooth()` zero-pads at epoch edges (issue pynapple#623). Features must be
  computed per epoch and edges handled explicitly (see §3.3).
- Machine: 8 cores, ~11 GB free RAM, CPU only, Python 3.12, WSL2.

## 1. Environment

```bash
cd /home/satvik/hdcompass
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip freeze > constraints.txt
```

Runtime deps: `numpy`, `scipy`, `pynapple`, `jax[cpu]`, `scikit-learn`, `ripser`, `matplotlib`.
Dev deps: `pytest`, `ruff`. Nothing else. Set `JAX_PLATFORMS=cpu` in tests (conftest).

## 2. Layout

```
hdcompass/
  pyproject.toml          # hatchling or setuptools, python>=3.10, [dev] extra, console script
  README.md
  HANDOFF.md
  constraints.txt
  .github/workflows/ci.yml  # ubuntu, py3.11+3.12, ruff check, pytest -q
  src/hdcompass/
    __init__.py
    circular.py   datasets.py  synth.py   features.py
    manifold.py   topology.py  align.py   tuning.py
    decode.py     dynamics.py  benchmark.py plots.py  cli.py
  tests/
    conftest.py  test_circular.py test_synth.py test_features.py test_topology.py
    test_manifold.py test_align.py test_decode.py test_dynamics.py test_datasets.py test_cli.py
  scripts/
    run_mouse32.py         # full real-data pipeline (written, NOT run in full)
    run_synthetic_sweep.py # recovery-vs-cells benchmark (written, NOT run in full)
  notebooks/
    figures.py             # jupytext-style percent script that regenerates all figures
```

All public functions take and return pynapple objects or numpy arrays, have numpy-style
docstrings, and are deterministic given `seed`.

## 3. Modules

### 3.1 `circular.py`
`wrap(a)` → [0, 2π); `circ_diff(a, b)` → (−π, π]; `circ_mean(a, w=None)`;
`circ_mae(a, b)`; `circ_corr(a, b)` (Fisher–Lee); `unwrap_speed(angle_tsd)` → angular
speed Tsd (rad/s) computed per epoch (never across epoch gaps).

### 3.2 `synth.py`
`simulate_hd(n_cells, duration, dt=0.001, kappa=4.0, peak_rate=30.0, base_rate=0.5,
sigma=1.0, jump_rate=0.0, seed=0) -> (TsGroup, Tsd)`
- Latent angle: wrapped Gaussian random walk, diffusion `sigma` rad/√s; optional Poisson
  jumps (`jump_rate` per s, uniform new angle) to mimic SWS-like discontinuities.
- Preferred directions evenly spaced + jitter; von Mises tuning
  `base + (peak-base)·exp(kappa(cos(θ-φ)-1))`.
- Spikes: Poisson counts per `dt` bin, spike times uniform within bin.
- Returns spikes with metadata `pref_angle`, and the true angle Tsd.

### 3.3 `datasets.py` and `features.py`
`load_mouse32(path=None, cache_dir="~/.cache/hdcompass") -> Session` where `Session` is a
frozen dataclass: `spikes` (ADn only), `angle`, `wake`, `rem`, `sws`. Download from OSF
only if the file is missing (stream to a temp file, then rename).

`population_rates(spikes, ep, bin_size=0.1, smooth_std=0.1, min_rate=1.0) -> TsdFrame`
- Drop units below `min_rate` (rate computed on `ep`).
- Count per epoch, `sqrt` transform, gaussian smooth **per epoch** with
  `windowsize = 6*smooth_std`, then **drop bins within 2·smooth_std of every epoch edge**
  (documented as the #623 edge guard). Z-score per unit over `ep`.

### 3.4 `topology.py`
`h1_persistence(X, n_landmarks=800, seed=0) -> dict` — farthest-point subsample, `ripser`
with `maxdim=1`, return H1 diagram and lifetimes sorted desc.
`ring_score(X)` = top H1 lifetime / second H1 lifetime.
`shuffle_null(rates, n_shuffles=20, seed=0)` — circularly shift each unit independently,
recompute `ring_score`; return null distribution. `is_ring(X, null)` → bool + p-value.

### 3.5 `manifold.py`
`embed(rates, method="isomap", n_components=2, n_neighbors=15, max_points=4000, seed=0)`
— PCA to 10 dims first, then scikit-learn Isomap fit on a subsample, transform all points.
`ring_angle(embedding) -> Tsd` — center, fit a circle by algebraic least squares (Kasa),
angle = `atan2`. Keep the same time index as `rates`.

### 3.6 `align.py`
Recovered angles have arbitrary offset and direction. `align(recovered, reference) ->
(aligned, {"sign": ±1, "offset": float})`: for each sign, offset = circ_mean of the
difference; choose the lower `circ_mae`. Fit only on the reference's support (wake); apply
the same transform everywhere else.

### 3.7 `tuning.py`
`fit_tuning(spikes, angle, ep, n_bins=60, smooth_bins=2) -> ndarray (units, n_bins)` via
`nap.compute_tuning_curves`, circular gaussian smoothing, floor at 0.05 Hz.
`tuning_from_ring(spikes, ring_angle, ep)` — same, but using the unsupervised angle (so
the whole pipeline can run with no behaviour at all).

### 3.8 `decode.py` — the latent model
Exact **grid HMM** on the circle (not a particle filter: the state is 1-D, so a
discretised grid is exact up to resolution, faster and has no Monte-Carlo noise — note
this reasoning in the docstring).
- State: `n_grid` (default = n_bins of tuning) angles. Transition: wrapped Gaussian with
  std `sigma*sqrt(bin_size)`, plus optional uniform jump prob `eps`; build as a circulant
  kernel applied by FFT convolution.
- Emission: Poisson log-likelihood of spike counts per bin given tuning × `bin_size`.
- `forward_backward(counts, tuning, bin_size, sigma, eps=0.0)` in JAX (`jax.lax.scan`,
  log-space normalisation) → posterior (T × n_grid), log marginal likelihood.
- `decode(spikes, tuning, ep, bin_size=0.02, sigma=None, eps=0.0) -> DecodeResult`:
  posterior mean angle (Tsd), circular variance (Tsd), log-likelihood; run **per epoch**,
  never across gaps. If `sigma is None`, choose it by max marginal likelihood on a log grid
  (`fit_sigma`, 12 values from 0.1 to 50 rad/√s).
- Baseline: `nap.decode_bayes`-style independent-bin decoder (`sigma=inf`) for comparison.

### 3.9 `dynamics.py`
`state_dynamics(result_by_state: dict[str, DecodeResult]) -> dict` — per state (wake, rem,
sws): fitted `sigma`, `eps`, median |angular speed|, fraction of bins with jumps > π/2,
mean posterior circular variance. Pure summary, no plotting.

### 3.10 `benchmark.py`
`recovery_sweep(n_cells=(8,16,32,64), kappas=(2,4,8), duration=600, seeds=(0,1,2)) -> list[dict]`
runs synth → features → topology → manifold → align, plus decode with true tuning.
Records `circ_mae`, `is_ring`, `ring_score`, runtime. Writes JSON with a config hash;
refuses to overwrite unless `overwrite=True`. `quick=True` → one config, 60 s duration.

### 3.11 `plots.py`
`plot_ring(embedding, color_angle)`, `plot_barcode(h1)`, `plot_decode(true, decoded, var)`,
`plot_state_dynamics(summary)`, `plot_sweep(results)`. Return figures; never `plt.show()`.

### 3.12 `cli.py`
`hdcompass mouse32 [--quick] --out DIR` and `hdcompass sweep [--quick] --out DIR`.
Writes `report.json` (+ PNGs). `--quick`: first 120 s of wake + 120 s of each sleep
state, 5 shuffles, 1 sigma grid of 4 values.

## 4. Tests (whole suite must run in < 90 s on CPU)

- `circular`: wrap edge cases, circ_diff antisymmetry, circ_mae of shifted copy = 0.
- `synth`: shape, determinism by seed, empirical rates near tuning within 20%.
- `features`: constant-rate input stays constant after smoothing (edge guard works);
  no bin crosses an epoch gap; units under `min_rate` dropped.
- `topology`: noisy circle → `is_ring` True; isotropic Gaussian blob → False.
- `manifold` + `align`: 40-cell synthetic, 300 s → aligned `circ_mae` < 0.35 rad;
  alignment recovers a known offset and a reflection exactly on noiseless data.
- `decode`: posterior rows sum to 1; synthetic 40 cells → `circ_mae` < 0.2 rad;
  marginal likelihood at the true `sigma` beats `sigma×10` and `sigma/10`;
  HMM error ≤ independent-bin decoder error.
- `dynamics`: synthetic with `jump_rate>0` shows higher jump fraction than without.
- `datasets`: skipped if the Mouse32 file is absent; else checks 31 ADn units, wake/rem/sws present.
- `cli`: `hdcompass sweep --quick` produces a valid report.json in a tmp dir.

## 5. Docs

- **README.md**: question, method diagram (text), install, `--quick` usage, full-run
  commands, a *Results* section that says "not yet run" until the user runs the full
  pipeline, limitations (one session, 31 cells, isomap hyperparameters, #623 edge guard),
  references (Peyrache 2015 Nat Neuro; Chaudhuri 2019 Nat Neuro; Rubin 2019 eLife).
- **HANDOFF.md**: what exists, exact commands, what was and was not executed, open
  questions, next steps (full Mouse32 run, synthetic sweep, CRCNS th-1 multi-session
  loader — th-1 is Neuroscope format, not NWB, needs the user's CRCNS account).

## 6. Acceptance (the only things to execute)

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/hdcompass sweep --quick --out /tmp/hdc_sweep
.venv/bin/hdcompass mouse32 --quick --out /tmp/hdc_m32
```

All four succeed. Then `git init`, one commit per module group on branch `main`
(no remote, no push unless the user asks).

## 7. Out of scope for this phase
Full sweeps, full-session runs, multi-session th-1, GPU, web UI, docs site.
