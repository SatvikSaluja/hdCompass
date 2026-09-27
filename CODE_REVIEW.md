# hdcompass — code review, system walkthrough and handoff

*Reviewed 2026-09-26 at commit `316945e` (branch `main`, 9 commits, no remote). Scope: all of
`src/`, `tests/`, `scripts/`, `notebooks/`, CI and packaging, ~1,700 lines.*

> **Status update 2026-09-27** (after the full runs; see `RESULTS.md`):
>
> | finding | status |
> |---|---|
> | R1 in-sample validation | **Measured:** held-out equals in-sample to within 0.005 rad on the full wake |
> | R2 sleep gain | **Measured:** uniform rescaling leaves σ unchanged and lowers log L; not the explanation |
> | R3 σ grid edge | **Checked:** no fit at a grid edge; still no automatic flag in `DecodeResult` |
> | R4 `is_ring` interpolation | **Fixed** (`method="higher"`), with a test |
> | R5 pipeline untested | **Fixed:** `run_pipeline(session)` plus a synthetic-session end-to-end test |
> | R6 CI never run | Open (no remote) |
> | C7 calibration | **Measured:** 64% coverage on real data, 90–92% on synthetic |
> | New | The as-specified H1 test is negative on the full real wake; density filtering fixes it (exploratory) |

This document has four parts:

1. [Verdict](#1-verdict): the short answer.
2. [How the system works](#2-how-the-system-works): a module-by-module explanation, with the
   maths, for whoever maintains this next.
3. [Review findings](#3-review-findings): ranked by severity, each with `file:line`, evidence
   and a fix.
4. [Handoff](#4-handoff): how to run it, what the full runs will cost, what to decide first, and
   next steps.

Every number below comes from a command run during this review or the build session. Numbers
labelled **estimate** are extrapolated from measured component timings, not measured end to
end. Nothing here is a scientific result: the full pipeline has not been run.

---

## 1. Verdict

**The code is sound. The pipeline's science needs three decisions before the full run.**

- Engineering quality is good. There are 22 tests (all pass, ~32 s). `ruff` is clean. Both
  `--quick` smoke runs work and are deterministic (two runs gave identical reports). Modules
  are small (largest is 223 lines), there are no circular imports, and docstrings are
  complete.
- No Critical findings: nothing crashes, loses data or has a security hole.
- Six **Required** findings. Four affect what the numbers mean rather than whether the code
  runs:
  - the wake validation is in-sample;
  - sleep is decoded with un-rescaled wake tuning, although population rates differ by 7–14%;
  - a σ that lands on the edge of its grid is reported silently;
  - `is_ring` can say "ring" when the score doesn't beat every shuffle.

  The other two are gaps in engineering hygiene:
  - the main pipeline function has no automated test;
  - CI has never actually run.
- Performance is fine for the machine. The full Mouse32 run is an **estimate** of 5–10 min at
  ~3–4 GB RAM; the full synthetic sweep is an **estimate** of 1–2 h, dominated by the ripser
  shuffle null.

---

## 2. How the system works

### 2.1 The question and the idea

The anterodorsal thalamus (ADn) holds head-direction (HD) cells. Each fires most when the head
points at its preferred angle. Together they behave like a ring attractor: population activity
lives on a 1-D circle, and a bump of activity moves around it. The bump keeps moving during
sleep, when there is no behaviour to label it with.

hdcompass recovers that circle **from spikes alone**, builds a decoder from it, and asks how the
compass moves in wake, REM and slow-wave sleep. Tracked head angle is used **only** to check
the answer.

### 2.2 Data flow

```
                         ┌──────────── synth.simulate_hd ─────────────┐ (tests, benchmark)
                         │ latent θ(t): wrapped Gaussian walk + jumps │
                         │ spikes: Poisson(von Mises(θ − φᵢ))         │
                         └──────────────────────┬─────────────────────┘
datasets.load_mouse32 ───► Session(spikes[ADn], angle, wake, rem, sws)
                                                │
features.population_rates(spikes, wake)         ▼
   per-epoch 0.1 s counts → √ → Gaussian smooth (σ=0.1 s) → drop 2σ edge bins → z-score
                                                │  TsdFrame (time × units)
             ┌──────────────────────────────────┼────────────────────────────┐
             ▼                                  ▼                            │
topology (is there a ring?)          manifold (where on it?)                 │
 PCA-10 → 800 landmarks → ripser H1   PCA-10 → Isomap-2D → Kåsa circle       │
 ring_score = life₁ / life₂           ring_angle θ̂(t) ∈ [0, 2π)              │
 shuffle_null: roll each unit                    │                           │
 is_ring: score vs null quantile                 ▼                           │
                                  tuning.tuning_from_ring(spikes, θ̂, wake)   │
                                     60-bin tuning curves, no behaviour      │
                                                 │                           │
                                                 ▼                           │
                              decode.decode(spikes, tuning, ep)  per state   │
                               grid HMM: σ by max marginal likelihood        │
                               → posterior-mean angle, circular variance     │
                                                 │                           │
                                                 ▼                           │
                              dynamics.state_dynamics → per-state summary    │
                                                                             │
validation only: align.align(θ̂ or decoded, head angle) → sign, offset ──────┘
                 align.angle_error → circular MAE (wake)
```

The orchestration lives in `cli.run_mouse32` (real data) and `benchmark.recovery_sweep`
(synthetic).

### 2.3 Module by module

#### `circular.py`: angle arithmetic
- `wrap` maps to [0, 2π). It includes a guard: `np.mod(-1e-17, 2π)` rounds to exactly 2π, which
  is outside the range, so any result ≥ 2π becomes 0.
- `circ_diff` gives the signed difference in (−π, π]. `circ_mean` is the angle of the weighted
  mean of e^{iθ}. `circ_mae` is the mean of |circ_diff|.
- `circ_corr` is the Fisher–Lee correlation. The O(n²) pairwise definition
  Σ_{i<j} sin(aᵢ−aⱼ) sin(bᵢ−bⱼ) / √(Σ sin²(aᵢ−aⱼ) Σ sin²(bᵢ−bⱼ)) is expanded with
  sin(x−y) = sin x cos y − cos x sin y into sums of products, which makes it O(n). It is +1 for
  b = a + c and −1 for b = −a + c (tested).
- `epoch_diffs` and `unwrap_speed` loop over the Tsd's time support, so no difference or
  gradient is ever taken across a gap between epochs.

#### `synth.py`: ground truth
- The latent angle is θ₀ + cumsum(N(0, σ²·dt)). Optional jumps are Poisson with rate
  `jump_rate`, and each one moves to a uniform new angle. Jumps are vectorised: a per-segment
  offset `U_k − base[k_s]` is added from each jump onward.
- Preferred angles are evenly spaced, plus a jitter of ±¼ spacing. Tuning is
  `base + (peak − base)·exp(κ(cos(θ−φ) − 1))` (`tuning_curves`, which is also used as "true
  tuning" by the benchmark and the tests).
- Spikes are Poisson counts per 1 ms step, with times uniform inside the step. The simulation
  loops one cell at a time, so memory stays at O(T), not O(T × cells).

#### `datasets.py`: Mouse32 loader
- `Session` is a frozen dataclass: ADn-only `TsGroup` (31 units), head angle `ry`, and the
  `wake`, `rem` and `sws` interval sets. `wake` is the epoch tagged `wake` intersected with
  `ry`'s support.
- It downloads from OSF only if the file is missing. The file streams to a `mkstemp` file in
  the same directory and is then renamed with `os.replace`. On any exception the temp file is
  deleted, so a partial download is never left behind.

#### `features.py`: rate vectors
- `bin_counts(spikes, ep, bin_size)` gives each epoch `floor(len / bin_size)` left-closed bins.
  Counts come from `np.diff(np.searchsorted(spike_times, edges))`, which is exact and O(n log n).
  Empty epochs yield empty arrays, so the output stays index-aligned with `ep`. This helper is
  shared by `population_rates` and the decoder.
- `population_rates` does five steps:
  1. Drop units below `min_rate` on `ep`.
  2. Take √counts, which roughly stabilises Poisson variance at ¼.
  3. Smooth each epoch with `gaussian_filter1d(mode="nearest", truncate=3)`, giving a 6σ window.
  4. Drop bins whose centre is within 2σ of an epoch edge. This is the pynapple#623 guard.
  5. Z-score each unit. Units whose std is effectively zero divide by 1 instead of amplifying
     round-off. That was a real bug, caught by a test during the build.

#### `topology.py`: is there a ring?
- `h1_persistence`:
  1. PCA to 10 dims when there are more columns (denoising).
  2. ripser's greedy-permutation subsample (`n_perm=800`) as the farthest-point landmarks, with
     `seed` rolling the start point.
  3. The Vietoris–Rips H1 diagram.

  It returns the diagram, lifetimes sorted descending, and `ring_score = life₁ / life₂`. A single
  dominant loop gives a large ratio; noise gives many similar small loops and a ratio near 1.
- `shuffle_null` rolls each unit's time series by an independent random lag. That keeps every
  unit's marginal statistics and autocorrelation but destroys the co-activation that forms the
  ring. It then recomputes the score.
- `is_ring` returns `score > quantile(null, 1 − α)` together with the permutation p-value
  `(1 + #{null ≥ score}) / (n + 1)`. See finding **R4**.

#### `manifold.py`: where on the ring?
- `embed`: PCA to 10 dims, then Isomap (15 neighbours, 2-D) fitted on at most 4000 random points
  and applied to all of them. `method="pca"` skips Isomap. The result keeps the rates' time index.
- `ring_angle` uses the Kåsa algebraic circle fit: solve x² + y² + a·x + b·y + c = 0 by least
  squares on centred coordinates. The centre is (−a/2, −b/2) and θ̂ = atan2 about that centre.
  The fit is linear and closed-form, with no iterations.

#### `align.py`: validation only
- A label-free angle has an arbitrary origin and direction. For sign ∈ {+1, −1}:
  offset = circ_mean(ref − sign·rec), and the sign with the lower MAE is kept. The fit uses only
  samples inside the reference's support; the transform is applied everywhere.
- The reference is sampled at the estimate's timestamps with pynapple `value_from` (nearest
  sample; `ry` is at 39 Hz).
- Only rotation and reflection are removed. Any **non-uniform warp** of θ̂ relative to true
  heading stays in the error (see **C9**).

#### `tuning.py`
- `fit_tuning` calls `nap.compute_tuning_curves` on 60 bins over [0, 2π). Unvisited bins are
  filled with the unit's mean, then smoothed circularly with `gaussian_filter1d(mode="wrap")`
  and floored at 0.05 Hz, so the Poisson log-rate is finite.
- `tuning_from_ring` is the same function called with θ̂ in place of head angle. This is what
  makes the whole pipeline label-free.

#### `decode.py`: the latent model
- **State:** the angle on a 60-point grid, gᵢ = (i + ½)·2π/60, spacing 0.105 rad (6°).
- **Transition:** a circulant kernel k[j] = P(move j steps). It is a wrapped Gaussian with
  std s = σ√Δt, summed over ±(⌈4s/2π⌉ + 1) wraps and normalised, then mixed with a uniform
  jump: k ← (1−ε)k + ε/n. σ = ∞ gives a uniform kernel, i.e. independent bins (the
  `decode_bayes`-style baseline).
- **Emission:** log p(cₜ | g) = Σᵤ cₜᵤ log(λᵤ(g)Δt) − Σᵤ λᵤ(g)Δt − Σᵤ log cₜᵤ!. This is one
  matrix product in NumPy.
- **Scaled forward–backward in JAX** (`_fb`, jit-compiled, two `lax.scan`s):
  - eₜ = exp(log pₜ − maxₜ) removes underflow, and the max goes back into the log-likelihood.
  - Forward: αₜ ∝ (k ⊛ αₜ₋₁) ⊙ eₜ, with cₜ = Σ(…) and αₜ normalised by cₜ. The convolution ⊛ is
    `irfft(rfft(α)·rfft(k))`, clipped at 0 to remove FFT round-off, with a floor of 1e-300 so a
    row can never be all zero.
  - Backward: βₜ = k ⊛ (eₜ₊₁ ⊙ βₜ₊₁) / cₜ₊₁. The kernel is symmetric, so kᵀ = k.
  - Posterior ∝ αₜ βₜ. log L = Σ log cₜ + Σ maxₜ − Σ log c!.
- **Epochs without recompiling.** All epochs of a state are concatenated with a `reset` flag
  at each epoch start. At a reset the prior is uniform and β is set to 1. That is exactly
  equivalent to running each epoch separately, and a test checks posterior and log L to
  1e-9. It means one JIT compile per distinct length instead of one per epoch.
- **Float64** is enabled globally at import (`jax_enable_x64`). In float32, posterior tails and
  FFT round-off collide.
- `fit_sigma` evaluates log L on a 12-point log grid from 0.1 to 50 rad/√s and takes the argmax.
  `decode` returns `DecodeResult(angle, variance, loglik, sigma, eps, bin_size)`. The angle is
  the posterior circular mean arg Σ p(g)e^{ig}; the variance is 1 − |Σ p(g)e^{ig}|.
- **Why a grid rather than a particle filter** (also in the module docstring): the state is 1-D,
  so a grid is exact up to its resolution, deterministic and cheaper.

#### `dynamics.py`
Per state it reports:
- `sigma` and `eps`, as used;
- `median_abs_speed`: the per-epoch gradient of the unwrapped posterior-mean angle;
- `jump_fraction`: the fraction of within-epoch steps larger than π/2;
- `mean_circ_variance` and `n_bins`.

#### `benchmark.py`
- `recovery_sweep` iterates over n_cells × κ × seeds. For each configuration it runs
  simulate → rates → null, score, `is_ring` → embed, ring angle, align → decode with the
  **true** tuning and fitted σ.
- It writes `{"config", "config_hash" (sha256[:12]), "results"}` as strict JSON: NaN and inf
  become `null`. It refuses to overwrite before any computation starts.
- `quick` means one configuration of 60 s, 5 shuffles and a 4-value σ grid.

#### `plots.py`
Each function builds a `matplotlib.figure.Figure` directly, without pyplot, so it never needs
a GUI backend and never calls `show`. Functions: `plot_ring`, `plot_barcode`, `plot_decode`,
`plot_state_dynamics`, `plot_sweep`.

#### `cli.py`
- `hdcompass mouse32` and `hdcompass sweep`, each with `--out`, `--quick` and `--overwrite`;
  `mouse32` also takes `--data`.
- `run_mouse32` runs the whole real-data pipeline: ring → tuning from ring → decode each state
  → dynamics → validation against head angle. That validation covers the ring angle, the HMM
  with ring tuning, the independent decoder with ring tuning, and the HMM with head tuning.
- Output: `report.json` plus `ring.png`, `barcode.png`, `decode_wake.png` and `dynamics.png`.
- `first_seconds(ep, s)` implements the `--quick` slicing (120 s per state).

#### `scripts/`, `notebooks/figures.py`
- The scripts are thin wrappers around `cli.main` with default output under `results/`.
- `figures.py` is a jupytext percent script. `QUICK=True` reproduces the smoke figures; `False`
  runs the full Mouse32 pipeline and plots the sweep from `results/sweep/report.json` if it
  exists.

### 2.4 `report.json` (mouse32)

| key | meaning |
|---|---|
| `n_adn_units`, `n_units_used` | 31 ADn units; how many pass `min_rate` on wake |
| `durations_s` | seconds analysed per state |
| `topology.{ring_score, null, p, is_ring, n_shuffles}` | ring test |
| `manifold.{align, ring_angle_mae_rad}` | unsupervised ring angle vs head angle (wake) |
| `decode_wake_validation.*` | wake MAE for the HMM (ring tuning), the independent decoder (ring tuning) and the HMM (head tuning), plus the alignment used |
| `loglik` | per-state log marginal likelihood |
| `dynamics.<state>.*` | σ, ε, median \|speed\|, jump fraction, mean circular variance, n_bins |

Angles and speeds in `dynamics` are in the **ring frame**. It equals the head frame up to sign
and offset only if θ̂ has no warp.

---

## 3. Review findings

Severity: **Critical** blocks merge (none found). **Required** should be addressed before the
full run or before publishing numbers. **Consider** is worth doing. **Nit** is optional.
**FYI** is informational.

### Required

**R1. The wake validation is in-sample, so the reported MAEs are optimistic.**
`src/hdcompass/cli.py:73-81`
- Tuning curves (from θ̂ and from head angle) and σ are fitted on the same wake bins that are
  then decoded and scored. With 60 bins × 19 units that is 1,140 fitted rates.
- In `--quick` mode head-angle coverage is very uneven. Measured occupancy per 30° bin in the
  first 120 s: `[733 568 288 123 1010 274 273 251 57 173 338 299]` samples at 39 Hz. So one bin
  rests on ~1.5 s of data.
- **Fix:** split wake 2-fold, alternating ~60 s blocks to respect autocorrelation. Fit tuning
  and σ on one half and score the other, then swap. Report both in-sample and held-out MAE. The
  same applies to `tc_head`.

**R2. Sleep is decoded with wake tuning. The state comparison, which is the headline output,
is confounded by gain.** `src/hdcompass/cli.py:73-74`
- Measured population rate over all 31 ADn units: wake 145.5 Hz, REM 135.6 Hz (−7%), SWS
  124.6 Hz (−14%).
- A rate mismatch changes the Poisson likelihood's −λΔt term. That shifts the fitted σ and the
  posterior variance in exactly the direction that would make SWS look "noisier".
- **Fix (pick one and record it):** (a) scale the tuning per state by that state's population
  rate ÷ the wake population rate (one line, keeps the model honest), or (b) fit a scalar gain
  per state by maximum likelihood alongside σ. Either way, report the gain used.

**R3. A σ optimum on the edge of the grid is reported without any warning.**
`src/hdcompass/decode.py:176-186`
- `fit_sigma` returns `argmax` over a fixed grid. Measured effective kernel std ÷ nominal s at
  n_grid=60, Δt=0.02 s:
  - σ = 0.1 → 0.00 (the kernel is a pure delta: "frozen");
  - σ = 0.3 → 0.73;
  - σ = 0.5 to 6.3 → 1.00 (accurate);
  - σ = 20 → 0.63 and σ = 50 → 0.26 (saturating toward uniform).
- So the grid's lowest points (0.1, 0.18) are indistinguishable from each other, and its top
  points are near-uniform per bin. An optimum at either end means "outside the resolvable
  range", not a measured σ.
- **Fix:** add `"sigma_at_grid_edge": bool` to `DecodeResult` and to the report, and warn when
  it is true. Optionally refine by golden-section search on log σ between the argmax
  neighbours. Consider narrowing the default grid to 0.3–30.

**R4. `is_ring` can return True when the score does not beat every shuffle.**
`src/hdcompass/topology.py:100`
- `np.quantile` interpolates. With 5 shuffles, `np.quantile([1, 1, 1, 1.4, 1.6], 0.95)` = 1.56.
  A score of 1.58 is then "ring" even though one shuffle beats it (p = 2/6 = 0.33).
- **Fix:** `np.quantile(null, 1 - alpha, method="higher")`. For n ≤ 1/α this is the null
  maximum; for n = 20 it is also the maximum. Add a two-line test with the null above.

**R5. The main pipeline, `run_mouse32`, and 4 of the 5 plot functions have no automated test.**
`src/hdcompass/cli.py:41-119`
- It is the most complex function in the repo, 80 lines. It was exercised only by the manual
  `--quick` run, and CI can never reach it because the NWB file is absent there.
- `plot_ring`, `plot_barcode`, `plot_decode` and `plot_state_dynamics` are covered only by that
  manual run.
- **Fix:** split out `run_pipeline(session, out, quick, …)` so `run_mouse32` only loads data
  and calls it. Then add a test that builds a synthetic `Session` (simulate 60 s, cut it into
  "wake/rem/sws" intervals, use the true angle as `angle`) and asserts on the report keys and
  the four PNGs. That is ~20 lines and runs in about 10 s.

**R6. CI has never run.** `.github/workflows/ci.yml`
- There is no remote, so the workflow is unverified.
- Two things to check on the first push, neither of which I could verify offline:
  1. `setup-python` with `cache: pip` looks for a dependency file. Set
     `cache-dependency-path: pyproject.toml` explicitly so it doesn't depend on the action's
     defaults.
  2. `requires-python = ">=3.10"`, but CI tests only 3.11 and 3.12, and I believe current JAX
     no longer supports 3.10. Either raise the floor to 3.11 or add 3.10 to the matrix.

### Consider

**C1. Parallelise the shuffle null; it dominates the run time.** `src/hdcompass/topology.py:70`
- Measured: one `h1_persistence` call on 19,500 points (full-wake size) takes 9.5 s. The full
  run makes 21 calls, ≈ 3.3 min (**estimate**). The sweep makes 21 per configuration × 36
  configurations, ≈ 1–2 h (**estimate**).
- The shuffles are independent. `concurrent.futures.ProcessPoolExecutor` (stdlib) over 8 cores
  would cut this by about 6–8×. Seed each worker from `rng.integers` so results stay
  deterministic.

**C2. `fit_sigma` does twice the necessary work.** `src/hdcompass/decode.py:185`
- It runs the full forward–backward, including the posterior, just to read log L. A forward-only
  jitted function would halve the cost of each grid point.
- `decode(sigma=None)` also bins the spikes twice (once in `fit_sigma`, once in `decode`).
- Measured: 300k bins take 5.1 s per forward–backward. Full SWS is 620k bins × 13 passes
  ≈ 2.3 min (**estimate**), so this saves about 1 min per full run. Low priority.

**C3. Verify the download's integrity.** `src/hdcompass/datasets.py:42`
- The download is HTTPS but not checksummed. Add `MOUSE32_SHA256` and check it before
  `os.replace`.
- Compute the digest once from the cached file:
  `sha256sum /home/satvik/side_project/cache/public/Mouse32-140822.nwb`.

**C4. One canonical helper for grid centres.** `(np.arange(n) + 0.5) * 2π / n` appears in 4
places: `decode.py:215`, `benchmark.py:108`, `tests/conftest.py:10` and `tests/test_dynamics.py:12`.
Add `decode.grid_angles(n)` and use it in all four.

**C5. Generic JSON I/O lives in `benchmark.py`.** `jsonable` and `save_json`
(`benchmark.py:19,34`) are imported by `cli.py`. Move them into `cli.py`, or into the
`pipeline.py` from R5, so the benchmark module stays about benchmarking.

**C6. Tuning rows are matched to units by count only.** `src/hdcompass/decode.py:162`
- `_counts` checks `tuning.shape[0] == len(spikes)` but not that row *i* belongs to unit
  `keys()[i]`. It works today because `compute_tuning_curves` returns units in key order.
- Consider keeping the xarray from `fit_tuning` (it carries the `unit` coordinate) and checking
  `tc.unit == spikes.keys()`.

**C7. Check whether the posterior is calibrated.**
- In the smoke run, mean wake circular variance was 0.0084, a posterior circular std of about
  0.13 rad, while the wake MAE was 0.32 rad.
- Part of that gap is R1 and C9, but independent-Poisson emissions also ignore noise
  correlations and overstate the evidence.
- Add a coverage check: the fraction of bins where the true angle falls inside the 90%
  posterior interval. Do it on synthetic data (where the model is correct and coverage should be
  ≈ 90%) and on real wake data.

**C8. Add a warp-invariant validation metric.**
- `align` removes only rotation and reflection. Report `circ_corr(θ̂, head)` too (measured −0.84
  on the 120 s slice; the sign just reflects the reflection).
- I measured whether warp is a large error source. A 2-fold cross-validated 30-bin map from θ̂
  to head angle gave MAE 0.30 and 0.37, against 0.34 for sign+offset. So at 120 s warp is **not**
  the dominant error. Re-check on the full wake.

**C9. Keep orchestration separate from the CLI.** `run_mouse32` mixes loading, pipeline,
reporting and plotting. The R5 split (`pipeline.py`) fixes this too.

**C10. Global side effect of `jax_enable_x64`.** `src/hdcompass/decode.py:18`
- Importing `hdcompass.cli` or `hdcompass.decode` switches the whole process to float64 JAX.
  That's fine for this package, but it would double memory for any other JAX code in the same
  process.
- Mention it in the README, or scope it with `jax.experimental.enable_x64()` around `_fb` calls.

### Nits

- **N1** `topology.py:98`: `np.isscalar` is False for a 0-d NumPy array, so
  `is_ring(np.float64(…))` works but `is_ring(np.array(3.0))` would crash. Use
  `np.ndim(X) == 0`.
- **N2** `decode.py:32`: `eps` is not checked to be in [0, 1].
- **N3** `pyproject.toml:11` declares MIT, but there is no `LICENSE` file.
- **N4** `circular.py`: `circ_corr` on constant input returns NaN with a divide warning.
- **N5** `scripts/*.py`: `"--out" in args` misses `--out=DIR`. That's harmless, because argparse
  keeps the last value and the user's value comes after the default.
- **N6** `cli.py:117-118`: the report is written before the figures. If a figure fails, the
  report exists and a rerun needs `--overwrite`. Save the figures first.
- **N7** `features.py:69,76`: the kernel is truncated at 3σ but the guard drops only 2σ, so bins
  between 2σ and 3σ from an edge still get at most 2.3% padded weight. This is documented and
  immaterial.

### FYI: checked and fine

- **Data integrity:** `ry` has 0 NaNs in 71,478 samples. REM (29 intervals) and SWS (44) have no
  touching or overlapping intervals (min gaps 55 s and 2 s). REM, SWS and wake are pairwise
  disjoint.
- **The `min_rate` drops are real, not a quick-mode artifact.** Over the full wake, 12 of 31 ADn
  units fire under 1 Hz (rates 0.02–0.70 Hz), so 19 are used in both quick and full mode. They
  may still be HD-tuned; lowering `min_rate` to about 0.3 Hz would keep 5 more (24 units) for decoding.
  This supersedes open question 3 in HANDOFF.md.
- **Decoder edge cases:** T = 1 and T = 2 give valid posteriors and a finite log L.
- **Determinism:** two `embed` calls on the same data differ by at most 1e-14 (threaded BLAS).
  Two full `--quick` Mouse32 runs gave identical reports.
- **Kernel accuracy** is within 0.2% for σ from 0.5 to 6.3 rad/√s, which covers the smoke-run
  fits (wake and REM 0.79, SWS 6.3).
- **Security:** no secrets and no shell or eval. Output goes only to the user's `--out`. The
  input NWB is parsed by pynwb/h5py; treat files from untrusted sources with the usual caution.
  The download is HTTPS (see C3).
- **Tests are meaningful.** They test behaviour, not implementation details. The strongest ones
  are the reset-equivalence test (it would catch any cross-epoch leakage in the HMM), the
  exact-alignment tests, and the ML-σ test.
- **Dead code:** none that matters. `DecodeResult.bin_size` and `circ_corr` are unused by the
  pipeline but were requested by the plan and are informative. `tuning_from_ring` is a named
  pass-through that the plan asked for.

### Plan compliance

| PLAN.md item | status |
|---|---|
| §1 env, deps, `JAX_PLATFORMS=cpu` in conftest | ✔ (constraints via `--exclude-editable`) |
| §2 layout | ✔ all files present |
| §3.1–3.12 signatures and behaviour | ✔. Additive changes: `n_pcs` (topology), `sigma_grid` (decode), `resets` (forward_backward), `--overwrite` and `--data` (CLI), `h1_persistence` also returns `ring_score`. The benchmark records `manifold_mae` and `decode_mae` instead of one `circ_mae`. |
| §4 test list | ✔ every listed test exists and passes; the suite takes ~32 s (budget 90 s) |
| §5 docs | ✔ README says "Results: not yet run". The plan's "Rubin 2019 eLife" is cited as *Nat Commun* 10:4745 (flagged, please verify). |
| §6 acceptance | ✔ all four commands ran, then git init with per-group commits |

---

## 4. Handoff

### 4.1 Where things are

```
src/hdcompass/     13 modules (walkthrough in §2.3)
tests/             11 files, 22 tests; conftest has the shared 40-cell/300 s fixture `hd40`
scripts/           run_mouse32.py, run_synthetic_sweep.py (full runs, not yet executed)
notebooks/         figures.py (jupytext percent script)
PLAN.md            original spec · HANDOFF.md short handoff · this file: review + detail
~/.cache/hdcompass/Mouse32-140822.nwb → symlink to /home/satvik/side_project/cache/public/…
```

### 4.2 Commands

```bash
cd /home/satvik/hdcompass
.venv/bin/ruff check . && .venv/bin/pytest -q                  # ~32 s
.venv/bin/hdcompass sweep   --quick --out /tmp/hdc_sweep       # ~15 s
.venv/bin/hdcompass mouse32 --quick --out /tmp/hdc_m32         # ~21–29 s
.venv/bin/python scripts/run_mouse32.py                        # full: estimate 5–10 min, ~3–4 GB
.venv/bin/python scripts/run_synthetic_sweep.py                # full: estimate 1–2 h
```

Never run Python from `/home/satvik/contri`, because the source clone there shadows the
installed pynapple.

### 4.3 Resource estimates for the full runs (from measured components)

| component | measured | full Mouse32 (**estimate**) |
|---|---|---|
| ripser, 800 landmarks of 19.5k points | 9.5 s/call | 21 calls ≈ 3.3 min |
| Isomap fit on 4000 + transform 19.5k | 42.5 s | once ≈ 0.7 min |
| forward–backward, 300k bins | 5.1 s, RSS 1.5 GB | SWS 620k bins × 13 ≈ 2.3 min; wake and REM ≈ 1 min; peak ≈ 3 GB |
| **total** | | **≈ 5–10 min, ≈ 3–4 GB** (machine: 8 cores, ~11 GB free) |

Sweep: per configuration, 21 ripser calls on 6000 points plus Isomap on 4000 plus 13 decoder
passes on 30k bins, ≈ 1.5–3 min × 36 configurations ≈ **1–2 h (estimate)**. Doing C1 first
would cut this several-fold.

### 4.4 Decide before the full Mouse32 run (in this order)

1. **R2 gain:** rescale the tuning per state, or fit the gain. Without one of these the SWS and
   REM σ and variance aren't comparable with wake.
2. **R1 held-out validation:** otherwise the MAEs in README *Results* will be in-sample.
3. **R3 σ grid edge flag:** otherwise an SWS σ at the grid ceiling would be reported as a number.
4. **`min_rate`:** keep 1 Hz (19 units) or lower it to about 0.3 Hz (24 units)?
5. **ε:** keep it fixed at 0, or fit it jointly with σ on a small 2-D grid? With ε = 0, SWS
   jumps are absorbed into a larger σ.

R4 (a one-token fix) and R5/R6 (tests and CI) are cheap and should go in the same pass.

### 4.5 Next steps

1. Apply R1–R4, with a test for each.
2. Refactor per R5 (`pipeline.py` plus a synthetic-Session test). Push to a remote and get CI
   green (R6).
3. C1 (parallel null), then run the full synthetic sweep. Check that recovery error falls with
   cell count and that `is_ring` is true for n ≥ 16 before trusting the real-data ring test.
4. Run the full Mouse32 pipeline. Fill in README *Results* **only** from `results/*/report.json`.
5. C7 calibration check, then report posterior coverage alongside the MAE.
6. CRCNS th-1 multi-session loader, `load_th1(session_dir) -> Session`. The data is Neuroscope
   format (`.res/.clu/.whl`), not NWB, and downloading it needs your CRCNS account. This is
   what lifts the "one session, 31 cells" limitation.

### 4.6 Smoke-run values (for orientation only, not results)

`/tmp/hdc_m32/report.json` (120 s per state, 5 shuffles, 4-value σ grid):
- ring_score 3.64 against a null max of 1.60;
- wake MAE: ring angle 0.34 rad, HMM (ring tuning) 0.32, independent decoder 0.56, HMM (head
  tuning) 0.21;
- σ: wake 0.79, REM 0.79, SWS 6.3 rad/√s;
- jump fraction: 0, 0.0005, 0.027;
- mean circular variance: 0.008, 0.011, 0.084.

Per R1 and R2 these are in-sample and gain-confounded.
