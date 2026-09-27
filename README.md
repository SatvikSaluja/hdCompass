# hdcompass

Recover the head-direction (HD) "internal compass" from anterodorsal thalamic spikes **without
behavioural labels**, decode it with uncertainty, and compare its dynamics across wake, REM and
slow-wave sleep (SWS).

## Question

The HD network is thought to be a ring attractor that keeps running during sleep (Peyrache et
al. 2015). Can we find that ring from spikes alone, turn it into a decoder with calibrated
uncertainty, and measure how the compass moves in each brain state — diffusion speed, jumps,
confidence — using no tracked head angle at all?

## Method

```
spikes (ADn units)
  │  population_rates: per-epoch counts → sqrt → Gaussian smooth → drop edge bins → z-score
  ▼
rate vectors (wake)
  ├─► topology: PCA-10 → farthest-point landmarks → ripser H1
  │        ring_score = top / second H1 lifetime, tested against a per-unit circular-shift null
  └─► manifold: PCA-10 → Isomap (2-D) → Kåsa circle fit → ring angle θ̂(t)
                     │
                     ▼
        tuning_from_ring: tuning curves against θ̂ (no behaviour used)
                     │
                     ▼
        decode: exact grid HMM on the circle, per epoch
                wrapped-Gaussian diffusion σ (fit by max marginal likelihood) + jump prob ε,
                Poisson emissions → posterior mean angle + circular variance
                     │
                     ▼
        dynamics per state (wake / REM / SWS): σ, median |speed|, jump fraction, mean variance

validation only: align θ̂ to tracked head angle (sign + offset fitted on wake) → circular MAE
```

Synthetic ground truth (`synth.simulate_hd`: von Mises tuning, wrapped random-walk latent,
optional Poisson jumps) is used by the tests and by the recovery-vs-cells benchmark.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'          # or add -c constraints.txt for the pinned set
```

Python ≥ 3.11, CPU only. Data: `Mouse32-140822.nwb` (Peyrache et al. 2015) is downloaded from
OSF (~37 MB) into `~/.cache/hdcompass/` on first use, or pass `--data PATH`.

## Quick smoke runs

```bash
.venv/bin/hdcompass sweep   --quick --out /tmp/hdc_sweep   # 1 synthetic config, 60 s
.venv/bin/hdcompass mouse32 --quick --out /tmp/hdc_m32     # 120 s of wake, REM and SWS each
```

Each writes `report.json` and PNG figures. Existing reports are never overwritten unless you
pass `--overwrite`.

## Full runs

```bash
.venv/bin/pip install -e '.[dev,dandi]'           # dandi extra: streaming for DANDI:000939
.venv/bin/python scripts/run_mouse32.py          # → results/mouse32/ (~12 min)
.venv/bin/python scripts/run_synthetic_sweep.py  # → results/sweep/ (4 sizes × 3 κ × 3 seeds, ~1 h)
.venv/bin/python scripts/validate.py real|synthetic|topology|sweep|decisions  # → results/validation/
.venv/bin/python scripts/run_dandi000939.py       # 22 post-subiculum sessions (~1 h, streams ~60 MB each)
```

`notebooks/figures.py` (jupytext percent format) regenerates all figures.

## Results

Full runs and validation completed 2026-09-27. Details, tables and caveats are in
[RESULTS.md](RESULTS.md). Headlines (Mouse32, 19 ADn units):

- **Label-free recovery works.** The unsupervised ring angle tracks head direction with a
  held-out error of 0.42 rad. The fully label-free HMM decoder reaches 0.41 rad, against 0.58
  for independent bins and 0.21 for a decoder trained on tracked head angle.
- **Compass dynamics by state:** σ = 0.54 (wake), 0.96 (REM), 2.97 (SWS) rad/√s. SWS is also
  about 10× less certain than wake. A uniform firing-rate gain drop doesn't explain this.
- **The ring test as first specified misses the ring** on the full wake (score 1.04,
  p = 0.48). A density filter (k-NN codensity, Carlsson et al. 2008) adopted by pre-registered
  checks detects it (4.92, p = 1/201) and is now the pipeline default.
- **It generalises to a second dataset and region.** On 22 post-subiculum sessions
  (DANDI:000939, analysis plan fixed in advance) the label-free decoder reaches a median
  0.36 rad. σ SWS > wake in 21/22 (p = 1.8e-5), SWS > REM in 19/20, and SWS variance > wake in
  21/22. REM ≈ wake there, and the ring test detects a ring in only 12/22.
- **Validated on ground truth.** Over 17,000 s Mouse32-shaped simulations (2 seeds), the
  label-free pipeline is within 0.012 rad of an oracle decoder in every state and recovers σ to
  the nearest grid point in wake and REM. It undercounts SWS jumps about 20×.
- **Real-data posteriors are overconfident:** 90% intervals cover 64% (synthetic: 90–92%).

## Limitations

- **One ADn session, few cells.** Mouse32-140822 has 31 ADn units; units under 1 Hz on wake
  are dropped (`min_rate`). The SWS result replicates on 22 post-subiculum sessions
  (DANDI:000939), but ADn replication needs more sessions (CRCNS th-1).
- **Isomap hyperparameters.** `n_neighbors=15`, PCA to 10 dims and the 4000-point fit
  subsample are fixed choices, not tuned; the ring angle can depend on them.
- **Epoch-edge guard.** pynapple's `smooth` zero-pads at epoch edges (pynapple#623). We smooth
  each epoch separately and drop bins within 2·`smooth_std` of every edge, which removes
  those bins from analysis rather than correcting them.
- **Sleep uses wake tuning.** Tuning curves are fitted on wake and applied unchanged to REM and
  SWS. A uniform gain correction was tested and rejected by likelihood (RESULTS.md §3.5);
  per-cell rate changes are untested.
- **Jump probability ε is fixed** (default 0), not fitted; only σ is chosen by likelihood.
- **Ring test resolution.** With `n` shuffles the permutation p-value cannot go below
  `1/(n+1)`; `is_ring` therefore decides by the null's 95th percentile and reports p
  separately (5 shuffles in `--quick`, 20 in full runs). The ring test is the least sensitive
  stage: it found a ring in only 12 of 22 DANDI:000939 sessions, though decoding worked in most.
- **σ is quantised** on a 12-point grid (factor 1.76 per step), and **posterior variances are
  overconfident on real data** (90% intervals cover 64%), so treat them as relative across
  states.

## References

- Peyrache A, Lacroix MM, Petersen PC, Buzsáki G (2015). Internally organized mechanisms of the
  head direction sense. *Nature Neuroscience* 18:569–575.
- Chaudhuri R, Gerçek B, Pandey B, Peyrache A, Fiete I (2019). The intrinsic attractor manifold
  and population dynamics of a canonical cognitive circuit across waking and sleep.
  *Nature Neuroscience* 22:1512–1520.
- Carlsson G, Ishkhanov T, de Silva V, Zomorodian A (2008). On the local behavior of spaces of
  natural images. *International Journal of Computer Vision* 76:1–12. (Codensity filter used
  by the ring test.)
- Duszkiewicz A, Skromne Carrasco S, Peyrache A (2026). Large-scale recordings of head direction
  cells in mouse postsubiculum (Version 0.260512.1701) [Data set]. DANDI Archive.
  https://doi.org/10.48324/dandi.000939/0.260512.1701 (CC-BY-4.0; original study Duszkiewicz
  et al. 2024, *Nature Neuroscience*).
- Rubin A, Sheintuch L, Brande-Eilat N, Pinchasof O, Rechavi Y, Geva N, Ziv Y (2019). Revealing
  neural correlates of behavior without behavioral measurements. *Nature Communications*
  10:4745.
