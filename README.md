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

Python ≥ 3.10, CPU only. Data: `Mouse32-140822.nwb` (Peyrache et al. 2015) is downloaded from
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
.venv/bin/python scripts/run_mouse32.py          # → results/mouse32/
.venv/bin/python scripts/run_synthetic_sweep.py  # → results/sweep/ (4 sizes × 3 κ × 3 seeds)
```

`notebooks/figures.py` (jupytext percent format) regenerates all figures.

## Results

Not yet run. The full Mouse32 pipeline and the synthetic sweep have not been executed; only
the `--quick` smoke runs above have.

## Limitations

- **One session, few cells.** Mouse32-140822 has 31 ADn units; units under 1 Hz on wake are
  dropped (`min_rate`). All conclusions are for this one animal and session.
- **Isomap hyperparameters.** `n_neighbors=15`, PCA to 10 dims and the 4000-point fit
  subsample are fixed choices, not tuned; the ring angle can depend on them.
- **Epoch-edge guard.** pynapple's `smooth` zero-pads at epoch edges (pynapple#623). We smooth
  each epoch separately and drop bins within 2·`smooth_std` of every edge, which removes
  those bins from analysis rather than correcting them.
- **Sleep uses wake tuning.** Tuning curves are fitted on wake and applied unchanged to REM and
  SWS; no correction for state-dependent gain.
- **Jump probability ε is fixed** (default 0), not fitted; only σ is chosen by likelihood.
- **Ring test resolution.** With `n` shuffles the permutation p-value cannot go below
  `1/(n+1)`; `is_ring` therefore decides by the null's 95th percentile and reports p
  separately (5 shuffles in `--quick`, 20 in full runs).

## References

- Peyrache A, Lacroix MM, Petersen PC, Buzsáki G (2015). Internally organized mechanisms of the
  head direction sense. *Nature Neuroscience* 18:569–575.
- Chaudhuri R, Gerçek B, Pandey B, Peyrache A, Fiete I (2019). The intrinsic attractor manifold
  and population dynamics of a canonical cognitive circuit across waking and sleep.
  *Nature Neuroscience* 22:1512–1520.
- Rubin A, Sheintuch L, Brande-Eilat N, Pinchasof O, Rechavi Y, Geva N, Ziv Y (2019). Revealing
  neural correlates of behavior without behavioral measurements. *Nature Communications*
  10:4745.
