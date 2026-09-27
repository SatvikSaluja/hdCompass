# hdcompass — results and validation (2026-09-27)

All numbers come from report files produced by the commands in §6, at commit `fbacf85`. The
files live under `results/`, which is git-ignored; rerun the commands to regenerate them.
Angles are in radians. Chance-level MAE for a random angle is π/2 ≈ 1.57 rad.

## 1. Summary

| question | answer | evidence |
|---|---|---|
| Can the HD angle be recovered from ADn spikes with no labels? | **Yes.** Error is 0.42 rad (≈24°) on held-out wake, stable across embedding settings | §3.1 |
| Does a label-free decoder work? | **Yes.** HMM with ring-derived tuning: 0.41 rad held-out, vs 0.58 for an independent-bin decoder and 0.21 for a decoder given tracked head angle | §3.1 |
| Does the pre-specified topology test detect the ring on the full wake? | **No.** Ring score 1.04, p = 0.48. It passes on synthetic data of the same length, and an exploratory density-filtered variant detects it strongly on real data | §3.2 |
| How does the compass move across states? | Faster and less certain in SWS than in REM, and in REM than in wake: σ = 0.54 / 0.96 / 2.97 rad/√s | §3.3 |
| Is the decoder's uncertainty trustworthy? | **On synthetic data yes** (90% intervals cover 90–92%). **On real data no**: they cover 64% | §3.4, §4 |
| Is the SWS result an artifact of lower firing rates? | Not of a uniform gain change. Rescaling tuning leaves σ unchanged and fits worse | §3.5 |
| Does the pipeline recover known ground truth over a full-length session? | **Yes** for ring, angle, wake/REM σ and error, over 17,000 s simulations and 2 seeds. Jumps are undercounted about 20× | §4 |

## 2. Data and settings

Mouse32-140822 (Peyrache et al. 2015). 31 ADn units, of which 19 fire ≥ 1 Hz during wake and
are used (the other 12 fire at 0.02–0.70 Hz). Durations analysed: wake 1958.9 s, REM 2565 s
(29 epochs), SWS 12402 s (44 epochs).

Settings: rates in 0.1 s bins smoothed at 0.1 s; PCA-10 → Isomap-2D with 15 neighbours; ripser
H1 on 800 landmarks with 20 shuffles; tuning on 60 bins; the HMM uses 20 ms bins, ε = 0, and
chooses σ by maximum likelihood on a 12-point grid from 0.1 to 50. Adjacent grid points differ
by a factor of 1.76, so each fitted σ is known to ± one grid step.

## 3. Mouse32 results

### 3.1 Label-free recovery and decoding (wake, validated against tracked head angle)

| estimate | in-sample MAE | held-out MAE (2-fold, 60 s blocks) |
|---|---|---|
| unsupervised ring angle (Isomap + circle fit) | 0.417 | **0.420** |
| HMM, tuning from ring angle (fully label-free) | 0.401 | **0.406** |
| independent-bin decoder, tuning from ring angle | 0.576 | 0.581 |
| HMM, tuning from tracked head angle (supervised reference) | 0.205 | 0.207 |

- Held-out and in-sample errors agree to within 0.005 rad, so fitting the tuning and σ on the
  same data inflates nothing at this length. This resolves review finding R1.
- The HMM's temporal prior cuts the label-free error by 30% relative to independent bins.
- Ring-angle error across Isomap settings (neighbours 10/15/30, seeds 0/1/2): 0.410–0.439.
- The gap to the supervised decoder (0.41 vs 0.21) is the price of not knowing the labels.
  At 120 s I measured that warping of the ring angle is not the main cause (review C8).

### 3.2 Topology: the ring test

| point cloud (real wake) | ring score | null max (shuffles) | verdict |
|---|---|---|---|
| **as specified**: FPS 800 landmarks of 19,585 points | **1.04** | 1.28 (20) | **no ring**, p = 0.48 |
| first 600 s | 1.30 | 1.31 (10) | no ring |
| random 6,000 points | 1.05 | 1.32 (10) | no ring |
| *exploratory:* keep the densest 50% (15-NN distance), then FPS 800 | **4.92** | 1.48 (10) | ring, beats all 10 shuffles (p = 1/11, the floor) |

The same four variants on a 2,000 s, 20-cell synthetic wake (truth: ring) all detect it
(scores 6.3, 6.3, 7.3 and 13.6, against null maxima of 1.1–1.4).

**Reading.** The pre-specified detector misses the ring in the real data, though the embedding
shows it clearly (`results/mouse32/ring.png`) and the ring angle decodes heading. The failure is
not about data length, since the same detector works on synthetic data of equal length. The
real point cloud has scattered off-ring states that fill the hole in a Rips complex; removing
low-density points recovers a strong H1 bar.

The density filter was chosen **after** seeing the failure, so it counts as exploratory evidence,
not a pre-registered test. It should be confirmed on other sessions before being adopted. (The
120 s smoke run's positive result, score 3.64, is consistent with this: a short slice has fewer
off-ring points.)

### 3.3 Compass dynamics by state (label-free decoder)

| state | σ (rad/√s) | median \|speed\| (rad/s) | jump fraction (>π/2 per 20 ms) | mean circular variance | bins |
|---|---|---|---|---|---|
| wake | 0.545 | 0.66 | 0 | 0.0058 | 97,943 |
| REM | 0.958 | 1.26 | 0.00012 | 0.0137 | 128,250 |
| SWS | 2.966 | 5.16 | 0.0019 | 0.0676 | 620,100 |

- The ordering on every metric is **SWS ≫ REM > wake**. None of the σ fits sits at a grid edge.
- Speeds are in the ring's frame; wake alignment to head direction is rotation-only, sign +1.
- The jump fraction is a **lower bound**. On synthetic SWS with known jumps the decoder reports
  about 1/20 of them (§4), because with ε = 0 the HMM renders jumps as fast sweeps.
- This matches the picture in Peyrache et al. (2015): the HD signal stays coherent in REM at
  near-wake speeds and moves much faster in SWS. The circular variance adds that the decoder is
  about 10× less certain in SWS than in wake.

### 3.4 Calibration

On held-out wake with head-angle tuning, the 90% highest-density posterior sets (mean width
0.43 rad) contain the true head angle in **64%** of bins. Because the identical decoder is
calibrated on synthetic data (§4), this is misfit between the model and the real spiking, not
a code bug. The likely causes are overdispersed or correlated spiking and tuning that isn't
stationary. **Treat the reported posterior variances as relative measures across states, not
absolute error bars.**

### 3.5 Firing-rate gain

Population rate of the 19 units relative to wake: REM ×0.864, SWS ×0.763. Decoding sleep with
wake tuning rescaled by that ratio:

| state | σ uncorrected → corrected | log-likelihood change | circular variance |
|---|---|---|---|
| REM | 0.958 → 0.958 | −1,378 (worse) | 0.0137 → 0.0137 |
| SWS | 2.966 → 2.966 | −12,320 (worse) | 0.068 → 0.075 |

A uniform gain change is rejected by likelihood and doesn't move σ. So the SWS/REM/wake
differences in §3.3 are **not explained by a uniform rate drop**. Non-uniform rate changes
(per-cell or tuning-width changes) were not tested.

## 4. Ground-truth validation on long simulations

`simulate_session` builds a Mouse32-shaped session: 20 cells, κ = 4, wake 2000 s, REM 2500 s
(90 s epochs), SWS 12000 s (300 s epochs), 17,000 s in total. Wake and REM have σ = 1 and a peak
of 30 Hz. SWS has σ = 4, jumps at 0.3/s and a peak of 26 Hz (a ~13% gain drop). This is run
through the **same full pipeline** and then scored against the true angle in every state,
including sleep. The "true model" rows use the true tuning, σ and jump probability, so they are
the best achievable.

| | seed 0 | seed 1 |
|---|---|---|
| ring test (as specified) | score 5.87, null max 1.32 → ring | 4.68, null max 1.08 → ring |
| ring-angle MAE (wake) | 0.151 | 0.160 |

| state | metric | seed 0 | seed 1 | true model (s0 / s1) |
|---|---|---|---|---|
| wake | decoded σ (true 1.0) | 0.958 | 0.958 | 0.958 / 0.958 |
| | MAE | 0.136 | 0.143 | 0.131 / 0.132 |
| | 90% coverage | 0.913 | 0.897 | 0.920 / 0.918 |
| REM | decoded σ (true 1.0) | 0.958 | 0.958 | 0.958 / 0.958 |
| | MAE | 0.135 | 0.142 | 0.131 / 0.131 |
| | 90% coverage | 0.916 | 0.901 | 0.919 / 0.920 |
| SWS | decoded σ (true 4.0) | 2.966 | 2.966 | 2.966 / 2.966 |
| | MAE | 0.315 | 0.317 | 0.309 / 0.309 |
| | 90% coverage | 0.822 | 0.820 | 0.864 / 0.866 |
| | jump fraction, decoded (true 0.0086) | 0.0004 | 0.0004 | — |
| | gain correction changes σ or MAE? | no | no | — |

What this establishes:

- **Label-free ≈ oracle.** In every state the label-free pipeline's error is within 0.012 rad
  of the true-model decoder. The unsupervised stages cost almost nothing when the data are a
  clean ring.
- **σ is recovered to the nearest grid point** in wake and REM. In SWS both the pipeline and the
  true-model fit choose 2.97 for a true 4.0, one grid step low: the jumps are absorbed by the
  ε = 0 fit and the grid is coarse. The SWS > REM ≈ wake ordering is recovered.
- **The decoder is calibrated when its model is right** (coverage 0.90–0.92 at nominal 0.90). It
  undercovers only for SWS with jumps (0.82–0.87).
- **Jump fraction undercounts by about 20×.** It is a qualitative indicator only.

## 5. Synthetic recovery sweep (600 s, 3 seeds per cell)

36 configurations: n_cells ∈ {8, 16, 32, 64} × κ ∈ {2, 4, 8} × 3 seeds, σ = 1.

| κ | cells | ring detected | ring-angle MAE (mean ± sd) | HMM (true tuning) MAE |
|---|---|---|---|---|
| 2 | 8 / 16 / 32 / 64 | 3/3 each | 0.203 / 0.170 / 0.147 / 0.136 | 0.185 / 0.154 / 0.129 / 0.109 |
| 4 | 8 / 16 / 32 / 64 | 3/3 each | 0.194 / 0.156 / 0.154 / 0.137 | 0.167 / 0.139 / 0.115 / 0.097 |
| 8 | 8 / 16 / 32 / 64 | **0/3**, 3/3, 3/3, 3/3 | 0.206 / 0.158 / 0.147 / 0.157 ± 0.025 | 0.157 / 0.128 / 0.106 / 0.089 |

- The ring is detected in 33 of 36 configurations. It fails only for 8 cells with very narrow
  tuning (κ = 8), where the circle is sparsely covered.
- HMM error falls monotonically with cell count for every κ. Ring-angle error falls too, except
  for κ = 8 between 32 and 64 cells (within 1 sd).
- The fitted σ is within one grid step of the true 1.0 in all 36 runs.
- Figure: `results/sweep/sweep.png`.

## 6. How these were produced

```bash
.venv/bin/python scripts/run_synthetic_sweep.py          # results/sweep/          1:04:51, 1.3 GB peak*
.venv/bin/python scripts/run_mouse32.py                  # results/mouse32/        12:00,   2.7 GB peak*
.venv/bin/python scripts/validate.py real                # results/validation/real.json         3:37, 2.8 GB
.venv/bin/python scripts/validate.py synthetic --seed 0  # …/synthetic_seed0.json  8:34, 3.1 GB
.venv/bin/python scripts/validate.py synthetic --seed 1  # …/synthetic_seed1.json  8:40, 3.1 GB
.venv/bin/python scripts/validate.py topology            # …/topology.json         5:23, 0.9 GB
.venv/bin/python scripts/validate.py sweep               # …/sweep_summary.json    seconds
```

\* The sweep and Mouse32 runs shared the machine with unrelated jobs (load average ~16 on 8
cores), so their wall times are pessimistic.

## 7. What these results do not show

- **One animal, one session, 19 units.** The state ordering is suggestive, not general. The
  CRCNS th-1 multi-session loader is the next step.
- **σ is quantised** at a factor of 1.76 per grid step. Values are ± one step.
- **Posterior variance is not an absolute error bar** on real data (§3.4).
- **Jump fraction is a lower bound** (§4). A model with fitted ε is needed to quantify SWS
  discontinuities.
- **The topology conclusion rests on an exploratory variant** (§3.2).
