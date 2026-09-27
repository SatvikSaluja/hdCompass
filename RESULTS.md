# hdcompass — results and validation (2026-09-27)

All numbers come from report files produced by the commands in §6, at commit `fbacf85`. The
files live under `results/`, which is git-ignored; rerun the commands to regenerate them.
Angles are in radians. Chance-level MAE for a random angle is π/2 ≈ 1.57 rad.

## 1. Summary

| question | answer | evidence |
|---|---|---|
| Can the HD angle be recovered from ADn spikes with no labels? | **Yes.** Error is 0.42 rad (≈24°) on held-out wake, stable across embedding settings | §3.1 |
| Does a label-free decoder work? | **Yes.** HMM with ring-derived tuning: 0.41 rad held-out, vs 0.58 for an independent-bin decoder and 0.21 for a decoder given tracked head angle | §3.1 |
| Does the topology test detect the ring on the full wake? | **As first specified, no** (score 1.04, p = 0.48). **With the density filter adopted by pre-registered checks, yes**: score 4.92, p = 1/201 with 200 shuffles | §3.2, §6.1 |
| How does the compass move across states? | Faster and less certain in SWS than in REM, and in REM than in wake: σ = 0.54 / 0.96 / 2.97 rad/√s | §3.3 |
| Is the decoder's uncertainty trustworthy? | **On synthetic data yes** (90% intervals cover 90–92%). **On real data no**: they cover 64% | §3.4, §4 |
| Is the SWS result an artifact of lower firing rates? | Not of a uniform gain change. Rescaling tuning leaves σ unchanged and fits worse | §3.5 |
| Should low-rate units be decoded too? | **No gain.** Held-out error is 0.406 rad with 19, 24 or 28 units (1, 0.3, 0.1 Hz floors), so the 1 Hz floor stays | §6.2 |
| Does the frozen ring test work on synthetic data? | **Yes:** 36 of 36 sweep configurations (the unfiltered test: 33/36) | §5 |
| Does it generalise to another dataset and region? | **Decoding and the SWS result, yes.** Across 22 post-subiculum sessions (DANDI:000939) the label-free decoder reaches a median 0.36 rad. σ SWS > wake in 21/22 sessions (p = 1.8e-5), SWS > REM in 19/20, and SWS variance > wake in 21/22. REM ≈ wake there. The ring test finds a ring in only 12/22 | §7 |
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

The density filter was chosen **after** seeing the failure, so this table alone is exploratory.
§6.1 then tested it with rules fixed in advance, and it passed. It should be confirmed on other sessions before being adopted. (The
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
- **Rerun with the frozen density filter** (§6; `results/sweep_frozen/`, config hash
  `c3f139c596c9`, 18:59 with 8 processes): the ring is detected in **36 of 36** configurations,
  including the three κ = 8, 8-cell misses (median score 5.4). Median scores rise to 5.4–25.6.
  Decoding and ring-angle errors are identical, since only the ring test changed.

## 6. Pre-registered decisions: the frozen pipeline

Two defaults were open after the first runs. Mouse32 was treated as the development session:
the rules below were written into `scripts/validate.py` (`DECISION_RULES`) before the checks
ran, applied automatically, and the resulting defaults frozen (commit `6b49e03`) before any
new dataset was analysed. Results: `results/validation/decisions.json` (30.8 min, 2.1 GB).

### 6.1 Ring test: adopt the density filter?

**Rule.** Adopt k = 15, keep = 0.5 (the setting first tried in §3.2, not the best one) only if
all of these hold:
- (a) real wake is a ring in ≥ 7 of 9 grid settings;
- (b) a synthetic ring is detected in 9 of 9;
- (c) a synthetic population with no shared angle is detected in ≤ 1 of 9. Each cell follows
  its own latent angle, and each test has a ~5% false-positive rate;
- (d) the 200-shuffle p-value at the frozen setting is below 0.05/4, correcting for the four
  variants tried in §3.2.

Ring score per setting (20 shuffles each; ✔ = beats every shuffle):

| setting | real Mouse32 wake | synthetic ring (2000 s) | no-ring control (2000 s) |
|---|---|---|---|
| no filter | ✘ 1.04 | ✔ 6.27 | ✘ 1.11 |
| k=10, keep 0.3 / 0.5 / 0.7 | ✔ 3.85 / ✔ 4.18 / ✔ 2.89 | ✔ 17.7 / 13.2 / 12.4 | ✘ 1.07 / **✔ 1.27** / ✘ 1.01 |
| k=15, keep 0.3 / 0.5 / 0.7 | **✘ 1.02** / ✔ 4.92 / ✔ 4.95 | ✔ 15.6 / 13.6 / 11.5 | ✘ 1.04 / ✘ 1.02 / ✘ 1.04 |
| k=30, keep 0.3 / 0.5 / 0.7 | ✔ 1.62 / ✔ 5.18 / ✔ 4.33 | ✔ 16.5 / 14.9 / 11.2 | ✘ 1.13 / ✘ 1.02 / ✘ 1.00 |

**Outcome: adopted.** The counts are 8/9, 9/9 and 1/9, and the 200-shuffle p-value is
1/201 ≈ 0.005 (null maximum 1.48).

- The one control false positive is weak: 1.27 against a null maximum of 1.18. Most real-wake
  settings score 2.9–5.2.
- Keeping only 30% of points can fragment the ring (k = 15 misses; k = 30 is weak at 1.62), so
  keep ≥ 0.5 is the safer range.

**Mechanism check.** In a synthetic ring, 30% of 2 s blocks were replaced by off-ring activity:

| off-ring activity | as specified | density filter |
|---|---|---|
| quiet (all cells at baseline) | ✘ 1.21 (null max 1.22) | ✔ 2.13 (null max 1.21) |
| desynchronised (each cell its own angle) | ✘ 1.02 (null max 1.27) | ✔ 10.5 (null max 1.15) |

A minority of off-ring states is enough to defeat the unfiltered test, and the filter recovers
the ring. On real data, though, the dropped bins are **not** quiet periods. Their mean population
z-score is +0.09 (kept: −0.09) and the head speed is similar (0.55 vs 0.49 rad/s). They are
sparse, slightly more active states, and what they are physiologically is open.

**Caveat found afterwards:** on the 120 s `--quick` slice (1,196 points) the frozen filter
*misses* the ring (1.03, null max 1.46), where the unfiltered test found it (3.64). The filter
was validated only at full length (~20,000 points). Halving an already small point cloud leaves
the ring too sparse, so `--quick` topology is a smoke test only.

**Frozen Mouse32 rerun** (`results/mouse32_frozen/`, 3 min 11 s with 8 processes): ring score
4.92 against a null maximum of 1.48, so a ring. Every other number is identical to §3, since
only the topology step changed.

### 6.2 Decoder rate floor

**Rule.** Take the lowest held-out (2-fold) label-free MAE among floors of 1, 0.3 and 0.1 Hz.
Keep 1 Hz unless another floor beats it by more than 0.005 rad.

| floor | units | label-free HMM MAE | head-tuned HMM MAE |
|---|---|---|---|
| 1 Hz | 19 | 0.4062 | 0.2074 |
| 0.3 Hz | 24 | 0.4064 | 0.2070 |
| 0.1 Hz | 28 | 0.4062 | 0.2065 |

**Outcome: 1 Hz kept.** The extra units fire too rarely to change the posterior.

## 7. Second dataset: DANDI:000939 (mouse post-subiculum, 22 sessions)

Duszkiewicz, Skromne Carrasco & Peyrache, DANDI:000939 v0.260512.1701
(https://doi.org/10.48324/dandi.000939/0.260512.1701, CC-BY-4.0; original study Duszkiewicz et
al. 2024): head-direction recordings in the post-subiculum during open-field exploration and
home-cage sleep, with scored wake, NREM and REM. It is a different brain region from Mouse32 (ADn) and a different lab setup.

**The analysis plan was fixed before any session was analysed** (docstring of
`scripts/run_dandi000939.py`, commit `1c9ceb8`):
- the 22 sessions without optogenetics;
- wake = the first open-field arena (`wake_square`); SWS/REM = `nrem`/`rem`;
- the frozen pipeline of §6, unchanged;
- REM counted in statistics only with ≥ 60 s (20 of 22 sessions qualify);
- three one-sided paired Wilcoxon tests.

The whole run took 52:41 with a 4.0 GB peak, streaming ~60 MB per session instead of ~23 GB.

Per session: 42–185 units (40–156 above the 1 Hz floor), of which 21–117 are flagged as HD
cells. The flags are never used by the pipeline. Wake lasts 1805–2635 s, SWS 1153–10160 s, and
REM 0–1040 s (median 294 s).

### 7.1 Pre-registered hypotheses: all three confirmed

| hypothesis | sessions in the predicted direction | median ratio | one-sided Wilcoxon p |
|---|---|---|---|
| σ SWS > σ wake | 21 of 22 | 3.1× | 1.8 × 10⁻⁵ |
| σ SWS > σ REM | 19 of 20 | 3.1× | 4.4 × 10⁻⁵ |
| circular variance SWS > wake | 21 of 22 | 6.5× | 4.8 × 10⁻⁷ |

The one reversal in the σ tests is A3723, with SWS σ 0.18 against wake 0.54. Its SWS decoding
may simply have failed; it isn't investigated here.

### 7.2 Label-free recovery

Medians over sessions:
- ring angle vs head direction: 0.40 rad;
- label-free HMM: **0.36 rad**;
- independent-bin decoder: 0.57 rad;
- HMM with head-angle tuning: 0.19 rad.

The HMM beats independent bins in **22 of 22** sessions.

The frozen ring test detects a ring in **12 of 22** sessions. **Exploratory** (not in the plan):
detection and accuracy track the number of HD cells. Sessions with a detected ring have a median
of 72 flagged HD cells, against 31.5 for the rest, and a median label-free error of 0.33 vs
0.46 rad. The Spearman correlation between HD-cell count and label-free error is −0.58
(p = 0.004). Post-subiculum sessions also contain many non-HD units (a median ~40%), which add
noise dimensions after z-scoring. That plausibly makes the topology harder than in ADn, but it
isn't tested.

### 7.3 Where this differs from Mouse32

- **REM ≈ wake here.** σ is equal at grid resolution in 15 of 20 sessions, and lower in 5.
  Mouse32 had REM σ 0.96 against wake 0.54. Region (post-subiculum vs ADn), the shorter REM
  periods and the coarse σ grid could each contribute; none is tested.
- **The ring test is weaker** (12/22) than decoding. It is the least reliable stage of the
  pipeline.

Per-session results (`results/dandi000939/<session>/report.json`, `summary.json`, `summary.png`;
\* = REM under 60 s, excluded from statistics; — = no REM):

| session | units (HD-flagged) | ring (score) | ring-angle MAE | label-free HMM MAE | head-tuned MAE | σ wake / REM / SWS |
|---|---|---|---|---|---|---|
| A3701 191119 | 102 (71) | ✔ 1.32 | 0.42 | 0.33 | 0.24 | 0.54 / 0.54 / 1.69 |
| A3702 191126 | 90 (61) | ✔ 1.41 | 0.39 | 0.36 | 0.17 | 0.54 / 0.54 / 1.69 |
| A3703 191215 | 103 (65) | ✘ 1.16 | 0.40 | 0.36 | 0.21 | 0.54 / 0.54 / 1.69 |
| A3705 200306 | 117 (66) | ✔ 2.11 | 0.21 | 0.20 | 0.18 | 0.54 / 0.54 / 2.97 |
| A3706 200313 | 126 (86) | ✔ 2.78 | 0.33 | 0.32 | 0.16 | 0.54 / 0.31 / 1.69 |
| A3707 200317 | 185 (117) | ✔ 2.33 | 0.38 | 0.39 | 0.22 | 0.54 / 0.54* / 2.97 |
| A3708 200317b | 78 (44) | ✘ 1.17 | 0.37 | 0.30 | 0.14 | 0.54 / 0.54 / 2.97 |
| A3709 200601 | 64 (21) | ✔ 2.35 | 0.60 | 0.53 | 0.18 | 0.54 / 0.54 / 1.69 |
| A3710 200609 | 107 (80) | ✔ 1.61 | 0.53 | 0.50 | 0.16 | 0.54 / 0.54 / 0.96 |
| A3711 200810b | 142 (90) | ✔ 1.88 | 0.29 | 0.29 | 0.22 | 0.54 / 0.31 / 0.96 |
| A3712 200903a | 126 (73) | ✔ 2.28 | 0.31 | 0.30 | 0.27 | 0.54 / 0.54 / 2.97 |
| A3713 200909a | 88 (25) | ✘ 1.01 | 0.39 | 0.35 | 0.21 | 0.54 / 0.54 / 1.69 |
| A3716 201015b | 147 (93) | ✔ 3.02 | 0.25 | 0.21 | 0.18 | 0.54 / 0.54 / 1.69 |
| A3717 201021 | 98 (59) | ✔ 3.71 | 0.32 | 0.31 | 0.18 | 0.54 / 0.31 / 1.69 |
| A3723 201115 | 68 (37) | ✘ 1.09 | 0.53 | 0.43 | 0.17 | 0.54 / 0.31 / 0.18 |
| A3728 210309b | 42 (27) | ✘ 1.10 | 0.85 | 0.79 | 0.33 | 0.96 / — / 2.97 |
| A3730 210323b | 77 (47) | ✘ 1.17 | 0.63 | 0.58 | 0.19 | 0.54 / 0.54 / 2.97 |
| A5505 200831 | 55 (30) | ✘ 1.10 | 0.39 | 0.34 | 0.19 | 0.54 / 0.54 / 1.69 |
| A5506 200914a | 49 (24) | ✘ 1.03 | 0.76 | 0.75 | 0.24 | 0.96 / 0.54 / 2.97 |
| A5507 200920 | 96 (43) | ✔ 1.63 | 0.49 | 0.46 | 0.19 | 0.54 / 0.54 / 2.97 |
| A5508 200930 | 52 (33) | ✘ 1.01 | 0.65 | 0.62 | 0.20 | 0.96 / 0.96 / 2.97 |
| A5801 200924 | 46 (26) | ✘ 1.02 | 0.59 | 0.50 | 0.23 | 0.54 / 0.54 / 1.69 |

## 8. How these were produced

```bash
.venv/bin/python scripts/run_synthetic_sweep.py          # results/sweep/          1:04:51, 1.3 GB peak*
.venv/bin/python scripts/run_mouse32.py                  # results/mouse32/        12:00,   2.7 GB peak*
.venv/bin/python scripts/validate.py real                # results/validation/real.json         3:37, 2.8 GB
.venv/bin/python scripts/validate.py synthetic --seed 0  # …/synthetic_seed0.json  8:34, 3.1 GB
.venv/bin/python scripts/validate.py synthetic --seed 1  # …/synthetic_seed1.json  8:40, 3.1 GB
.venv/bin/python scripts/validate.py topology            # …/topology.json         5:23, 0.9 GB
.venv/bin/python scripts/validate.py sweep               # …/sweep_summary.json    seconds
.venv/bin/python scripts/validate.py decisions --jobs 8  # …/decisions.json        30:49, 2.1 GB
.venv/bin/python scripts/run_mouse32.py --out results/mouse32_frozen   # frozen    3:11, 2.7 GB
.venv/bin/python scripts/run_dandi000939.py --jobs 8     # results/dandi000939/   52:41, 4.0 GB
.venv/bin/python scripts/run_synthetic_sweep.py --out results/sweep_frozen    # frozen 18:59, 1.3 GB
```

\* The sweep and Mouse32 runs shared the machine with unrelated jobs (load average ~16 on 8
cores), so their wall times are pessimistic.

## 9. What these results do not show

- **ADn is still one animal and one session** (19 units). The SWS result replicates across 22
  post-subiculum sessions (§7), but more ADn sessions need CRCNS th-1. Of each session's
  `.tar.gz` we need `*.res.N`, `*.clu.N`, `*.xml`, `*.ang` and `*.states.SWS/REM/Wake`, plus
  `docs/crcns_th-1_session_info.xls`. The open DANDI copy (000056) lacks sleep scoring and unit
  regions.
- **REM differs between datasets** (above wake in Mouse32, equal in DANDI:000939), unexplained.
- **σ is quantised** at a factor of 1.76 per grid step. Values are ± one step.
- **Posterior variance is not an absolute error bar** on real data (§3.4).
- **Jump fraction is a lower bound** (§4). A model with fitted ε is needed to quantify SWS
  discontinuities.
- **The density filter was adopted on Mouse32 alone.** The pre-registered checks (§6.1) guard
  against false positives, but only other sessions can confirm it.
