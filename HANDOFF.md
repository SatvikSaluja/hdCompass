# HANDOFF — hdcompass v0.1.0 (2026-09-26)

## What exists

Everything in `PLAN.md` §2: the package (`src/hdcompass/`, 13 modules), 11 test files,
`scripts/run_mouse32.py`, `scripts/run_synthetic_sweep.py`, `notebooks/figures.py`, CI
workflow, `constraints.txt`, README. Git repo on `main`, no remote.

## Commands

```bash
cd /home/satvik/hdcompass
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/hdcompass sweep   --quick --out /tmp/hdc_sweep
.venv/bin/hdcompass mouse32 --quick --out /tmp/hdc_m32
# heavy, not yet run:
.venv/bin/python scripts/run_mouse32.py            # results/mouse32/
.venv/bin/python scripts/run_synthetic_sweep.py    # results/sweep/
```

Add `--overwrite` to replace an existing `report.json`.

## Executed vs not executed

Executed (all succeeded):

- `ruff check .`: clean.
- `pytest -q`: 22 passed (the Mouse32 dataset test runs because the file is cached), ~32 s.
- `hdcompass sweep --quick`: 15–16 s wall (two runs).
- `hdcompass mouse32 --quick`: 21–29 s wall (two runs).

The smoke runs are 60–120 s slices, useful only to show the pipeline runs end to end, not as
results. From `/tmp/hdc_m32/report.json` (120 s per state, 5 shuffles, 4-value σ grid):
19 of 31 ADn units passed `min_rate`; ring_score 3.64 vs null max 1.60 (`is_ring` true,
p = 1/6, the floor for 5 shuffles); wake MAE vs head angle: ring angle 0.34 rad, HMM with
ring tuning 0.32 rad, independent-bin decoder 0.56 rad, HMM with head-angle tuning 0.21 rad.

**Not executed:** full Mouse32 run, full synthetic sweep, `notebooks/figures.py`, the two
scripts (syntax-checked only). README *Results* says "not yet run" on purpose.

## Choices beyond or different from the plan

- **Data path.** `~/.cache/hdcompass/Mouse32-140822.nwb` is a symlink to
  `/home/satvik/side_project/cache/public/Mouse32-140822.nwb`, so the default loader finds it
  without downloading.
- **Topology** projects onto 10 PCs before ripser (`n_pcs`, same as `embed`), for denoising.
  Farthest-point subsampling uses ripser's own `n_perm` (greedy permutation); `seed` picks the
  start point.
- **`is_ring`** decides by `score > quantile(null, 0.95)` and reports the permutation p
  separately, because with 5 shuffles p can never be < 0.05. It also accepts a precomputed
  score, and `h1_persistence` returns `ring_score`, so ripser isn't run twice.
- **Smoothing** uses `scipy.ndimage.gaussian_filter1d` (nearest padding, truncate 3σ = 6σ
  window) per epoch, plus the 2σ edge drop. pynapple's `smooth` is not used.
- **Decoder** concatenates epochs and resets the prior at each epoch start. A test checks this
  is identical to separate runs, and it means one JIT compile instead of one per epoch. JAX
  runs in float64 (`jax_enable_x64` set when `hdcompass.decode` is imported).
- **Benchmark** records `manifold_mae` and `decode_mae` (both circular MAE) rather than one
  `circ_mae`.
- **CLI** has `--overwrite` on both commands and `--data` on `mouse32`; `mouse32` also refuses
  to overwrite.
- **constraints.txt** comes from `pip freeze --exclude-editable`, so it has no local
  `file://` line. CI installs unpinned: the pinned numpy 2.5 may not support Python 3.11.
- **Reference:** PLAN.md cites "Rubin 2019 eLife". As far as I know that paper is in
  *Nature Communications* 10:4745, so the README cites it there. Please check.

## Open questions

1. **Sleep gain.** Wake tuning is applied unchanged to REM/SWS. If ADn rates differ by state,
   the SWS posterior variance and σ partly reflect the gain mismatch. Should tuning be scaled
   per state (e.g. by the population-rate ratio)?
2. **ε is fixed at 0.** SWS jumps are then absorbed into a larger σ. Should ε be fitted jointly
   with σ (small 2-D grid)?
3. **`min_rate=1.0` dropped 12/31 ADn units** on the 120 s wake slice. Is that intended? A lower
   threshold keeps more cells for decoding.
4. **The HMM seems overconfident** (mean wake circular variance ≈ 0.008 with 0.32 rad error in
   the smoke run). Independent-Poisson emissions ignore noise correlations. A calibration
   check (coverage of posterior intervals) would settle it.

## Next steps

1. Full Mouse32 run: `scripts/run_mouse32.py` (whole wake, all REM/SWS, 20 shuffles,
   12-value σ grid).
2. Synthetic sweep: `scripts/run_synthetic_sweep.py`, then fill README *Results* from the two
   reports.
3. CRCNS th-1 multi-session loader. th-1 is Neuroscope format (`.res/.clu/.whl`), not NWB,
   and needs your CRCNS account to download. Add `load_th1(session_dir)` returning the same
   `Session`.
