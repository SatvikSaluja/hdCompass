# %% [markdown]
# # hdcompass figures
#
# Regenerates every figure. `QUICK = True` reproduces the smoke-test figures (120 s per state,
# one synthetic configuration). `QUICK = False` runs the full Mouse32 pipeline (slow) and
# plots the full sweep from its saved `report.json` (run `scripts/run_synthetic_sweep.py` first,
# or it will be computed here, which is much slower).
#
# Open with jupytext (`jupytext --to notebook figures.py`) or run cell by cell in VS Code.

# %%
import json
from pathlib import Path

from hdcompass.cli import run_mouse32, run_sweep
from hdcompass.plots import plot_sweep

QUICK = True
OUT = Path("figures") / ("quick" if QUICK else "full")

# %% [markdown]
# ## Mouse32: label-free ring, decoding and state dynamics

# %%
report, figs = run_mouse32(OUT / "mouse32", quick=QUICK, overwrite=True)
print(json.dumps(report["topology"], indent=1))
figs["ring"]

# %%
figs["barcode"]

# %%
figs["decode_wake"]

# %%
figs["dynamics"]

# %% [markdown]
# ## Synthetic recovery sweep

# %%
saved = Path("results/sweep/report.json")
if not QUICK and saved.exists():
    results = json.loads(saved.read_text())["results"]
else:
    results, _ = run_sweep(OUT / "sweep", quick=QUICK, overwrite=True)
fig = plot_sweep(results)
fig.savefig(OUT / "sweep.png", dpi=120)
fig
