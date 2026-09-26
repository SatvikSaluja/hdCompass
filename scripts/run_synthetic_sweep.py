"""Full recovery-vs-cells sweep: 4 sizes x 3 kappas x 3 seeds, 600 s each. Slow; run manually.

Usage: .venv/bin/python scripts/run_synthetic_sweep.py [--out DIR] [--overwrite]
"""

import sys

from hdcompass.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    main(["sweep", *([] if "--out" in args else ["--out", "results/sweep"]), *args])
