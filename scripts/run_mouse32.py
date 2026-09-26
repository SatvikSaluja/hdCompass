"""Full Mouse32 pipeline: all wake/REM/SWS, 20 shuffles, 12-value sigma grid. Slow; run manually.

Usage: .venv/bin/python scripts/run_mouse32.py [--out DIR] [--data PATH] [--overwrite]
"""

import sys

from hdcompass.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    main(["mouse32", *([] if "--out" in args else ["--out", "results/mouse32"]), *args])
