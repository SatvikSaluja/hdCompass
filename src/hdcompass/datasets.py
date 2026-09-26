"""Loader for the Peyrache et al. (2015) Mouse32-140822 session."""

import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pynapple as nap

MOUSE32_URL = "https://osf.io/jb2gd/download"
MOUSE32_FILE = "Mouse32-140822.nwb"


@dataclass(frozen=True)
class Session:
    """One recording session.

    Attributes
    ----------
    spikes : nap.TsGroup
        Anterodorsal thalamic (ADn) units only.
    angle : nap.Tsd
        Tracked head angle (rad), wake only. Used for validation, never for recovery.
    wake, rem, sws : nap.IntervalSet
        Behavioural-state epochs; ``wake`` is restricted to the tracked period.
    """

    spikes: nap.TsGroup
    angle: nap.Tsd
    wake: nap.IntervalSet
    rem: nap.IntervalSet
    sws: nap.IntervalSet


def _download(url, dest):
    """Stream ``url`` to a temp file next to ``dest``, then rename (no partial files left)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=60) as r:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_mouse32(path=None, cache_dir="~/.cache/hdcompass"):
    """Load Mouse32-140822, downloading it from OSF only if it is not cached.

    Parameters
    ----------
    path : str or Path, optional
        Explicit NWB path. If given, nothing is downloaded.
    cache_dir : str or Path
        Where ``Mouse32-140822.nwb`` is looked for / downloaded to when ``path`` is None.

    Returns
    -------
    Session
    """
    if path is None:
        path = Path(cache_dir).expanduser() / MOUSE32_FILE
        if not path.exists():
            _download(MOUSE32_URL, path)
    data = nap.load_file(str(path))

    units = data["units"]
    spikes = units[units["location"] == "adn"]
    angle = data["ry"]
    epochs = data["epochs"]
    wake = epochs[epochs["tags"] == "wake"].intersect(angle.time_support)
    return Session(spikes=spikes, angle=angle, wake=wake, rem=data["rem"], sws=data["sws"])
