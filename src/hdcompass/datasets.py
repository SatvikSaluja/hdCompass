"""Loader for the Peyrache et al. (2015) Mouse32-140822 session."""

import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pynapple as nap

from .circular import wrap

DANDI_API = "https://api.dandiarchive.org/api"
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


def _read_000939(f):
    """Extract spikes, head direction, epochs and sleep states from a DANDI:000939 NWB file."""

    def txt(a):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])

    u, ep, ss = f["units"], f["intervals/epochs"], f["intervals/sleep_states"]
    hd = f["processing/behavior/CompassDirection/head-direction"]
    tags, tag_end = txt(ep["tags"][()]), ep["tags_index"][()]
    return {
        "unit_ids": u["id"][()],
        "spike_times": u["spike_times"][()],
        "spike_index": u["spike_times_index"][()],
        "is_head_direction": u["is_head_direction"][()].astype(bool),
        "is_excitatory": u["is_excitatory"][()].astype(bool),
        "is_fast_spiking": u["is_fast_spiking"][()].astype(bool),
        "epoch_start": ep["start_time"][()],
        "epoch_stop": ep["stop_time"][()],
        "epoch_tag": np.array(
            [",".join(tags[a:b]) for a, b in zip(np.r_[0, tag_end[:-1]], tag_end, strict=True)]
        ),
        "state_start": ss["start_time"][()].astype(float),
        "state_stop": ss["stop_time"][()].astype(float),
        "state": txt(ss["state"][()]),
        "hd_t": hd["timestamps"][()],
        "hd": hd["data"][()],
    }


def load_dandi000939(asset, cache_dir="~/.cache/hdcompass", wake_tag="wake_square"):
    """Load one session of DANDI:000939 (Duszkiewicz et al. 2024, mouse postsubiculum).

    Only the arrays the pipeline needs (spikes, head direction, epochs, sleep states: a few MB)
    are streamed from the ~20 GB file, then cached as ``<cache_dir>/dandi000939/<asset>.npz``.
    Streaming needs the optional ``remfile`` package (``pip install 'hdcompass[dandi]'``).

    Parameters
    ----------
    asset : str or Path
        DANDI asset id, or a local NWB path.
    cache_dir : str or Path
    wake_tag : str
        Epoch tag used as wake (the first open-field arena by default: tuning can rotate between
        arenas, so arenas are not pooled).

    Returns
    -------
    Session
        All units (metadata ``is_head_direction``, ``is_excitatory``, ``is_fast_spiking``, which
        are never used by the label-free pipeline); ``angle`` is the tracked head direction with
        tracking-loss NaNs dropped; ``rem``/``sws`` are the ``rem``/``nrem`` sleep states.
    """
    import h5py

    local = Path(str(asset)).exists()
    cache = Path(cache_dir).expanduser() / "dandi000939" / f"{Path(str(asset)).stem}.npz"
    if cache.exists():
        with np.load(cache) as z:
            d = dict(z)
    else:
        if local:
            src = h5py.File(asset, "r")
        else:
            try:
                import remfile
            except ImportError as e:
                raise ImportError("streaming needs remfile: pip install 'hdcompass[dandi]'") from e
            src = h5py.File(remfile.File(f"{DANDI_API}/assets/{asset}/download/"), "r")
        with src as f:
            d = _read_000939(f)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".part.npz")
        np.savez(tmp, **d)
        os.replace(tmp, cache)

    trains = np.split(d["spike_times"], d["spike_index"][:-1])
    spikes = nap.TsGroup(
        {int(i): nap.Ts(t) for i, t in zip(d["unit_ids"], trains, strict=True)},
        metadata={k: d[k] for k in ("is_head_direction", "is_excitatory", "is_fast_spiking")},
    )
    is_wake = d["epoch_tag"] == wake_tag
    if not is_wake.any():
        raise ValueError(f"no epoch tagged {wake_tag!r}; tags: {sorted(set(d['epoch_tag']))}")
    ok = np.isfinite(d["hd"])
    t = d["hd_t"][ok]
    wake = nap.IntervalSet(d["epoch_start"][is_wake], d["epoch_stop"][is_wake]).intersect(
        nap.IntervalSet(t[0], t[-1])
    )

    def states(name):
        m = d["state"] == name
        return nap.IntervalSet(d["state_start"][m], d["state_stop"][m])

    return Session(
        spikes=spikes,
        angle=nap.Tsd(t=t, d=wrap(d["hd"][ok]), time_support=wake),
        wake=wake,
        rem=states("rem"),
        sws=states("nrem"),
    )
