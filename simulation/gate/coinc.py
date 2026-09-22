"""GATE's coincidences to the decoder's event table, and the terms it implies.

Per coincidence:

  crystals   `PostPosition` after readout is a crystal centre; it becomes a
             GE id through `CrystalLookup` (nearest GE azimuth and ring)
  TOF        `tof_bin = round(c (t2 - t1) / 2 / 13.38 mm)`: single 1 is
             `xtal_a`, and a later second single means the pair was nearer
             `xtal_a`, which is GE's positive bin (see `events_io`)
  clock      `t_ms = bed_start_ticks + GlobalTime1 / 1 ms`
  FOV        pairs outside the sinogram are dropped, as the scanner's own
             FOV filter does (`cpm_fov_filtered` in the singles log)
  norm       GATE's crystals are identical; GE's efficiency pattern is
             imposed by keeping a pair in bin b with probability
             `min(1, normdt_b / mean over views of normdt)` -- only the
             relative losses, since GATE supplies the geometry itself
  truth      random = EventID differs; scatter = a Compton or Rayleigh step
             in the patient on either photon
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from utils.binmap import BinMap
from utils.scanner import NXTAL

from .. import events_io as eio
from ..crystals import CrystalLookup

BRANCHES = ("EventID", "GlobalTime", "PostPosition_X", "PostPosition_Y",
            "PostPosition_Z", "TotalEnergyDeposit", "PhantomCompton",
            "PhantomRayleigh", "PreStepUniqueVolumeID")


def _pos(c: dict, k: str = ""):
    return np.stack([c[f"PostPosition_X{k}"], c[f"PostPosition_Y{k}"],
                     c[f"PostPosition_Z{k}"]], 1)


def _crystals(c: dict, lookup: CrystalLookup, k: str = "") -> np.ndarray:
    return lookup(_pos(c, k), c.get(f"PreStepUniqueVolumeID{k}"))


def _spelling(k: str) -> str:
    """`PostPosition1_X` (online sorter) to `PostPosition_X1` (offline sorter)."""
    if k[-2:] in ("_X", "_Y", "_Z") and k[-3] in "12":
        return k[:-3] + k[-2:] + k[-3]
    return k


def _base(k: str) -> str:
    k = _spelling(k)
    return k[:-1] if k[-1] in "12" else k


def _normalise(arrs: dict) -> dict:
    return {_spelling(k): np.asarray(v) for k, v in arrs.items()}


def iterate(root_file: Path, tree: str, step: str = "400 MB"):
    import uproot

    f = uproot.open(str(root_file))
    names = [k for k in f.keys() if k.split(";")[0] == tree]
    if not names:
        return
    t = f[names[0]]
    want = [b for b in t.keys() if _base(b) in BRANCHES]
    for chunk in t.iterate(want, step_size=step, library="np"):
        yield _normalise(chunk)


def norm_acceptance(normdt_flat, binmap: BinMap) -> np.ndarray:
    """`min(1, normdt / its mean over views)` per bin; 0 where normdt is 0."""
    n = normdt_flat.reshape(binmap.shape).astype(np.float64)
    m = n.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(m > 0, n / m, 0.0)
    return np.clip(p, 0.0, 1.0).reshape(-1).astype(np.float32)


def pairs_to_events(c: dict, lookup: CrystalLookup, bed_start_ticks: int):
    """`(xa, xb, tof, t_ms, is_random, is_scatter)` for one chunk of coincidences."""
    xa, xb = _crystals(c, lookup, "1"), _crystals(c, lookup, "2")
    tof = eio.tof_bin_from_dt(c["GlobalTime2"] - c["GlobalTime1"])
    t_ms = bed_start_ticks + np.floor(c["GlobalTime1"] / 1e6).astype(np.int64)
    rnd = c["EventID1"] != c["EventID2"]
    sc = np.zeros(len(xa), bool)
    for k in ("PhantomCompton", "PhantomRayleigh"):
        if f"{k}1" in c:
            sc |= (c[f"{k}1"] > 0) | (c[f"{k}2"] > 0)
    return xa, xb, tof, t_ms, rnd, sc & ~rnd


def convert(chunk_dirs, binmap: BinMap, accept, bed_start_ticks: int, rng, out=print):
    """All prompts of all chunks as events, plus the delays count, after FOV and norm."""
    lookup = CrystalLookup()
    keep_cols = {k: [] for k in ("xa", "xb", "tof", "t", "rnd", "sc")}
    n_raw = n_fov = delays = delays_raw = 0
    for d in chunk_dirs:
        f = Path(d) / "coinc.root"
        for c in iterate(f, "Prompts"):
            xa, xb, tof, t, rnd, sc = pairs_to_events(c, lookup, bed_start_ticks)
            n_raw += len(xa)
            b = binmap.flat(xa, xb)
            ok = b >= 0
            n_fov += int(ok.sum())
            ok &= rng.random(len(xa)) < np.where(ok, accept[np.maximum(b, 0)], 0.0)
            for k, v in zip(keep_cols, (xa, xb, tof, t, rnd, sc)):
                keep_cols[k].append(v[ok])
        for c in iterate(f, "Delays"):
            b = binmap.flat(_crystals(c, lookup, "1"), _crystals(c, lookup, "2"))
            ok = b >= 0
            delays_raw += len(b)
            delays += int((rng.random(len(b)) < np.where(ok, accept[np.maximum(b, 0)], 0.0)).sum())
    cols = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in keep_cols.items()}
    out(f"  prompts: {n_raw:,} sorted, {n_fov:,} inside the sinogram, "
        f"{len(cols['xa']):,} after the norm acceptance; delays {delays:,} "
        f"(of {delays_raw:,})")
    return cols, {"prompts_sorted": n_raw, "prompts_in_fov": n_fov,
                  "prompts_kept": int(len(cols["xa"])), "delays": delays,
                  "delays_sorted": delays_raw}


def singles_rate(singles_dir: Path, seconds: float) -> np.ndarray:
    """`(13824,)` singles per second per GE crystal, from the short singles run."""
    lookup = CrystalLookup()
    counts = np.zeros(NXTAL, np.int64)
    for c in iterate(Path(singles_dir) / "singles.root", "Singles"):
        counts += np.bincount(_crystals(c, lookup), minlength=NXTAL)
    return counts / float(seconds)


def smooth_scatter(xa, xb, binmap: BinMap, scale: float, sigma=(1.5, 8.0, 5.0)):
    """Scatter-flagged pairs to a smooth per-bin expectation, `(plane, view, tang)`.

    Too few scatter events land in a bin to use raw, so the histogram is
    smoothed within each segment's planes, circularly over views and along
    the tangential axis, then scaled by `scale` (frame over simulated time).
    """
    from scipy.ndimage import gaussian_filter

    b = binmap.flat(xa, xb)
    h = np.bincount(b[b >= 0], minlength=binmap.n_bin).astype(np.float32)
    h = h.reshape(binmap.shape)
    out = np.empty_like(h)
    start = 0
    for n in binmap.hdr.axial:
        seg = h[start:start + n]
        out[start:start + n] = gaussian_filter(seg, sigma, mode=("nearest", "wrap", "nearest"))
        start += n
    tot = float(h.sum())
    if out.sum() > 0:
        out *= tot / float(out.sum())
    return out * np.float32(scale)


def tof_profile(xa, xb, tof, binmap: BinMap, n_tof: int = eio.TOF_HALF * 2 + 1):
    """`(n_tang, n_tof)` TOF shape of a set of pairs per tangential bin, rows summing to 1.

    The index is parallelproj's sinogram bin for the LOR run from det1 to det2
    of the bin, `27 - tof_bin` when `xtal_a` is on det1 -- what `pp` multiplies
    its scatter by.
    """
    b, swap = binmap.flat(xa, xb, with_swap=True)
    ok = b >= 0
    t = np.where(swap, -tof, tof)[ok]
    j = np.clip(eio.TOF_HALF - t, 0, n_tof - 1)
    u = (b[ok] % binmap.n_tang).astype(np.int64)
    P = np.bincount(u * n_tof + j, minlength=binmap.n_tang * n_tof).reshape(
        binmap.n_tang, n_tof).astype(np.float64)
    P += 1e-3
    from scipy.ndimage import gaussian_filter1d

    P = gaussian_filter1d(P, 3.0, axis=0, mode="nearest")
    return (P / P.sum(axis=1, keepdims=True)).astype(np.float32)


def save_summary(path: Path, **kw) -> None:
    np.savez_compressed(path, **kw)


def load_summary(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def decay_integral(t0: float, t1: float, half_life_s: float, power: int = 1) -> float:
    lam = power * math.log(2.0) / half_life_s
    return (math.exp(-lam * t0) - math.exp(-lam * t1)) / lam


def read_run(d: Path) -> dict:
    return json.loads((Path(d) / "run.json").read_text())
