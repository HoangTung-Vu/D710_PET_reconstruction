"""Just enough Interfile to memory-map a `.s` without loading it.

A real bed is 553 x 288 x 381; the five correction terms plus the prompts come
to 1.4 GB.  `sirf.AcquisitionData` reads all of that into RAM, which is fine in
the notebook -- one bed at a time -- and wasteful in a test that only wants
per-plane sums.  These read the shape out of the header and hand back a memmap.
"""

from __future__ import annotations

import os
import re

import numpy as np

NUMBER_FORMAT = {("float", 4): "<f4",
                 ("signed integer", 2): "<i2",
                 ("signed integer", 4): "<i4",
                 ("unsigned integer", 2): "<u2"}


def keys(hs: str) -> dict[str, str]:
    """`{key: value}`, with the `!` and case noise stripped off the keys."""
    out = {}
    with open(hs) as f:
        for line in f:
            if ":=" not in line:
                continue
            k, v = line.split(":=", 1)
            out[k.strip().lstrip("!").strip().lower()] = v.strip()
    return out


def _check_layout(hs: str, k: dict[str, str]) -> None:
    """Only the plane-major layout can be one flat array.

    `decode.sh` and `to_stir.py` both write ``segment, axial, view, tangential``
    with every segment's planes concatenated, so the file is one dense
    ``(planes, views, tang)`` block.  A file **SIRF** wrote is
    ``segment, view, axial, tangential`` -- the axial length changes per
    segment, so it is ragged and there is no single memmap for it.  Read those
    through `sirf.AcquisitionData`, which canonicalises the order anyway.
    """
    if k.get("matrix axis label [3]", "").lower() != "axial coordinate":
        raise ValueError(
            f"{hs}: axis 3 is {k.get('matrix axis label [3]')!r}, not 'axial "
            "coordinate' -- this is SIRF's own segment-major layout, which is "
            "ragged.  Load it with sirf.AcquisitionData instead.")


def shape(hs: str) -> tuple[int, int, int, int]:
    """`(tof, axial, view, tangential)` -- STIR's own storage order.

    Axis 5 is the timing-position axis and is absent from a non-TOF header, so
    it defaults to 1.  It has to be **read**, not assumed: a TOF `.s` is
    `n_tof` times the size, and hard-coding 1 builds a memmap over the first
    slice of the file only -- which looks like a perfectly ordinary sinogram
    holding a fraction of the counts.
    """
    k = keys(hs)
    _check_layout(hs, k)
    axial = [int(n) for n in re.findall(r"-?\d+", k["matrix size [3]"])]
    n_tof = int(k.get("matrix size [5]", 1))
    return (n_tof, sum(axial), int(k["matrix size [2]"]),
            int(k["matrix size [1]"]))


def axial_sizes(hs: str) -> list[int]:
    """Axial positions per segment, in storage order."""
    k = keys(hs)
    _check_layout(hs, k)
    return [int(n) for n in re.findall(r"-?\d+", k["matrix size [3]"])]


def dtype(hs: str) -> str:
    k = keys(hs)
    fmt = (k["number format"].lower(), int(k["number of bytes per pixel"]))
    if fmt not in NUMBER_FORMAT:
        raise ValueError(f"{hs}: unsupported {fmt}")
    return NUMBER_FORMAT[fmt]


def data_file(hs: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(hs)),
                        keys(hs)["name of data file"])


def load(hs: str) -> np.memmap:
    """The sinogram as a read-only memmap of shape `(tof, axial, view, tang)`."""
    return np.memmap(data_file(hs), dtype=dtype(hs), mode="r", shape=shape(hs))


def per_plane(hs: str) -> np.ndarray:
    """Sum over TOF, view and tangential, in float64, one value per stored plane.

    **Over TOF as well**, so the result is per LOR whether or not the file has a
    timing axis.  Every invariant built on this compares prompts against
    non-TOF terms (randoms, scatter), and taking one timing bin of the prompts
    would compare a fraction of them against the whole of a correction.
    """
    a = load(hs)
    return a.sum(axis=(0, 2, 3), dtype=np.float64)


def edge_distance(hs: str) -> np.ndarray:
    """Planes from the nearest end of the plane's own segment, per plane.

    0 at a segment's outermost axial position.  Those planes see the fewest
    ring pairs of anything in the acquisition, so they are where every
    estimated term is least constrained.
    """
    out = []
    for n in axial_sizes(hs):
        out.append(np.minimum(np.arange(n), np.arange(n)[::-1]))
    return np.concatenate(out)
