"""A miniature D710 -- same michelogram rules, small enough to hold in RAM.

The real bed is 553 x 288 x 381 int16 = 121 MB, which no test should carry
around.  Every geometric rule this project depends on is a rule about *span 2*,
not about 24 rings, so the tests scale the scanner down and keep the rules:

* segment 0 spans ring difference -1..+1, so its odd axial positions gather
  **two** ring pairs and its even ones gather one;
* segment +k spans ``[2k, 2k+1]`` and -k spans ``[-(2k+1), -2k]``;
* segments are stored ``0, +1, -1, +2, -2, ...`` with ``(2R-1) - 4|k|`` axial
  positions each.

With ``R`` rings that gives ``2R-1`` planes in segment 0 and a maximum ring
difference of ``2K+1``.  The default ``R=6`` yields 31 planes total against the
real scanner's 553.
"""

from __future__ import annotations

import os

import numpy as np

#: Values that are the thing under test elsewhere keep their real numbers.
PLANE_MM = 3.2699997
BIN_SIZE_CM = 0.21306


def segments(num_rings: int) -> list[tuple[int, int, int, int]]:
    """``(segment, min_rd, max_rd, num_axial_poss)`` in STIR storage order."""
    n0 = 2 * num_rings - 1
    out = [(0, -1, 1, n0)]
    k = 1
    while n0 - 4 * k >= 1:
        out.append((k, 2 * k, 2 * k + 1, n0 - 4 * k))
        out.append((-k, -(2 * k + 1), -2 * k, n0 - 4 * k))
        k += 1
    return out


def header(num_rings: int = 6, num_det: int = 16, num_tang: int = 9,
           data_file: str = "mini.s", number_format: str = "signed integer",
           bytes_per_pixel: int = 2) -> str:
    segs = segments(num_rings)
    return "\n".join([
        "!INTERFILE  :=",
        "!imaging modality := PT",
        f"name of data file := {data_file}",
        "originating system := Discovery 710",
        "!version of keys := STIR3.0",
        "!GENERAL DATA :=",
        "!GENERAL IMAGE DATA :=",
        "!type of data := PET",
        "imagedata byte order := LITTLEENDIAN",
        "!PET STUDY (General) :=",
        "!PET data type := Emission",
        "applied corrections := {None}",
        f"!number format := {number_format}",
        f"!number of bytes per pixel := {bytes_per_pixel}",
        "number of dimensions := 4",
        "matrix axis label [4] := segment",
        f"!matrix size [4] := {len(segs)}",
        "matrix axis label [3] := axial coordinate",
        "!matrix size [3] := {%s}" % ",".join(str(s[3]) for s in segs),
        "matrix axis label [2] := view",
        f"!matrix size [2] := {num_det // 2}",
        "matrix axis label [1] := tangential coordinate",
        f"!matrix size [1] := {num_tang}",
        "minimum ring difference per segment := {%s}"
        % ",".join(str(s[1]) for s in segs),
        "maximum ring difference per segment := {%s}"
        % ",".join(str(s[2]) for s in segs),
        "number of time frames := 1",
        "image duration (sec)[1] := 90",
        "Scanner parameters:=",
        "Scanner type := Discovery 710",
        f"Number of rings := {num_rings}",
        f"Number of detectors per ring := {num_det}",
        "Inner ring diameter (cm) := 81.0200",
        "Average depth of interaction (cm) := 0.9400",
        f"Distance between rings (cm) := {2 * PLANE_MM / 10:.4f}",
        f"Default bin size (cm) := {BIN_SIZE_CM}",
        "View offset (degrees) := -5.0210",
        f"Maximum number of non-arc-corrected bins := {num_tang}",
        f"Default number of arc-corrected bins := {num_tang}",
        "end scanner parameters:=",
        "!END OF INTERFILE :=",
        "",
    ])


def num_planes(num_rings: int = 6) -> int:
    return sum(s[3] for s in segments(num_rings))


def write(stem: str, num_rings: int = 6, num_det: int = 16, num_tang: int = 9,
          data: np.ndarray | None = None) -> str:
    """Write ``<stem>.hs``/``<stem>.s``; returns the header path.

    ``data`` is ``(1, planes, views, tang)`` -- STIR's own axis order.  Left
    out, the sinogram is zeros.
    """
    stem = str(stem)
    base = os.path.basename(stem)
    shape = (1, num_planes(num_rings), num_det // 2, num_tang)
    a = np.zeros(shape, dtype="<i2") if data is None else data.astype("<i2")
    if a.shape != shape:
        raise ValueError(f"data is {a.shape}, this header describes {shape}")
    with open(stem + ".hs", "w") as f:
        f.write(header(num_rings, num_det, num_tang, data_file=base + ".s"))
    a.tofile(stem + ".s")
    return stem + ".hs"
