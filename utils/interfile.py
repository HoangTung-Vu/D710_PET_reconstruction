"""Reader for STIR projdata headers that does not require STIR."""

from __future__ import annotations

import re


def keys(hs) -> dict:
    """`{key: value}` of an Interfile header, keys lower-cased and stripped of `!`."""
    out = {}
    for line in open(hs, errors="replace"):
        if ":=" not in line or line.lstrip().startswith(";"):
            continue
        k, v = line.split(":=", 1)
        out[k.strip().lstrip("!").strip().lower()] = v.strip()
    return out


def _ints(v: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", v)]


class Header:
    """The geometry of one projdata file."""

    def __init__(self, hs):
        k = keys(hs)
        self.path = str(hs)
        self.n_view = int(k["matrix size [2]"])
        self.n_tang = int(k["matrix size [1]"])
        self.n_rings = int(k["number of rings"])
        self.n_det = int(k["number of detectors per ring"])
        self.axial = _ints(k["matrix size [3]"])
        self.min_rd = _ints(k["minimum ring difference per segment"])
        self.max_rd = _ints(k["maximum ring difference per segment"])
        self.n_tof = int(k.get("matrix size [5]", 1))
        self.tof_mash = int(k.get("tof mashing factor", 1))
        self.data_name = k["name of data file"]
        self.axis3 = k.get("matrix axis label [3]", "axial coordinate").lower()
        if not len(self.axial) == len(self.min_rd) == len(self.max_rd):
            raise SystemExit(f"error: {hs} lists {len(self.axial)} axial sizes "
                             f"but {len(self.min_rd)} ring differences")

    @property
    def plane_major(self) -> bool:
        return self.axis3.startswith("axial")

    def require_plane_major(self) -> None:
        if not self.plane_major:
            raise SystemExit(
                f"error: {self.path} is in SIRF's own segment order (axis 3 is "
                f"{self.axis3!r}, segments ascending).\n"
                "  Nothing here reads with SIRF any more, so every term must be "
                "in the decoded layout.\n"
                f"  delete it and rebuild:  rm {self.path[:-3]}.hs "
                f"{self.path[:-3]}.s && d710 attn --case <name>")

    @property
    def n_plane(self) -> int:
        return sum(self.axial)

    def segments(self):
        out = []
        for i, (lo, hi, n) in enumerate(zip(self.min_rd, self.max_rd, self.axial)):
            s = 0 if i == 0 else (i + 1) // 2 * (1 if i % 2 else -1)
            out.append((s, lo, hi, n))
        return out

    def ring_pairs(self):
        """Per plane, the `(a, b)` ring pairs merged into it, with `a - b` the
        header's ring difference.

        The header declares the ring difference with STIR's sign, so `a` is
        the ring STIR calls `ring2` and `b` the one it calls `ring1` -- it is
        `b` that sits on `det1`.  Nothing should turn a plane back into a
        geometric line from here: go through
        `utils.binmap.BinMap.ring_pairs_by_plane()`, which pairs these rings
        with detectors and is what `d710 lm check` proves bit-exact against
        GE's own sinogram.  `utils/binmap.py` holds the measurement, and the
        warning about headers written before 2026-09-18.
        """
        out = []
        for _s, lo, hi, n in self.segments():
            z0 = min(abs(d) for d in range(lo, hi + 1))
            for a in range(n):
                z = z0 + a
                out.append([((z - d) // 2 + d, (z - d) // 2)
                            for d in range(lo, hi + 1)
                            if (z - d) % 2 == 0
                            and 0 <= (z - d) // 2 < self.n_rings
                            and 0 <= (z - d) // 2 + d < self.n_rings])
        return out

    def data_file(self):
        from pathlib import Path

        return Path(self.path).parent / self.data_name
