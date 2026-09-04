"""Read a STIR projdata header without STIR.

`lm/` and `lowdose/` run in the PyTomography environment, which has no SIRF and
no STIR -- those live in the `sirf-local` image (`d710_isolate_stir.sh`). The
header states everything the bin map needs, so it is parsed here rather than
imported.

`tests/test_lm_geom.py` checks every number this returns against STIR's own
`ProjDataInfo` whenever STIR happens to be importable, so the parser is verified
rather than trusted.
"""

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
        """True when the file runs `(tof, plane, view, tang)` -- the decoded layout.

        SIRF's own writer puts view before axial and stores segments ascending,
        so `as_array()` hides the difference but `np.fromfile` does not. Files in
        that layout cannot be read here.
        """
        return self.axis3.startswith("axial")

    def require_plane_major(self) -> None:
        if not self.plane_major:
            raise SystemExit(
                f"error: {self.path} is in SIRF's own segment order (axis 3 is "
                f"{self.axis3!r}, segments ascending).\n"
                "  `lm` reads with numpy, not SIRF, so it needs the decoded "
                "layout every other term is in.\n"
                f"  delete it and rebuild:  rm {self.path[:-3]}.hs "
                f"{self.path[:-3]}.s && d710 attn --case <name>")

    @property
    def n_plane(self) -> int:
        return sum(self.axial)

    def segments(self):
        """`(segment, min_rd, max_rd, n_axial)` in the order the file stores them.

        The header lists them in storage order and the segment number is implied
        by the sign of the ring difference, exactly as STIR writes it:
        `0, +1, -1, +2, -2, ...`
        """
        out = []
        for i, (lo, hi, n) in enumerate(zip(self.min_rd, self.max_rd, self.axial)):
            s = 0 if i == 0 else (i + 1) // 2 * (1 if i % 2 else -1)
            out.append((s, lo, hi, n))
        return out

    def ring_pairs(self):
        """The ring pairs summed into each plane, as `(ring of pos1, ring of pos2)`.

        Same rule as `utils.geometry.plane_ring_pairs`, which derives it from
        STIR: the pair is `(r + d, r)` because STIR's signed segment follows
        `pos1 - pos2`, and segment 0 of span 2 merges two ring pairs into every
        odd axial position.
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
