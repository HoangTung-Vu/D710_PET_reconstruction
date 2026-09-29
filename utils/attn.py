from __future__ import annotations

import numpy as np

from . import attenuation, attn_proj, interfile
from .scanner import DR_MM, GEOMETRY, PLANE_MM, XY

PROVENANCE = "; d710 attn ring pairing := utils.binmap"

GEOMETRY_STAMP = f"; d710 attn geometry := {GEOMETRY}"


def check_same_exam(ct, hdr) -> None:
    if ct.meta.get("format") == "nifti":
        return
    got, want = ct.meta["frame_of_reference_uid"], hdr["sop_instance_uid"]
    if got != want:
        raise SystemExit(
            "error: this CT does not belong to the same exam as this bed\n"
            f"  CT  FrameOfReferenceUID {got}\n"
            f"  RDF sop_instance_uid    {want}")


class Attenuation:
    def __init__(self, case, ct_dir: str, xy: int = XY, dr_mm: float = DR_MM,
                 device: str = "auto", verbose: bool = True):
        self.case = case
        self.ct = attenuation.load(ct_dir)
        self.xy = xy
        self.dr_mm = dr_mm
        self.device = device
        self.verbose = verbose
        self._cache: dict[int, np.ndarray] = {}

    def describe(self) -> str:
        return self.ct.describe()

    def af(self, n: int) -> np.ndarray:
        if n in self._cache:
            return self._cache[n]

        hdr = self.case.header(n)
        check_same_exam(self.ct, hdr)

        path = self.case.work_bed(n) / "attn.hs"
        if _complete(path):
            self._cache[n] = read(path)
            if self.verbose:
                print(f"  bed {n}: attn.hs already present "
                      f"af mean {self._cache[n].mean():.4f}")
            return self._cache[n]
        if path.exists() and self.verbose:
            txt = path.read_text(errors="replace")
            why = ("was built before the ring pairing was corrected on "
                   "2026-09-18" if PROVENANCE not in txt else
                   "was built for another scanner geometry than "
                   f"{GEOMETRY}" if GEOMETRY_STAMP not in txt else
                   "is missing its data file or the wrong size")
            print(f"  bed {n}: attn.hs {why} -- rebuilding")

        path.parent.mkdir(parents=True, exist_ok=True)
        mu = attenuation.mu_map(self.ct, hdr["table_position_mm"], self.xy,
                                self.dr_mm)
        if self.verbose:
            print(f"  bed {n}: table {hdr['table_position_mm']:>8.2f} mm  "
                  f"mu max {mu.max():.4f} 1/cm", flush=True)
        af = attn_proj.factors(mu, self.case.prompt(n), dr_mm=self.dr_mm,
                               plane_mm=PLANE_MM, device=self.device,
                               out=print if self.verbose else (lambda *_: None))
        _write_like_the_others(af, self.case, n, path)
        self._cache[n] = af[None]
        return self._cache[n]

    def all(self, beds) -> dict:
        return {n: self.af(n) for n in beds}


def read(hs) -> np.ndarray:
    h = interfile.Header(hs)
    h.require_plane_major()
    a = np.fromfile(h.data_file(), "<f4")
    want = h.n_plane * h.n_view * h.n_tang
    if a.size != want:
        raise SystemExit(f"error: {hs} holds {a.size:,} values, its header "
                         f"describes {want:,}")
    return a.reshape(1, h.n_plane, h.n_view, h.n_tang)


def _complete(hs) -> bool:
    s = hs.with_suffix(".s")
    if not (hs.exists() and s.exists()):
        return False
    txt = hs.read_text(errors="replace")
    if PROVENANCE not in txt or GEOMETRY_STAMP not in txt:
        return False
    want = hs.parent / "normdt.s"
    if want.exists():
        return s.stat().st_size == want.stat().st_size
    return s.stat().st_size > 0


def _write_like_the_others(a, case, n: int, path) -> None:
    import re

    src = (case.work_bed(n) / "normdt.hs")
    if not src.exists():
        src = case.prompt(n)
        if int(interfile.keys(src).get("matrix size [5]", 1)) != 1:
            raise SystemExit(
                f"error: the prompts of bed {n} are TOF, so their header cannot "
                f"describe the non-TOF\n  attenuation, and {case.work_bed(n)}/"
                f"normdt.hs is not there to borrow one from.\n"
                f"  run: d710 tostir --case {case.name} --bed {n}")
    hdr = src.read_text()
    hdr = re.sub(r"(?im)^(\s*name of data file\s*:=).*$", r"\1 attn.s", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number format\s*:=).*$", r"\1 float", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number of bytes per pixel\s*:=).*$", r"\1 4", hdr)
    hdr = hdr.rstrip("\n") + "\n" + PROVENANCE + "\n" + GEOMETRY_STAMP + "\n"
    np.ascontiguousarray(a, "<f4").tofile(path.with_suffix(".s"))
    path.write_text(hdr)
