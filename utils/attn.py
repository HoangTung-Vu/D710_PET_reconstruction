"""Per-bed attenuation factors, the one term not taken from GE's kernel."""

from __future__ import annotations

from . import attenuation


def check_same_exam(ct, hdr) -> None:
    """Require the CT and the exam to agree by UID, not by adjacency on disk."""
    got, want = ct.meta["frame_of_reference_uid"], hdr["sop_instance_uid"]
    if got != want:
        raise SystemExit(
            "error: this CT does not belong to the same exam as this bed\n"
            f"  CT  FrameOfReferenceUID {got}\n"
            f"  RDF sop_instance_uid    {want}")


class Attenuation:
    """`af` for each bed of a case, cached on disk and in memory."""

    def __init__(self, case, ct_dir: str, image, acq_template, verbose: bool = True):
        self.case = case
        self.ct = attenuation.load(ct_dir)
        self.image = image
        self.acq = acq_template
        self.verbose = verbose
        self._cache: dict[int, object] = {}

    def describe(self) -> str:
        return self.ct.describe()

    def _nontof_template(self, n: int):
        import sirf.STIR as pet

        if int(self.acq.dimensions()[0]) == 1:
            return self.acq
        p = self.case.work_bed(n) / "normdt.hs"
        if not p.exists():
            raise SystemExit(
                "error: the prompts of bed %d are TOF, so the attenuation "
                "factors need a\n"
                "  non-TOF template, and %s is missing.\n"
                "  run: d710 tostir --case %s --bed %d"
                % (n, p, self.case.name, n))
        return pet.AcquisitionData(str(p))

    def af(self, n: int):
        import sirf.STIR as pet

        if n in self._cache:
            return self._cache[n]

        hdr = self.case.header(n)
        check_same_exam(self.ct, hdr)

        path = self.case.work_bed(n) / "attn.hs"
        if _complete(path):
            self._cache[n] = pet.AcquisitionData(str(path)).as_array()
            if self.verbose:
                print(f"  bed {n}: attn.hs already present "
                      f"af mean {self._cache[n].mean():.4f}")
            return self._cache[n]
        if path.exists() and self.verbose:
            print(f"  bed {n}: attn.hs is there but attn.s is missing or the "
                  f"wrong size -- rebuilding")

        path.parent.mkdir(parents=True, exist_ok=True)
        mu = attenuation.mu_image(self.ct, hdr["table_position_mm"], self.image)
        af, _acf = attenuation.factors(self._nontof_template(n), mu)
        self._cache[n] = af.as_array()
        _write_like_the_others(self._cache[n], self.case, n, path)
        if self.verbose:
            m = mu.as_array()
            print(f"  bed {n}: table {hdr['table_position_mm']:>8.2f} mm  "
                  f"mu max {m.max():.4f} 1/cm  "
                  f"af mean {self._cache[n].mean():.4f}  -> ghi attn.hs")
        return self._cache[n]

    def all(self, beds) -> dict:
        return {n: self.af(n) for n in beds}


def _complete(hs) -> bool:
    s = hs.with_suffix(".s")
    if not (hs.exists() and s.exists()):
        return False
    want = hs.parent / "normdt.s"
    if want.exists():
        return s.stat().st_size == want.stat().st_size
    return s.stat().st_size > 0


def _write_like_the_others(a, case, n: int, path) -> None:
    import re

    import numpy as np

    src = (case.work_bed(n) / "normdt.hs")
    src = src if src.exists() else case.prompt(n)
    hdr = src.read_text()
    hdr = re.sub(r"(?im)^(\s*name of data file\s*:=).*$", r"\1 attn.s", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number format\s*:=).*$", r"\1 float", hdr)
    hdr = re.sub(r"(?im)^(\s*!?\s*number of bytes per pixel\s*:=).*$", r"\1 4", hdr)
    np.ascontiguousarray(a, "<f4").tofile(path.with_suffix(".s"))
    path.write_text(hdr)
