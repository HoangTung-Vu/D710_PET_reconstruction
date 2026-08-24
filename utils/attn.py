"""Per-bed attenuation factors — the ONE term not taken from GE's kernel.

`vendor/estimate.py` does use a mu-map (it has to, in order to simulate
scatter), but what it exports is **scatter**, not attenuation factors. So `af` is
rebuilt here from the same CT series, via `utils/attenuation.py`.

Cached to `work/bed<n>/attn.hs` so it sits alongside the other three terms
(randoms / scatter / normdt) — after the first run all four terms of the model
are present on disk and can be opened with any STIR tool. Recomputing costs
~16 s/bed.

⚠ Delete `attn.hs`/`attn.s` if the CT or the image grid changes — the file cannot
tell by itself.
"""

from __future__ import annotations

from . import attenuation


def check_same_exam(ct, hdr) -> None:
    """Which CT belongs to which exam is decided by UID, not by sitting next to it.

    This is an identity, not an approximate comparison: an image directory next
    to a raw directory is **no** guarantee of the same exam (`11082026/` holds
    images from two different exams).
    """
    got, want = ct.meta["frame_of_reference_uid"], hdr["sop_instance_uid"]
    if got != want:
        raise SystemExit(
            "error: CT không thuộc cùng exam với bed này\n"
            f"  CT  FrameOfReferenceUID {got}\n"
            f"  RDF sop_instance_uid    {want}")


class Attenuation:
    """`af` for each bed of a case, cached on disk and in RAM.

        at = Attenuation(case, ct_dir, template_image, template_acq)
        af4 = at.af(4)                 # (1, 553, 288, 381) numpy array
    """

    def __init__(self, case, ct_dir: str, image, acq_template, verbose: bool = True):
        self.case = case
        self.ct = attenuation.load(ct_dir)
        self.image = image
        self.acq = acq_template
        self.verbose = verbose
        self._cache: dict[int, object] = {}

    def describe(self) -> str:
        return self.ct.describe()

    def af(self, n: int):
        """Attenuation factors for bed `n` — survival probability ∈ (0, 1]."""
        import sirf.STIR as pet

        if n in self._cache:
            return self._cache[n]

        hdr = self.case.header(n)
        check_same_exam(self.ct, hdr)

        path = self.case.work_bed(n) / "attn.hs"
        if path.exists():
            self._cache[n] = pet.AcquisitionData(str(path)).as_array()
            if self.verbose:
                print(f"  bed {n}: attn.hs có sẵn         "
                      f"af mean {self._cache[n].mean():.4f}")
            return self._cache[n]

        path.parent.mkdir(parents=True, exist_ok=True)
        mu = attenuation.mu_image(self.ct, hdr["table_position_mm"], self.image)
        af, _acf = attenuation.factors(self.acq, mu)
        af.write(str(path))                          # -> attn.hs + attn.s
        self._cache[n] = af.as_array()
        if self.verbose:
            m = mu.as_array()
            print(f"  bed {n}: table {hdr['table_position_mm']:>8.2f} mm  "
                  f"mu max {m.max():.4f} 1/cm  "
                  f"af mean {self._cache[n].mean():.4f}  -> ghi attn.hs")
        return self._cache[n]

    def all(self, beds) -> dict:
        return {n: self.af(n) for n in beds}
