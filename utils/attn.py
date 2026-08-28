"""Per-bed attenuation factors — the ONE term not taken from GE's kernel.

`vendor/estimate.py` does use a mu-map (it has to, in order to simulate
scatter), but what it exports is **scatter**, not attenuation factors. So `af` is
rebuilt here from the same CT series, via `utils/attenuation.py`.

Cached to `work/bed<n>/attn.hs` so it sits alongside the other three terms
(randoms / scatter / normdt) — after the first run all four terms of the model
are present on disk and can be opened with any STIR tool. Recomputing costs
~16 s/bed.

`af` is **always non-TOF**, even when the prompts are not. That is not a
simplification: attenuation is the survival probability of a photon PAIR along
the LOR and does not depend on when either photon arrived, which is why GE has
one `COsemTofMain::GetAttnViewData(view, out)` with no TOF parameter and
multiplies the same non-TOF buffer into every TOF bin. STIR agrees, and says so
out loud — `BinNormalisationFromAttenuationImage` refuses TOF input with
"currently can only handle non_TOF data", so the template used below has to be
a non-TOF one whatever was passed in.

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
            "error: this CT does not belong to the same exam as this bed\n"
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

    def _nontof_template(self, n: int):
        """A non-TOF acquisition template with this bed's geometry.

        `self.acq` is normally the prompts, which are TOF, and
        `compute_attenuation_factors` rejects those outright. `normdt.hs` is the
        obvious stand-in: `vendor/to_stir.py` writes it non-TOF from the same
        header, for the same bed, so the geometry is identical by construction
        and `terms.load` already requires it to exist.
        """
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
                print(f"  bed {n}: attn.hs already present "
                      f"af mean {self._cache[n].mean():.4f}")
            return self._cache[n]

        path.parent.mkdir(parents=True, exist_ok=True)
        mu = attenuation.mu_image(self.ct, hdr["table_position_mm"], self.image)
        af, _acf = attenuation.factors(self._nontof_template(n), mu)
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
