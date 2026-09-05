"""Every number that describes the D710 or the grid it is reconstructed on.

One file, so the two algorithm packages (`osem/`, `lm/`) cannot drift apart.
Nothing here is a tuning knob: each value is measured, read off a vendor header,
or derived from one that is. Constants plus three small helpers, with every
import inside a function — safe to read from either runtime.
"""

from __future__ import annotations

# ----------------------------------------------------------------- detectors
NRINGS, NDET = 24, 576
NXTAL = NRINGS * NDET

#: Ring radius, mm. Header: `Inner ring diameter (cm) := 81.02` / 2.
R_MM = 405.10

#: Azimuth of GE crystal 0 in the gantry frame. cmcfg.XR.xml:721 and every RDF.
XTAL0_OFFSET_DEG = -5.0210

#: Crystal / view pitch; GE's own `deltaAngle`.
XTAL_PITCH_DEG = 360.0 / NDET          # 0.625

#: STIR's `psi_offset`, NOT the same quantity as XTAL0_OFFSET_DEG, and PROVISIONAL.
#: Was -5.0210, which cost 10.04 deg of image rotation on both paths.
#: Derivation, measurements and the open +4.4 / +5.7 conflict: GEOMETRY_AUDIT.md.
VIEW_OFFSET_DEG = -(XTAL0_OFFSET_DEG + XTAL_PITCH_DEG)      # +4.3960

#: STIR places a LOR at `R_MM + DOI_MM`. GE: SYS_EFF_RING_DIAMETER 827.0
#: (cmcfg.XR.xml:677, RDF effectiveRingDiameter) -> 413.50 - 405.10. Was 9.4,
#: STIR's Discovery 690 default. See GEOMETRY_AUDIT.md.
DOI_MM = 8.4

#: Where STIR/GE put the LOR. Both paths must use this, not R_MM.
R_EFF_MM = R_MM + DOI_MM        # 413.50

#: Plane spacing = half the ring spacing, from `Distance between rings (cm)`.
PLANE_MM = 3.2699997

#: NOT `axial_fov_mm / nrings` (156.70/24 = 6.529): that is 0.17 % short and
#: leaves the crystal LUT just inside the outermost image plane. `lm.recon`
#: no longer depends on it for that — `axial_mask` rounds instead of flooring,
#: which is what kept plane 46 alive — but the value is still the one the
#: header states and the one the projector geometry has to use.
RING_PITCH_MM = 2 * PLANE_MM

#: Direct planes (segment 0) in the 553-plane sinogram = 2·NRINGS − 1.
NSEG0 = 47

#: GE numbers crystals opposite to STIR, offset by half a ring: calibrated by
#: histogramming NEMA bed 2 list-mode through all 1152 candidates against the
#: vendor-decoded sinogram of that same bed (corr 0.990, runner-up 0.981).
CRYSTAL_REVERSE = True
CRYSTAL_OFFSET = 288

# ----------------------------------------------------------------------- TOF
#: 55 bins of one coincidence-timing LSB; timing resolution 675 ps FWHM.
#: Was 550.0 (STIR's Discovery 690 placeholder, 23 % too narrow). The machine's
#: own value: sharcAp.cfg:46 TIMING_RESOLUTION 675.  See GEOMETRY_AUDIT.md.
N_TOF_RAW, TOF_LSB_PS, TIMING_PS = 55, 89.2459, 675.0
C_MM_PS = 0.299792458

#: Total TOF range in mm — fixed by the hardware, so mashing changes the number
#: of bins and never this.
TOF_RANGE_MM = N_TOF_RAW * C_MM_PS * TOF_LSB_PS / 2

#: Sinogram bin width, mm. Header: `Default bin size (cm) := 0.21306`.
BIN_MM = 2.1306

# ---------------------------------------------------------------- image grid
#: Transverse voxel, mm. Equal to `BIN_MM` by construction — that is STIR at
#: zoom 1. Both SIRF builds reach it at `XY`; `lm` sets it outright.
DR_MM = BIN_MM

#: Transverse matrix size. Measured, not chosen: the `sirf-local` image pins the
#: FOV at 718.01 mm and scales the voxel with `xy`, the host build pins the voxel
#: at DR_MM and scales the FOV. 337 is the one size that gives 2.130600 mm in
#: BOTH (256 gave 2.8047 mm in the container and 2.1306 on the host — the same
#: `--xy` producing two different scales, which is what `K` then absorbed).
XY = 337
FOV_MM = XY * DR_MM

# ------------------------------------------------- reconstruction defaults
#: 288 views, so the number of subsets must divide 288.
N_SUBSETS, N_ITERATIONS = 24, 2

#: Rays per tangential bin. The single biggest cost knob: 5 -> 1 is a ~5x
#: speedup for a coarser transaxial model. Use `--lors 1` for a smoke test only.
TANGENTIAL_LORS = 5

#: GE's own transaxial PSF FWHM (mm), from the PT private tags.
PSF_MM = 6.4

#: Post-filter, GE's own setting for this protocol, read off the private tags of
#: the vendor's reconstruction of this very exam: `(0009,10BB) post_filt_parm`
#: with `(0009,10BA) post_filter = 1` saying it is on.
POST_FILTER_FWHM_MM = 6.4

#: Axial post-filter from `(0009,10DC) ir_z_filter_ratio` (enabled by
#: `(0009,10DB) ir_z_filter_flag = 2`). The vendor tag gives the RATIO; the
#: three-tap `[1, ratio, 1]` shape is INFERRED — at ratio 4 it normalises to
#: `[0.1666667, 0.6666667, 0.1666667]`, character for character the
#: `PSF_AXIAL_KERNEL[]` in `sharcAp.cfg.XR`. A good reconstruction of GE's
#: filter, not decoded ground truth.
POST_FILTER_Z_RATIO = 4.0

# -------------------------------------------------------------- CT -> mu-map
#: Carney bilinear HU -> mu(511 keV), 1/mm.
MU_WATER_511 = 0.0096
MU_BONE_511 = 0.0172
CARNEY_B = {80: 0.681, 100: 0.755, 120: 0.837, 140: 1.0}

# -------------------------------------------------------------- calibration
#: ASSUMED unit convention between `hrActivityFactor` and (Bq/mL)/(count/voxel).
#: Not yet derived; only a NEMA measurement will settle it.
WCC_UNIT_SCALE = 1e4

#: Export's own `K`, in (Bq/mL)/(count/voxel). INDEPENDENT of the scanner's WCC:
#: measured against GE's own BQML reconstruction of the same exams
#: (`tools/compare_vendor.py` -> `tools/calib_k.py`). `None` = fall back to the
#: WCC, then to the dose-based bound.
#:
#: ⚠ Valid ONLY for the exact chain it was measured with:
#:   * F-18, and the GE series-start time reference (`quant.scan_start_factor`);
#:   * `XY` 337 at `DR_MM` 2.1306 mm, `PLANE_MM` axially — the projector
#:     accumulates along the voxel STEP, so another voxel size is another `K`
#:     (that is why `DR_MM` is pinned above rather than left to `--xy`);
#:   * `N_SUBSETS` x `N_ITERATIONS`, unregularised OSEM;
#:   * the post-filter below, on;
#:   * GE's own randoms / scatter / normdt / norm_only (`d710 estimate`).
#:
#: TWO constants, because the two reconstructors do not put the same number of
#: counts in a voxel: `K_EXPORT` is the non-TOF sinogram path (`d710 osem`),
#: `K_EXPORT_LM` the 55-bin TOF list-mode path (`d710 lm recon`). Using one for
#: both was measured wrong by ~2.1x.
K_EXPORT = 63_002.1      # 5 ca, tản 8.3 %, r >= 0.962
K_EXPORT_LM = 124_178.0      # 5 ca, tản 10.8 %, r >= 0.977


def fov_radius_mm(n_tang: int, ndet: int = NDET,
                  r_mm: float = R_MM + DOI_MM) -> float:
    """How far off the axis a LOR actually reaches, mm.

    The outermost of `n_tang` non-arc-corrected tangential bins is number
    `(n_tang - 1) / 2`, so its chord has half-width
    `r sin(pi (n_tang - 1) / 2 ndet)` — with `r` measured to the depth of
    interaction, which is where STIR puts the LOR. D710: 381 of 576 -> 355.8250
    mm, equal to `geometry.tangential_s_mm(hs).max()` to every digit STIR
    prints (`tests/test_geometry.py`) — for a header carrying the same DOI. It
    was 356.6855 while `DOI_MM` was STIR's D690 default 9.4.

    Past that radius NO bin crosses the voxel, so its sensitivity is not small
    but meaningless — and OSEM divides by it.
    """
    import math

    return r_mm * math.sin(math.pi * (n_tang - 1) / (2 * ndet))


def fov_mask(xy: int, n_tang: int, dr_mm: float = DR_MM):
    """`(xy, xy)` bool, True inside `fov_radius_mm`.

    The grid is square and the scanner is round: at `XY` the corners sit at
    506 mm against a reach of 357. Left alone, list-mode OSEM put **34 % of ped
    bed 1's counts** out there, with a peak 52x the real one, because the
    sensitivity is 2e-05 against a peak of 1.16e+04. GE's own `PT` series are
    likewise a round FOV in a square matrix.

    Used as the INITIAL estimate, not as a clip afterwards: OSEM is
    multiplicative, so zero at the start stays zero and the counts land inside
    the patient instead of being thrown away at the end.
    """
    import numpy as np

    y, x = np.mgrid[0:xy, 0:xy] - (xy - 1) / 2.0
    return np.hypot(x, y) * dr_mm <= fov_radius_mm(n_tang)


def sirf_grid(acq, xy: int = XY, dr_mm: float = DR_MM, out=print):
    """Uniform image with `dr_mm` transverse voxels, in either SIRF build.

    `XY` hits `dr_mm` in both; any other `xy` is rescaled here rather than
    silently changing the scale of the result (and with it `K`).
    """
    x = acq.create_uniform_image(1.0, xy)
    got = float(x.voxel_sizes()[1])
    if abs(got - dr_mm) <= 1e-3:
        return x
    n = max(1, round(xy * got / dr_mm))
    out(f"grid: xy {xy} -> {n}, this SIRF pins the FOV ({xy * got:.2f} mm) "
        f"and would have given {got:.4f} mm voxels")
    return acq.create_uniform_image(1.0, n)
