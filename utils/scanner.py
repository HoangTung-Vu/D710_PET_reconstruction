from __future__ import annotations

NRINGS, NDET = 24, 576
NXTAL = NRINGS * NDET

R_MM = 405.10

XTAL0_OFFSET_DEG = -5.0210

XTAL_PITCH_DEG = 360.0 / NDET

VIEW_OFFSET_DEG = -(XTAL0_OFFSET_DEG + XTAL_PITCH_DEG)

DOI_MM = 8.4

R_EFF_MM = R_MM + DOI_MM

PLANE_MM = 3.2699997

RING_PITCH_MM = 2 * PLANE_MM

NSEG0 = 47

CRYSTAL_REVERSE = True
CRYSTAL_OFFSET = 288

N_TOF_RAW, TOF_LSB_PS, TIMING_PS = 55, 89.2459, 675.0
C_MM_PS = 0.299792458

TOF_RANGE_MM = N_TOF_RAW * C_MM_PS * TOF_LSB_PS / 2

BIN_MM = 2.1306

DR_MM = BIN_MM

XY = 337
FOV_MM = XY * DR_MM

N_SUBSETS, N_ITERATIONS = 24, 2

TANGENTIAL_LORS = 5

PSF_XY_MM = 4.87
PSF_Z_MM = 4.45
PSF_FWHM_MM = (PSF_XY_MM, PSF_XY_MM, PSF_Z_MM)

POST_FILTER_FWHM_MM = 6.4

POST_FILTER_Z_RATIO = 4.0

MU_WATER_511 = 0.0093
MU_BONE_511 = 0.0166

CARNEY_B = {80: 0.681, 100: 0.755, 120: 0.837, 140: 1.0}

WCC_UNIT_SCALE = 1e4

K_EXPORT = 63_002.1
K_EXPORT_LM = 124_178.0


def fov_radius_mm(n_tang: int, ndet: int = NDET,
                  r_mm: float = R_MM + DOI_MM) -> float:
    """How far off axis a LOR reaches, in mm."""
    import math

    return r_mm * math.sin(math.pi * (n_tang - 1) / (2 * ndet))


def fov_mask(xy: int, n_tang: int, dr_mm: float = DR_MM):
    """`(xy, xy)` bool, true inside `fov_radius_mm`."""
    import numpy as np

    y, x = np.mgrid[0:xy, 0:xy] - (xy - 1) / 2.0
    return np.hypot(x, y) * dr_mm <= fov_radius_mm(n_tang)


def sirf_grid(acq, xy: int = XY, dr_mm: float = DR_MM, out=print):
    """A uniform image with `dr_mm` transverse voxels, in either SIRF build."""
    x = acq.create_uniform_image(1.0, xy)
    got = float(x.voxel_sizes()[1])
    if abs(got - dr_mm) <= 1e-3:
        return x
    n = max(1, round(xy * got / dr_mm))
    out(f"grid: xy {xy} -> {n}, this SIRF pins the FOV ({xy * got:.2f} mm) "
        f"and would have given {got:.4f} mm voxels")
    x = acq.create_uniform_image(1.0, n)

    got = float(x.voxel_sizes()[1])
    if abs(got - dr_mm) > 1e-3:
        raise SystemExit(
            f"error: cannot reach {dr_mm} mm voxels on this SIRF build -- "
            f"xy {n} gives {got:.6f} mm.\n"
            f"  K is calibrated for {dr_mm} mm and scales with the voxel step, "
            f"so a reconstruction on this grid would be quantitatively wrong.")
    return x
