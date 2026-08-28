"""CT -> mu-map on the bed's image grid. This is scatter's **input**, not its output.

Scatter needs to know where the material is; this module builds exactly that and
no more — it does not perform the attenuation correction for the reconstruction.
"""

from __future__ import annotations

import glob
import os

import numpy as np

#: PET plane spacing, and the number of planes in one bed (= 2·num_rings − 1).
PLANE_MM = 3.2699997
PLANES_PER_BED = 47

# Carney bilinear HU -> mu(511 keV), 1/mm.
MU_WATER_511 = 0.0096
MU_BONE_511 = 0.0172
CARNEY_B = {80: 0.681, 100: 0.755, 120: 0.837, 140: 1.0}


def hu_to_mu(hu: np.ndarray, kvp: float = 120.0) -> np.ndarray:
    """Carney bilinear HU -> mu(511 keV), 1/mm."""
    b = CARNEY_B.get(int(round(kvp)), 0.837)
    hu = np.asarray(hu, dtype=np.float32)
    soft = MU_WATER_511 * (1.0 + hu / 1000.0)
    bone = MU_WATER_511 + hu * (MU_BONE_511 - MU_WATER_511) / (1000.0 * b)
    return np.clip(np.where(hu <= 0, soft, bone), 0.0, None).astype(np.float32)


def to_radiological(arr: np.ndarray) -> np.ndarray:
    """Flip STIR's y axis to DICOM patient y; it is its own inverse."""
    return np.flip(np.asarray(arr), axis=1)


class CTAC:
    """One CT series: an HU volume ``[slice, row, col]`` in DICOM order, plus its geometry."""

    def __init__(self, hu, z, x0, y0, pixel_mm, kvp, meta):
        self.hu, self.z, self.x0, self.y0 = hu, z, x0, y0
        self.pixel_mm, self.kvp, self.meta = pixel_mm, kvp, meta

    @property
    def dz(self) -> float:
        return float(np.diff(self.z).mean())

    def describe(self) -> str:
        return (f"CT {self.meta['series_description']}: {self.hu.shape[0]} slice "
                f"{self.hu.shape[1]}x{self.hu.shape[2]} @ {self.pixel_mm:.4f} mm, "
                f"{self.kvp:.0f} kVp, z {self.z[0]:.2f}..{self.z[-1]:.2f} "
                f"step {self.dz:.4f} mm")


def load(path: str) -> CTAC:
    """Read one CT series directory."""
    import pydicom

    ds = []
    for f in sorted(glob.glob(os.path.join(path, "*"))):
        if not os.path.isfile(f):
            continue
        try:
            d = pydicom.dcmread(f)
        except Exception:
            continue
        if getattr(d, "Modality", None) == "CT":
            ds.append(d)
    if not ds:
        raise SystemExit(f"error: no CT instance under {path}")
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    iop = [float(v) for v in ds[0].ImageOrientationPatient]
    if not np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-6):
        raise SystemExit(f"error: {path} is tilted (IOP {iop}); resample it first")

    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    step = np.round(np.diff(z), 3)
    # An export missing slices would be interpolated straight across the gap into
    # a mu-map that is wrong but looks plausible.  Diagnose it; do not average.
    if len(step) and step.std() > 0.05:
        modal = float(np.bincount((step * 100).astype(int)).argmax()) / 100
        gaps = step[np.abs(step - modal) > 0.01]
        raise SystemExit(
            f"error: {path} is an incomplete export, not an evenly sparse series.\n"
            f"  {len(ds)} slices, step {modal:.2f} mm over {len(step) - len(gaps)}/"
            f"{len(step)} intervals, {len(gaps)} gaps up to {gaps.max():.1f} mm.\n"
            f"  One bed needs {PLANES_PER_BED * PLANE_MM:.1f} mm of continuous coverage.")

    hu = np.stack([d.pixel_array * float(getattr(d, "RescaleSlope", 1))
                   + float(getattr(d, "RescaleIntercept", 0)) for d in ds])
    return CTAC(hu=hu.astype(np.float32), z=z,
                x0=float(ds[0].ImagePositionPatient[0]),
                y0=float(ds[0].ImagePositionPatient[1]),
                pixel_mm=float(ds[0].PixelSpacing[0]),
                kvp=float(getattr(ds[0], "KVP", 120.0) or 120.0),
                meta={"path": path,
                      "series_description": str(getattr(ds[0], "SeriesDescription", "?")),
                      "frame_of_reference_uid":
                          str(getattr(ds[0], "FrameOfReferenceUID", "")),
                      "num_slices": len(ds)})


def mu_image(ct: CTAC, table_position_mm: float, template, edge_tol_planes: float = 1.5):
    """A SIRF ``ImageData`` holding the bed's mu-map, 1/cm, on ``template``'s grid.

    ``edge_tol_planes`` lets a bed hang a few planes off the end of the CT. The
    first and last bed of a case almost always overhang by a few mm — the
    paediatric case's bed 1 sits at -767.7 mm while the CT only starts at
    -765.4 mm — and rejecting the whole bed over those 2 mm loses a bed entirely.
    The overhang is **clamped to the outermost CT slice** (repeating it) rather
    than filled with air: there is still body there, and treating it as air would
    under-correct attenuation. Beyond ``edge_tol_planes`` it still raises.
    """
    from scipy.ndimage import map_coordinates

    shape = tuple(int(s) for s in template.shape)
    if shape[0] != PLANES_PER_BED or shape[1] != shape[2]:
        raise SystemExit(f"error: image grid {shape} is not (47, xy, xy)")
    vz, vy, vx = (float(v) for v in template.voxel_sizes())
    if abs(vy - vx) > 1e-3:
        raise SystemExit(f"error: transaxial voxels are not isotropic {vy} × {vx}")

    # Axially: PET plane p sits at table_position + p·PLANE_MM.
    zc = table_position_mm + np.arange(PLANES_PER_BED) * PLANE_MM
    gz = (zc - ct.z[0]) / ct.dz
    # The overhang past the outermost CT slice, measured in mm and only then
    # converted to PET PLANES. The first half CT slice still interpolates, so it
    # does not count. The tolerance must be a DISTANCE, not a slice count:
    # measured in slice indices, a 1.25 mm CT would be held 2.6× tighter than a
    # 3.27 mm CT for the same bed.
    out_mm = max(ct.z[0] - zc.min(), zc.max() - ct.z[-1], 0.0)
    over = max(out_mm - ct.dz / 2, 0.0) / PLANE_MM
    if over > edge_tol_planes:
        raise SystemExit(
            f"error: bed at {table_position_mm:.2f} mm needs CT z "
            f"{zc[0]:.1f}..{zc[-1]:.1f} mm, the series only covers "
            f"{ct.z[0]:.1f}..{ct.z[-1]:.1f} mm "
            f"(overhang {out_mm:.1f} mm = {over:.2f} planes > tolerance "
            f"{edge_tol_planes})")
    if over > 0:
        # Report the ACTUAL overhang (mm), not the amount over tolerance.
        print(f"  warning: bed {table_position_mm:.2f} mm overhangs the CT by "
              f"{out_mm:.1f} mm; clamping to the outermost CT slice")
        gz = np.clip(gz, 0.0, len(ct.z) - 1.0)

    # Transversely: the PET grid is centred at index xy//2, the scanner axis at DICOM (x, y) = (0, 0).
    xy = shape[1]
    c = (np.arange(xy) - xy // 2) * vy
    g = np.meshgrid(gz, (c - ct.y0) / ct.pixel_mm, (c - ct.x0) / ct.pixel_mm,
                    indexing="ij")
    hu = map_coordinates(ct.hu, [x.ravel() for x in g], order=1,
                         mode="constant", cval=-1000.0).reshape(PLANES_PER_BED, xy, xy)
    mu = hu_to_mu(hu, ct.kvp) * 10.0                   # 1/mm -> 1/cm for STIR

    out = template.get_uniform_copy(0)
    out.fill(np.ascontiguousarray(to_radiological(mu), dtype=np.float32))
    return out


def factors(ad, mu_img):
    """``(af, acf)`` — the survival probability and its inverse, as AcquisitionData."""
    import sirf.STIR as pet

    return pet.AcquisitionSensitivityModel.compute_attenuation_factors(ad, mu_img)
