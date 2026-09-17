"""CT series to a mu-map on the bed's image grid."""

from __future__ import annotations

import glob
import os

import numpy as np

from .scanner import (CARNEY_B, MU_BONE_511, MU_WATER_511,
                      PLANE_MM)
from .scanner import NSEG0 as PLANES_PER_BED


def hu_to_mu(hu: np.ndarray, kvp: float = 120.0) -> np.ndarray:
    """Carney bilinear conversion from HU to mu at 511 keV, in 1/mm."""
    b = CARNEY_B.get(int(round(kvp)), 0.837)
    hu = np.asarray(hu, dtype=np.float32)
    soft = MU_WATER_511 * (1.0 + hu / 1000.0)
    bone = MU_WATER_511 + hu * (MU_BONE_511 - MU_WATER_511) / (1000.0 * b)
    return np.clip(np.where(hu <= 0, soft, bone), 0.0, None).astype(np.float32)


def to_radiological(arr: np.ndarray) -> np.ndarray:
    """Flip STIR's y axis to DICOM patient y; the operation is its own inverse."""
    return np.flip(np.asarray(arr), axis=1)


class CTAC:
    """One CT series: an HU volume `[slice, row, col]` in DICOM order, with its geometry."""

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
    """A SIRF `ImageData` holding the bed's mu-map, in 1/cm, on `template`'s grid."""
    from scipy.ndimage import map_coordinates

    shape = tuple(int(s) for s in template.shape)
    if shape[0] != PLANES_PER_BED or shape[1] != shape[2]:
        raise SystemExit(f"error: image grid {shape} is not (47, xy, xy)")
    vz, vy, vx = (float(v) for v in template.voxel_sizes())
    if abs(vy - vx) > 1e-3:
        raise SystemExit(f"error: transaxial voxels are not isotropic {vy} × {vx}")

    zc = table_position_mm + np.arange(PLANES_PER_BED) * PLANE_MM
    gz = (zc - ct.z[0]) / ct.dz
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
        print(f"  warning: bed {table_position_mm:.2f} mm overhangs the CT by "
              f"{out_mm:.1f} mm; clamping to the outermost CT slice")
        gz = np.clip(gz, 0.0, len(ct.z) - 1.0)

    xy = shape[1]
    c = (np.arange(xy) - xy // 2) * vy
    g = np.meshgrid(gz, (c - ct.y0) / ct.pixel_mm, (c - ct.x0) / ct.pixel_mm,
                    indexing="ij")
    hu = map_coordinates(ct.hu, [x.ravel() for x in g], order=1,
                         mode="constant", cval=-1000.0).reshape(PLANES_PER_BED, xy, xy)
    mu = hu_to_mu(hu, ct.kvp) * 10.0

    out = template.get_uniform_copy(0)
    out.fill(np.ascontiguousarray(to_radiological(mu), dtype=np.float32))
    return out


def factors(ad, mu_img):
    """`(af, acf)`: the survival probability and its inverse, as `AcquisitionData`."""
    import sirf.STIR as pet

    return pet.AcquisitionSensitivityModel.compute_attenuation_factors(ad, mu_img)
