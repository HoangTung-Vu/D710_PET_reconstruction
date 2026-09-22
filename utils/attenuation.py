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


def resample_to_bed(vol, z, x0: float, y0: float, pixel_mm: float,
                    table_position_mm: float, xy: int, dr_mm: float,
                    first_plane: int = 0, n_planes: int = PLANES_PER_BED,
                    cval: float = -1000.0, edge_tol_planes: float = 1.5,
                    what: str = "CT", clamp_edges: bool = False) -> np.ndarray:
    """A DICOM-ordered `[slice, row, col]` volume on the bed's grid, `(n_planes, xy, xy)`.

    Plane `p` sits at patient z `table_position_mm + p * PLANE_MM`, for `p` from
    `first_plane`; x and y are centred on the gantry axis. The result is still in
    DICOM row order -- `to_radiological` turns it into the image's. Only the
    bed's own 47 planes are held to `edge_tol_planes`; planes outside them (a
    margin asked for with `first_plane < 0`) take `cval` where the series ends.

    `clamp_edges` also clamps a bed plane lying less than half a slice beyond
    the series. Without it such a plane is sampled outside the input and
    `map_coordinates(mode="constant")` returns `cval`: bed 1 of fdg26081008
    starts exactly on the PET's first slice, float32 rounding in the NIfTI
    affine puts it 2e-5 mm outside, and its plane 0 came out empty. `mu_map`
    keeps the old behaviour.
    """
    from scipy.ndimage import map_coordinates

    z = np.asarray(z, np.float64)
    dz = float(np.diff(z).mean())
    vy = float(dr_mm)
    zc = table_position_mm + np.arange(first_plane, first_plane + n_planes) * PLANE_MM
    gz = (zc - z[0]) / dz
    own = (np.arange(first_plane, first_plane + n_planes) >= 0) & \
        (np.arange(first_plane, first_plane + n_planes) < PLANES_PER_BED)
    zo = zc[own]
    out_mm = max(z[0] - zo.min(), zo.max() - z[-1], 0.0)
    over = max(out_mm - dz / 2, 0.0) / PLANE_MM
    if over > edge_tol_planes:
        raise SystemExit(
            f"error: bed at {table_position_mm:.2f} mm needs {what} z "
            f"{zo[0]:.1f}..{zo[-1]:.1f} mm, the series only covers "
            f"{z[0]:.1f}..{z[-1]:.1f} mm "
            f"(overhang {out_mm:.1f} mm = {over:.2f} planes > tolerance "
            f"{edge_tol_planes})")
    if over > 0:
        print(f"  warning: bed {table_position_mm:.2f} mm overhangs the {what} by "
              f"{out_mm:.1f} mm; clamping to the outermost {what} slice")
    if over > 0 or clamp_edges:
        gz[own] = np.clip(gz[own], 0.0, len(z) - 1.0)

    c = (np.arange(xy) - xy // 2) * vy
    g = np.meshgrid(gz, (c - y0) / pixel_mm, (c - x0) / pixel_mm, indexing="ij")
    return map_coordinates(vol, [x.ravel() for x in g], order=1,
                           mode="constant", cval=cval).reshape(n_planes, xy, xy)


def mu_map(ct: CTAC, table_position_mm: float, xy: int, dr_mm: float,
           edge_tol_planes: float = 1.5) -> np.ndarray:
    """The bed's mu-map in 1/cm, as `(47, xy, xy)` in the image's `(plane, y, x)` order."""
    hu = resample_to_bed(ct.hu, ct.z, ct.x0, ct.y0, ct.pixel_mm,
                         table_position_mm, xy, dr_mm,
                         edge_tol_planes=edge_tol_planes)
    mu = hu_to_mu(hu, ct.kvp) * 10.0
    return np.ascontiguousarray(to_radiological(mu), dtype=np.float32)
