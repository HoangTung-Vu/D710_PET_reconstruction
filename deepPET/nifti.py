"""NIfTI volumes, and their in-plane resampling onto the DeepPET grid.

`load` mirrors `_load_nifti` of `simulation/phantom.py` (on the `simulation`
branch): any axis-aligned affine is turned into LPS and the array into DICOM
`[slice, row, col]` order. It is repeated here so that `deepPET` stands on
`main` alone; keep the two in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scanner2d import FOV_MM


@dataclass
class Volume:
    """`data[slice, row, col]` in LPS: col along +x (left), row along +y (posterior)."""

    data: np.ndarray
    z: np.ndarray
    x0: float
    y0: float
    pixel_mm: float
    path: str

    @property
    def dz(self) -> float:
        return float(np.diff(self.z).mean()) if len(self.z) > 1 else 0.0


def load(path) -> Volume:
    """An axis-aligned NIfTI as a `Volume` in LPS."""
    import nibabel as nib

    img = nib.load(str(path))
    a = np.asarray(img.get_fdata(dtype=np.float32))
    if a.ndim != 3:
        raise SystemExit(f"error: {path} is {a.ndim}-D, not a 3-D volume")
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ img.affine
    rot = lps[:3, :3]
    if not np.allclose(rot, np.diag(np.diag(rot)), atol=1e-6):
        raise SystemExit(f"error: {path} is oblique; only axis-aligned volumes are supported")
    step = np.diag(rot).astype(np.float64)
    org = lps[:3, 3].astype(np.float64)
    for ax in range(3):
        if step[ax] < 0:
            a = np.flip(a, ax)
            org[ax] += step[ax] * (a.shape[ax] - 1)
            step[ax] = -step[ax]
    if abs(step[0] - step[1]) > 1e-4:
        raise SystemExit(f"error: {path} has non-square pixels {step[:2]}")
    data = np.ascontiguousarray(a.transpose(2, 1, 0), dtype=np.float32)
    z = org[2] + np.arange(data.shape[0]) * step[2]
    return Volume(data, z, float(org[0]), float(org[1]), float(step[0]), str(path))


def to_grid(slices, x0: float, y0: float, pixel_mm: float, grid: int,
            cval: float = 0.0) -> np.ndarray:
    """`(n, rows, cols)` LPS slices onto the `(n, grid, grid)` DeepPET grid.

    The grid spans `FOV_MM` centred on the gantry axis, in the frame of
    `scanner2d`: output pixel `(r, c)` is STIR `(x, y) = (c', r')` with
    `c' = (c - (grid-1)/2) * v`, and STIR y is minus patient y. Downsampling
    is anti-aliased with a Gaussian of sigma `sqrt(ratio^2 - 1) / 2` source
    pixels.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    a = np.asarray(slices, np.float32)
    v = FOV_MM / grid
    ratio = v / pixel_mm
    if ratio > 1.0 + 1e-6:
        s = 0.5 * np.sqrt(ratio * ratio - 1.0)
        a = gaussian_filter(a, sigma=(0.0, s, s), mode="constant", cval=cval)
    c = (np.arange(grid) - (grid - 1) / 2.0) * v
    rows = (-c - y0) / pixel_mm
    cols = (c - x0) / pixel_mm
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    out = np.empty((a.shape[0], grid, grid), np.float32)
    for i in range(a.shape[0]):
        out[i] = map_coordinates(a[i], [rr, cc], order=1, mode="constant", cval=cval)
    return out


def at_z(vol: Volume, z_mm) -> np.ndarray:
    """Slices of `vol` linearly interpolated at patient `z_mm`, clamped to its range."""
    g = np.clip((np.asarray(z_mm, np.float64) - vol.z[0]) / vol.dz, 0.0, len(vol.z) - 1.0)
    i0 = np.floor(g).astype(int)
    i1 = np.minimum(i0 + 1, len(vol.z) - 1)
    w = (g - i0)[:, None, None].astype(np.float32)
    return (1 - w) * vol.data[i0] + w * vol.data[i1]
