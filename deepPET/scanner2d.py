"""The D710 as a 2D scanner: one direct plane, 288 views x 371 tangential bins.

The LORs are the real sinogram's, from `utils.geometry.det_pair_map` and
`detector_xy_mm` -- the STIR frame that `d710 lm check` proves bit-exact -- so a
sinogram projected here lines up bin for bin with plane `k` of a real
`bed<n>.s` rebinned to 2D.

The tangential axis is cropped to the bins whose LOR passes within
`FOV_RADIUS_MM` of the axis: 371 of 381 (5..375). DeepPET cropped for the same
reason (the bins outside the image circle can hold no trues); its 269 bins do
not carry over, they would cut this geometry's FOV to r = 276 mm.

Images are `(y, x)` arrays in that frame, pixel `i` centred at
`(i - (n - 1) / 2) * voxel`. Row 0 is the most negative STIR y, which is the
most positive patient (LPS) y: `nifti.to_grid` does that flip, as
`utils.attenuation.to_radiological` does for the reconstruction.
"""

from __future__ import annotations

import numpy as np

from utils.geometry import det_pair_map, detector_xy_mm
from utils.scanner import NDET

N_VIEW, N_TANG_RAW = 288, 381

FOV_RADIUS_MM = 350.0

FOV_MM = 2 * FOV_RADIUS_MM

GRIDS = (128, 256)


def tangential_distance_mm() -> np.ndarray:
    """Signed distance from the axis of each of the 381 tangential bins (view 0)."""
    d1, d2 = det_pair_map(N_VIEW, N_TANG_RAW, NDET)
    xy = detector_xy_mm().astype(np.float64)
    a, b = xy[d1[0]], xy[d2[0]]
    return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.linalg.norm(b - a, axis=1)


def tangential_crop(radius_mm: float = FOV_RADIUS_MM) -> slice:
    """The contiguous tangential bins with |s| <= `radius_mm`."""
    keep = np.nonzero(np.abs(tangential_distance_mm()) <= radius_mm + 1e-6)[0]
    return slice(int(keep[0]), int(keep[-1]) + 1)


TANG = tangential_crop()

N_TANG = TANG.stop - TANG.start


def fov_mask(grid: int, radius_mm: float = FOV_RADIUS_MM) -> np.ndarray:
    """`(grid, grid)` bool, true for pixel centres inside the bore."""
    v = FOV_MM / grid
    c = (np.arange(grid) - (grid - 1) / 2.0) * v
    return np.hypot(c[:, None], c[None, :]) <= radius_mm


class Scanner2D:
    """Joseph forward and back projection of one plane with the D710's LORs."""

    def __init__(self, grid: int = 128):
        d1, d2 = det_pair_map(N_VIEW, N_TANG_RAW, NDET)
        d1, d2 = d1[:, TANG], d2[:, TANG]
        xy = detector_xy_mm()
        self.xs = np.zeros((N_VIEW, N_TANG, 3), np.float32)
        self.xe = np.zeros((N_VIEW, N_TANG, 3), np.float32)
        self.xs[..., :2] = xy[d1]
        self.xe[..., :2] = xy[d2]
        self.grid = int(grid)
        self.voxel_mm = FOV_MM / self.grid
        o = -(self.grid - 1) / 2.0 * self.voxel_mm
        self.origin = np.array([o, o, 0.0], np.float32)
        self.voxsize = np.full(3, self.voxel_mm, np.float32)
        self.shape = (N_VIEW, N_TANG)

    def _lors(self, views):
        if views is None:
            return self.xs.reshape(-1, 3), self.xe.reshape(-1, 3), N_VIEW
        return (np.ascontiguousarray(self.xs[views].reshape(-1, 3)),
                np.ascontiguousarray(self.xe[views].reshape(-1, 3)), len(views))

    def fwd(self, img_yx, views=None) -> np.ndarray:
        """`(n_views, N_TANG)` line integrals of a `(grid, grid)` image, in image units x mm."""
        import parallelproj

        xs, xe, nv = self._lors(views)
        img = np.ascontiguousarray(np.asarray(img_yx, np.float32).T[:, :, None])
        p = parallelproj.joseph3d_fwd(xs, xe, img, self.origin, self.voxsize)
        return np.asarray(p, np.float32).reshape(nv, N_TANG)

    def back(self, sino, views=None) -> np.ndarray:
        """The adjoint of `fwd`: a `(grid, grid)` image from `(n_views, N_TANG)`."""
        import parallelproj

        xs, xe, nv = self._lors(views)
        s = np.ascontiguousarray(np.asarray(sino, np.float32).reshape(-1))
        b = parallelproj.joseph3d_back(xs, xe, (self.grid, self.grid, 1),
                                       self.origin, self.voxsize, s)
        return np.ascontiguousarray(np.asarray(b, np.float32)[:, :, 0].T)
