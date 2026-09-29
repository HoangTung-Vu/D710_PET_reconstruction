from __future__ import annotations

import numpy as np
import parallelproj
import pytomography
import torch
from pytomography.metadata import ObjectMeta
from pytomography.projectors import SystemMatrix
from pytomography.transforms.shared import GaussianFilter

from lm.recon import psf_fwhm
from utils.attn_proj import image_origin
from utils.binmap import BinMap
from utils.geometry import det_pair_map, detector_xy_mm, ring_z_mm
from utils.scanner import DR_MM, NSEG0, PLANE_MM, PSF_FWHM_MM, XY

MAX_RAYS = 8_000_000


def object_meta(xy: int = XY, n_plane: int = NSEG0, dr_mm: float = DR_MM,
                plane_mm: float = PLANE_MM) -> ObjectMeta:
    return ObjectMeta(dr=(dr_mm, dr_mm, plane_mm), shape=(xy, xy, n_plane))


def psf_transforms(psf=PSF_FWHM_MM) -> list:
    f = psf_fwhm(psf)
    return [GaussianFilter(f)] if f else []


class SinogramSystemMatrix(SystemMatrix):
    def __init__(self, binmap: BinMap, sensitivity=None, xy: int = XY,
                 n_plane: int = NSEG0, psf=PSF_FWHM_MM, max_rays: int = MAX_RAYS,
                 device=None, meta: ObjectMeta | None = None):
        self.device = torch.device(device or pytomography.device)
        meta = meta or object_meta(xy, n_plane)
        super().__init__(meta, None, obj2obj_transforms=psf_transforms(psf),
                         proj2proj_transforms=[])
        self.binmap = binmap
        self.n_plane, self.n_view, self.n_tang = binmap.shape
        self.max_rays = int(max_rays)

        d1, d2 = det_pair_map(binmap.n_view, binmap.n_tang, binmap.ndet)
        xy_det = detector_xy_mm(binmap.ndet)
        z = ring_z_mm(binmap.nrings)
        r1, r2, plane = binmap.ring_pairs_by_plane()
        if not np.array_equal(np.bincount(plane, minlength=self.n_plane),
                              binmap.mult):
            raise SystemExit(f"error: {binmap.hdr.path} -- the ring pairs read "
                             f"back out of the bin map do not match its own "
                             f"multiplicity")

        dev = self.device
        self.xy1 = torch.from_numpy(xy_det[d1]).to(dev)
        self.xy2 = torch.from_numpy(xy_det[d2]).to(dev)
        self.z1 = torch.from_numpy(z[r1]).to(dev)
        self.z2 = torch.from_numpy(z[r2]).to(dev)
        self.plane = torch.from_numpy(plane.astype(np.int64)).to(dev)
        self.inv_mult = torch.from_numpy(
            (1.0 / binmap.mult).astype(np.float32))[:, None, None].to(dev)

        self.shape = tuple(int(s) for s in meta.shape)
        self.origin = torch.from_numpy(image_origin(self.shape, meta.dr)).to(dev)
        self.voxel = torch.tensor([float(v) for v in meta.dr],
                                  dtype=torch.float32, device=dev)
        self.sensitivity = (None if sensitivity is None else
                            torch.as_tensor(sensitivity, dtype=torch.float32).to(dev))
        self.subsets = [torch.arange(self.n_view, device=dev)]

    def _views(self, subset_idx):
        if subset_idx is None:
            return torch.arange(self.n_view, device=self.device)
        return self.subsets[subset_idx]

    def _chunks(self, n_views: int):
        per = max(1, self.max_rays // (n_views * self.n_tang))
        n = self.plane.numel()
        return [slice(a, min(a + per, n)) for a in range(0, n, per)]

    def _endpoints(self, rp: slice, views):
        xs = torch.empty((rp.stop - rp.start, views.numel(), self.n_tang, 3),
                         dtype=torch.float32, device=self.device)
        xe = torch.empty_like(xs)
        xs[..., :2] = self.xy1[views][None]
        xe[..., :2] = self.xy2[views][None]
        xs[..., 2] = self.z1[rp][:, None, None]
        xe[..., 2] = self.z2[rp][:, None, None]
        return xs.reshape(-1, 3), xe.reshape(-1, 3)

    def forward(self, object, subset_idx=None):
        x = object.to(self.device)
        for t in self.obj2obj_transforms:
            x = t.forward(x)
        x = x.contiguous()
        views = self._views(subset_idx)
        out = torch.zeros((self.n_plane, views.numel(), self.n_tang),
                          dtype=torch.float32, device=self.device)
        for rp in self._chunks(views.numel()):
            xs, xe = self._endpoints(rp, views)
            p = parallelproj.joseph3d_fwd(xs, xe, x, self.origin, self.voxel)
            out.index_add_(0, self.plane[rp],
                           p.reshape(rp.stop - rp.start, views.numel(), self.n_tang))
        return out * self.inv_mult

    def backward(self, proj, subset_idx=None):
        views = self._views(subset_idx)
        g = (proj.to(self.device) * self.inv_mult).contiguous()
        bp = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        for rp in self._chunks(views.numel()):
            xs, xe = self._endpoints(rp, views)
            bp += parallelproj.joseph3d_back(xs, xe, self.shape, self.origin,
                                             self.voxel,
                                             g[self.plane[rp]].reshape(-1))
        for t in self.obj2obj_transforms[::-1]:
            bp = t.backward(bp)
        return bp

    def set_n_subsets(self, n_subsets: int):
        n = max(1, int(n_subsets))
        self.subsets = [torch.arange(k, self.n_view, n, device=self.device)
                        for k in range(n)]
        return self.subsets

    def get_subset_splits(self, n_subsets: int):
        return self.set_n_subsets(n_subsets)

    def get_projection_subset(self, projections, subset_idx):
        if subset_idx is None:
            return projections
        return projections[:, self.subsets[subset_idx].to(projections.device)]

    def get_weighting_subset(self, subset_idx):
        if subset_idx is None:
            return 1.0
        return self.subsets[subset_idx].numel() / self.n_view

    def compute_normalization_factor(self, subset_idx=None):
        if self.sensitivity is None:
            s = torch.ones((self.n_plane, self._views(subset_idx).numel(),
                            self.n_tang), dtype=torch.float32, device=self.device)
        else:
            s = self.get_projection_subset(self.sensitivity, subset_idx)
        return self.backward(s, subset_idx)
