from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

TINY = 1e-30


def bn(n: int) -> nn.BatchNorm3d:
    return nn.BatchNorm3d(n, track_running_stats=False)


class ResUnit3D(nn.Module):
    def __init__(self, depth: int = 3, kernels: int = 32, kernel_size: int = 3):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        pad = kernel_size // 2
        layers = [nn.Conv3d(1, kernels, kernel_size, padding=pad), bn(kernels),
                  nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv3d(kernels, kernels, kernel_size, padding=pad),
                       bn(kernels), nn.ReLU(inplace=True)]
        layers += [nn.Conv3d(kernels, 1, kernel_size, padding=pad), bn(1)]
        self.dcnn = nn.Sequential(*layers)
        nn.init.zeros_(self.dcnn[-1].weight)

    def forward(self, x):
        return torch.relu(self.dcnn(x) + x)


class FBSEMNet(nn.Module):
    def __init__(self, depth: int = 3, kernels: int = 32):
        super().__init__()
        self.config = {"depth": int(depth), "kernels": int(kernels)}
        self.reg = ResUnit3D(depth, kernels)
        self.gamma = nn.Parameter(torch.rand(1))
        self.register_buffer("u", torch.ones(()))
        self.register_buffer("k", torch.ones(()))

    def regularise(self, x):
        return self.u * self.reg((x / self.u)[None, None])[0, 0]

    def fuse(self, x_em, x_reg, s):
        inv_s = torch.where(s > 0, 1.0 / torch.where(s > 0, s, 1.0), 0.0)
        d = self.gamma * self.k * inv_s
        a = 1.0 - d * x_reg
        b = 4.0 * d * x_em
        r = torch.sqrt((a * a + b).clamp_min(TINY))
        pos = a >= 0
        den = torch.where(pos, a + r, 1.0).clamp_min(TINY)
        d_neg = torch.where(pos, 1.0, d)
        return torch.where(pos, 2.0 * x_em / den, (r - a) / (2.0 * d_neg))

    def state(self, x, x_em, s):
        return self.fuse(x_em, self.regularise(x), s)

    def forward(self, bed, n_it: int):
        x = bed.initial()
        grad = torch.is_grad_enabled()
        for _ in range(n_it):
            for m in range(bed.n_sub):
                x_em = bed.em(x.detach(), m)
                if grad:
                    x = checkpoint(self.state, x, x_em, bed.s[m],
                                   use_reentrant=False)
                else:
                    x = self.state(x, x_em, bed.s[m])
        return x * bed.mask

    @torch.no_grad()
    def clamp_gamma(self):
        g = self.gamma
        if not torch.isfinite(g).all() or float(g) < 0:
            g.fill_(0.01)

    @torch.no_grad()
    def set_units(self, u: float, k: float):
        self.u.fill_(float(u))
        self.k.fill_(float(k))


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def save(path, net: FBSEMNet, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": net.config, "state_dict": net.state_dict(), **extra}, tmp)
    tmp.replace(path)


def load(path, device=None):
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    net = FBSEMNet(**ck["config"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device or "cpu"), ck
