from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .project import LMBackward, LMForward

N_PHASE = 8

DUAL_WIDTHS = (64, 64, 16)

PRIMAL_WIDTHS = (64, 128, 256, 64)

DUAL_FEATURES = ("raw", "scaled", "log")


def dual_features(mode: str, q, a, scale):
    if mode == "raw":
        return q, a
    if mode == "scaled":
        return q / scale, a / scale
    if mode == "log":
        return torch.log1p(q / scale), torch.log1p(a / scale)
    raise ValueError(f"dual_feature must be one of {DUAL_FEATURES}, "
                     f"got {mode!r}")


class DualNet(nn.Module):

    def __init__(self, n_in: int = 3, widths=DUAL_WIDTHS):
        super().__init__()
        layers, c = [], n_in
        for w in widths:
            layers += [nn.Linear(c, w), nn.PReLU()]
            c = w
        layers.append(nn.Linear(c, 1))
        self.block = nn.Sequential(*layers)

    def forward(self, h, q, a):
        return h + self.block(torch.cat((h, q, a), dim=1))


class PrimalNet3D(nn.Module):

    def __init__(self, in_ch: int = 3, widths=PRIMAL_WIDTHS,
                 ckpt_block: bool = False):
        super().__init__()
        blocks, c = [], in_ch
        for w in widths:
            blocks.append(nn.Sequential(nn.Conv3d(c, w, 3, padding=1),
                                        nn.BatchNorm3d(w), nn.PReLU()))
            c = w
        blocks.append(nn.Sequential(nn.Conv3d(c, 1, 3, padding=1),
                                    nn.BatchNorm3d(1)))
        self.blocks = nn.ModuleList(blocks)
        self.ckpt_block = ckpt_block

    def forward(self, u, feats):
        x = torch.cat((u, feats), dim=1)
        for b in self.blocks:
            if self.ckpt_block and torch.is_grad_enabled():
                x = checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return u + x


class LMPDNet3D(nn.Module):

    def __init__(self, n_phase: int = N_PHASE, dual_widths=DUAL_WIDTHS,
                 primal_widths=PRIMAL_WIDTHS, ckpt_phase: bool = True,
                 ckpt_block: bool = False, amp: bool = True,
                 dual_feature: str = "scaled"):
        super().__init__()
        if dual_feature not in DUAL_FEATURES:
            raise ValueError(f"dual_feature must be one of {DUAL_FEATURES}, "
                             f"got {dual_feature!r}")
        self.n_phase = n_phase
        self.ckpt_phase = ckpt_phase
        self.amp = amp
        self.dual_feature = dual_feature
        self.dual = nn.ModuleList(
            DualNet(3, dual_widths) for _ in range(n_phase))
        self.primal = nn.ModuleList(
            PrimalNet3D(3, primal_widths, ckpt_block) for _ in range(n_phase))

    def set_checkpointing(self, phase=None, block=None, amp=None):
        if phase is not None:
            self.ckpt_phase = bool(phase)
        if block is not None:
            for p in self.primal:
                p.ckpt_block = bool(block)
        if amp is not None:
            self.amp = bool(amp)
        return self

    def _autocast(self, device):
        if self.amp and device.type == "cuda":
            return torch.autocast("cuda", torch.float16)
        return contextlib.nullcontext()

    def phase(self, k, sm, a, s_n, s_max, kappa, scale, h, u, subset_idx=None):
        q = kappa * LMForward.apply(sm, u[0, 0], subset_idx).unsqueeze(1) + a
        qf, af = dual_features(self.dual_feature, q, a, scale)
        h = self.dual[int(k)](h, qf, af)
        bp = LMBackward.apply(sm, h[:, 0].contiguous(), subset_idx) / s_max
        feats = torch.stack((bp, s_n)).unsqueeze(0)
        with self._autocast(u.device):
            u = self.primal[int(k)](u, feats)
        return h, u.float()

    def forward(self, sm, a, s, kappa):
        a = a.reshape(-1, 1)
        s_max = s.max()
        s_n = s / s_max
        support = (s > 0).to(s.dtype)
        h = torch.zeros_like(a)
        u = torch.zeros((1, 1) + tuple(s.shape), dtype=s.dtype,
                        device=s.device)
        scale = torch.clamp(a.mean(), min=1e-12)
        for k in range(self.n_phase):
            if self.ckpt_phase and torch.is_grad_enabled():
                h, u = checkpoint(self.phase, k, sm, a, s_n, s_max, kappa,
                                  scale, h, u, use_reentrant=False)
            else:
                h, u = self.phase(k, sm, a, s_n, s_max, kappa, scale, h, u)
        return u[0, 0].clamp(min=0) * support


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
