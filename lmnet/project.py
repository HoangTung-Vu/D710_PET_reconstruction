from __future__ import annotations

import torch


def _f32(t):
    return t.detach().to(torch.float32).contiguous()


class LMForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sm, x, subset_idx=None):
        ctx.sm = sm
        ctx.subset_idx = subset_idx
        ctx.dev = x.device
        ctx.dtype = x.dtype
        y = sm.forward(_f32(x), subset_idx)
        return y.to(device=ctx.dev, dtype=ctx.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = ctx.sm.backward(_f32(grad_output), ctx.subset_idx)
        return None, g.to(device=ctx.dev, dtype=ctx.dtype), None


class LMBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sm, y, subset_idx=None):
        ctx.sm = sm
        ctx.subset_idx = subset_idx
        ctx.dev = y.device
        ctx.dtype = y.dtype
        x = sm.backward(_f32(y), ctx.subset_idx)
        return x.to(device=ctx.dev, dtype=ctx.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = ctx.sm.forward(_f32(grad_output), ctx.subset_idx)
        return None, g.to(device=ctx.dev, dtype=ctx.dtype), None


def project(sm, x, subset_idx=None):
    return LMForward.apply(sm, x, subset_idx)


def backproject(sm, y, subset_idx=None):
    return LMBackward.apply(sm, y, subset_idx)
