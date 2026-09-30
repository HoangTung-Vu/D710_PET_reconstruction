from __future__ import annotations

import os

DEVICES = ("auto", "cpu", "cuda")
ENV = "D710_SIM_DEVICE"
FREE_FRACTION = 0.35


def device(name: str | None = None):
    import torch

    name = (name or os.environ.get(ENV) or "auto").lower()
    if name not in DEVICES:
        raise SystemExit(f"error: device {name!r} is not one of {DEVICES}")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: device cuda was asked for, but torch sees no GPU")
    return torch.device(name)


def batch_for(dev, bytes_per_item: float, cpu: int, cap: int) -> int:
    import torch

    if dev.type != "cuda":
        return int(cpu)
    free, _ = torch.cuda.mem_get_info(dev)
    return int(max(cpu, min(cap, FREE_FRACTION * free // max(bytes_per_item, 1.0))))


def is_torch(a) -> bool:
    return type(a).__module__.startswith("torch")
