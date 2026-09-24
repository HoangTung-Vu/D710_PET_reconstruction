"""Train DeepPET on sinograms simulated on the fly from the prepared slices.

    python -m deepPET.train --name g128 [--grid 128] [--mode paper] [--epochs 100]
    python -m deepPET.train --name g128 --resume
    # smoke test on the laptop:
    python -m deepPET.train --name smoke --limit-studies 3 --epochs 1 --steps 20 \
                            --batch 2 --workers 2 --device cpu

The recipe is the paper's: MSE, SGD with momentum 0.9, lr 0.005 halved every
20 epochs, batch 30, BN momentum 0.2, 100 epochs, and the checkpoint with the
lowest validation loss kept. The loss is taken inside the 350 mm bore only.

Output in `$D710_OUT/deeppet/runs/<name>/`: `args.json`, `log.csv` (one row per
epoch), `last.pt`, `best.pt`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .simulate import COUNT_RANGE, COUNTS_PER_SUV_MM, MODES, SCALES


def run_dir(name: str, out=None) -> Path:
    from utils.paths import out_root

    return (Path(os.path.expanduser(out)) if out else out_root() / "deeppet" / "runs") / name


def pick_device(want: str):
    import torch

    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def memory_estimate(model, batch: int, grid: int) -> str:
    """A lower bound on training memory: weights, grads, momentum, and saved activations."""
    import torch

    x = torch.zeros(1, 1, 288, 371)
    acts = []
    hooks = [m.register_forward_hook(lambda m, i, o: acts.append(o.numel()))
             for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    per_sample = 4 * 2 * sum(acts)
    params = 4 * 3 * model.n_params()
    gib = (params + batch * per_sample) / 2 ** 30
    return (f"{model.n_params() / 1e6:.1f} M params, {model.n_conv()} conv layers; "
            f"~{params / 2 ** 30:.2f} GiB weights+grads+momentum, "
            f"~{per_sample / 2 ** 30:.3f} GiB activations per sample -> "
            f"~{gib:.1f} GiB at batch {batch} (fp32, lower bound)")


def masked_loss(pred, target, mask, kind: str):
    d = (pred - target) * mask
    e = d * d if kind == "mse" else d.abs()
    return e.sum() / (mask.sum() * pred.shape[0])


def evaluate_loader(model, dl, mask, device, loss_kind: str, amp: bool):
    import torch

    model.eval()
    tot, rr, n = 0.0, 0.0, 0
    m = mask.bool()[0, 0]
    with torch.no_grad():
        for x, t, _ in dl:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                p = model(x)
            p = p.float()
            tot += float(masked_loss(p, t, mask, loss_kind)) * x.shape[0]
            for b in range(x.shape[0]):
                g = t[b, 0][m]
                rr += float(torch.sqrt(((p[b, 0][m] - g) ** 2).mean()) / g.mean().clamp_min(1e-6))
            n += x.shape[0]
    return tot / max(n, 1), rr / max(n, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=None, help="default: $D710_OUT/deeppet/data")
    ap.add_argument("--out", default=None, help="runs root; default $D710_OUT/deeppet/runs")
    ap.add_argument("--grid", type=int, default=128, choices=(128, 256))
    ap.add_argument("--mode", default="paper", choices=MODES)
    ap.add_argument("--loss", default="mse", choices=("mse", "l1"))
    ap.add_argument("--scale", default="physical", choices=SCALES,
                    help="physical: calibrated counts per SUV.mm (calib.json); "
                         "counts: prompts per slice drawn from --count-min..--count-max")
    ap.add_argument("--count-scale", type=float, default=1.0,
                    help="physical scale only: multiple of the calibrated counts "
                         "(0.25 = a quarter of the injected dose; randoms scale by its square)")
    ap.add_argument("--count-min", type=float, default=COUNT_RANGE[0])
    ap.add_argument("--count-max", type=float, default=COUNT_RANGE[1])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr-step", type=int, default=20)
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--bn-momentum", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=None, help="cap on steps per epoch")
    ap.add_argument("--val-items", type=int, default=2000, help="fixed val subset size")
    ap.add_argument("--limit-studies", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sim-threads", type=int, default=1, help="OpenMP threads per worker")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch

    from .dataset import SinoDataset, default_data, loader
    from .model import DeepPET
    from .scanner2d import fov_mask

    data = a.data or str(default_data())
    rd = run_dir(a.name, a.out)
    rd.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    amp = bool(a.amp and device.type == "cuda")
    if a.amp and not amp:
        print("note: --amp is a no-op off CUDA; training in fp32")
    torch.manual_seed(a.seed)

    model = DeepPET(a.grid, a.bn_momentum)
    print(memory_estimate(model, a.batch, a.grid))
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum)
    sched = torch.optim.lr_scheduler.StepLR(opt, a.lr_step, a.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best = 0, math.inf
    if a.resume:
        ck = torch.load(rd / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start}, best val {best:.5g}")
    else:
        (rd / "args.json").write_text(json.dumps(vars(a), indent=1))

    cr = (a.count_min, a.count_max)
    kw = {"scale": a.scale, "count_scale": a.count_scale, "count_range": cr,
          "limit_studies": a.limit_studies}
    tr = SinoDataset(data, "train", a.grid, a.mode, train=True, **kw)
    va = SinoDataset(data, "val", a.grid, a.mode, train=False, max_items=a.val_items,
                     seed=a.seed, **kw)
    dl_tr = loader(tr, a.batch, a.workers, True, a.sim_threads, drop_last=len(tr) > a.batch)
    dl_va = loader(va, a.batch, a.workers, False, a.sim_threads)
    mask = torch.from_numpy(fov_mask(a.grid).astype(np.float32))[None, None].to(device)
    n_steps = min(len(dl_tr), a.steps or len(dl_tr))
    print(f"train {len(tr)} slices ({len(tr.store.studies)} studies), val {len(va)}; "
          f"{n_steps} steps/epoch on {device}, grid {a.grid}, mode {a.mode}, "
          + (f"physical scale {a.count_scale:g} x {COUNTS_PER_SUV_MM:.4f} counts per SUV.mm"
             if a.scale == "physical" else f"counts {cr[0]:.0e}..{cr[1]:.0e}")
          + f", loss {a.loss}")

    log = rd / "log.csv"
    if not log.exists():
        log.write_text("epoch,lr,train_loss,val_loss,val_rrmse,seconds\n")
    for ep in range(start, a.epochs):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for step, (x, t, _) in enumerate(dl_tr):
            if step >= n_steps:
                break
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                p = model(x)
            loss = masked_loss(p.float(), t, mask, a.loss)
            if not torch.isfinite(loss):
                raise SystemExit(f"error: loss is {float(loss)} at epoch {ep} step {step}; "
                                 f"lower --lr or check the data")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.shape[0]
            seen += x.shape[0]
            if step % 50 == 0:
                print(f"  epoch {ep} step {step}/{n_steps}  loss {loss.item():.5g}  "
                      f"{(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
        vl, vr = evaluate_loader(model, dl_va, mask, device, a.loss, amp)
        sched.step()
        dt = time.time() - t0
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep, opt.param_groups[0]["lr"], run / max(seen, 1), vl, vr,
                                    round(dt, 1)])
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": ep, "best": min(best, vl),
              "args": vars(a)}
        torch.save(ck, rd / "last.pt")
        if vl < best:
            best = vl
            torch.save(ck, rd / "best.pt")
        print(f"epoch {ep}: train {run / max(seen, 1):.5g}  val {vl:.5g}  "
              f"val rRMSE {vr:.4f}  best {best:.5g}  {dt:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
