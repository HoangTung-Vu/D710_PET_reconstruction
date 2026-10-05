from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from utils.paths import out_root
from utils.scanner import N_ITERATIONS, N_SUBSETS, PSF_FWHM_MM, XY

COLUMNS = ["epoch", "step", "case", "bed", "loss", "nrmse", "osem_nrmse", "gamma",
           "seconds", "peak_gib"]
VAL_COLUMNS = ["epoch", "step", "val_loss", "val_nrmse", "osem_nrmse", "gamma",
               "minutes", "best"]
VAL_BED_COLUMNS = ["epoch", "step", "case", "bed", "loss", "nrmse", "osem_nrmse",
                   "seconds"]


def parser():
    ap = argparse.ArgumentParser(prog="fbsem train")
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", help="default $D710_OUT")
    ap.add_argument("--sets", nargs="+", default=["thyr_trainset"])
    ap.add_argument("--sim", default="sim_an_s1")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--val-sets", nargs="*", default=["thyr_testset"],
                    help="evaluated during training; no value disables it")
    ap.add_argument("--val-cases", nargs="+")
    ap.add_argument("--val-limit", type=int, help="only the first N val cases")
    ap.add_argument("--val-every", type=int, default=0, metavar="N",
                    help="also evaluate every N samples (0: after each epoch only)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--beds-per-case", type=int, default=1)
    ap.add_argument("--n-it", type=int, default=N_ITERATIONS)
    ap.add_argument("--n-sub", type=int, default=N_SUBSETS)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--kernels", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--psf", type=float, nargs="+", default=list(PSF_FWHM_MM))
    ap.add_argument("--xy", type=int, default=XY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache", help="default <root>/fbsem/cache")
    ap.add_argument("--max-rays", type=int)
    ap.add_argument("--device")
    return ap


def body_of(lab):
    import torch

    return lab > 0.05 * torch.quantile(lab.flatten(), 0.999)


def errors(x, lab, body) -> tuple[float, float]:
    import torch

    d = (x - lab)[body]
    return (float(torch.mean((x - lab) ** 2)),
            float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(lab[body])))


def osem_nrmse(a, bed, lab, body) -> float:
    import torch

    if a.get("osem") is None:
        return math.nan
    return errors(torch.from_numpy(a["osem"]).to(lab.device) * bed.mask, lab, body)[1]


def val_samples(root, args) -> list:
    from . import data

    if not args.val_sets:
        return []
    cases = data.sim_cases(root, args.val_sets, args.sim, args.val_cases)
    out = [(C, beds[len(beds) // 2]) for C, beds in cases]
    return out[:args.val_limit] if args.val_limit else out


def validate(net, samples, args, cache, dev) -> list:
    import torch

    from . import bed as B
    from . import data

    rows = []
    net.eval()
    with torch.no_grad():
        for a in data.prefetch(samples, args.n_sub, args.psf, cache, xy=args.xy):
            t0 = time.time()
            bed = B.build(a, args.n_sub, args.psf, dev, args.max_rays, args.xy)
            lab = torch.from_numpy(a["label"]).to(dev) * bed.mask
            body = body_of(lab)
            loss, nrmse = errors(net(bed, args.n_it), lab, body)
            osem = osem_nrmse(a, bed, lab, body)
            rows.append({"case": a["case"].root.parent.name, "bed": a["bed"],
                         "loss": loss, "nrmse": nrmse, "osem_nrmse": osem,
                         "seconds": time.time() - t0})
            del bed, lab
    net.train()
    return rows


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    import pytomography
    import torch
    import torch.nn.functional as F

    from . import bed as B
    from . import data
    from . import model as M

    root = out_root(args.root)
    run = root / "fbsem" / "runs" / args.run
    cache = Path(args.cache) if args.cache else root / "fbsem" / "cache"
    dev = torch.device(args.device or pytomography.device)
    cases = data.sim_cases(root, args.sets, args.sim, args.cases)
    if not cases:
        raise SystemExit(f"error: no {args.sets} case under {root} has a complete "
                         f"{args.sim}/ bed with x_true")
    vals = val_samples(root, args)
    n_beds = sum(len(b) for _, b in cases)
    print(f"train {len(cases)} cases, {n_beds} beds; val {len(vals)} beds  ->  {run}")

    torch.manual_seed(args.seed)
    net = M.FBSEMNet(args.depth, args.kernels).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    start, step, units, best = 0, 0, False, math.inf
    last = run / "last.pt"
    if args.resume and last.exists():
        net, ck = M.load(last, dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        opt.load_state_dict(ck["optimizer"])
        start, step, units = ck["epoch"] + 1, ck["step"], True
        best = ck.get("best_val", math.inf)
        print(f"resumed {last} at epoch {start}")
    elif run.exists() and any(run.glob("*.pt")):
        raise SystemExit(f"error: {run} already holds checkpoints; pass --resume "
                         f"or another --run")
    run.mkdir(parents=True, exist_ok=True)

    def table(name, cols):
        p = run / name
        new = not p.exists()
        fh = open(p, "a", newline="")
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        return fh, w

    fh, w = table("log.csv", COLUMNS)
    vfh, vw = table("val.csv", VAL_COLUMNS)
    bfh, bw = table("val_beds.csv", VAL_BED_COLUMNS)

    meta = {"n_it": args.n_it, "n_sub": args.n_sub, "psf": args.psf, "xy": args.xy,
            "sim": args.sim, "args": vars(args),
            "val": [[str(C.root), n] for C, n in vals]}

    def checkpoint(epoch):
        return {**meta, "epoch": epoch, "step": step, "best_val": best,
                "optimizer": opt.state_dict()}

    def run_val(epoch):
        nonlocal best
        if not vals:
            return
        t0 = time.time()
        rows = validate(net, vals, args, cache, dev)
        for r in rows:
            bw.writerow([epoch, step, r["case"], r["bed"], f"{r['loss']:.6g}",
                         f"{r['nrmse']:.5f}", f"{r['osem_nrmse']:.5f}",
                         f"{r['seconds']:.1f}"])
        bfh.flush()
        v_loss = float(np.mean([r["loss"] for r in rows]))
        v_err = float(np.mean([r["nrmse"] for r in rows]))
        o_err = float(np.nanmean([r["osem_nrmse"] for r in rows]))
        is_best = v_err < best
        if is_best:
            best = v_err
            M.save(run / "best.pt", net, **checkpoint(epoch), val_nrmse=v_err)
        vw.writerow([epoch, step, f"{v_loss:.6g}", f"{v_err:.5f}",
                     f"{o_err:.5f}", f"{net.gamma.item():.6g}",
                     f"{(time.time() - t0) / 60:.1f}", int(is_best)])
        vfh.flush()
        print(f"  val @ step {step}: nrmse {v_err:.4f}  (osem {o_err:.4f})  "
              f"loss {v_loss:.4g}{'  best' if is_best else ''}", flush=True)

    done, validated = False, -1
    for epoch in range(start, args.epochs):
        rng = np.random.default_rng([args.seed, epoch])
        samples = data.epoch_samples(cases, args.beds_per_case, rng)
        t_ep, losses, errs = time.time(), [], []
        for a in data.prefetch(samples, args.n_sub, args.psf, cache, xy=args.xy):
            t0 = time.time()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            bed = B.build(a, args.n_sub, args.psf, dev, args.max_rays, args.xy)
            lab = torch.from_numpy(a["label"]).to(dev) * bed.mask
            body = body_of(lab)
            if not units:
                u = float(lab[body].mean())
                s_bar = float((bed.s * bed.mask).sum() / (bed.mask.sum() * bed.n_sub))
                net.set_units(u, s_bar / u)
                meta.update(u=u, s_bar=s_bar, k=s_bar / u, units_from=[
                    str(a["case"].root), a["bed"]])
                units = True
                (run / "args.json").write_text(json.dumps(
                    {**meta, "n_params": M.n_params(net),
                     "cases": [str(C.root) for C, _ in cases]}, indent=1))

            net.train()
            opt.zero_grad(set_to_none=True)
            x = net(bed, args.n_it)
            loss = F.mse_loss(x, lab)
            loss.backward()
            opt.step()
            net.clamp_gamma()
            step += 1
            done = bool(args.max_steps and step >= args.max_steps)

            with torch.no_grad():
                _, nrmse = errors(x, lab, body)
                osem = osem_nrmse(a, bed, lab, body)
            peak = (torch.cuda.max_memory_allocated(dev) / 2**30
                    if dev.type == "cuda" else 0.0)
            losses.append(loss.item())
            errs.append(nrmse)
            w.writerow([epoch, step, a["case"].root.parent.name, a["bed"],
                        f"{losses[-1]:.6g}", f"{nrmse:.5f}", f"{osem:.5f}",
                        f"{net.gamma.item():.6g}", f"{time.time() - t0:.1f}",
                        f"{peak:.2f}"])
            fh.flush()
            del bed, x, loss, lab
            if args.val_every and step % args.val_every == 0 and not done:
                run_val(epoch)
                validated = step
            if done:
                break

        print(f"epoch {epoch}: loss {np.mean(losses):.4g}  nrmse {np.mean(errs):.4f}"
              f"  gamma {net.gamma.item():.4g}  {(time.time() - t_ep) / 60:.1f} min",
              flush=True)
        if validated != step:
            run_val(epoch)
            validated = step
        M.save(run / f"epoch_{epoch:02d}.pt", net, **checkpoint(epoch))
        M.save(last, net, **checkpoint(epoch))
        if done:
            break
    for f in (fh, vfh, bfh):
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
