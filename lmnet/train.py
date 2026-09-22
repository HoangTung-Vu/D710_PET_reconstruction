"""Train LMPDNet3D on real beds thinned on the fly; `prep` builds the per-bed caches.

  python -m lmnet.train prep  --cases fdg26072803 fdg26080604 ...
  python -m lmnet.train train --name trial1 --epochs 10
  python -m lmnet.train train --name trial1 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils.paths import case as get_case
from utils.paths import out_root
from utils.scanner import N_ITERATIONS, N_SUBSETS, N_TOF_RAW, NSEG0, PSF_MM, XY

TRAIN_CASES = ("fdg26072803", "fdg26080604", "fdg26081213", "fdg26081902")

VAL_CASES = ("fdg26081008",)

EV_DIR = "lmnet_ev"

TERMS = ("normdt", "attn", "randoms", "scatter")


def ev_dir(C, bed: int):
    return C.work_bed(bed) / EV_DIR


def stamp(C, bed: int) -> str:
    paths = [C.decoded / f"bed{bed}.lm.npy", C.prompt(bed),
             C.work_bed(bed) / "lm.npz"]
    for t in TERMS:
        paths += [C.work_bed(bed) / f"{t}{x}" for x in (".hs", ".s")]
    out = []
    for p in paths:
        if p.exists():
            st = p.stat()
            out.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(out)


def prepared(C, bed: int):
    p = ev_dir(C, bed) / "meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    if m.get("stamp") != stamp(C, bed):
        return None
    if not all((ev_dir(C, bed) / f).exists() for f in ("ids.npy", "add.npy")):
        return None
    return m


def prep_bed(C, bed: int, rebuild: bool = False):
    from lm import events as ev
    from lm import geom, terms
    from lmnet import sens

    m = None if rebuild else prepared(C, bed)
    if m is not None:
        print(f"  bed {bed}: cached ({m['n_kept']:,} events)")
        return m
    for p in (C.decoded / f"bed{bed}.lm.npy", C.work_bed(bed) / "lm.npz"):
        if not p.exists():
            print(f"  bed {bed}: skipped, no {p}")
            return None

    t0 = time.perf_counter()
    binmap = geom.BinMap(C.prompt(bed))
    e = ev.load(C.decoded / f"bed{bed}.lm.npy")
    keep, _w, add = terms.event_terms(C, bed, e, binmap, N_TOF_RAW)
    ids = ev.detector_ids(e, N_TOF_RAW, 1)[keep]
    if int(ids.max()) >= 2 ** 15:
        raise SystemExit(f"error: a detector id of {C.name} bed {bed} does "
                         f"not fit int16")
    d = ev_dir(C, bed)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "ids.npy", ids.astype(np.int16))
    np.save(d / "add.npy", np.ascontiguousarray(add, np.float32))
    sens.get(C, bed, binmap)

    lab = np.load(C.work_bed(bed) / "lm.npz", allow_pickle=False)["img"]
    m = {"stamp": stamp(C, bed), "n_all": int(len(e)),
         "n_kept": int(keep.sum()), "n_tang": int(binmap.n_tang),
         "tof_bins": N_TOF_RAW, "tof_sign": 1,
         "label_shape": list(lab.shape), "s": time.perf_counter() - t0}
    (d / "meta.json").write_text(json.dumps(m, indent=2))
    print(f"  bed {bed}: {m['n_kept']:,} of {m['n_all']:,} events kept, "
          f"{m['s']:.0f} s")
    return m


class Bed:

    def __init__(self, name, n_all, ids, add, raw, s, label, n_tang,
                 sm_kw=None):
        self.name = name
        self.n_all = int(n_all)
        self.ids = ids
        self.add = add
        self.raw = raw
        self.s = s
        self.sum_s = float(np.asarray(s, np.float64).sum())
        self.label = label
        self.n_tang = int(n_tang)
        self.sm_kw = sm_kw or dict(xy=XY, n_plane=NSEG0, psf=PSF_MM)

    @property
    def n_kept(self) -> int:
        return len(self.add)


def load_bed(C, bed: int) -> Bed:
    from lmnet import sens

    m = prepared(C, bed)
    if m is None:
        raise SystemExit(f"error: {C.name} bed {bed} is not prepared, or has "
                         f"changed since\n  run: python -m lmnet.train prep "
                         f"--cases {C.name}")
    raw = sens.load(C, bed)
    if raw is None:
        raise SystemExit(f"error: no current sens_lm.npz for {C.name} bed "
                         f"{bed}\n  run: python -m lmnet.train prep --cases "
                         f"{C.name}")
    lab = np.load(C.work_bed(bed) / "lm.npz", allow_pickle=False)["img"]
    lab = np.ascontiguousarray(lab.transpose(2, 1, 0), np.float32)
    if lab.shape != raw.shape:
        raise SystemExit(f"error: {C.name} bed {bed}: label {lab.shape} vs "
                         f"sensitivity {raw.shape}")
    d = ev_dir(C, bed)
    return Bed(f"{C.name}/bed{bed}", m["n_all"],
               np.load(d / "ids.npy", mmap_mode="r"),
               np.load(d / "add.npy", mmap_mode="r"), raw,
               sens.image(raw, m["n_tang"]), lab, m["n_tang"])


def case_beds(name: str, only=None):
    C = get_case(name)
    beds = [n for n in C.decoded_beds() if prepared(C, n) is not None]
    if only:
        beds = [n for n in beds if n in only]
    if not beds:
        raise SystemExit(f"error: no prepared bed in {C.root}\n  run: python "
                         f"-m lmnet.train prep --cases {name}")
    return [load_bed(C, n) for n in beds]


def make_sample(bed: Bed, n_target: int, rng, n_splits: int = 8):
    import torch

    from lm import recon

    f = min(1.0, n_target / bed.n_all)
    if f < 1.0:
        sel = np.flatnonzero(rng.random(bed.n_kept, dtype=np.float32) < f)
        ids, add = bed.ids[sel], bed.add[sel]
    else:
        ids, add = bed.ids, bed.add
    ids = np.ascontiguousarray(ids, np.int32)
    n = len(ids)
    kappa = n / bed.sum_s
    sm = recon.build_sm(ids, N_TOF_RAW, n_splits=n_splits,
                        sensitivity=torch.from_numpy(bed.raw), **bed.sm_kw)
    return {"bed": bed, "f": f, "n": n, "kappa": kappa, "sm": sm,
            "ids": ids, "a": np.ascontiguousarray(f * add, np.float32),
            "target": np.ascontiguousarray(bed.label * (f / kappa),
                                           np.float32)}


def to_device(smp, dev):
    import torch

    return (torch.from_numpy(smp["a"]).reshape(-1, 1).to(dev),
            torch.from_numpy(smp["bed"].s).to(dev),
            torch.from_numpy(smp["target"]).to(dev))


def nmse(y, t, sup) -> float:
    return float(((y - t)[sup] ** 2).sum() / (t[sup] ** 2).sum().clamp(min=1e-30))


def osem_baseline(smp, n_it: int, n_sub: int, n_splits: int = 8):
    import torch
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood

    from lm import recon

    bed = smp["bed"]
    sm = recon.build_sm(smp["ids"], N_TOF_RAW, n_splits=n_splits,
                        sensitivity=torch.from_numpy(bed.raw), **bed.sm_kw)
    ll = PoissonLogLikelihood(sm, additive_term=torch.from_numpy(smp["a"]))
    x = OSEM(ll, object_initial=recon._initial(sm, bed.n_tang))(
        n_iters=n_it, n_subsets=n_sub)
    return np.ascontiguousarray(x.cpu().numpy() / smp["kappa"], np.float32)


class _FrozenBN:

    def __init__(self, model):
        import torch.nn as nn

        self.bns = [m for m in model.modules()
                    if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        self.saved = [m.momentum for m in self.bns]
        for m in self.bns:
            m.momentum = 0.0

    def __exit__(self, *exc):
        for m, v in zip(self.bns, self.saved):
            m.momentum = v


def validate(model, vals, dev):
    import torch

    rows, keep = [], None
    with torch.no_grad():
        for i, v in enumerate(vals):
            a, s, t = to_device(v, dev)
            sup = s > 0
            t0 = time.perf_counter()
            model.eval()
            y_ev = model(v["sm"], a, s, v["kappa"])
            dt = time.perf_counter() - t0
            with _FrozenBN(model):
                model.train()
                y_tr = model(v["sm"], a, s, v["kappa"])
            model.eval()
            row = {"bed": v["bed"].name, "events": v["n"],
                   "nmse_eval": nmse(y_ev, t, sup),
                   "nmse_trainbn": nmse(y_tr, t, sup), "s": dt}
            if v.get("osem") is not None:
                row["nmse_osem"] = nmse(torch.from_numpy(v["osem"]).to(dev),
                                        t, sup)
            rows.append(row)
            if i == 0:
                keep = {"pred": y_ev.cpu().numpy().astype(np.float16),
                        "pred_trainbn": y_tr.cpu().numpy().astype(np.float16),
                        "target": v["target"].astype(np.float16),
                        "osem": (v["osem"].astype(np.float16)
                                 if v.get("osem") is not None else None),
                        "bed": v["bed"].name, "events": v["n"]}
    model.train()
    return rows, keep


def lr_lambda(total: int, warmup: int, floor: float = 0.05):
    def f(k):
        if k < warmup:
            return (k + 1) / warmup
        p = min(1.0, (k - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def log_uniform(rng, lo: int, hi: int) -> int:
    if hi <= lo:
        return int(lo)
    return int(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def run(args, train, vals, run_dir, dev):
    import torch

    from lmnet.model import LMPDNet3D, n_parameters

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = LMPDNet3D(n_phase=args.phases, dual_feature=args.dual_feature,
                      ckpt_block=args.block, amp=not args.no_amp).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    per_epoch = len(train) * args.samples_per_bed
    total = args.epochs * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total, args.warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=model.amp
                                  and dev.type == "cuda")

    start, best = 0, float("inf")
    last_p, best_p = run_dir / "last.pt", run_dir / "best.pt"
    if args.resume and last_p.exists():
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed {last_p} at epoch {start}, best val nmse {best:.4f}")
    (run_dir / "config.json").write_text(json.dumps(
        {**vars(args), "train_beds": [b.name for b in train],
         "val": [(v["bed"].name, v["n"]) for v in vals],
         "parameters": n_parameters(model)}, indent=2, default=str))

    def save(p, epoch):
        tmp = p.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "args": vars(args)}, tmp)
        tmp.replace(p)

    def job(epoch, i, b):
        rng = np.random.default_rng([args.seed, epoch, i])
        bed = train[b]
        return make_sample(bed, log_uniform(rng, args.min_events,
                                            args.max_events), rng,
                           args.n_splits)

    print(f"\n{len(train)} train beds, {len(vals)} val samples, "
          f"{n_parameters(model):,} parameters, {per_epoch} steps per epoch, "
          f"epochs {start}..{args.epochs - 1}\n")
    tlog = open(run_dir / "train.jsonl", "a")
    vlog = open(run_dir / "val.jsonl", "a")
    pool = ThreadPoolExecutor(max_workers=1)
    epoch = start
    n_bad = 0
    try:
        for epoch in range(start, args.epochs):
            rng = np.random.default_rng([args.seed, epoch])
            order = [b for _ in range(args.samples_per_bed)
                     for b in rng.permutation(len(train))]
            fut = pool.submit(job, epoch, 0, order[0])
            t_ep = time.perf_counter()
            losses = []
            for i in range(len(order)):
                t0 = time.perf_counter()
                smp = fut.result()
                wait = time.perf_counter() - t0
                if i + 1 < len(order):
                    fut = pool.submit(job, epoch, i + 1, order[i + 1])
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                a, s, t = to_device(smp, dev)
                sup = s > 0
                opt.zero_grad(set_to_none=True)
                out = model(smp["sm"], a, s, smp["kappa"])
                d = (out - t)[sup]
                loss = (d ** 2).mean() if args.loss == "mse" else d.abs().mean()
                if not torch.isfinite(loss):
                    n_bad += 1
                    print(f"  step {i}: non-finite loss, skipped ({n_bad})")
                    if n_bad >= 5:
                        raise SystemExit("error: 5 non-finite losses in a row")
                    continue
                n_bad = 0
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                g_dual = float(torch.sqrt(sum(
                    (p.grad.float() ** 2).sum() for p in model.dual.parameters()
                    if p.grad is not None)))
                g_prim = float(torch.sqrt(sum(
                    (p.grad.float() ** 2).sum()
                    for p in model.primal.parameters() if p.grad is not None)))
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak = (torch.cuda.max_memory_allocated() / 2 ** 30
                        if dev.type == "cuda" else float("nan"))
                row = {"epoch": epoch, "i": i, "bed": smp["bed"].name,
                       "events": smp["n"], "f": smp["f"],
                       "loss": float(loss.detach()),
                       "nmse": nmse(out.detach(), t, sup),
                       "g_dual": g_dual, "g_primal": g_prim,
                       "lr": sched.get_last_lr()[0], "s": dt, "wait_s": wait,
                       "peak_GiB": peak}
                losses.append(row["nmse"])
                tlog.write(json.dumps(row) + "\n")
                tlog.flush()
                print(f"ep {epoch:3d} {i + 1:3d}/{len(order)}  "
                      f"{row['bed']:<18} {smp['n'] / 1e6:5.2f}M  "
                      f"loss {row['loss']:9.4g}  nmse {row['nmse']:6.3f}  "
                      f"|g| d {g_dual:.1e} p {g_prim:.1e}  "
                      f"{dt:5.1f} s (wait {wait:4.1f})  {peak:4.1f} GiB",
                      flush=True)
                del smp, out, loss, d, a, s, t

            print(f"epoch {epoch}: mean train nmse {np.mean(losses):.4f}, "
                  f"{(time.perf_counter() - t_ep) / 60:.1f} min")
            if vals and ((epoch + 1) % args.val_every == 0
                         or epoch + 1 == args.epochs):
                rows, img = validate(model, vals, dev)
                ev = float(np.mean([r["nmse_eval"] for r in rows]))
                tb = float(np.mean([r["nmse_trainbn"] for r in rows]))
                osem = [r["nmse_osem"] for r in rows if "nmse_osem" in r]
                vlog.write(json.dumps({"epoch": epoch, "nmse_eval": ev,
                                       "nmse_trainbn": tb, "rows": rows})
                           + "\n")
                vlog.flush()
                print(f"  val: nmse {ev:.4f} (eval BN)  {tb:.4f} (batch BN)"
                      + (f"  OSEM {np.mean(osem):.4f}" if osem else ""))
                np.savez_compressed(run_dir / "val_last.npz",
                                    **{k: v for k, v in img.items()
                                       if v is not None})
                if ev < best:
                    best = ev
                    save(best_p, epoch)
                    np.savez_compressed(run_dir / "val_best.npz",
                                        **{k: v for k, v in img.items()
                                           if v is not None})
                    print(f"  new best -> {best_p}")
            save(last_p, epoch)
    except KeyboardInterrupt:
        print(f"\ninterrupted in epoch {epoch}; last.pt holds the end of "
              f"epoch {epoch - 1}, --resume restarts epoch {epoch}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        tlog.close()
        vlog.close()
    return best


def val_samples(beds, counts, seed: int, n_splits: int, osem: bool):
    vals = []
    for j, bed in enumerate(beds):
        for k, n in enumerate(counts):
            rng = np.random.default_rng([seed, 10 ** 6, j, k])
            v = make_sample(bed, n, rng, n_splits)
            if osem:
                t0 = time.perf_counter()
                v["osem"] = osem_baseline(v, N_ITERATIONS, N_SUBSETS,
                                          n_splits)
                print(f"  {bed.name} {v['n']:,} events: OSEM "
                      f"{N_ITERATIONS}x{N_SUBSETS} baseline "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
            vals.append(v)
    return vals


def cmd_prep(args) -> int:
    from lmnet import probe

    probe.set_mode(args.mode)
    for name in args.cases:
        C = get_case(name)
        beds = C.decoded_beds()
        print(f"{name}: beds {beds}")
        if not beds:
            print(f"  nothing decoded under {C.root}")
        for n in beds:
            if args.beds and n not in args.beds:
                continue
            prep_bed(C, n, args.rebuild)
    return 0


def cmd_train(args) -> int:
    import torch

    from lmnet import probe
    from tools import ram_estimate

    dev = torch.device(args.device)
    probe.set_mode(args.mode)
    both = set(args.train) & set(args.val)
    if both:
        raise SystemExit(f"error: {sorted(both)} in both --train and --val")

    ram_estimate.require(device=str(dev), force=args.force, xy=XY,
                         n_plane=NSEG0, events=args.max_events,
                         n_phase=args.phases, ckpt_phase=True,
                         ckpt_block=args.block,
                         amp=(not args.no_amp) and dev.type == "cuda",
                         n_splits=args.n_splits)

    t0 = time.perf_counter()
    train = [b for c in args.train for b in case_beds(c, args.beds)]
    val_beds = [b for c in args.val for b in case_beds(c, args.beds)]
    print(f"loaded {len(train)} train + {len(val_beds)} val beds in "
          f"{time.perf_counter() - t0:.0f} s")
    for b in train + val_beds:
        print(f"  {b.name:<18} {b.n_kept:>12,} events")
    vals = val_samples(val_beds, args.val_events, args.seed, args.n_splits,
                       not args.no_osem)

    run_dir = out_root(args.out) / "lmnet" / "runs" / args.name
    best = run(args, train, vals, run_dir, dev)
    print(f"\nbest val nmse {best:.4f}; run in {run_dir}")
    return 0


def main(argv=None) -> int:
    ev_int = lambda x: int(float(x))  # noqa: E731
    ap = argparse.ArgumentParser(prog="lmnet.train")
    ap.add_argument("cmd", choices=("prep", "train"))
    ap.add_argument("--out")
    ap.add_argument("--mode", default="cuda", choices=("cpu", "hybrid", "cuda"))
    ap.add_argument("--beds", type=int, nargs="+",
                    help="use only these bed numbers of every case")
    ap.add_argument("--cases", nargs="+", default=list(TRAIN_CASES + VAL_CASES))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--train", nargs="+", default=list(TRAIN_CASES))
    ap.add_argument("--val", nargs="*", default=list(VAL_CASES))
    ap.add_argument("--name", default="trial")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples-per-bed", type=int, default=1)
    ap.add_argument("--min-events", type=ev_int, default=1_000_000)
    ap.add_argument("--max-events", type=ev_int, default=4_000_000)
    ap.add_argument("--val-events", type=ev_int, nargs="+",
                    default=[2_000_000])
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--no-osem", action="store_true")
    ap.add_argument("--phases", type=int, default=8)
    ap.add_argument("--dual-feature", default="scaled",
                    choices=("raw", "scaled", "log"))
    ap.add_argument("--block", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--loss", default="mse", choices=("mse", "l1"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return {"prep": cmd_prep, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
