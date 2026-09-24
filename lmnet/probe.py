from __future__ import annotations

import argparse
import json
import time

import numpy as np

from utils.paths import case as get_case
from utils.paths import out_root
from utils.scanner import NDET, NRINGS, NSEG0, N_TOF_RAW, PSF_FWHM_MM, XY

MODES = ("cpu", "hybrid", "cuda")

_NCUDA = {}


def _backend():
    import parallelproj.backend as be

    return be


def _ncuda() -> int:
    if "n" not in _NCUDA:
        _NCUDA["n"] = int(_backend().num_visible_cuda_devices)
    return _NCUDA["n"]


def available_modes() -> list[str]:
    modes = ["cpu"]
    if _ncuda() > 0:
        modes.append("hybrid")
        try:
            import cupy  # noqa: F401

            modes.append("cuda")
        except Exception:
            pass
    return modes


def set_mode(mode: str) -> str:
    import pytomography

    if mode not in MODES:
        raise SystemExit(f"error: unknown mode {mode!r}, expected {MODES}")
    be = _backend()
    be.num_visible_cuda_devices = 0 if mode == "cpu" else _ncuda()
    pytomography.device = "cuda" if mode == "cuda" else "cpu"
    return pytomography.device


def probe_dir(out=None):
    d = out_root(out) / "lmnet" / "probe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(name: str, payload: dict, out=None):
    p = probe_dir(out) / f"{name}.json"
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {p}")
    return p


def timed(fn, n_rep: int = 3, cuda: bool = False):
    import torch

    def sync():
        if cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    out = fn()
    sync()
    best = float("inf")
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best, out


def synthetic_ids(n: int, n_tof: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    ra = rng.integers(0, NRINGS, n)
    rb = rng.integers(0, NRINGS, n)
    da = rng.integers(0, NDET, n)
    spread = NDET // 6
    db = (da + NDET // 2 + rng.integers(-spread, spread + 1, n)) % NDET
    t = rng.integers(0, n_tof, n)
    return np.ascontiguousarray(
        np.stack([ra * NDET + da, rb * NDET + db, t], 1).astype(np.int32))


def unit_image(xy: int, n_plane: int, n_tang: int = 281):
    from utils import scanner

    m = scanner.fov_mask(xy, n_tang)[:, :, None]
    return np.ascontiguousarray(
        np.repeat(m, n_plane, 2).astype(np.float32) * 1e-3)


def cmd_projector(args) -> int:
    import torch

    from lm import recon

    modes = args.mode or available_modes()
    if "cpu" not in modes:
        modes = ["cpu"] + list(modes)
    print(f"cuda devices visible to parallelproj: {_ncuda()}")
    print(f"modes: {modes}\n")

    x = torch.from_numpy(unit_image(args.xy, NSEG0))
    rows, ref = [], {}

    for n in args.events:
        ids = synthetic_ids(n, args.tof_bins)
        for mode in modes:
            dev = set_mode(mode)
            sens = torch.ones((args.xy, args.xy, NSEG0), dtype=torch.float32)
            try:
                sm = recon.build_sm(ids, args.tof_bins, xy=args.xy,
                                    psf=args.psf, n_splits=args.n_splits,
                                    sensitivity=sens, device=dev)
                xf = x.to(dev)
                t_f, y = timed(lambda: sm.forward(xf), args.repeat,
                               mode == "cuda")
                t_b, bp = timed(lambda: sm.backward(y), args.repeat,
                                mode == "cuda")
            except Exception as exc:
                rows.append({"mode": mode, "events": n,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {mode:>6} {n:>10,}: {type(exc).__name__}: {exc}")
                continue

            yn = y.double().cpu().numpy()
            bn = bp.double().cpu().numpy()
            if mode == "cpu":
                ref[n] = (yn, bn)
            d_f = float(np.abs(yn - ref[n][0]).max()
                        / max(np.abs(ref[n][0]).max(), 1e-30))
            d_b = float(np.abs(bn - ref[n][1]).max()
                        / max(np.abs(ref[n][1]).max(), 1e-30))
            row = {"mode": mode, "events": n, "fwd_s": t_f, "back_s": t_b,
                   "fwd_s_per_1e6": t_f * 1e6 / n,
                   "back_s_per_1e6": t_b * 1e6 / n,
                   "rel_fwd_vs_cpu": d_f, "rel_back_vs_cpu": d_b}
            rows.append(row)
            print(f"  {mode:>6} {n:>10,}: fwd {t_f:8.3f} s  back {t_b:8.3f} s"
                  f"   ({t_f * 1e6 / n:6.3f} / {t_b * 1e6 / n:6.3f} s per 1e6)"
                  f"   rel {d_f:.2e} / {d_b:.2e}")

    ok = [r for r in rows if "fwd_s" in r and r["mode"] != "cpu"]
    best = min(ok, key=lambda r: r["fwd_s_per_1e6"] + r["back_s_per_1e6"],
               default=None)
    if best:
        print(f"\nfastest GPU path: {best['mode']}  "
              f"(set pytomography.device = "
              f"{'cuda' if best['mode'] == 'cuda' else 'cpu'})")
    write_json("projector", {"cuda_devices": _ncuda(), "modes": modes,
                             "xy": args.xy, "tof_bins": args.tof_bins,
                             "psf": recon.psf_fwhm(args.psf),
                             "n_splits": args.n_splits,
                             "rows": rows,
                             "recommended": best["mode"] if best else None},
               args.out)
    return 0


def cmd_sens(args) -> int:
    from lm import geom, recon
    from lmnet import sens as cache

    set_mode(args.mode[0] if args.mode else "cpu")
    C = get_case(args.case, args.out)
    binmap = geom.BinMap(C.prompt(args.bed))

    t0 = time.perf_counter()
    a = cache.get(C, args.bed, binmap, xy=args.xy, psf=args.psf, rebuild=True)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = cache.get(C, args.bed, binmap, xy=args.xy, psf=args.psf)
    t_load = time.perf_counter() - t0

    img = cache.image(a, binmap.n_tang)
    payload = {"case": C.name, "bed": args.bed, "xy": args.xy,
               "psf": recon.psf_fwhm(args.psf), "build_s": t_build,
               "load_s": t_load,
               "path": str(cache.path(C, args.bed, args.xy, args.psf)),
               "identical": bool(np.array_equal(a, b)),
               "sentinel_voxels": int((a >= cache.SENTINEL - 1).sum()),
               "support_voxels": int((img > 0).sum()),
               "sum_fov": float(img.sum()), "max": float(img.max())}
    print(f"build {t_build:.1f} s, cached load {t_load:.2f} s, "
          f"identical {payload['identical']}")
    print(f"support {payload['support_voxels']:,} voxels, "
          f"sum {payload['sum_fov']:.6g}, max {payload['max']:.6g}")
    write_json("sens", payload, args.out)
    return 0


def _bed_inputs(args, need_label: bool = False):
    import torch

    from lm import events as ev
    from lm import geom, recon, terms
    from lmnet import sens as cache

    C = get_case(args.case, args.out)
    npy = C.decoded / f"bed{args.bed}.lm.npy"
    if not npy.exists():
        raise SystemExit(f"error: no {npy}\n  run: d710 decode ... --listmode")

    binmap = geom.BinMap(C.prompt(args.bed))
    e = ev.load(npy)
    n_all = len(e)

    tof_scatter = np.load(args.tof_scatter) if args.tof_scatter else None
    if args.tof_bins > 1 and tof_scatter is None:
        tof_scatter, note = terms.scatter_tof_weights(
            C, args.bed, binmap, args.tof_bins, e)
        print(f"  TOF scatter profile from the full event table: {note}")

    f = 1.0 if not args.events else min(1.0, args.events / n_all)
    if f < 1.0:
        rng = np.random.default_rng(args.seed)
        e = np.asarray(e[rng.random(n_all) < f])
    else:
        e = np.asarray(e)

    keep, _w, add = terms.event_terms(C, args.bed, e, binmap, args.tof_bins,
                                      tof_scatter)
    ids = ev.detector_ids(e, args.tof_bins, args.tof_sign)[keep]
    n_kept = int(keep.sum())

    raw = cache.get(C, args.bed, binmap, xy=args.xy, psf=args.psf)
    s_np = cache.image(raw, binmap.n_tang)
    kappa = cache.kappa(n_kept, s_np)

    sm = recon.build_sm(ids, args.tof_bins, xy=args.xy, psf=args.psf,
                        n_splits=args.n_splits,
                        sensitivity=torch.from_numpy(raw))

    label = None
    if need_label:
        p = C.work_bed(args.bed) / "lm.npz"
        if not p.exists():
            raise SystemExit(f"error: no {p}\n  run: d710 lm recon --case "
                             f"{C.name} --beds {args.bed}")
        img = np.load(p, allow_pickle=False)["img"]
        label = np.ascontiguousarray(img.transpose(2, 1, 0), np.float32)
        if label.shape != (args.xy, args.xy, NSEG0):
            raise SystemExit(
                f"error: {p} holds a {label.shape} label but this run is on "
                f"{(args.xy, args.xy, NSEG0)}.\n"
                f"  reconstruct that bed at --xy {args.xy}, or pass "
                f"--no-label (step only measures, it does not need one)")

    print(f"case {C.name!r} bed {args.bed}: {n_all:,} events, kept "
          f"{n_kept:,} (f = {f:.4g}), kappa = {kappa:.6g}")
    return {"case": C, "sm": sm, "a": (f * add).astype(np.float32),
            "s": s_np, "kappa": kappa, "f": f, "n_kept": n_kept,
            "n_all": n_all, "label": label, "binmap": binmap}


def cmd_scales(args) -> int:
    import torch

    from lmnet.project import LMForward

    dev = set_mode(args.mode[0] if args.mode else "cpu")
    d = _bed_inputs(args, need_label=True)

    s = torch.from_numpy(d["s"])
    a = torch.from_numpy(d["a"])
    kappa, f = d["kappa"], d["f"]

    u = torch.from_numpy(
        np.ascontiguousarray(d["label"] * f / kappa, np.float32))

    q = kappa * LMForward.apply(d["sm"], u.to(dev)) + a.to(dev)
    bp = d["sm"].backward(1.0 / torch.clamp(q, min=1e-12))

    s_max = float(s.max())
    s_n = (s / s_max).numpy()
    bp_n = (bp.cpu() / s_max).numpy()
    fov = d["s"] > 0

    pct = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]

    def row(name, v):
        v = np.asarray(v, np.float64).ravel()
        out = {"name": name, "mean": float(v.mean()), "min": float(v.min()),
               "max": float(v.max()),
               "pct": {str(p): float(np.percentile(v, p)) for p in pct}}
        print(f"  {name:>14}: " + "  ".join(
            f"p{p}={out['pct'][str(p)]:.4g}" for p in (1, 50, 99)) +
            f"   min={out['min']:.4g} max={out['max']:.4g}")
        return out

    print("\npercentiles:")
    rows = [row("q", q.cpu().numpy()), row("a", d["a"]),
            row("a/q", (d["a"] / np.maximum(q.cpu().numpy(), 1e-12))),
            row("kappa*Pu", (q.cpu().numpy() - d["a"])),
            row("s_n (FOV)", s_n[fov]), row("bp/s_max (FOV)", bp_n[fov]),
            row("bp/s (FOV)", bp_n[fov] / np.maximum(s_n[fov], 1e-12)),
            row("u (FOV)", u.numpy()[fov])]

    payload = {"case": d["case"].name, "bed": args.bed, "kappa": kappa,
               "f": f, "n_kept": d["n_kept"], "n_all": d["n_all"],
               "s_max": s_max, "tof_bins": args.tof_bins, "xy": args.xy,
               "rows": rows}
    write_json("scales", payload, args.out)
    return 0


def _combos(which):
    all_of = [(p, b, m) for p in (True, False) for b in (True, False)
              for m in (True, False)]
    if which == "all":
        return all_of
    return [(True, False, True), (True, False, False), (True, True, True),
            (False, False, True)]


def cmd_step(args) -> int:
    import torch

    from lmnet.model import LMPDNet3D, n_parameters

    from tools import ram_estimate

    dev_pt = set_mode(args.mode[0] if args.mode else "cpu")
    device = torch.device(args.device)

    combos = _combos(args.combos)
    budget = ram_estimate.available_gib(str(device))
    print(f"grid {args.xy}x{args.xy}x{NSEG0}, up to {args.events:,} events, "
          f"{args.phases} phases; {budget:.1f} GiB available on {device}")
    fits = []
    for ckpt_phase, ckpt_block, amp in combos:
        e = ram_estimate.check(budget, str(device), args.headroom,
                               xy=args.xy, n_plane=NSEG0, events=args.events,
                               n_phase=args.phases, ckpt_phase=ckpt_phase,
                               ckpt_block=ckpt_block,
                               amp=amp and device.type == "cuda",
                               n_splits=args.n_splits)
        ok = e["fits"] or args.force
        print(f"  phase={int(ckpt_phase)} block={int(ckpt_block)} "
              f"amp={int(amp)}: peak {e['peak_GiB']:6.1f} GiB "
              f"(need {e['needed_GiB']:6.1f})  {'ok' if ok else 'SKIP'}")
        if ok:
            fits.append((ckpt_phase, ckpt_block, amp, e))
    if not fits:
        raise SystemExit(
            f"error: no configuration fits in {budget:.1f} GiB on {device}.\n"
            f"  the primal alone needs "
            f"{ram_estimate.estimate(xy=args.xy, n_plane=NSEG0, events=1, n_phase=1)['primal_per_phase_GiB']:.1f}"
            f" GiB per phase in fp32 at this grid\n"
            f"  cut --events, --xy or --phases, or run it on the workstation; "
            f"--force overrides")
    if device.type != "cuda":
        print("  note: amp is a no-op off CUDA, so those rows are fp32")

    d = _bed_inputs(args, need_label=not args.no_label)

    s = torch.from_numpy(d["s"]).to(device)
    a = torch.from_numpy(d["a"]).reshape(-1, 1).to(device)
    label = (torch.from_numpy(
        np.ascontiguousarray(d["label"] * d["f"] / d["kappa"], np.float32)
    ).to(device) if d["label"] is not None else None)
    support = (s > 0)

    model = LMPDNet3D(n_phase=args.phases,
                      dual_feature=args.dual_feature).to(device)
    print(f"\nLMPDNet3D: {args.phases} phases, "
          f"{n_parameters(model):,} parameters, pytomography.device={dev_pt}, "
          f"model on {device}, dual_feature={args.dual_feature}\n")

    import copy
    import resource

    init_state = copy.deepcopy(model.state_dict())

    rows = []
    for ckpt_phase, ckpt_block, amp, est in fits:
        name = (f"phase={int(ckpt_phase)} block={int(ckpt_block)} "
                f"amp={int(amp)}")
        model.load_state_dict(init_state)
        model.set_checkpointing(phase=ckpt_phase, block=ckpt_block, amp=amp)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp
                                      and device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            out = model(d["sm"], a, s, d["kappa"])
            tgt = label if label is not None else torch.zeros_like(out)
            loss = ((out - tgt)[support] ** 2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                kind = "cuda"
            else:
                peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        * 1024 / 2 ** 30)
                kind = "rss-highwater"
            row = {"ckpt_phase": ckpt_phase, "ckpt_block": ckpt_block,
                   "amp": amp, "step_s": dt, "peak_GiB": peak,
                   "peak_kind": kind, "estimated_GiB": est["peak_GiB"],
                   "loss": float(loss.detach())}
            print(f"  {name}: {dt:8.2f} s   peak {peak:7.2f} GiB ({kind}, "
                  f"estimated {est['peak_GiB']:.1f})   "
                  f"loss {float(loss.detach()):.6g}")
        except Exception as exc:
            row = {"ckpt_phase": ckpt_phase, "ckpt_block": ckpt_block,
                   "amp": amp, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: {type(exc).__name__}: {exc}")
        rows.append(row)
        del opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json("step", {"case": d["case"].name, "bed": args.bed,
                        "events": d["n_kept"], "phases": args.phases,
                        "parameters": n_parameters(model), "xy": args.xy,
                        "tof_bins": args.tof_bins, "device": str(device),
                        "pytomography_device": dev_pt, "rows": rows},
               args.out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lmnet.probe")
    ap.add_argument("cmd", choices=("projector", "sens", "scales", "step"))
    ap.add_argument("--case")
    ap.add_argument("--out")
    ap.add_argument("--bed", type=int, default=1)
    ap.add_argument("--events", type=int, nargs="+",
                    default=[1_000_000, 9_000_000])
    ap.add_argument("--mode", nargs="+", choices=MODES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--xy", type=int, default=XY)
    ap.add_argument("--psf", type=float, nargs="+", default=list(PSF_FWHM_MM),
                    metavar="MM",
                    help="XY [Z] mm FWHM; one value = isotropic; 0 disables")
    ap.add_argument("--tof-bins", type=int, default=N_TOF_RAW)
    ap.add_argument("--tof-sign", type=int, choices=(1, -1), default=1)
    ap.add_argument("--tof-scatter", metavar="PROF.npy")
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--phases", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--combos", choices=("default", "all"), default="default")
    ap.add_argument("--dual-feature", default="scaled",
                    choices=("raw", "scaled", "log"))
    ap.add_argument("--headroom", type=float, default=1.35)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd != "projector":
        if not args.case:
            raise SystemExit(f"error: {args.cmd} needs --case")
        args.events = args.events[0] if args.events else 0

    return {"projector": cmd_projector, "sens": cmd_sens,
            "scales": cmd_scales, "step": cmd_step}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
