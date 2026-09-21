from __future__ import annotations

import argparse
import json
import sys

GIB = float(2 ** 30)

DUAL_WIDTHS = (64, 64, 16)

PRIMAL_WIDTHS = (64, 128, 256, 64)

PARAM_BYTES_PER_SLOT = 4

ADAM_SLOTS = 4

RUNTIME_OVERHEAD_GIB = 0.5

HEADROOM = 1.35


def primal_saved_channels(in_ch: int, widths, ckpt_block: bool):
    chans = list(widths)
    blocks = []
    c = in_ch
    for w in chans:
        blocks.append((c, w, True))
        c = w
    blocks.append((c, 1, False))

    if ckpt_block:
        carried = sum(c_in for c_in, _c_out, _act in blocks)
        live = max(c_in + 2 * c_out if act else c_in + c_out
                   for c_in, c_out, act in blocks)
        return carried + live

    total = 0
    for c_in, c_out, act in blocks:
        total += c_in + (2 * c_out if act else c_out)
    return total


def dual_saved_floats(n_in: int, widths):
    total = 0
    c = n_in
    for w in widths:
        total += c
        total += w
        c = w
    total += c
    return total


def primal_parameters(in_ch: int, widths):
    total = 0
    c = in_ch
    for w in list(widths) + [1]:
        total += c * w * 27 + w
        total += 2 * w
        if w != 1:
            total += w
        c = w
    return total


def dual_parameters(n_in: int, widths):
    total = 0
    c = n_in
    for w in widths:
        total += c * w + w + w
        c = w
    return total + c * 1 + 1


def estimate(xy: int = 337, n_plane: int = 47, events: int = 9_000_000,
             n_phase: int = 8, primal_widths=PRIMAL_WIDTHS,
             dual_widths=DUAL_WIDTHS, ckpt_phase: bool = True,
             ckpt_block: bool = False, amp: bool = False,
             n_splits: int = 8, optimizer: bool = True,
             overhead_gib: float = RUNTIME_OVERHEAD_GIB):
    v = float(xy) * xy * n_plane
    n = float(events)
    conv_bytes = 2.0 if amp else 4.0

    prim_ch = primal_saved_channels(3, primal_widths, ckpt_block)
    primal = prim_ch * v * conv_bytes

    dual = dual_saved_floats(3, dual_widths) * n * 4.0

    projector = (2.0 * n + v) * 4.0 + (n / max(n_splits, 1)) * 3 * 4.0 * 2

    live_phase = (n + v) * 4.0 * 3

    carry = n_phase * (n + v) * 4.0 if ckpt_phase else 0.0

    n_params = n_phase * (primal_parameters(3, primal_widths)
                          + dual_parameters(3, dual_widths))
    params = n_params * PARAM_BYTES_PER_SLOT * (ADAM_SLOTS if optimizer else 2)

    inputs = (2 * n + 3 * v) * 4.0

    if ckpt_phase:
        activations = primal + dual
    else:
        activations = n_phase * (primal + dual)

    overhead = overhead_gib * GIB
    peak = (activations + carry + projector + live_phase + params + inputs
            + overhead)

    return {"xy": xy, "n_plane": n_plane, "voxels": int(v), "events": int(n),
            "n_phase": n_phase, "ckpt_phase": ckpt_phase,
            "ckpt_block": ckpt_block, "amp": amp,
            "parameters": int(n_params),
            "primal_saved_channels": prim_ch,
            "dual_saved_floats": dual_saved_floats(3, dual_widths),
            "primal_per_phase_GiB": primal / GIB,
            "dual_per_phase_GiB": dual / GIB,
            "activations_GiB": activations / GIB,
            "phase_carry_GiB": carry / GIB,
            "projector_GiB": projector / GIB,
            "live_GiB": live_phase / GIB,
            "params_optimizer_GiB": params / GIB,
            "inputs_GiB": inputs / GIB,
            "overhead_GiB": overhead / GIB,
            "peak_GiB": peak / GIB}


def available_gib(device: str = "cpu") -> float:
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            return 0.0
        i = 0 if ":" not in device else int(device.split(":")[1])
        free, _total = torch.cuda.mem_get_info(i)
        return free / GIB
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    return 0.0


def check(budget_gib: float | None = None, device: str = "cpu",
          headroom: float = HEADROOM, **kw):
    e = estimate(**kw)
    budget = available_gib(device) if budget_gib is None else budget_gib
    need = e["peak_GiB"] * headroom
    e["budget_GiB"] = budget
    e["needed_GiB"] = need
    e["fits"] = bool(budget > 0 and need <= budget)
    return e


def require(budget_gib: float | None = None, device: str = "cpu",
            headroom: float = HEADROOM, force: bool = False, **kw):
    e = check(budget_gib, device, headroom, **kw)
    print(f"  memory estimate: peak {e['peak_GiB']:.1f} GiB "
          f"(x{headroom:g} = {e['needed_GiB']:.1f} GiB), "
          f"available {e['budget_GiB']:.1f} GiB on {device}")
    if not e["fits"] and not force:
        raise SystemExit(
            f"error: this configuration needs about {e['needed_GiB']:.1f} GiB "
            f"and only {e['budget_GiB']:.1f} GiB is available on {device}.\n"
            f"  primal {e['primal_per_phase_GiB']:.1f} GiB per phase, "
            f"dual {e['dual_per_phase_GiB']:.1f} GiB, "
            f"phase carry {e['phase_carry_GiB']:.1f} GiB\n"
            f"  cut it with: --events (dual and carry scale with N), "
            f"--xy / --phases, ckpt_phase, ckpt_block, amp\n"
            f"  re-run with --force to try anyway")
    return e


def measure(xy: int, n_plane: int, events: int, n_phase: int, primal_widths,
            dual_widths, ckpt_phase: bool, ckpt_block: bool, amp: bool,
            device: str = "cpu"):
    import resource

    import torch

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                           .parent.parent))
    from lmnet.model import LMPDNet3D

    dev = torch.device(device)
    v = (xy, xy, n_plane)
    s = torch.rand(v, dtype=torch.float32, device=dev) + 0.1
    a = torch.rand((events, 1), dtype=torch.float32, device=dev) * 1e-3 + 1e-5

    class FakeSM:
        def forward(self, x, subset_idx=None):
            return torch.full((events,), float(x.mean()), device=x.device)

        def backward(self, y, subset_idx=None):
            return torch.full(v, float(y.mean()), device=y.device)

    m = LMPDNet3D(n_phase=n_phase, primal_widths=primal_widths,
                  dual_widths=dual_widths, ckpt_phase=ckpt_phase,
                  ckpt_block=ckpt_block, amp=amp).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = 0.0
    else:
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB

    opt.zero_grad(set_to_none=True)
    m(FakeSM(), a, s, 1.0).pow(2).mean().backward()
    opt.step()

    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / GIB
    else:
        peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB
                - base)
    return peak


VALIDATION = (
    (57, 15, 50_000, 2, (8, 16, 8), (8, 8), False),
    (57, 15, 50_000, 2, (8, 16, 8), (8, 8), True),
    (81, 23, 200_000, 2, (16, 32, 16), (16, 16), False),
    (81, 23, 200_000, 3, (16, 32, 16), (16, 16), True),
)


def run_validation(device: str = "cpu"):
    import subprocess

    print("\nvalidation against a measured step (synthetic projector):")
    print(f"  {'grid':>12} {'events':>9} {'ph':>3} {'ckpt':>5} "
          f"{'predicted':>10} {'measured':>9} {'ratio':>7}")
    for xy, npl, n, ph, pw, dw, ckpt in VALIDATION:
        pred = estimate(xy=xy, n_plane=npl, events=n, n_phase=ph,
                        primal_widths=pw, dual_widths=dw, ckpt_phase=ckpt,
                        ckpt_block=False, amp=False,
                        overhead_gib=0.0)["peak_GiB"]
        cmd = [sys.executable, __file__, "--measure-one",
               "--xy", str(xy), "--n-plane", str(npl), "--events", str(n),
               "--phases", str(ph), "--device", device,
               "--primal-widths", *map(str, pw),
               "--dual-widths", *map(str, dw)]
        if ckpt:
            cmd.append("--ckpt-phase")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {xy}^2x{npl:<4} FAILED: {r.stderr.strip().splitlines()[-1:]}")
            continue
        got = float(r.stdout.strip().splitlines()[-1])
        print(f"  {f'{xy}^2x{npl}':>12} {n:>9,} {ph:>3} {int(ckpt):>5} "
              f"{pred * 1024:>9.0f}M {got * 1024:>8.0f}M "
              f"{got / max(pred, 1e-12):>7.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ram_estimate")
    ap.add_argument("--xy", type=int, default=337)
    ap.add_argument("--n-plane", type=int, default=47)
    ap.add_argument("--events", type=int, nargs="+", default=[9_000_000])
    ap.add_argument("--phases", type=int, default=8)
    ap.add_argument("--primal-widths", type=int, nargs="+",
                    default=list(PRIMAL_WIDTHS))
    ap.add_argument("--dual-widths", type=int, nargs="+",
                    default=list(DUAL_WIDTHS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float)
    ap.add_argument("--headroom", type=float, default=HEADROOM)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--measure-one", action="store_true")
    ap.add_argument("--ckpt-phase", action="store_true")
    ap.add_argument("--ckpt-block", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args(argv)

    if args.measure_one:
        print(measure(args.xy, args.n_plane, args.events[0], args.phases,
                      tuple(args.primal_widths), tuple(args.dual_widths),
                      args.ckpt_phase, args.ckpt_block, args.amp,
                      args.device))
        return 0

    budget = available_gib(args.device) if args.budget is None else args.budget
    print(f"grid {args.xy}x{args.xy}x{args.n_plane} "
          f"({args.xy * args.xy * args.n_plane:,} voxels), "
          f"{args.phases} phases, primal {tuple(args.primal_widths)}, "
          f"dual {tuple(args.dual_widths)}")
    print(f"available on {args.device}: {budget:.1f} GiB "
          f"(headroom x{args.headroom:g})\n")

    rows = []
    hdr = (f"{'events':>12}  {'ckpt':>4} {'blk':>3} {'amp':>3}  "
           f"{'primal/ph':>9} {'dual/ph':>8} {'carry':>7} {'peak':>8}  fits")
    print(hdr)
    print("-" * len(hdr))
    for n in args.events:
        for ckpt_phase in (True, False):
            for ckpt_block in (True, False):
                for amp in (True, False):
                    e = check(budget, args.device, args.headroom,
                              xy=args.xy, n_plane=args.n_plane, events=n,
                              n_phase=args.phases,
                              primal_widths=tuple(args.primal_widths),
                              dual_widths=tuple(args.dual_widths),
                              ckpt_phase=ckpt_phase, ckpt_block=ckpt_block,
                              amp=amp)
                    rows.append(e)
                    print(f"{n:>12,}  {int(ckpt_phase):>4} {int(ckpt_block):>3}"
                          f" {int(amp):>3}  "
                          f"{e['primal_per_phase_GiB']:>9.2f} "
                          f"{e['dual_per_phase_GiB']:>8.2f} "
                          f"{e['phase_carry_GiB']:>7.2f} "
                          f"{e['peak_GiB']:>8.2f}  "
                          f"{'yes' if e['fits'] else 'NO'}")

    if args.validate:
        run_validation(args.device)

    if args.json:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
