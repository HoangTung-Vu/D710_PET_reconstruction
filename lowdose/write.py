"""Writing a thinned exam as a complete, ordinary case."""

from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

from utils import terms as sino_terms

from . import thin

COPY = ("normdt", "norm_only", "attn")

COPY_FILES = ("scatter_tof.npy",)

TOF_PROFILE_BINS = 55

SCALED = ("randoms", "scatter", "background")

SINOGRAM = ("derived", "binomial")


def link_into_place(target, link) -> None:
    """Point `link` at `target` as a relative symlink, replacing whatever is there."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def clone_to_stir(src_dir, dst_dir, prompts: int, f: float, randoms_power: int) -> None:
    """Write a `to_stir.json` describing this case rather than the one it was thinned from."""
    p = src_dir / "to_stir.json"
    if not p.exists():
        return
    m = json.loads(p.read_text())
    m["verified"] = {**m.get("verified", {}), "prompts": prompts}
    m.pop("stats", None)
    m["lowdose"] = {"dose_fraction": f, "randoms_power": randoms_power,
                    "scaled": {"randoms": f ** randoms_power, "scatter": f}}
    (dst_dir / "to_stir.json").write_text(json.dumps(m, indent=2, sort_keys=True))


def clone_header(src_hs, dst_hs, data_name: str) -> None:
    hdr = re.sub(r"(?im)^(\s*name of data file\s*:=).*$", r"\1 " + data_name,
                 src_hs.read_text())
    dst_hs.write_text(hdr)


def scale_term(src_dir, dst_dir, stem: str, factor: float) -> float:
    """Scale the `stem.{hs,s}` pair by `factor`."""
    a = np.fromfile(src_dir / f"{stem}.s", "<f4") * np.float32(factor)
    a.tofile(dst_dir / f"{stem}.s")
    clone_header(src_dir / f"{stem}.hs", dst_dir / f"{stem}.hs", f"{stem}.s")
    return float(a.sum(dtype=np.float64))


def readme(dst, src_name: str, f: float, mode: str, seed: int, sinogram: str,
           rho_axis: str = "tangential", tof_rho: str = "model",
           window: str = "uniform") -> None:
    """Write a note in `raw_simulation/` recording what these files are."""
    power = thin.randoms_power(mode)
    dst.raw_sim.mkdir(parents=True, exist_ok=True)
    (dst.raw_sim / "README.txt").write_text(f"""\
Simulated low-count / low-dose raw data -- written by `d710 lowdose`.
NOT measured data. Thinned from case {src_name!r} at dose fraction f = {f:g}
(DRF {1 / f:g}), mode {thin.canonical(mode)!r}, window {window!r}, seed {seed},
sinogram {sinogram!r}\
{f", rho per {rho_axis}, TOF rho {tof_rho}" if thin.canonical(mode) == "low-dose" else ""}.\
{chr(10) + "The window is the FIRST " + format(f, "g") + " of each frame, so frame_duration_ms"
 + chr(10) + "is shorter and randoms/scatter carry decay-weighted factors, not f."
 if window == "time" else ""}

  bed<n>.hs / bed<n>.s   thinned prompt sinogram
  bed<n>.lm.npy          thinned event table (the same draw, event for event)
  bed<n>/randoms.{{hs,s}}  source randoms x f^{power}
  bed<n>/scatter.{{hs,s}}  source scatter x f
  bed<n>/background.{{hs,s}}  randoms + scatter, rebuilt rather than scaled

Everything here is symlinked into ../decoded/ and ../work/bed<n>/ under its
usual name, so the case reconstructs like any other. Terms NOT listed above
(normdt, norm_only, attn) are copies of the source: a sensitivity does not
depend on the dose. So is ../work/bed<n>/{sino_terms.TOF_PROFILE_FILE}, which is the
scatter's TOF shape measured on the FULL-COUNT source -- these thinned tails are
too sparse to measure it from. `../lowdose.json` carries the numbers, and export
applies K x {1 / f:g} from it on its own.
""")


def carry_tof_profile(src, dst, n: int, e, binmap) -> str | None:
    """Measure the scatter's TOF profile on the source and carry it across."""
    from lm import terms as lmterms

    if (src.work_bed(n) / "scatter_tof.npy").exists():
        return None
    try:
        w, note = lmterms.scatter_tof_weights(src, n, binmap, TOF_PROFILE_BINS, e)
    except SystemExit as exc:
        return f"NOT carried -- {str(exc).splitlines()[0]}"
    np.save(dst.work_bed(n) / sino_terms.TOF_PROFILE_FILE, np.asarray(w, np.float64))
    return note


def term_scales(hdr, f: float, mode: str, window: str):
    """`(randoms factor, scatter factor, frame factor)` for one bed."""
    if window == "uniform":
        return f ** thin.randoms_power(mode), f, 1.0
    lin, quad = thin.decay_scales(hdr["half_life_s"], hdr["frame_duration_ms"], f)
    return quad, lin, f


def bed(src, dst, n: int, e, mask, binmap, f: float, mode: str,
        sinogram: str = "derived", q=None, rng=None, window: str = "uniform"):
    """Write one bed of the thinned case."""
    from lm import events as ev
    from lm import interfile

    if sinogram not in SINOGRAM:
        raise ValueError(f"sinogram must be one of {SINOGRAM}, got {sinogram!r}")
    power = thin.randoms_power(mode)
    n_tof = interfile.Header(src.prompt(n)).n_tof
    raw, rawbed = dst.raw_sim, dst.raw_sim_bed(n)
    rawbed.mkdir(parents=True, exist_ok=True)
    dst.decoded.mkdir(parents=True, exist_ok=True)

    if sinogram == "derived":
        kept = e[mask] if mask is not None else e
        h, dropped = ev.histogram(kept, binmap, n_tof)
        np.save(raw / f"bed{n}.lm.npy", kept)
        link_into_place(raw / f"bed{n}.lm.npy", dst.decoded / f"bed{n}.lm.npy")
        events = int(len(kept))
    else:
        if q is None:
            raise ValueError("sinogram 'binomial' needs q, the keep probability "
                             "per plane or per (plane, tangential bin) -- a scalar "
                             "f will not do in low-dose mode, where it varies")
        y = np.fromfile(src.decoded / f"bed{n}.s", "<i2").reshape(
            -1, binmap.n_plane, binmap.n_view * binmap.n_tang)
        h, dropped, events = thin.binomial_sinogram(y, q, rng), 0, None

    h.tofile(raw / f"bed{n}.s")
    clone_header(src.prompt(n), raw / f"bed{n}.hs", f"bed{n}.s")
    for ext in (".hs", ".s"):
        link_into_place(raw / f"bed{n}{ext}", dst.decoded / f"bed{n}{ext}")

    hdr = src.header(n)
    r_fac, s_fac, fr_fac = term_scales(hdr, f, mode, window)
    hdr["prompts"] = int(h.sum(dtype=np.int64))
    hdr["delays"] = int(round(hdr["delays"] * r_fac))
    hdr["frame_duration_ms"] = hdr["frame_duration_ms"] * fr_fac
    (dst.decoded / f"bed{n}.json").write_text(json.dumps(hdr, indent=2))

    sw, dw = src.work_bed(n), dst.work_bed(n)
    dw.mkdir(parents=True, exist_ok=True)
    for stem in COPY:
        for ext in (".hs", ".s"):
            if (sw / (stem + ext)).exists():
                shutil.copy2(sw / (stem + ext), dw / (stem + ext))
    for name in COPY_FILES:
        if (sw / name).exists():
            shutil.copy2(sw / name, dw / name)

    clone_to_stir(sw, dw, hdr["prompts"], f, power)
    tof_note = carry_tof_profile(src, dst, n, e, binmap) if e is not None else None
    r = scale_term(sw, rawbed, "randoms", r_fac)
    s = scale_term(sw, rawbed, "scatter", s_fac)
    (np.fromfile(rawbed / "randoms.s", "<f4")
     + np.fromfile(rawbed / "scatter.s", "<f4")).tofile(rawbed / "background.s")
    clone_header(sw / "randoms.hs", rawbed / "background.hs", "background.s")
    for stem in SCALED:
        for ext in (".hs", ".s"):
            link_into_place(rawbed / f"{stem}{ext}", dw / f"{stem}{ext}")

    return {"bed": n, "events": events, "prompts": hdr["prompts"],
            "dropped": dropped, "randoms": r, "scatter": s,
            "randoms_scale": r_fac, "scatter_scale": s_fac,
            "frame_duration_ms": hdr["frame_duration_ms"],
            "scatter_tof_profile": tof_note}


def manifest(dst, src_name: str, f: float, mode: str, seed: int, rows: list,
             replicate=None, sinogram: str = "derived",
             rho_axis: str = "tangential", tof_rho: str = "model",
             window: str = "uniform") -> None:
    low_dose = thin.canonical(mode) == "low-dose"
    (dst.root / "lowdose.json").write_text(json.dumps({
        "source_case": src_name, "dose_fraction": f, "drf": 1.0 / f,
        "mode": thin.canonical(mode), "seed": seed, "k_scale": 1.0 / f,
        "randoms_power": thin.randoms_power(mode), "sinogram": sinogram,
        "window": window,
        "rho_axis": rho_axis if low_dose else None,
        "tof_rho": tof_rho if low_dose else None,
        "replicate": replicate, "beds": rows,
    }, indent=2))
