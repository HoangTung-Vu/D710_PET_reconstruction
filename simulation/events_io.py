"""A simulated bed written as an ordinary case, beside the real one it copies.

The event table has the decoder's dtype -- `xtal_a`, `xtal_b` (GE crystal id
`ring * 576 + trans`), `tof_bin` (GE's signed bin, -27..27) and `t_ms` (the
scanner's millisecond clock, starting at the bed's `bed_start_ticks`). The
sinogram is histogrammed from those events by `lm.events.histogram`, which is
proven bit-exact against the decoder's own sinogram on every real bed
(`d710 lm check`), so a simulated and a real `bed<n>.s` compare bin for bin.

TOF sign. `tof_bin > 0` means the annihilation was nearer `xtal_a`: the
reconstruction hands parallelproj the signed bin `-tof_bin` with the LOR
running from `xtal_a` to `xtal_b` (PyTomography subtracts `(n - 1) // 2` from
`tof_to_stir(tof_bin) = 27 - tof_bin`), and parallelproj's positive bins run
towards the LOR's end -- measured, see `tests/test_simulation.py`.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

from utils import interfile
from utils.binmap import BinMap
from utils.paths import Case
from utils.scanner import C_MM_PS, N_TOF_RAW, TOF_RANGE_MM

EVENT_DTYPE = np.dtype([("xtal_a", "<u2"), ("xtal_b", "<u2"),
                        ("tof_bin", "i1"), ("t_ms", "<u4")])

TOF_HALF = N_TOF_RAW // 2

TOF_BIN_MM = TOF_RANGE_MM / N_TOF_RAW

C_MM_NS = C_MM_PS * 1e3

COPIED_TERMS = ("normdt", "norm_only")

TERM_TEMPLATE = "randoms"


def sim_case(real: Case, method: str, seed: int) -> Case:
    return Case(f"{real.name}_sim_{method}_s{seed}", real.root.parent)


def sim_root(real: Case) -> Path:
    """`$D710_OUT/<case>_sim/`: the phantom and the comparison, shared by both methods."""
    return real.root.parent / f"{real.name}_sim"


def events(xa, xb, tof_bin, t_ms) -> np.ndarray:
    e = np.empty(len(xa), EVENT_DTYPE)
    e["xtal_a"], e["xtal_b"] = xa, xb
    e["tof_bin"] = np.clip(np.asarray(tof_bin), -TOF_HALF, TOF_HALF)
    e["t_ms"] = t_ms
    return e


def tof_bin_from_dt(dt_ns) -> np.ndarray:
    """GE `tof_bin` of a pair from `dt = t_b - t_a`, the second single's time minus the first's."""
    return np.rint(np.asarray(dt_ns) * C_MM_NS / 2.0 / TOF_BIN_MM).astype(np.int64)


def swap_randomly(xa, xb, tof, rng):
    """Put the two crystals in either order, mirroring the TOF bin with them."""
    s = rng.random(len(xa)) < 0.5
    return (np.where(s, xb, xa), np.where(s, xa, xb), np.where(s, -tof, tof))


def clone_header(src_hs: Path, dst_hs: Path, data_name: str) -> None:
    txt = re.sub(r"(?im)^(\s*name of data file\s*:=).*$", r"\1 " + data_name,
                 Path(src_hs).read_text())
    Path(dst_hs).write_text(txt)


def write_term(real: Case, dst: Case, bed: int, name: str, arr) -> float:
    """A per-bin float32 term, header cloned from the real bed's randoms."""
    w = dst.work_bed(bed)
    w.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr, dtype="<f4")
    a.tofile(w / f"{name}.s")
    clone_header(real.work_bed(bed) / f"{TERM_TEMPLATE}.hs", w / f"{name}.hs", f"{name}.s")
    return float(a.sum(dtype=np.float64))


def scatter_tof_profile(ev, is_scatter, binmap: BinMap) -> np.ndarray:
    """`(55,)` TOF shape of the scatter-labelled events, summing to 1.

    Indexed as `lm` indexes the scatter's TOF weights (`lm.events.tof_index`):
    `27 + t`, with `t` the TOF bin oriented from det1 to det2 of the event's
    sinogram bin. Written as `work/bed<n>/scatter_tof_profile.npy`, the file
    `utils.terms.measured_tof_weights` reads, so `d710 lm recon` uses the
    simulation's own scatter TOF shape instead of measuring one from tails.
    """
    e = ev[np.asarray(is_scatter, bool)]
    b, swap = binmap.flat(e["xtal_a"], e["xtal_b"], with_swap=True)
    t = np.where(swap, -e["tof_bin"].astype(np.int64), e["tof_bin"].astype(np.int64))
    h = np.bincount(t[b >= 0] + TOF_HALF, minlength=N_TOF_RAW).astype(np.float64)
    h += 1e-6
    return h / h.sum()


def clone_to_stir(real: Case, dst: Case, bed: int, prompts: int) -> None:
    """The real bed's `to_stir.json`, describing the simulated bed.

    `d710 lm recon` reads the CT path from it (`utils.terms.ct_dir`). The
    real bed's statistics are dropped, as `lowdose` drops them.
    """
    p = real.work_bed(bed) / "to_stir.json"
    if not p.exists():
        return
    m = json.loads(p.read_text())
    m.pop("stats", None)
    m["verified"] = {**m.get("verified", {}), "prompts": prompts}
    m["simulated"] = True
    (dst.work_bed(bed) / "to_stir.json").write_text(json.dumps(m, indent=2, sort_keys=True))


def k_scale(real: Case, bed: int, sim_seconds: float) -> float:
    """`T_real / T_sim`: what `export` multiplies K by, as it does for `d710 lowdose`.

    The stitch's decay factor already refers each bed to the mean activity
    over its own frame, so counts differ only by the frame length.
    """
    return float(real.header(bed)["frame_duration_ms"]) / 1000.0 / float(sim_seconds)


def write_bed(real: Case, dst: Case, bed: int, ev: np.ndarray, terms: dict,
              header: dict, truth: dict | None = None, readme: str = "") -> dict:
    """Write one simulated bed; returns a summary row."""
    from lm.events import histogram

    dst.mkdirs()
    raw = dst.raw_sim
    raw.mkdir(parents=True, exist_ok=True)
    order = np.argsort(ev["t_ms"], kind="stable")
    ev = ev[order]
    np.save(dst.decoded / f"bed{bed}.lm.npy", ev)
    if truth:
        np.savez_compressed(raw / f"bed{bed}_truth.npz",
                            **{k: np.asarray(v)[order] for k, v in truth.items()})

    binmap = BinMap(real.prompt(bed))
    n_tof = interfile.Header(real.prompt(bed)).n_tof
    h, dropped = histogram(ev, binmap, n_tof)
    h.tofile(dst.decoded / f"bed{bed}.s")
    clone_header(real.prompt(bed), dst.prompt(bed), f"bed{bed}.s")

    hdr = real.header(bed)
    hdr.update(header)
    hdr["prompts"] = int(h.sum(dtype=np.int64))
    hdr["simulated"] = {**header.get("simulated", {}), "events": int(len(ev)),
                        "outside_sinogram": int(dropped)}
    (dst.decoded / f"bed{bed}.json").write_text(json.dumps(hdr, indent=2))

    w = dst.work_bed(bed)
    w.mkdir(parents=True, exist_ok=True)
    clone_to_stir(real, dst, bed, hdr["prompts"])
    for stem in COPIED_TERMS:
        for ext in (".hs", ".s"):
            p = real.work_bed(bed) / f"{stem}{ext}"
            if p.exists():
                shutil.copy2(p, w / p.name)
    sums = {k: write_term(real, dst, bed, k, v) for k, v in terms.items()
            if v is not None}
    if "randoms" in terms and "scatter" in terms:
        sums["background"] = write_term(real, dst, bed, "background",
                                        terms["randoms"] + terms["scatter"])
    if truth:
        sc = (truth["is_scatter"] if "is_scatter" in truth
              else np.asarray(truth["label"]) == 1)
        if np.any(sc):
            from utils.terms import TOF_PROFILE_FILE

            np.save(w / TOF_PROFILE_FILE,
                    scatter_tof_profile(ev, np.asarray(sc)[order], binmap))
    if readme:
        (raw / "README.txt").write_text(readme)
    return {"bed": bed, "events": int(len(ev)), "prompts": hdr["prompts"],
            "delays": hdr.get("delays"), "outside_sinogram": int(dropped),
            "k_scale": k_scale(real, bed, hdr["frame_duration_ms"] / 1000.0),
            "terms": sums}


def manifest(dst: Case, info: dict) -> None:
    p = dst.root / "simulation.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    beds = {str(r["bed"]): r for r in old.get("beds", [])}
    for r in info.pop("beds", []):
        beds[str(r["bed"])] = r
    old.update(info)
    old["beds"] = [beds[k] for k in sorted(beds, key=int)]
    ks = {round(float(r.get("k_scale", 0.0)), 9) for r in old["beds"]}
    old["k_scale"] = ks.pop() if len(ks) == 1 else None
    p.write_text(json.dumps(old, indent=2, default=float))
