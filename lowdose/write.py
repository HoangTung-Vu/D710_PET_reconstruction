"""Write a thinned exam as a complete, ordinary case.

The output is not a special format: it is a `$D710_OUT/<case>/` tree with the
same files `d710 exam` produces, so `d710 osem`, `d710 lm` and `d710 export` run
on it unchanged. What differs is only what physics says must:

    prompts     thinned event by event
    scatter     x f          (linear in activity)
    randoms     x f, or x f^2 in randoms-aware mode
    normdt      unchanged    -- sensitivity does not depend on dose
    attn        unchanged
    K           x 1/f        -- recorded in lowdose.json, applied by export

`to_stir.json` is rewritten rather than copied: see `clone_to_stir`.

Headers are **cloned** from the source, never regenerated: a fresh header
desynchronises ExamInfo and STIR only complains much later, inside
`make_Poisson_loglikelihood`.
"""

from __future__ import annotations

import json
import re
import shutil

import numpy as np

#: Copied as they are -- a sensitivity does not change with the dose.
COPY = ("normdt", "norm_only", "attn")

#: Copied because it is a *shape*, not an amplitude.
COPY_FILES = ("scatter_tof.npy",)


def clone_to_stir(src_dir, dst_dir, prompts: int, f: float, randoms_power: int) -> None:
    """`to_stir.json` describing THIS case, not the one it was thinned from.

    It has to travel (`utils.terms.ct_dir` and `d710_isolate_stir.sh` read
    `estimate.ct` out of it) but most of it is a measurement of the source: the
    copy would claim a bit-exactness proof for prompts this case does not have,
    and `stats` for terms that have since been scaled.
    """
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
    """`stem.{hs,s}` x `factor`. Element-wise, so the segment layout is irrelevant."""
    clone_header(src_dir / f"{stem}.hs", dst_dir / f"{stem}.hs", f"{stem}.s")
    a = np.fromfile(src_dir / f"{stem}.s", "<f4") * np.float32(factor)
    a.tofile(dst_dir / f"{stem}.s")
    return float(a.sum(dtype=np.float64))


def bed(src, dst, n: int, e, mask, binmap, f: float, randoms_power: int):
    """One bed. Returns the row of the summary table."""
    from lm import events as ev
    from lm import interfile

    n_tof = interfile.Header(src.prompt(n)).n_tof
    kept = e[mask] if mask is not None else e
    h, dropped = ev.histogram(kept, binmap, n_tof)

    dst.decoded.mkdir(parents=True, exist_ok=True)
    clone_header(src.prompt(n), dst.prompt(n), f"bed{n}.s")
    h.tofile(dst.decoded / f"bed{n}.s")
    np.save(dst.decoded / f"bed{n}.lm.npy", kept)

    hdr = src.header(n)
    hdr["prompts"] = int(h.sum(dtype=np.int64))
    hdr["delays"] = int(round(hdr["delays"] * f ** randoms_power))
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

    clone_to_stir(sw, dw, hdr["prompts"], f, randoms_power)
    r = scale_term(sw, dw, "randoms", f ** randoms_power)
    s = scale_term(sw, dw, "scatter", f)
    # b = randoms + scatter, rebuilt rather than scaled: the two halves no longer
    # share a factor once randoms go as f^2.
    clone_header(sw / "randoms.hs", dw / "background.hs", "background.s")
    (np.fromfile(dw / "randoms.s", "<f4")
     + np.fromfile(dw / "scatter.s", "<f4")).tofile(dw / "background.s")
    return {"bed": n, "events": int(len(kept)), "prompts": hdr["prompts"],
            "dropped": dropped, "randoms": r, "scatter": s}


def manifest(dst, src_name: str, f: float, mode: str, seed: int, rows: list,
             replicate=None) -> None:
    (dst.root / "lowdose.json").write_text(json.dumps({
        "source_case": src_name, "dose_fraction": f, "drf": 1.0 / f,
        "mode": mode, "seed": seed, "k_scale": 1.0 / f,
        "randoms_power": 2 if mode == "randoms" else 1,
        "replicate": replicate, "beds": rows,
    }, indent=2))
