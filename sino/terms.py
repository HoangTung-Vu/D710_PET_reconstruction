from __future__ import annotations

import numpy as np

from utils import interfile
from utils.binmap import BinMap

DTYPES = {("signed integer", 2): "<i2", ("unsigned integer", 2): "<u2",
          ("signed integer", 4): "<i4", ("float", 4): "<f4"}

NEEDED = ("normdt", "attn", "background")


def read(hs) -> np.ndarray:
    h = interfile.Header(hs)
    h.require_plane_major()
    k = interfile.keys(hs)
    fmt = (k.get("number format", "float").lower(),
           int(k.get("number of bytes per pixel", 4)))
    if fmt not in DTYPES:
        raise SystemExit(f"error: {hs} stores {fmt[0]} x {fmt[1]} bytes; "
                         f"only {sorted(DTYPES)} are read here")
    a = np.fromfile(h.data_file(), DTYPES[fmt])
    shape = (h.n_tof, h.n_plane, h.n_view, h.n_tang)
    if a.size != int(np.prod(shape)):
        raise SystemExit(f"error: {h.data_file()} holds {a.size:,} values, "
                         f"{hs} describes {shape}")
    return a.reshape(shape)


def paths(case, bed: int) -> list:
    out = [case.prompt(bed)]
    for name in NEEDED:
        out.append(case.work_bed(bed) / f"{name}.hs")
    return out


def load(case, bed: int):
    missing = [p for p in paths(case, bed) if not p.exists()]
    if missing:
        how = (f"d710 attn --case {case.name}"
               if any(p.name == "attn.hs" for p in missing) else
               f"d710 exam --raw <...> --ct <...> --case {case.name}")
        raise SystemExit("error: bed %d is missing %s\n  run: %s"
                         % (bed, ", ".join(str(p) for p in missing), how))

    binmap = BinMap(case.prompt(bed))
    y = read(case.prompt(bed)).sum(axis=0, dtype=np.float32)
    got = {}
    for name in NEEDED:
        a = read(case.work_bed(bed) / f"{name}.hs")
        if a.shape[0] != 1:
            raise SystemExit(f"error: work/bed{bed}/{name}.hs has a TOF axis; "
                             f"`d710 sino` is non-TOF and reads non-TOF terms")
        got[name] = a[0].astype(np.float32, copy=False)
    for name, a in (("prompts", y), *got.items()):
        if a.shape != binmap.shape:
            raise SystemExit(f"error: bed {bed} {name} is {a.shape}, the bin map "
                             f"is {binmap.shape}")

    sens = got["normdt"] * got["attn"]
    live = sens > 0
    add = np.where(live, got["background"] / np.where(live, sens, 1.0), 0.0)
    y = np.where(live, y, 0.0)
    return binmap, y.astype(np.float32), sens.astype(np.float32), \
        add.astype(np.float32), int((~live).sum())
