from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from utils.paths import Case
from utils.scanner import NSEG0, PLANE_MM

from . import events_io as eio
from . import phantom as ph

TEMPLATE_BED = 4
BED_SPAN_MM = (NSEG0 - 1) * PLANE_MM
IDENTITY = ("patient_name", "patient_id", "patient_birth_date", "accession_number",
            "institution", "path", "patient_height_m")
LINKED = ("normdt", "norm_only")
HEADER_ONLY = ("randoms",)
KEPT_TO_STIR = ("mapping", "sensitivity_term", "background_term")


def table_positions(z0: float, z1: float, step: float) -> list[float]:
    first, last = float(z0), float(z1) - BED_SPAN_MM
    if last < first - 1e-6:
        raise SystemExit(f"error: the PET covers {z1 - z0:.1f} mm, less than one bed "
                         f"({BED_SPAN_MM:.1f} mm)")
    if last - first < 1e-6:
        return [first]
    n = math.ceil((last - first) / step - 1e-9) + 1
    return [first + k * (last - first) / (n - 1) for k in range(n)]


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_to_stir(template: Case, template_bed: int, w: Path, ct, table_position_mm: float) -> None:
    src = template.work_bed(template_bed) / "to_stir.json"
    m = json.loads(src.read_text()) if src.exists() else {}
    v = {k: m[k] for k in KEPT_TO_STIR if k in m}
    v["estimate"] = {"ct": str(ct), "table_position_mm": float(table_position_mm),
                     "norm": m.get("estimate", {}).get("norm")}
    v["virtual"] = {"template": template.name, "template_bed": template_bed}
    (w / "to_stir.json").write_text(json.dumps(v, indent=2, sort_keys=True))


def build(template: Case, name: str, ct, pet, pet_units: str, root: Path,
          template_bed: int = TEMPLATE_BED, margin_mm: float = ph.DEFAULT_MARGIN_MM,
          kvp: float | None = None, out=print) -> Case:
    V = Case(name, root)
    if any(V.decoded.glob("bed*.json")):
        raise SystemExit(f"error: {V.root} already holds a case; pick another --name")
    tbeds = template.decoded_beds()
    if template_bed not in tbeds:
        raise SystemExit(f"error: template {template.name} has no bed {template_bed}")
    for stem in LINKED + HEADER_ONLY:
        if not (template.work_bed(template_bed) / f"{stem}.hs").exists():
            raise SystemExit(f"error: template bed {template_bed} has no {stem}.hs\n"
                             f"  run: d710 tostir --case {template.name} --bed {template_bed}")
    hdrs = [template.header(b) for b in tbeds]
    tps = sorted(float(h["table_position_mm"]) for h in hdrs)
    starts = sorted(float(h["bed_start_time"]) for h in hdrs)
    step = float(np.median(np.diff(tps))) if len(tps) > 1 else BED_SPAN_MM
    dt = float(np.median(np.diff(starts))) if len(starts) > 1 else 0.0
    vol = ph.load_volume(pet, "PT")
    z = np.asarray(vol.z, np.float64)
    pos = table_positions(z.min(), z.max(), step)
    th = template.header(template_bed)
    V.mkdirs()
    for n, tp in enumerate(pos, start=1):
        h = {k: v for k, v in th.items() if k not in IDENTITY}
        h.update(bed_number=n, bed_index=n - 1, table_position_mm=float(tp),
                 bed_start_time=starts[0] + (n - 1) * dt,
                 bed_start_ticks=int(th.get("bed_start_ticks", 0)) + int(round((n - 1) * dt * 1000)),
                 prompts=0, delays=0,
                 virtual={"template": template.name, "template_bed": template_bed})
        (V.decoded / f"bed{n}.json").write_text(json.dumps(h, indent=2))
        eio.clone_header(template.prompt(template_bed), V.prompt(n), f"bed{n}.s")
        w = V.work_bed(n)
        w.mkdir(parents=True, exist_ok=True)
        src = template.work_bed(template_bed)
        for stem in LINKED:
            for ext in (".hs", ".s"):
                _link(src / f"{stem}{ext}", w / f"{stem}{ext}")
        for stem in HEADER_ONLY:
            eio.clone_header(src / f"{stem}.hs", w / f"{stem}.hs", f"{stem}.s")
        write_to_stir(template, template_bed, w, ct, tp)
    info = {"template": template.name, "template_bed": template_bed, "ct": str(ct),
            "pet": str(pet), "pet_units": pet_units, "pet_z_mm": [float(z.min()), float(z.max())],
            "bed_step_mm": step, "table_positions_mm": [float(p) for p in pos]}
    (V.root / "virtual.json").write_text(json.dumps(info, indent=2))
    for n in range(1, len(pos) + 1):
        out(f"phantom bed {n}: table {pos[n - 1]:.2f} mm")
        ph.build(V, n, ct, pet, pet_units, ph.directory(eio.sim_root(V), n),
                 margin_mm=margin_mm, kvp=kvp, out=out)
    return V
