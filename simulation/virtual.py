from __future__ import annotations

import datetime as dt
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
EXAM_KEYS = ("dose_mbq", "residual_dose_mbq", "patient_weight_kg", "half_life_s",
             "radiopharm_start_datetime")
MANIFEST_KEYS = ("radionuclide_total_dose_Bq", "patient_weight_kg", "radionuclide_half_life_s",
                 "injection_time", "scan_time")
MAX_UPTAKE_S = 6 * 3600


def exam_from_manifest(path: Path, utc_offset_h: float = 7.0) -> dict:
    p = Path(path)
    s = json.loads(p.read_text()).get("SUV") or {}
    missing = [k for k in MANIFEST_KEYS if s.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"error: {p} has no SUV.{', SUV.'.join(missing)}")
    if str(s.get("decay_correction", "START")).upper() != "START":
        raise SystemExit(f"error: {p} is decay-corrected to {s['decay_correction']!r}, not START")
    tz = dt.timezone(dt.timedelta(hours=float(utc_offset_h)))

    def utc(stamp: str) -> dt.datetime:
        t = dt.datetime.fromisoformat(str(stamp))
        return (t if t.tzinfo else t.replace(tzinfo=tz)).astimezone(dt.timezone.utc)

    t_inj, t_scan = utc(s["injection_time"]), utc(s["scan_time"])
    uptake = (t_scan - t_inj).total_seconds()
    if not 0 < uptake < MAX_UPTAKE_S:
        raise SystemExit(f"error: {p}: uptake {uptake / 60:.1f} min between injection and scan")
    dose = float(s["radionuclide_total_dose_Bq"])
    half = float(s["radionuclide_half_life_s"])
    decayed = dose * 2.0 ** (-uptake / half)
    if s.get("decayed_dose_Bq") and abs(decayed / float(s["decayed_dose_Bq"]) - 1.0) > 1e-4:
        raise SystemExit(f"error: {p}: decayed dose {decayed:.6g} Bq, the manifest says "
                         f"{float(s['decayed_dose_Bq']):.6g}")
    return {"dose_mbq": dose / 1e6, "residual_dose_mbq": 0.0,
            "patient_weight_kg": float(s["patient_weight_kg"]), "half_life_s": half,
            "radiopharm_start_datetime": t_inj.strftime("%Y%m%d%H%M%S.00"),
            "bed_start_time": t_scan.timestamp(), "uptake_min": uptake / 60.0,
            "bqml_per_suv": decayed / (float(s["patient_weight_kg"]) * 1000.0),
            "source": str(p), "utc_offset_h": float(utc_offset_h)}


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
          kvp: float | None = None, exam: dict | None = None, out=print) -> Case:
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
    step_s = float(np.median(np.diff(starts))) if len(starts) > 1 else 0.0
    t0 = float(exam["bed_start_time"]) if exam else starts[0]
    vol = ph.load_volume(pet, "PT")
    z = np.asarray(vol.z, np.float64)
    pos = table_positions(z.min(), z.max(), step)
    th = template.header(template_bed)
    V.mkdirs()
    for n, tp in enumerate(pos, start=1):
        h = {k: v for k, v in th.items() if k not in IDENTITY}
        h.update(bed_number=n, bed_index=n - 1, table_position_mm=float(tp),
                 bed_start_time=t0 + (n - 1) * step_s,
                 bed_start_ticks=int(th.get("bed_start_ticks", 0)) + int(round((n - 1) * step_s * 1000)),
                 prompts=0, delays=0,
                 virtual={"template": template.name, "template_bed": template_bed})
        if exam:
            h.update({k: exam[k] for k in EXAM_KEYS})
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
            "bed_step_mm": step, "table_positions_mm": [float(p) for p in pos],
            "exam": exam}
    if exam:
        out(f"exam from {exam['source']}: {exam['dose_mbq']:.2f} MBq, "
            f"{exam['patient_weight_kg']:g} kg, uptake {exam['uptake_min']:.1f} min, "
            f"{exam['bqml_per_suv']:.1f} Bq/mL per SUV")
    (V.root / "virtual.json").write_text(json.dumps(info, indent=2))
    for n in range(1, len(pos) + 1):
        out(f"phantom bed {n}: table {pos[n - 1]:.2f} mm")
        ph.build(V, n, ct, pet, pet_units, ph.directory(eio.sim_root(V), n),
                 margin_mm=margin_mm, kvp=kvp, out=out)
    return V
